import copy
import functools
import os
from typing import List

import blobfile as bf
import numpy as np
import torch
from torch.optim import AdamW
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter
import tqdm

from . import CONSTANTS
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossSecondMomentResampler, UniformSampler

DEVICE_ID = CONSTANTS.DEVICE_ID
INITIAL_LOG_LOSS_SCALE = CONSTANTS.INITIAL_LOG_LOSS_SCALE


class TrainLoop:

    def __init__(
            self,
            *,
            model,
            diffusion,
            data_train: torch.utils.data.DataLoader,
            data_val: torch.utils.data.DataLoader,
            num_epochs: int,
            batch_size: int,
            microbatch: int,
            lr: float,
            ema_rate,
            save_interval: int,
            resume_checkpoint: str,
            logdir: str,
            writer: SummaryWriter,
            use_fp16: bool = False,
            fp16_scale_growth: float = 1e-3,
            schedule_sampler=None,
            weight_decay: float = 0.0,
            lr_anneal_steps: int = 0,
            compute_val: bool = False,
            subset_timesteps: int = None
            ) -> None:
        '''
        Inputs:
        ------
            model (UNetModel): UNetModel

            diffusion (_WrappedModel): diffusion model

            data_train (torch.utils.data.DataLoader): dataloader

            data_val (torch.utils.data.DataLoader): dataloader

            num_epochs (int): the number of training epochs

            batch_size (int): training batch size

            microbatch (int): microbatch size; if -1 then batch_size is used

            lr (float): initial learning rate

            ema_rate (float/str): either a float or a comma-separated list of
                rates to compute smoothed version(s) of model using an
                Exponential Moving Average (EMA).
                (e.g., https://pytorch.org/ignite/generated/ignite.handlers.ema_handler.EMAHandler.html)

            save_interval (int): interval at which to save model weights to .pt file

            resume_checkpoint (str): path to model to resume training from,
                if desired

            logdir (str): path to model folder

            writer (SummaryWriter): tensorboard Writer object

            use_fp16 (bool): whether to convert model weights to 16-bytes
                during training.

            fp16_scale_growth (float):

            schedule_sampler (str): defines the ScheduleSampler to sample
                diffusion timesteps/weights

            weight_decay (float): weight decay coefficient in AdamW method

            lr_anneal_steps (int): step decay for learning rate annealing

            compute_val (bool): if True, compute the validation loss at
                each epoch (can be very long!).

            subset_timesteps (int): if compute_val is True and subset_timesteps
                is not None, calculate the validation loss on the last
                'subset_timesteps' diffusion time steps
                (i.e. {0, ..., subset_timesteps-1})
        '''
        if not os.path.exists(logdir):
            raise ValueError('The log folder does not exist')
        self.logdir = logdir

        self.model = model
        self.diffusion = diffusion
        self.data_train = data_train
        self.data_val = data_val
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.compute_val = compute_val
        self.subset_timesteps = subset_timesteps

        self.writer = writer

        # Print training loss every 100 batches
        self.print_train_loss = 100

        self.step = 0  # number of iterations within epoch
        self.step_total = 0  # total number of iterations
        self.resume_step = 0
        self.epoch_index = 0
        self.resume_epoch_index = 0

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = torch.cuda.is_available()

        # Load model parameters if resume_checkpoint is provided
        self._load_and_sync_parameters()

        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(
            self.master_params,
            lr=self.lr,
            weight_decay=self.weight_decay)

        if self.resume_step:
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.step_total = self.resume_step
            self.epoch_index = self.resume_epoch_index + 1
            self.num_epochs += self.resume_epoch_index
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if torch.cuda.is_available():
            self.ddp_model = self.model.to(DEVICE_ID)
        else:
            self.ddp_model = self.model

    def _load_and_sync_parameters(self) -> None:
        """
        Loads and synchronises model parameters if resume_checkpoint
        """
        if self.resume_checkpoint is not None:

            if not os.path.exists(self.resume_checkpoint):
                raise ValueError(f'Model {self.resume_checkpoint} does not exist.')

            # Retrieve last epoch and number of steps
            self.resume_epoch_index, self.resume_step =\
                parse_resume_step_from_filename(self.resume_checkpoint)

            print(f"Loading model from checkpoint: {self.resume_checkpoint}.")
            params = torch.load(self.resume_checkpoint, map_location=DEVICE_ID)
            try:
                self.model.load_state_dict(params)
            except Exception:
                raise ValueError('Error with parameter match! Please check.')

    def _load_ema_parameters(
            self,
            rate) -> List[Parameter]:
        '''
        Loads ema model from a "resume_checkpoint" model path.

        From the url of the model path/to/modelNNNNNN.pt,
        the method builds the url of the ema model:
        path/to/ema_{rate}_NNNNNN.pt
        '''
        # Initialise ema_params
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(
            main_checkpoint,
            self.resume_epoch_index,
            rate)

        if not os.path.exists(ema_checkpoint) or ema_checkpoint is None:
            raise ValueError(f'The EMA url {ema_checkpoint} does not exist')

        print(f"loading EMA from checkpoint: {ema_checkpoint}...")
        state_dict = torch.load(self.resume_checkpoint, map_location=DEVICE_ID)
        ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_epoch_index}.pt"
        )

        if not bf.exists(opt_checkpoint):
            raise ValueError(f'The opt url {opt_checkpoint} does not exist')

        print(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        state_dict = torch.load(opt_checkpoint, map_location=DEVICE_ID)
        self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def calc_val_loss(self) -> tuple:
        '''
        Calculates validation loss on the validation dataset.
        Calculates the full VB loss, unlike in training (which
        relies on importance sampling to calculate the loss).
        '''
        # Initialise lists of metrics
        all_bpd, all_xstart_mse = [], []
        # Loop on validation dataset
        for (batch_val, model_kwargs) in tqdm.tqdm(self.data_val):

            batch_val = batch_val.to(DEVICE_ID)
            model_kwargs = {k: v.to(DEVICE_ID) for k, v in model_kwargs.items()}

            # Calculate loss metrics on this batch
            minibatch_metrics = self.diffusion.calc_bpd_loop(
                self.model,
                batch_val,
                clip_denoised=True,
                subset_timesteps=self.subset_timesteps,
                model_kwargs=model_kwargs
            )

            # Average metrics across the batch
            # VB
            total_bpd = minibatch_metrics["total_bpd"].mean()
            all_bpd.append(total_bpd.item())
            # xstart_MSE
            xstart_mse = minibatch_metrics["xstart_mse"].mean()
            all_xstart_mse.append(xstart_mse.item())

        return np.mean(np.array(all_bpd)), np.mean(np.array(all_xstart_mse))

    def run_loop(self):
        '''
        Defines the training loop to run the model
        '''
        while (
            self.epoch_index < self.num_epochs
            and (
                not self.lr_anneal_steps
                or self.step_total < self.lr_anneal_steps
            )
        ):
            print(f"Epoch {self.epoch_index}...")

            # Enumerate on the train dataloader
            for (batch, cond) in self.data_train:
                self.run_step(batch, cond)
                if self.save_interval is not None:
                    if self.step % self.save_interval == 0:
                        self.save_within_epoch()

            # Calculate validation loss
            if self.compute_val:
                vb_val_loss, mse_val_loss = self.calc_val_loss()
                print(f"Epoch {self.epoch_index}: val loss={np.round(vb_val_loss, 5)}")

                if self.epoch_index == 0:
                    best_score = vb_val_loss * 1.01

                # Save model
                if vb_val_loss < best_score:
                    best_score = vb_val_loss
                    print("Saving model...")
                    self.save(vb_val_loss, mse_val_loss)
                    # Run for a finite amount of time in integration tests.
                    if (os.environ.get("DIFFUSION_TRAINING_TEST", "")
                            and self.epoch_index > 0):
                        return
            else:
                self.save(None, None)
                # Run for a finite amount of time in integration tests.
                if (os.environ.get("DIFFUSION_TRAINING_TEST", "")
                        and self.epoch_index > 0):
                    return

            self.epoch_index += 1
            self.step = 0

        # Save the last checkpoint if it wasn't already saved.
        if self.epoch_index == self.num_epochs:
            if self.compute_val:
                self.save(vb_val_loss, mse_val_loss)
            else:
                self.save(None, None)

    def run_step(
            self,
            batch: torch.Tensor,
            cond: dict):
        '''
        Calculate loss and update model weights on a single batch,
        within an epoch.
        '''
        self.forward_backward(batch, cond)
        # Update learning rate and write stuff to tensorboard
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        # Write more stuff to tensorboard
        self.log_step()
        # Update model step
        self.step += 1
        self.step_total += 1

    def _log_loss_dict(
            self,
            diffusion,
            ts: torch.Tensor,
            losses: dict):
        '''
        Writes loss terms to tensorboard writer

        Inputs:
        ------
            diffusion (SpacedDiffusion): diffusion model.

            ts (torch.Tensor): sample of diffusion timesteps

            losses (dict): dictionary of model losses, with keys 'vb', 'mse'
                and 'loss' if diffusion.loss_type in [LossType.KL,
                LossType.RESCALED_KL], else 'loss'
        '''

        for key, values in losses.items():
            self.writer.add_scalar(key, values.mean().item(), self.step_total)
            self.writer.flush()
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                self.writer.add_scalar(f"{key}_q{quartile}", sub_loss, self.step_total)
                self.writer.flush()

    def forward_backward(
            self,
            batch: torch.Tensor,
            cond: dict
            ) -> None:
        '''
        Calculates model loss and updates model weights.

        Inputs:
        ------
            batch, cond are the outputs of the dataloader.
            NOTE batch is a tensor of shape (batch_size, num_channels, H, W)
        '''
        # Not sure why we need to do this...
        zero_grad(self.model_params)

        batch_loss = 0.0
        # Divide the batch into microbatches
        for i in range(0, batch.shape[0], self.microbatch):

            # Define microbatch
            micro = batch[i: i + self.microbatch].to(DEVICE_ID)
            micro_cond = {
                k: v[i: i + self.microbatch].to(DEVICE_ID)
                for k, v in cond.items()
            }

            # Sample diffusion time steps using self.schedule_sampler
            # Gives more weight to time steps with large loss
            t, weights = self.schedule_sampler.sample(
                micro.shape[0], DEVICE_ID)

            # Loss calculation
            # partial function which sets the value of several
            # arguments of self.diffusion.training_losses()
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,  # UNetModel
                micro,  # batch of input images
                t,  # batch of diffusion time steps
                model_kwargs=micro_cond,  # batch of labels for micro
            )

            losses = compute_losses()

            # If schedule_sampler uses importance sampling,
            # update past model losses.
            if isinstance(self.schedule_sampler, LossSecondMomentResampler):
                # self.schedule_sampler.update_with_local_losses(
                self.schedule_sampler.update_with_all_losses(
                    t, losses["loss"].detach()
                )

            # Average model losses across the batch using the weights
            loss = (losses["loss"] * weights).mean()
            batch_loss += loss.item()

            # Write loss terms to tensorboard writer
            self._log_loss_dict(
                self.diffusion,
                t,
                {k: v * weights for k, v in losses.items()}
            )

            # Update model weights
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

        # Print model loss
        if (self.step > 0) and (self.step % self.print_train_loss == 0):
            print(f'Batch {self.step}: Loss = {np.round(batch_loss, 5)}')

    def optimize_normal(self) -> None:
        # Write to tensorboard
        self._log_grad_norm()
        # Update learning rate
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def optimize_fp16(self) -> None:
        if any(not torch.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            print(f"Found NaN,decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def _log_grad_norm(self) -> None:
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        self.writer.add_scalar('grad_norm', np.sqrt(sqsum), self.step_total)
        self.writer.flush()

    def _anneal_lr(self) -> None:
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step_total) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self) -> None:
        self.writer.add_scalar('step', self.step_total, self.step_total)
        self.writer.flush()
        self.writer.add_scalar('samples', (self.step_total + 1) * self.batch_size, self.step_total)
        self.writer.flush()  # Write to disk immediately
        if self.use_fp16:
            self.writer.add_scalar('lg_loss_scale', self.lg_loss_scale, self.step_total)
            self.writer.flush()

    def save_within_epoch(
            self
            ) -> None:
        '''
        Saves the model as .pt file during an epoch
        '''

        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)

            print(f"Saving model {rate}...")
            if not rate:
                filename = f"model_epoch={(self.epoch_index)}_step={self.step}.pt"
            else:
                filename = f"ema_rate={rate}_{(self.epoch_index)}_step={self.step}.pt"
            with bf.BlobFile(bf.join(self.logdir, filename), "wb") as f:
                torch.save(state_dict, f)

        # master_params -> model parameters
        save_checkpoint(0, self.master_params)
        # ema_params -> EMA model parameters (i.e. with EMA smoothing)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

    def save(
            self,
            vb_val_loss: float = None,
            mse_val_loss: float = None
            ) -> None:
        '''
        Saves the model as .pt file
        '''

        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)

            print(f"Saving model {rate}...")
            if not rate:
                if vb_val_loss is None or mse_val_loss is None:
                    filename = f"model_epoch={(self.epoch_index)}_iter={self.step_total}.pt"
                else:
                    filename = f"model_epoch={(self.epoch_index)}_iter={self.step_total}_vb={np.round(vb_val_loss, 4)}_mse={np.round(mse_val_loss, 4)}.pt"
            else:
                if vb_val_loss is None or mse_val_loss is None:
                    filename = f"ema_rate={rate}_{(self.epoch_index)}.pt"
                else:
                    filename = (f"ema_rate={rate}_{(self.epoch_index)}_vb"
                                f"={np.round(vb_val_loss, 4)}_mse"
                                f"={np.round(mse_val_loss, 4)}.pt")
            with bf.BlobFile(bf.join(self.logdir, filename), "wb") as f:
                torch.save(state_dict, f)

        # master_params -> model parameters
        save_checkpoint(0, self.master_params)
        # ema_params -> EMA model parameters (i.e. with EMA smoothing)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if vb_val_loss is None or mse_val_loss is None:
            with bf.BlobFile(
                bf.join(self.logdir, f"opt{(self.epoch_index)}.pt"),
                "wb",
            ) as f:
                torch.save(self.opt.state_dict(), f)
        else:
            with bf.BlobFile(
                bf.join(self.logdir, f"opt{(self.epoch_index)}_vb={np.round(vb_val_loss, 4)}_mse={np.round(mse_val_loss, 4)}.pt"),
                "wb",
            ) as f:
                torch.save(self.opt.state_dict(), f)

    def _master_params_to_state_dict(
            self,
            master_params
            ) -> dict:
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(
            self,
            state_dict: dict
            ) -> list:
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        return make_master_params(params) if self.use_fp16 else params


def parse_resume_step_from_filename(
        filename: str):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    num_resume_epoch = filename.split("epoch=")[-1].split('_')[0]
    num_resume_iter = filename.split("iter=")[-1].split("_")[0].split('.p')[0]
    try:
        return int(num_resume_epoch), int(num_resume_iter)
    except ValueError:
        return 0


def find_ema_checkpoint(
        main_checkpoint: str,
        epoch_index,
        rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_rate={rate}_{epoch_index}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    return path if bf.exists(path) else None
