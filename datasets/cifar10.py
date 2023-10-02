import os
import shutil
import tempfile

import torchvision
from tqdm.auto import tqdm

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def main(
        data_folder='data/',
        pct_val: float = 0.1
        ) -> None:
    """
    Download the cifar10 dataset, split it into train/, val/ and test/ folders,
    and store them in './data/'.
    """
    for split in ["train", "test"]:
        out_dir = f"{data_folder}/cifar_{split}"
        if os.path.exists(out_dir):
            print(f"skipping split {split} since {out_dir} already exists.")
            continue

        print("Downloading...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.CIFAR10(
                root=tmp_dir, train=split == "train", download=True
            )

        print("Dumping images...")
        os.mkdir(out_dir)
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            filename = os.path.join(out_dir, f"{CLASSES[label]}_{i:05d}.png")
            image.save(filename)

    # Put aside some validation data from the test data
    out_dir = f"{data_folder}/cifar_val"
    if os.path.exists(out_dir):
        print(f"Skipping split val since {out_dir} already exists.")
        return

    print("Dumping images...")
    os.mkdir(out_dir)
    l_test_files = os.listdir(f"{data_folder}/cifar_test/")
    l_test_files = l_test_files[:int(len(l_test_files) * pct_val)]
    for url in l_test_files:
        shutil.move(f"{data_folder}/cifar_test/{url}",
                    f"{out_dir}/{url}")


if __name__ == "__main__":

    main()
