import argparse
import imghdr
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

ROOT = "labels"  # JSON root

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


@dataclass
class Choice:
    name: str
    labels: List["Label"]

    def __init__(self, choice):
        if isinstance(choice, dict):
            self.name = choice["choice_name"]
            self.labels = (
                [Label(l) for l in choice["labels"]]
                if "labels" in choice.keys()
                else None
            )

        elif isinstance(choice, str):
            self.name = choice
            self.labels = None

    def __str__(self):
        return "Choice " + self.name


@dataclass
class Label:
    type: str
    name: str
    choices: List["Choice"]

    def __init__(self, label_dict: dict):
        self.name = label_dict["label_name"]
        type = label_dict.get("type", "leaf")
        assert type in [
            "categorical",
            "boolean",
            "leaf",
        ], f"Label type {type} not supported."
        self.type = type

        self.choices = (
            [Choice(c) for c in label_dict["choices"]]
            if "choices" in label_dict.keys()
            else None
        )
        if self.type == "boolean" and not self.choices:
            self.choices = [Choice("true"), Choice("false")]

    def __str__(self):
        return "Label " + self.name


def main(images_directory, csv_filename, json_labels_filename):

    print(" - Image Labelling Tool - ")
    print(f"Source Directory: {images_directory}")

    assert os.path.exists(json_labels_filename) and os.path.isfile(json_labels_filename)

    with open(json_labels_filename, "r") as f:
        labels = json.load(f)

    try:
        dataset = pd.read_csv(csv_filename)
        print(f"Found {csv_filename}, resuming operation.")

    except FileNotFoundError:
        print(f"File {csv_filename} not found. Created a new one.")
        dataset = pd.DataFrame()

    # Count number of valid images in `images_directory` folder
    # To handle more image formats you need Pillow installed
    total_images_filenames = [
        f
        for f in os.listdir(images_directory)
        if imghdr.what(os.path.join(images_directory, f)) == "png"
    ]

    total_images = len(total_images_filenames)

    try:
        remaining_images_filenames = set(total_images_filenames) - set(
            dataset["filename"].values
        )
    except KeyError:
        remaining_images_filenames = list(total_images_filenames)

    remaining_images = len(remaining_images_filenames)

    print(f"{remaining_images}/{total_images}")

    user_name = input("Please insert your name: ")

    fig, ax = plt.subplots(1, 1)

    for i, filename in enumerate(remaining_images_filenames):

        filename_absolute = os.path.join(images_directory, filename)

        # Show the image
        image = mpimg.imread(filename_absolute)
        im = ax.imshow(image)
        im.set_data(image)
        fig.canvas.draw_idle()
        plt.show(block=False)

        results = OrderedDict({"user_name": user_name, "filename": filename})

        label_list = [Label(l) for l in labels[ROOT]]

        while label_list:
            label = label_list.pop()

            print(f"{GREEN}\t{label.name}{RESET}")

            for i, c in enumerate(label.choices):
                print(f"[{i}]\t{c.name}")

            print("[-1]\t-- Can't be assessed --")

            choice_i = -2
            while choice_i not in range(-1, len(label.choices)):
                try:
                    choice_i = int(input("Insert corresponding number: "))
                except ValueError as e:
                    print(e)

            print()
            if choice_i == -1:
                continue

            choice = label.choices[choice_i]

            if label.type == "categorical":
                results[label.name] = choice.name

            elif label.type == "boolean":
                results[label.name] = choice_i

            if choice.labels:
                label_list.extend(choice.labels)

        dataset = dataset.append(results, ignore_index=True)
        dataset.to_csv(csv_filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image labelling tool")
    parser.add_argument("images_directory", type=str, help="Input images directory")
    parser.add_argument(
        "json_labels_filename", type=str, help="Json with labels structure"
    )
    parser.add_argument("csv_filename", type=str, help="Filename of labels csv")

    args = parser.parse_args()
    images_directory = args.images_directory
    csv_filename = args.csv_filename
    json_labels_filename = args.json_labels_filename

    main(images_directory, csv_filename, json_labels_filename)
