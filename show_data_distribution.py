import json
import os
import matplotlib.pyplot as plt
import pandas as pd

# This dictionary is based on the one in src/dataloader.py
CLASS_ID_TO_NAME = {
    0: "call",
    1: "dislike",
    2: "like",
    3: "ok",
    4: "one",
    5: "palm",
    6: "peace",
    7: "rock",
    8: "stop",
    9: "three",
}

def get_class_distribution(dataset_path):
    """
    Calculates the class distribution for a given dataset.

    Args:
        dataset_path (str): The path to the dataset directory, which should
                            contain an 'image_info.json' file.

    Returns:
        pandas.Series: A pandas Series with class names as the index and
                       their counts as values. Returns an empty Series
                       if the JSON file is not found.
    """
    image_info_path = os.path.join(dataset_path, "image_info.json")
    if not os.path.exists(image_info_path):
        print(f"Warning: 'image_info.json' not found in {dataset_path}")
        return pd.Series(dtype=int)

    with open(image_info_path, "r") as f:
        image_info = json.load(f)

    class_ids = [item["class_id"] for item in image_info]
    class_names = [CLASS_ID_TO_NAME[cid] for cid in class_ids]
    
    return pd.Series(class_names).value_counts().sort_index()

def plot_distribution(distribution, title, output_filename):
    """
    Plots and saves a bar chart of the class distribution.

    Args:
        distribution (pandas.Series): A pandas Series with class distribution data.
        title (str): The title for the plot.
        output_filename (str): The path to save the output plot image.
    """
    if distribution.empty:
        print(f"Skipping plot for '{title}' due to empty distribution.")
        return
        
    plt.figure(figsize=(12, 7))
    distribution.plot(kind='bar')
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    plt.savefig(output_filename)
    plt.close()
    print(f"Saved plot to {output_filename}")

def main():
    """
    Main function to analyze and visualize dataset distributions.
    """
    base_dir = "dataset/dataset_v1"
    output_dir = "outputs/data_distribution_analysis"

    paths = {
        "train": os.path.join(base_dir, "train"),
        "val": os.path.join(base_dir, "val"),
        "test": os.path.join(base_dir, "test"),
    }

    for split, path in paths.items():
        print(f"Analyzing {split} dataset at: {path}")
        distribution = get_class_distribution(path)
        
        if not distribution.empty:
            print(f"\n--- {split.capitalize()} Distribution ---")
            print(distribution)
            print("---------------------------------\n")
            
            plot_filename = os.path.join(output_dir, f"{split}_distribution.png")
            plot_distribution(distribution, f"{split.capitalize()} Set Class Distribution", plot_filename)

if __name__ == "__main__":
    main()
