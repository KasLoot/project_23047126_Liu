import os
from src.utils import get_class_distribution, plot_distribution

def main():
    base_dir = "dataset/dataset_v1"
    output_dir = "results/data_distribution_analysis"

    paths = {
        "train": os.path.join(base_dir, "train"),
        "val": os.path.join(base_dir, "val"),
        "test": os.path.join(base_dir, "test"),
    }

    for split, path in paths.items():
        print(f"Analyzing {split} dataset at: {path}")
        distribution = get_class_distribution(path)

        if distribution:
            print(f"\n--- {split.capitalize()} Distribution ---")
            print(distribution)
            print("---------------------------------\n")

            plot_filename = os.path.join(output_dir, f"{split}_distribution.png")
            plot_distribution(distribution, f"{split.capitalize()} Set Class Distribution", plot_filename)

if __name__ == "__main__":
    main()
