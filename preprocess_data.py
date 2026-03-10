from src.utils import (
    balance_data_distribution,
    get_class_distribution,
    gether_images_and_masks,
    image_to_tensor,
)


def main() -> None:
    origin_dataset_path = input("Paste origin dataset folder path: ").strip()
    output_dataset_path = input("Paste output dataset folder path: ").strip()

    if not origin_dataset_path or not output_dataset_path:
        print("Both origin and output paths are required.")
        return

    gether_images_and_masks(origin_dataset_path, output_dataset_path)
    image_to_tensor(output_dataset_path)
    balance_data_distribution(output_dataset_path)

    distribution = get_class_distribution(output_dataset_path)
    print("Class distribution:")
    for class_name, count in distribution.items():
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()