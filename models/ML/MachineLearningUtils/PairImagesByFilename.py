import os

def pair_images_by_filename(base_rgb_dir: str, base_disease_dir: str, base_leaf_dir: str) -> list[tuple[str, str, str, str]]:
    paired_images = []
    for disease in os.listdir(base_rgb_dir):
        rgb_dir = os.path.join(base_rgb_dir, disease)
        disease_dir = os.path.join(base_disease_dir, disease)
        leaf_dir = os.path.join(base_leaf_dir, disease)

        if not os.path.isdir(rgb_dir) or not os.path.isdir(disease_dir) or not os.path.isdir(leaf_dir):
            print(
                f"One of the directories is invalid: {rgb_dir}, {disease_dir}, {leaf_dir}")
            continue

        for file_name in os.listdir(rgb_dir):
            if file_name.endswith('.png'):
                rgb_path = os.path.join(rgb_dir, file_name)
                disease_path = os.path.join(disease_dir, file_name)
                leaf_path = os.path.join(leaf_dir, file_name)

                if os.path.exists(rgb_path) and os.path.exists(disease_path) and os.path.exists(leaf_path):
                    paired_images.append(
                        (rgb_path, leaf_path, disease_path, disease))
                else:
                    print(f"Missing image for {file_name} in {disease}")

    return paired_images