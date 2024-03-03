import os

def pair_images_by_filename(base_rgb_dir: str, base_disease_dir: str, base_leaf_dir: str) -> list[tuple[str, str, str, str]]:
    paired_images = []
    for disease_name in os.listdir(base_rgb_dir):
        rgb_dir = os.path.join(base_rgb_dir, disease_name)
        disease_dir = os.path.join(base_disease_dir, disease_name)
        leaf_dir = os.path.join(base_leaf_dir, disease_name)

        # Check if directories for RGB, disease, and leaf images exist
        if not os.path.isdir(rgb_dir) or not os.path.isdir(disease_dir) or not os.path.isdir(leaf_dir):
            print(f"One of the directories is invalid: {rgb_dir}, {disease_dir}, {leaf_dir}")
            continue

        # Iterate over RGB images and match with corresponding disease and leaf images
        for file_name in os.listdir(rgb_dir):
            if file_name.endswith('.png'):
                rgb_path = os.path.join(rgb_dir, file_name)
                disease_path = os.path.join(disease_dir, file_name)
                leaf_path = os.path.join(leaf_dir, file_name)

                # Ensure paths for RGB, disease, and leaf images exist before adding
                if os.path.exists(rgb_path) and os.path.exists(disease_path) and os.path.exists(leaf_path):
                    paired_images.append(
                        (rgb_path, leaf_path, disease_path, disease_name))
                else:
                    print(f"Missing image for {file_name} in {disease_name}")

    return paired_images