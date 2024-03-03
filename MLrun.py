import os
import sys

from models.ML.KMeans import KMeansSegmentation
# from models.ML.RandomForest import RandomForest

# rgb_dirs: list[str], disease_segmented_dirs: list[str], leaf_segmented_dirs: list[str], val_split: float, dataset_name: str

def run_all_models(base_directories):
    for base_dir in base_directories:
        base_rgb_dir = os.path.join(base_dir, 'RGB')
        base_disease_dir = os.path.join(base_dir, 'GroundTruth_Disease')
        base_leaf_dir = os.path.join(base_dir, 'GroundTruth_Leaf')

        # Base output directory
        last_part = os.path.basename(base_dir)
        number = 1
        while True:
            if not os.path.exists(os.path.join('outputs', str(number), 'outputs' + last_part)):
                os.makedirs(os.path.join('outputs', str(
                    number), 'outputs' + last_part))
                break
            number += 1
        base_output_dir = os.path.join(
            'outputs', str(number), 'outputs' + last_part)

        KMeans_output_dir = os.path.join(base_output_dir, 'KMeansSegmentation')
        KMeans_Algorithm = KMeansSegmentation(base_rgb_dir, base_disease_dir, base_leaf_dir, val_split=0.2, dataset_name=last_part)
        KMeans_Results = KMeans_Algorithm.compile_and_train(base_output_dir)
        
    
    
if __name__ == '__main__':
    if len(sys.argv) < 1:
        print ("Usage: python MLrun.py base_directory1 basedirectory2 ...")
        sys.exit(1)
        
base_directories = sys.argv[1:]
run_all_models(base_directories)