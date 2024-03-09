import os
import sys

from models.ML.KMeans import KMeansSegmentation
from models.ML.RandomForest import ImageRandomForestSegmenter

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
            if not os.path.exists(os.path.join('ML_outputs', str(number), 'outputs' + last_part)):
                os.makedirs(os.path.join('ML_outputs', str(
                    number), 'outputs' + last_part), exist_ok=True)
                break
            number += 1
        base_output_dir = os.path.join(
            'ML_outputs', str(number), 'outputs' + last_part)
        try:
            RandomForest_output_dir = os.path.join(base_output_dir, 'RandomForest')
            RandomForest_Algorithm = ImageRandomForestSegmenter(base_rgb_dir, base_disease_dir, base_leaf_dir, val_split=0.2, dataset_name=last_part)
            RandomForest_Results = RandomForest_Algorithm.compile_and_train(RandomForest_output_dir)
            print("Finished running RandomForest model")
        except Exception as e:
            print(f"Failed to run RandomForest model: {e}")
        try:
            KMeans_output_dir = os.path.join(base_output_dir, 'KMeansSegmentation')
            KMeans_Algorithm = KMeansSegmentation(base_rgb_dir, base_disease_dir, base_leaf_dir, val_split=0.2, dataset_name=last_part)
            KMeans_Results = KMeans_Algorithm.compile_and_train(KMeans_output_dir)
            print("Finished running KMeans model")
        except Exception as e:
            print(f"Failed to run KMeans model: {e}")
    
    
if __name__ == '__main__':
    if len(sys.argv) < 1:
        print ("Usage: python MLrun.py base_directory1 basedirectory2 ...")
        sys.exit(1)
        
    base_directories = sys.argv[1:]
    run_all_models(base_directories)