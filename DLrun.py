import os
import sys

from InceptionResNetV2Model import InceptionResNetV2Model
from AlexNetModel import AlexNetModel
from ResNet50Model import ResNet50Model

def run_all_models(rgb_dir, disease_segmented_dir, leaf_segmented_dir):
    # Base output directory
    base_output_dir = 'outputs'

    # Run InceptionResNetV2Model
    inception_resnet_v2_output_dir = os.path.join(base_output_dir, 'InceptionResNetV2')
    inception_resnet_v2_model = InceptionResNetV2Model(rgb_dir, disease_segmented_dir, leaf_segmented_dir)
    inception_resnet_v2_history = inception_resnet_v2_model.compile_and_train(epochs=10, batch_size=32, output_dir=inception_resnet_v2_output_dir)

    # Run AlexNetModel
    alexNet_output_dir = os.path.join(base_output_dir, 'AlexNet')
    alexNet_model = AlexNetModel(rgb_dir, disease_segmented_dir, leaf_segmented_dir)
    alexNet_history = alexNet_model.compile_and_train(epochs=10, batch_size=32, output_dir=alexNet_output_dir)

    # Run ResNet50Model
    resNet50_output_dir = os.path.join(base_output_dir, 'ResNet50')
    resNet50_model = ResNet50Model(rgb_dir, disease_segmented_dir, leaf_segmented_dir)
    resNet50_history = resNet50_model.compile_and_train(epochs=10, batch_size=32, output_dir=resNet50_output_dir)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python DLrun.py <RGB directory> <Disease Segmented directory> <Leaf Segmented directory>")
        sys.exit(1)

    run_all_models(sys.argv[1], sys.argv[2], sys.argv[3])
