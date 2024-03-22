import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from PIL import Image
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

# CLASSMAPPING = {
#     'background': 0, 
#     'Tomato_Healthy': 1, 
#     'Tomato_Late_Blight': 2, 
#     'Tomato_Yellow_Leaf_Curl_Virus': 3,
#     'Tomato_Bacterial_Spot': 4,
# 		'Tomato_Early_Blight': 5,
# 		'Tomato_Leaf_Mold': 6,
# 		'Tomato_Mosaic_Virus': 7,
# 		'Tomato_Septoria_Leaf_Spot': 8,
# 		'Tomato_Spider_Mites': 9,
# 		'Tomato_Target_Spot': 10
# }

CLASSMAPPING = {
    'background': 0, 
    'Tomato_Healthy': 1, 
    'Diseased' : 2
}

output_dir = 'combined_masks'
os.makedirs(output_dir, exist_ok=True)

class PlantDiseaseDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, leaf_mask_paths, disease_mask_paths, batch_size, image_size, num_classes, shuffle=True, augmentations=None):
        self.image_paths = image_paths
        self.leaf_mask_paths = leaf_mask_paths
        self.disease_mask_paths = disease_mask_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.augmentations = augmentations if augmentations else ImageDataGenerator()

        if self.shuffle:
            np.random.shuffle(self.indexes)


    def __len__(self):
        # Compute the number of batches to yield
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_image_paths = [self.image_paths[k] for k in indexes]
        batch_mask_paths = [self.leaf_mask_paths[k] for k in indexes]
        disease_mask_paths = [self.disease_mask_paths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_image_paths, batch_mask_paths, disease_mask_paths)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def apply_transformations(self, img, leaf_mask, disease_mask):
        # Convert PIL images to numpy arrays
        img_array = np.array(img)
        leaf_mask_array = np.array(leaf_mask)
        disease_mask_array = np.array(disease_mask)
        
        # Rotation
        angle = np.random.uniform(-30, 30)
        center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
        scale = np.random.uniform(0.9, 1.1)  # Random scaling
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # Translation
        tx = random.randint(-10, 10)  # Random translation along x
        ty = random.randint(-10, 10)  # Random translation along y
        translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

        # Apply transformations
        img_array = cv2.warpAffine(img_array, rotation_matrix, (img_array.shape[1], img_array.shape[0]))
        img_array = cv2.warpAffine(img_array, translation_matrix, (img_array.shape[1], img_array.shape[0]))

        leaf_mask_array = cv2.warpAffine(leaf_mask_array, rotation_matrix, (leaf_mask_array.shape[1], leaf_mask_array.shape[0]), flags=cv2.INTER_NEAREST)
        leaf_mask_array = cv2.warpAffine(leaf_mask_array, translation_matrix, (leaf_mask_array.shape[1], leaf_mask_array.shape[0]), flags=cv2.INTER_NEAREST)

        disease_mask_array = cv2.warpAffine(disease_mask_array, rotation_matrix, (disease_mask_array.shape[1], disease_mask_array.shape[0]), flags=cv2.INTER_NEAREST)
        disease_mask_array = cv2.warpAffine(disease_mask_array, translation_matrix, (disease_mask_array.shape[1], disease_mask_array.shape[0]), flags=cv2.INTER_NEAREST)

        # Convert numpy arrays back to PIL images
        img = Image.fromarray(img_array)
        leaf_mask = Image.fromarray(leaf_mask_array)
        disease_mask = Image.fromarray(disease_mask_array)

        return img, leaf_mask, disease_mask


    def __data_generation(self, batch_image_paths, batch_leaf_mask_paths, batch_disease_mask_paths):
        X = np.empty((self.batch_size, *self.image_size, 3))
        y = np.empty((self.batch_size, *self.image_size), dtype=int)

        for i, (img_path, leaf_mask_path, disease_mask_path) in enumerate(zip(batch_image_paths, batch_leaf_mask_paths, batch_disease_mask_paths)):
            img = Image.open(img_path).convert('RGB')
            leaf_mask = Image.open(leaf_mask_path).convert('L')
            disease_mask = Image.open(disease_mask_path).convert('L')

            img, leaf_mask, disease_mask = self.apply_transformations(img, leaf_mask, disease_mask)

            X[i, ] = np.array(img) / 255.0

            leaf_mask_array = np.array(leaf_mask)

            disease_mask_array = np.array(disease_mask)
            disease_name = disease_mask_path.split(os.sep)[-2]  # Extract disease name
            disease_class = CLASSMAPPING['Diseased']

            # Combine the masks
            combined_mask = np.full_like(leaf_mask_array, CLASSMAPPING['background'])  # Set all to background initially
            combined_mask[leaf_mask_array == 255] = CLASSMAPPING['Tomato_Healthy']  # Set leaf area to healthy
            combined_mask[disease_mask_array == 255] = disease_class  # Set disease area

            y[i, ] = combined_mask
            
            # if i == 0:
            #   combined_mask_image = Image.fromarray(combined_mask.astype(np.uint8))
            #   combined_mask_image.save(os.path.join(output_dir, f"combined_mask_{i}.png"))
              # plt.figure()
              # plt.imshow(combined_mask_image, cmap=matplotlib.colors.ListedColormap(['black', 'green', 'red']))
              # plt.colorbar()
              # plt.show()
              
        # One-hot encoding for masks
        y = to_categorical(y, num_classes=len(CLASSMAPPING))

        return X, y

