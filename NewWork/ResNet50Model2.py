from keras.layers import Input, Conv2D, UpSampling2D, concatenate
from keras.models import Model
from keras.applications import ResNet50
import os
from Preprocessing import PlantDiseaseDataGenerator
from sklearn.model_selection import train_test_split
from keras.metrics import MeanIoU
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from meanIoU import MeanIoUMetric
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import MeanIoU
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.layers import multiply, GlobalAveragePooling2D, Reshape, Dense
from keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from keras import backend as K
import optuna
from keras.models import load_model


# CLASSMAPPING = {
#     'background': 0, 
#     'Tomato_Healthy': 1, 
#     'Tomato_Late_Blight': 2, 
#     'Tomato_Yellow_Leaf_Curl_Virus': 3,
#     'Tomato_Bacterial_Spot': 4,
#     'Tomato_Early_Blight': 5,
#     'Tomato_Leaf_Mold': 6,
#     'Tomato_Mosaic_Virus': 7,
#     'Tomato_Septoria_Leaf_Spot': 8,
#     'Tomato_Spider_Mites': 9,
#     'Tomato_Target_Spot': 10
# }

CLASSMAPPING = {
    'background': 0, 
    'Tomato_Healthy': 1, 
    'Diseased' : 2
}

import optuna

def objective(trial):
    # Define the hyperparameters to be optimized
    num_filters = trial.suggest_categorical('num_filters', [[256, 128, 64, 32], [512, 256, 128, 64]])
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    input_size = (256, 256, 3)
    
    # Create the model
    model = create_model(input_size, len(CLASSMAPPING), num_filters, activation)

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=weighted_categorical_crossentropy(weights), metrics=['accuracy', MeanIoUMetric(num_classes=len(CLASSMAPPING))])
    
    # Fit the model
    history = model.fit(train_gen, epochs=10, validation_data=val_gen, verbose=0, callbacks=[early_stopping_callback, reduce_lr])
    
    # Evaluate the model
    val_loss, val_accuracy, val_iou = model.evaluate(val_gen, verbose=0)
    
    # Return the metric to be optimized
    return val_iou

# Custom weighted loss function
def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, axis=-1)
        return loss
    return loss

# Function to extract class weights
def extract_class_weights(generator, num_samples, num_classes):
    label_count = np.zeros(num_classes)
    for _ in range(num_samples):
        _, masks = next(generator)
        for i in range(num_classes):
            label_count[i] += np.sum(masks[:, :, :, i])
    class_weights = compute_class_weight('balanced', np.arange(num_classes), label_count)
    return dict(enumerate(class_weights))

def calculate_class_weights(generator, num_classes):
    class_counts = np.zeros(num_classes)
    
    # Iterate over the dataset to count the labels
    for _, mask in generator:
        labels = np.argmax(mask, axis=-1).flatten()  # Assuming mask is one-hot encoded
        for c in range(num_classes):
            class_counts[c] += np.sum(labels == c)
    
    # Calculate class weights
    total_counts = np.sum(class_counts)
    class_weights = total_counts / (num_classes * class_counts + 1e-6)  # Adding a small value to avoid division by zero
    
    # Normalize weights so that the minimum weight is 1.0
    class_weights = class_weights / np.min(class_weights)

    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    return class_weight_dict

def squeeze_excite_block(input_tensor, ratio=16):
    # Squeeze and Excitation block definition
    init = input_tensor
    channel_axis = -1  # assuming channels last
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([init, se])
    return x

def calculate_iou(y_true, y_pred, num_classes):
    iou_list = []
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)

    for c in range(num_classes):
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))
        denominator = TP + FP + FN
        IoU = TP / float(denominator) if denominator > 0 else 0
        iou_list.append(IoU)
    mean_iou = np.mean(iou_list)
    return mean_iou


# Function to create the modified ResNet50 model
# def create_model(input_size, num_classes):
#     inputs = Input(input_size)
#     base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

#     # Encoder - ResNet50
#     skip_connection_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']
#     encoder_outputs = [base_model.get_layer(name).output for name in skip_connection_names]

#     # Decoder
#     x = base_model.output
#     x = squeeze_excite_block(x) 

#     # Decoder Blocks
#     num_decoder_filters = [256, 128, 64, 32]
#     for i, filter in enumerate(num_decoder_filters):
#         if i < len(encoder_outputs):
#             skip_output = encoder_outputs[-(i + 1)]
#             x = UpSampling2D((2, 2))(x)  # Upsample
#             # Ensure the dimensions here match before concatenation
#             x = concatenate([x, skip_output])  
#         else:
#             x = UpSampling2D((2, 2))(x)  # Additional upsampling
#         x = Conv2D(filter, (3, 3), activation='relu', padding='same')(x)
#         if i < len(num_decoder_filters) - 1:
#             x = squeeze_excite_block(x)  # Apply SE block 

#     # Output layer
#     x = UpSampling2D((2, 2))(x)
#     x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)

#     model = Model(inputs=inputs, outputs=x)
#     return model

def create_model(input_size, num_classes, num_decoder_filters, activation):
    inputs = Input(input_size)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Encoder - ResNet50
    skip_connection_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']
    encoder_outputs = [base_model.get_layer(name).output for name in skip_connection_names]

    # Decoder
    x = base_model.output
    x = squeeze_excite_block(x) 

    # Adjusting the number of filters and activation in the Decoder Blocks based on the function's parameters
    for i, filter_size in enumerate(num_decoder_filters):
        if i < len(encoder_outputs):
            skip_output = encoder_outputs[-(i + 1)]
            x = UpSampling2D((2, 2))(x)  # Upsample
            x = concatenate([x, skip_output])  # Concatenate
        else:
            x = UpSampling2D((2, 2))(x)  # Additional upsampling if required
        x = Conv2D(filter_size, (3, 3), activation=activation, padding='same')(x)
        if i < len(num_decoder_filters) - 1:
            x = squeeze_excite_block(x)  # Apply SE block 

    # Output layer
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


dataset_path = os.path.join('C:', os.sep, 'Users', 'josep', 'OneDrive', 'Desktop', 'Final-URCA-DATASET', 'highlight_tomato')
image_dir = os.path.join(dataset_path, 'RGB')
mask_dir = os.path.join(dataset_path, 'GroundTruth_Disease')
leaf_mask_dir = os.path.join(dataset_path, 'GroundTruth_Leaf')

########################################### Limit CPU Usage #########################################

# # Get the number of available CPU cores
# num_cores = os.cpu_count()

# # Calculate the number of cores to use (50% of available cores)
# num_cores_to_use = max(1, num_cores // 3)  # Ensure at least one core is used

# # Set TensorFlow's intra- and inter-op parallelism threads
# tf.config.threading.set_intra_op_parallelism_threads(num_cores_to_use)
# tf.config.threading.set_inter_op_parallelism_threads(num_cores_to_use)

######################################################################################################

image_paths = []
mask_paths = []
leaf_mask_paths = []

for class_dir in os.listdir(leaf_mask_dir):
    class_leaf_mask_dir = os.path.join(leaf_mask_dir, class_dir)
    if os.path.isdir(class_leaf_mask_dir):
        leaf_mask_paths.extend([os.path.join(class_leaf_mask_dir, file) for file in os.listdir(class_leaf_mask_dir) if os.path.isfile(os.path.join(class_leaf_mask_dir, file))])

for class_dir in os.listdir(image_dir):
    class_image_dir = os.path.join(image_dir, class_dir)
    if os.path.isdir(class_image_dir):
        image_paths.extend([os.path.join(class_image_dir, file) for file in os.listdir(class_image_dir) if os.path.isfile(os.path.join(class_image_dir, file))])

for class_dir in os.listdir(mask_dir):
    class_mask_dir = os.path.join(mask_dir, class_dir)
    if os.path.isdir(class_mask_dir):
        mask_paths.extend([os.path.join(class_mask_dir, file) for file in os.listdir(class_mask_dir) if os.path.isfile(os.path.join(class_mask_dir, file))])


# Splitting the data
train_data, test_data = train_test_split(list(zip(image_paths, leaf_mask_paths, mask_paths)), test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Unzipping the tuples
image_paths_train, leaf_mask_paths_train, disease_mask_paths_train = zip(*train_data)
image_paths_val, leaf_mask_paths_val, disease_mask_paths_val = zip(*val_data)
image_paths_test, leaf_mask_paths_test, disease_mask_paths_test = zip(*test_data)

# Creating the data generators
train_gen = PlantDiseaseDataGenerator(list(image_paths_train), list(leaf_mask_paths_train), list(disease_mask_paths_train), batch_size=32, image_size=(256, 256), num_classes=len(CLASSMAPPING))
val_gen = PlantDiseaseDataGenerator(list(image_paths_val), list(leaf_mask_paths_val), list(disease_mask_paths_val), batch_size=32, image_size=(256, 256), num_classes=len(CLASSMAPPING))


# Define the checkpoint directory and filename
checkpoint_filepath = 'RESNET_model_checkpoint.keras'

# Create a ModelCheckpoint callback
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# Define early stopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=150,
    verbose=1,
    restore_best_weights=True
)

# Define learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.00001, verbose=1)

# Initialize a list to store all the labels
all_labels = []

# Iterate over the batches in train_gen
for i in range(len(train_gen)):
    _, y = train_gen[i]
    # Convert one-hot encoded y to class labels
    labels = np.argmax(y, axis=-1).flatten()
    all_labels.extend(labels)

class_weights = calculate_class_weights(train_gen, len(CLASSMAPPING))

weights = np.array([class_weights[i] for i in range(len(CLASSMAPPING))])

print("Weights: {}".format(weights))
print("Class Weight: {}".format(class_weights))

# Optimization Stuff #################################################################

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10) 

print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('  Value: ', trial.value)
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
    
best_decoder_filters = trial.params['num_filters']
best_activation = trial.params['activation']
best_learning_rate = trial.params.get('learning_rate', 0.001)
    
######################################################################################$

epochs = 500

# Instantiate and compile the model
input_size = (256, 256, 3)
model = create_model(input_size, len(CLASSMAPPING), best_decoder_filters, best_activation)
model.compile(optimizer=Adam(learning_rate=best_learning_rate), 
              loss=weighted_categorical_crossentropy(weights), 
              metrics=['accuracy', MeanIoUMetric(num_classes=len(CLASSMAPPING))])

history = model.fit(
    train_gen, 
    epochs=epochs, 
    validation_data=val_gen,
    callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr],
    verbose=1,
    class_weight=class_weights
    )

# Evaluate the model
test_gen = PlantDiseaseDataGenerator(list(image_paths_test), list(leaf_mask_paths_test), list(disease_mask_paths_test), batch_size=32, image_size=(256, 256), num_classes=len(CLASSMAPPING))

# Define the custom_objects dictionary
custom_objects = {
    'MeanIoUMetric': MeanIoUMetric(num_classes=len(CLASSMAPPING)),
    'loss': weighted_categorical_crossentropy(weights)  
}

# Load your model
saved_model = load_model(checkpoint_filepath, custom_objects=custom_objects)

results = saved_model.evaluate(test_gen)
print("Test loss, Test accuracy, Test IoU:", results)

# Predict masks using the saved model
predictions = saved_model.predict(test_gen)

# Convert predictions to class indices
predicted_class_indices = np.argmax(predictions, axis=-1)

output_base_dir = "output_images"

for i in range(100):
    output_dir = os.path.join(output_base_dir, str(i))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        break

# Iterate over the predictions and ground truth to calculate per-image IoU, accuracy, and save the images
for i, (input_image_path, true_mask_path) in enumerate(zip(image_paths_test, disease_mask_paths_test)):
    input_image = np.array(Image.open(input_image_path).resize((256, 256))) / 255.0
    pred_mask = model.predict(np.expand_dims(input_image, axis=0))
    pred_mask = np.argmax(pred_mask, axis=-1).squeeze()

    true_mask_img = np.array(Image.open(true_mask_path).resize((256, 256)))
    true_mask = np.argmax(to_categorical(true_mask_img, num_classes=len(CLASSMAPPING)), axis=-1)
    
    # Ensure true_mask is 2D before calculating accuracy
    if true_mask.ndim == 3:
        true_mask = np.argmax(true_mask, axis=-1)
    
    iou = calculate_iou(true_mask, pred_mask, len(CLASSMAPPING))
    print(f"Image {i} - IoU: {iou}")
    
    accuracy = np.mean(pred_mask == true_mask)
    print(f"Image {i} - Accuracy: {accuracy}")

    # Save the input image
    input_image_file = os.path.join(output_dir, f"input_image_{i}.png")
    Image.fromarray((input_image * 255).astype(np.uint8)).save(input_image_file)

    # Save the true mask
    true_mask_file = os.path.join(output_dir, f"true_mask_{i}.png")
    Image.fromarray((true_mask * 255).astype(np.uint8)).save(true_mask_file)

    # Save the predicted mask
    pred_mask_file = os.path.join(output_dir, f"predicted_mask_{i}.png")
    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(pred_mask_file)

