    def process_folder(self, folder_path):
        original_folder = os.path.join(folder_path, 'original')
        disease_folder = os.path.join(folder_path, 'segmented_disease')
        metrics = {'accuracy': [], 'recall': [], 'precision': []}

        for image_name in os.listdir(original_folder):
            original_image_path = os.path.join(original_folder, image_name)
            disease_image_path = os.path.join(disease_folder, image_name)

            original_image = self.load_and_preprocess_image(original_image_path)
            disease_image = self.load_and_preprocess_image(disease_image_path)

            # Ensure disease_image is preprocessed into a binary mask and has the same dimensions as original_image
            # Example: disease_image = preprocess_ground_truth(disease_image, original_image.shape)

            labels = self.apply_kmeans(original_image)
            self.visualize_segmentation(original_image, labels)

            # Flatten and compare arrays
            ground_truth = disease_image.flatten()
            predicted = labels.flatten()

            if ground_truth.size == predicted.size:
                metrics['accuracy'].append(accuracy_score(ground_truth, predicted))
                metrics['recall'].append(recall_score(ground_truth, predicted, average='macro'))
                metrics['precision'].append(precision_score(ground_truth, predicted, average='macro'))
            else:
                print(f"Size mismatch in image {image_name}: ground truth {ground_truth.size}, predicted {predicted.size}")

        return metrics