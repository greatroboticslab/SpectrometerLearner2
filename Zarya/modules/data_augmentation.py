import numpy as np
import pandas as pd

class DataAugmentation:
    def __init__(self, X_train, y_train, num_augmented_samples=1000):
        self.X_train = X_train.reset_index(drop=True)
        self.y_train = y_train
        self.num_augmented_samples = num_augmented_samples

    def apply_data_augmentation(self):
        augmented_data = []
        augmented_labels = []

        for i in range(self.num_augmented_samples):
            # Randomly select an index from the original data
            index = np.random.randint(len(self.X_train))
            original_sample = self.X_train.iloc[index]
            label = self.y_train[index]

            # Apply data augmentation (you can customize these operations)
            augmented_sample = original_sample + np.random.normal(0, 0.1, original_sample.shape)  # Adding random noise
            augmented_sample *= np.random.uniform(0.9, 1.1)  # Scaling by a random factor
            if np.random.rand() < 0.5:
                augmented_sample = np.flip(augmented_sample)  # Randomly flipping the sample

            augmented_data.append(augmented_sample)
            augmented_labels.append(label)

        # Convert the augmented data and labels to numpy arrays
        augmented_data = np.array(augmented_data)
        augmented_labels = np.array(augmented_labels)

        # Concatenate the original data with the augmented data
        X_train_augmented = np.concatenate((self.X_train.values, augmented_data), axis=0)
        y_train_augmented = np.concatenate((self.y_train, augmented_labels), axis=0)

        return pd.DataFrame(X_train_augmented), y_train_augmented