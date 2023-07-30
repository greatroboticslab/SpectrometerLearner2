import os
import pandas as pd
import matplotlib.pyplot as plt

input_folder = 'new_data'

output_folder = 'Plots'

os.makedirs(output_folder, exist_ok=True)

i = 0
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)


        subject_name = df['Subject'].unique()[0]

        subject_data = df[df['Subject'] == subject_name]

        returned_wavelength = subject_data['Returned Wavelength']
        returned_intensity = subject_data['Returned Intensity']


        plt.plot(returned_wavelength, returned_intensity, marker='o', linestyle='-')

        plt.xlabel('Returned Wavelength')
        plt.ylabel('Returned Intensity')
        plt.title(f'Subject: {subject_name}')

        output_file_path = os.path.join(output_folder, f'{subject_name}_{i}_plot.png')
        i += 1
        plt.savefig(output_file_path)

        plt.close()

print("Plots saved successfully.")
