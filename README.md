# CNN Arabic 22 Letter HMBD-v1

## Image Classification Using CNN (Convolutional Neural Networks)

### Authors
- [Rafiq7M Al Mohammady](https://github.com/Rafiq7M)
- [Abdulrahman](https://github.com/AbdulrahmanAlmikhlafi)

## Project Overview
This project implements a Convolutional Neural Network (CNN) for classifying 22 isolated Arabic letters. The model is trained on a custom dataset of handwritten Arabic letters, demonstrating the application of deep learning techniques in Arabic character recognition.

## Dataset

The dataset consists of 22 classes of isolated Arabic letters.
This database was taken from a huge database containing all the letters of the Arabic language written by hand. The link to this database is [HossamBalaha/HMBD-v1](https://github.com/HossamBalaha/HMBD-v1). We thank everyone who prepared and equipped this database, which we benefited from in this project.

# Dataset Information
| # | Directory Name | Number of Images | Example Image |
|---|----------------|------------------|---------------|
| 0 | Dataset\Ain_Isolated | 462 | ![AHCR_00092_Ain_Isolated_53.jpg](Dataset\Ain_Isolated\AHCR_00092_Ain_Isolated_53.jpg) |
| 1 | Dataset\Alf_Hamza_Above_Isolated | 476 | ![AHCR_00010_Alf_Hamza_Above_Isolated_1.jpg](Dataset\Alf_Hamza_Above_Isolated\AHCR_00010_Alf_Hamza_Above_Isolated_1.jpg) |
| 2 | Dataset\Alf_Hamza_Under_Isolated | 474 | ![AHCR_00117_Alf_Hamza_Under_Isolated_49.jpg](Dataset\Alf_Hamza_Under_Isolated\AHCR_00117_Alf_Hamza_Under_Isolated_49.jpg) |
| 3 | Dataset\Alf_Isolated | 480 | ![AHCR_00063_Alf_Isolated_7.jpg](Dataset\Alf_Isolated\AHCR_00063_Alf_Isolated_7.jpg) |
| 4 | Dataset\Baa_Isolated | 468 | ![AHCR_00075_Baa_Isolated_17.jpg](Dataset\Baa_Isolated\AHCR_00075_Baa_Isolated_17.jpg) |
| 5 | Dataset\Baa_Middle | 460 | ![AHCR_00081_Baa_Middle_65.jpg](Dataset\Baa_Middle\AHCR_00081_Baa_Middle_65.jpg) |
| 6 | Dataset\Daad_Isolated | 455 | ![AHCR_00033_Daad_Isolated_22.jpg](Dataset\Daad_Isolated\AHCR_00033_Daad_Isolated_22.jpg) |
| 7 | Dataset\Dal_Isolated | 472 | ![AHCR_00074_Dal_Isolated_45.jpg](Dataset\Dal_Isolated\AHCR_00074_Dal_Isolated_45.jpg) |
| 8 | Dataset\Faa_Isolated | 464 | ![AHCR_00068_Faa_Isolated_1.jpg](Dataset\Faa_Isolated\AHCR_00068_Faa_Isolated_1.jpg) |
| 9 | Dataset\Gem_Isolated | 472 | ![AHCR_00030_Gem_Isolated_3.jpg](Dataset\Gem_Isolated\AHCR_00030_Gem_Isolated_3.jpg) |
| 10 | Dataset\Gem_Start | 472 | ![AHCR_00011_Gem_Start_7.jpg](Dataset\Gem_Start\AHCR_00011_Gem_Start_7.jpg) |
| 11 | Dataset\Gen_Isolated | 920 | ![AHCR_00021_Gen_Isolated_77.jpg](Dataset\Gen_Isolated\AHCR_00021_Gen_Isolated_77.jpg) |
| 12 | Dataset\Hamza_Isolated | 466 | ![AHCR_00091_Hamza_Isolated_36.jpg](Dataset\Hamza_Isolated\AHCR_00091_Hamza_Isolated_36.jpg) |
| 13 | Dataset\Kaf_Isolated | 464 | ![AHCR_00103_Kaf_Isolated_21.jpg](Dataset\Kaf_Isolated\AHCR_00103_Kaf_Isolated_21.jpg) |
| 14 | Dataset\Lam_Alf_Hamza_Isolated | 459 | ![AHCR_00036_Lam_Alf_Hamza_Isolated_44.jpg](Dataset\Lam_Alf_Hamza_Isolated\AHCR_00036_Lam_Alf_Hamza_Isolated_44.jpg) |
| 15 | Dataset\Mem_Isolated | 468 | ![AHCR_00036_Mem_Isolated_48.jpg](Dataset\Mem_Isolated\AHCR_00036_Mem_Isolated_48.jpg) |
| 16 | Dataset\Qaf_Isolated | 467 | ![AHCR_00023_Qaf_Isolated_6.jpg](Dataset\Qaf_Isolated\AHCR_00023_Qaf_Isolated_6.jpg) |
| 17 | Dataset\Raa_Isolated | 476 | ![AHCR_00082_Raa_Isolated_52.jpg](Dataset\Raa_Isolated\AHCR_00082_Raa_Isolated_52.jpg) |
| 18 | Dataset\Saad_Isolated | 459 | ![AHCR_00095_Saad_Isolated_24.jpg](Dataset\Saad_Isolated\AHCR_00095_Saad_Isolated_24.jpg) |
| 19 | Dataset\Sin_Isolated | 468 | ![AHCR_00055_Sin_Isolated_68.jpg](Dataset\Sin_Isolated\AHCR_00055_Sin_Isolated_68.jpg) |
| 20 | Dataset\Taa_Isolated | 467 | ![AHCR_00005_Taa_Isolated_31.jpg](Dataset\Taa_Isolated\AHCR_00005_Taa_Isolated_31.jpg) |
| 21 | Dataset\Taa_Middle | 462 | ![AHCR_00019_Taa_Middle_66.jpg](Dataset\Taa_Middle\AHCR_00019_Taa_Middle_66.jpg) |

## Project Structure
- **Important Note**: Due to the size of the file "X_Arabic_22_letter_64.pickle,y_Arabic_22_letter_64.pickle" it has not been uploaded to the gtb, so you can run the project, and these files will be created and then you can train the model based on these files.
```
CNN ARABIC 22 LETTER HMBD -V1/
├── datasave/
│   ├── checkpoints/
│   ├── model_logs/
│   │   ├── train/
│   │   └── validation/
│   ├── model_acc_Arabic_22_letter_64.h5
│   ├── weights_model_acc_Arabic_22_letter_64.h5
│   ├── X_Arabic_22_letter_64.pickle
│   └── y_Arabic_22_letter_64.pickle
├── Dataset/
│   ├── Ain_Isolated/
│   ├── Alf_Hamza_Above_Isolated/
│   ├── ...
│   └── Taa_Middle/
└── CNN Arabic 22 Letter HMBD .ipynb
```

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- opencv-python (cv2)
- tensorflow
- keras
- scikit-learn
- pickle

## Model Architecture

The CNN model architecture is as follows:

```python
KerasModel = keras.models.Sequential([
    keras.layers.Conv2D(8, kernel_size=(5, 5), activation='relu', input_shape=(s, s, 3)),
    keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
    keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(22, activation='softmax')
])
```

## Training

The model is compiled using the Adam optimizer and sparse categorical crossentropy loss function:

```python
KerasModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
![graph](/dataSave/output_graph.png)
![graph](/dataSave/output_graph2.png)

## Results

The training and validation accuracy curves, as well as the loss curves, are provided in the notebook. These visualizations help in understanding the model's performance and identifying potential overfitting or underfitting.
All results and detailed drawings will be found in the [CNN Arabic 22 Letter HMBD-v1](CNN%20Arabic%2022%20Letter%20HMBD%20.ipynb) file.
The following image shows the results of the model and the accuracy of classifying letters.
![graph](/dataSave/output.png)

## Usage

To use this project:

1. Clone the repository
2. Install the required dependencies
3. Run the Jupyter notebook `CNN Arabic 22 Letter HMBD .ipynb`

## Future Work

- Experiment with different model architectures
- Implement data augmentation techniques to improve model generalization
- Explore transfer learning approaches using pre-trained models
- Create models to recognize handwritten Arabic words.
 
## Acknowledgements

This project was developed for discussion to obtain practical grades in the Neural Networks course as part of the Bachelors' of Software Engineering major at Taiz University.

We would like to thank our teachers and colleagues for their support and feedback throughout the development process.
Also, thanks to everyone who contributed to the preparation and publication of the dataset HMBD-v1.

## License
[MIT License](https://opensource.org/licenses/MIT)