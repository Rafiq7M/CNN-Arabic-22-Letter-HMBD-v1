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
####  Dataset Information
| # | Directory Name | Number of Images | Example Image |
|---|----------------|------------------|---------------|
| 0 | Dataset\Ain_Isolated | 462 | <img src="Dataset\Ain_Isolated\AHCR_00084_Ain_Isolated_60.jpg" width="100" height="100"> |
| 1 | Dataset\Alf_Hamza_Above_Isolated | 476 | <img src="Dataset\Alf_Hamza_Above_Isolated\AHCR_00101_Alf_Hamza_Above_Isolated_9.jpg" width="100" height="100"> |
| 2 | Dataset\Alf_Hamza_Under_Isolated | 474 | <img src="Dataset\Alf_Hamza_Under_Isolated\AHCR_00121_Alf_Hamza_Under_Isolated_49.jpg" width="100" height="100"> |
| 3 | Dataset\Alf_Isolated | 480 | <img src="Dataset\Alf_Isolated\AHCR_00023_Alf_Isolated_6.jpg" width="100" height="100"> |
| 4 | Dataset\Baa_Isolated | 468 | <img src="Dataset\Baa_Isolated\AHCR_00085_Baa_Isolated_24.jpg" width="100" height="100"> |
| 5 | Dataset\Baa_Middle | 460 | <img src="Dataset\Baa_Middle\AHCR_00078_Baa_Middle_65.jpg" width="100" height="100"> |
| 6 | Dataset\Daad_Isolated | 455 | <img src="Dataset\Daad_Isolated\AHCR_00099_Daad_Isolated_22.jpg" width="100" height="100"> |
| 7 | Dataset\Dal_Isolated | 472 | <img src="Dataset\Dal_Isolated\AHCR_00105_Dal_Isolated_36.jpg" width="100" height="100"> |
| 8 | Dataset\Faa_Isolated | 464 | <img src="Dataset\Faa_Isolated\AHCR_00095_Faa_Isolated_8.jpg" width="100" height="100"> |
| 9 | Dataset\Gem_Isolated | 472 | <img src="Dataset\Gem_Isolated\AHCR_00118_Gem_Isolated_10.jpg" width="100" height="100"> |
| 10 | Dataset\Gem_Start | 472 | <img src="Dataset\Gem_Start\AHCR_00041_Gem_Start_6.jpg" width="100" height="100"> |
| 11 | Dataset\Gen_Isolated | 920 | <img src="Dataset\Gen_Isolated\AHCR_00106_Gen_Isolated_77.jpg" width="100" height="100"> |
| 12 | Dataset\Hamza_Isolated | 466 | <img src="Dataset\Hamza_Isolated\AHCR_00058_Hamza_Isolated_36.jpg" width="100" height="100"> |
| 13 | Dataset\Kaf_Isolated | 464 | <img src="Dataset\Kaf_Isolated\AHCR_00046_Kaf_Isolated_28.jpg" width="100" height="100"> |
| 14 | Dataset\Lam_Alf_Hamza_Isolated | 459 | <img src="Dataset\Lam_Alf_Hamza_Isolated\AHCR_00043_Lam_Alf_Hamza_Isolated_37.jpg" width="100" height="100"> |
| 15 | Dataset\Mem_Isolated | 468 | <img src="Dataset\Mem_Isolated\AHCR_00048_Mem_Isolated_49.jpg" width="100" height="100"> |
| 16 | Dataset\Qaf_Isolated | 467 | <img src="Dataset\Qaf_Isolated\AHCR_00028_Qaf_Isolated_6.jpg" width="100" height="100"> |
| 17 | Dataset\Raa_Isolated | 476 | <img src="Dataset\Raa_Isolated\AHCR_00117_Raa_Isolated_60.jpg" width="100" height="100"> |
| 18 | Dataset\Saad_Isolated | 459 | <img src="Dataset\Saad_Isolated\AHCR_00006_Saad_Isolated_16.jpg" width="100" height="100"> |
| 19 | Dataset\Sin_Isolated | 468 | <img src="Dataset\Sin_Isolated\AHCR_00063_Sin_Isolated_68.jpg" width="100" height="100"> |
| 20 | Dataset\Taa_Isolated | 467 | <img src="Dataset\Taa_Isolated\AHCR_00046_Taa_Isolated_23.jpg" width="100" height="100"> |
| 21 | Dataset\Taa_Middle | 462 | <img src="Dataset\Taa_Middle\AHCR_00084_Taa_Middle_75.jpg" width="100" height="100"> |


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
<img src="/dataSave/output_graph.png">
<img src="/dataSave/output_graph2.png">

## Results

The training and validation accuracy curves, as well as the loss curves, are provided in the notebook. These visualizations help in understanding the model's performance and identifying potential overfitting or underfitting.
All results and detailed drawings will be found in the [CNN Arabic 22 Letter HMBD-v1](CNN%20Arabic%2022%20Letter%20HMBD%20.ipynb) file.
The following image shows the results of the model and the accuracy of classifying letters.
<img src="/dataSave/output.png">

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