# CNN_SorghumWeed_Classifier
'CNN_SorghumWeed_Classifier' is an artificial intelligence (AI) based software that can differentiate a sorghum sampling image from its associated weeds images. This repository releases the source code for pre-processing, augmenting, and normalizing the 'SorghumWeedDataset_Classification' dataset. It also contains the code for training, validating, and testing the AI model using transfer learning. The reproducible code of the CNN_SorghumWeed_Classifier is also available at https://codeocean.com/capsule/9796503/tree.

## Dataset utilized
CNN_SorghumWeed_Classifier is constructed using 'SorghumWeedDataset_Classification,' a crop-weed research dataset. The dataset is cloned in the source code for further processing and model building. The following references relate to the dataset: <br/>
<ul>
  <li>First appeared at https://data.mendeley.com/datasets/4gkcyxjyss/1</li>
  <li>GitHub repository: https://github.com/JustinaMichael/SorghumWeedDataset_Classification.git</li>
  <li>Detailed description of the data acquisition process: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4593178</li>
</ul>

## Language and Dependencies
Language: Python 3.10.12 </br>
Dependencies: <br/>
<ul>
  <li> Tensorflow: 2.14.0</li>
  <li> Scikit-learn: 1.2.2</li>
  <li>Seaborn: 0.12.2</li>
  <li>Matplotlib: 3.7.1</li>
  <li>Scipy: 1.11.3</li>
  <li>Numpy: 1.23.5</li>
  <li>Pandas: 1.5.3</li>
</ul>

## CNN_Sorghum_Weed_Classifier.ipynb
The complete source code for pre-processing the dataset and creating the model is included in the interactive Python notebook "CNN_Sorghum_Weed_Classifier.ipynb." 

### Getting started
'CNN_Sorghum_Weed_Classifier.ipynb' can be opened in the 'Google colaboratory' (or any other Jupyter Notebook environment). The runtime of the source code is configured to 'T4 GPU' to expedite the model training process.

### Dataset cloning
The dataset is cloned from the respective GitHub repository using the following command: 
```python
!git clone https://github.com/JustinaMichael/SorghumWeedDataset_Classification.git
```

### Data pre-processing
The necessary libraries and packages are installed followed by initializing the tuned hyper-parameter values. The data is augmented and normalized before building the model. The following code snippet augments and normalizes the training data:
```python
train_datagen = ImageDataGenerator(rescale = 1./255,
                                       rotation_range = 45,
                                       width_shift_range = 0.3,
                                       shear_range = 0.25,
                                       zoom_range = 0.25,
                                       height_shift_range = 0.3,
                                       horizontal_flip = True,
                                       brightness_range=(0.2, 0.9),
                                       vertical_flip = True,
                                       fill_mode = 'reflect')
```

### Model building
Using transfer learning, the classifier is trained, validated, and tested on the following four pre-trained Convolutional Neural Network (CNN) models, whose codes are provided sequentially.  
<ul>
  <li>VGG19</li>
  <li>MobileNetV2</li>
  <li>DenseNet201</li>
  <li>ResNet152V2</li>
</ul>
The model is trained using the SorghumWeedDataset_Classification dataset using the following code snippet:

```python
history = model.fit(x = training_set,
                    batch_size = batch_size,
                    epochs = epochs,
                    callbacks = cd,
                    validation_data = valid_set,
                    steps_per_epoch = len(training_set),
                    validation_steps = len(valid_set),
                    validation_batch_size = batch_size,
                    validation_freq = 1)
```
'EarlyStopping' is triggered by the following code, which prevents overfitting even after the model has been initialized for 50 training epochs:

```python
es = EarlyStopping(monitor = "val_accuracy",
                   min_delta = 0.01,
                   patience = 5,
                   verbose = 1,
                   mode = 'auto')
```

### Evaluating the best-performing model
The following code is used to evaluate each of the four models, and the results are compared. With an accuracy of 94.19%, MobileNetV2 produced the best results out of the four models. The results are presented graphically for easy comprehension.
```python
evaluate_test_data = model.evaluate(test_set)
```

## Licence
This project is licensed under the APACHE LICENSE, VERSION 2.0.

## Citation 
If you find this dataset helpful and use it in your work, kindly cite the dataset using “Michael, Justina; M, Thenmozhi (2023), “SorghumWeedDataset_Classification”, Mendeley Data, V1, doi: 10.17632/4gkcyxjyss.1”

## Contributors profile <br/>
1. Justina Michael. J <br/>
        Google Scholar: https://scholar.google.com/citations?user=pEEzO14AAAAJ&hl=en&oi=ao <br/>
        ORCID: https://orcid.org/0000-0001-8072-3230 </br>
2. Dr. M. Thenmozhi <br/>
        Google Scholar: https://scholar.google.com/citations?user=Es49w08AAAAJ&hl=en&oi=ao <br/>
        ORCID: https://orcid.org/0000-0002-8064-5938 <br/>

