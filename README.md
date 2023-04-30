# Cardiothoracic Disease Diagnosis: Using Machine Learning for Accurate Disease Detection from Chest Scans

![_2__Cardiothoracic_Diagnosis-1](https://user-images.githubusercontent.com/78103390/235333113-aa6d692d-41ef-4f6a-8451-2072ffb0a015.png)

## Introduction

Cardiovascular diseases were rare before the 20th century, but have become the leading cause of death worldwide since the 1960s. Early and accurate diagnosis is crucial for effective treatment, but interpreting chest scans can be challenging, leading to misdiagnoses. To address this challenge, we applied a 2D CNN to diagnose cardiothoracic diseases. Furthermore, we employed inductive transfer learning of YOLOv7 and EfficientDet and designed a customized DETR transformer model for detection of health issues that lie within the patient's scan. During this, we focused on treating three conditions: Cardiomegaly, Pneumonia, and Covid-19. Cardiomegaly is a key indicator of potential heart disease which can be difficult to detect in chest radiographs due to its subtle presentation. We also decided to look into Covid-19 and Pneumonia, given their prevalence in today's population and the significant impact they have on mortality rates. 

## Data Collection
To create our main dataset for training, we gathered data from various sources, including the NCBI and NIH. Each image was preprocessed with a bounding box to indicate the disease location. With this data, we trained our models to accurately detect COVID-19, Pneumonia, and Cardiomegaly in chest scans. We used industry-standard data augmentation techniques, such as random cropping, rotation, and horizontal flips, to diversify our data and generate a larger pool of accurate data.These augmentation methods enabled us to generate data that accurately reflects the common complexities and impurities in diagnostic imaging. The use of this technique allowed for a significant increase in our pool of data.

![Training Data Sample](https://user-images.githubusercontent.com/78103390/235333139-92c97aa8-6a6b-4df1-9dfd-6bb22b1ba030.png)

<sub>This is an example of our training data.</sub>

## Models
Our Choices of Models:
 - DETR (Detection Transformer): DETR, or Detection Transformer, is a cutting-edge object detection framework that utilizes a transformer-based architecture to reason about the relationship of objects and the global context of the image. Unlike traditional object detection algorithms that rely on region proposal networks and post-processing steps, DETR directly predicts the set of objects present in an image as well as their corresponding bounding boxes in a single forward pass. This approach eliminates the need for heuristic post-processing steps and enables end-to-end training of the model. The backbone of DETR consists of a ResNet-50, which is a popular convolutional neural network architecture for image classification tasks. However, to combine the strengths of global and local feature extraction, we have developed a ResNet-50/EfficientNet hybrid backbone. This allows the model to extract both low-level and high-level features from the input image, which improves the overall performance of the detector. The DETR framework has shown remarkable results on the COCO benchmark and has demonstrated its potential for real-world applications such as robotics, autonomous driving, and surveillance.
- YOLOv7: YOLOv7 is a state-of-the-art object detection algorithm that has gained significant attention in recent years due to its high efficiency and accuracy. This model serves as an industry-standard detection model, and its comparison with the customized DETR transformer provides us with valuable insights into the performance of different object detection approaches. The YOLOv7 model achieves high efficiency by using a single neural network to predict the bounding boxes and class probabilities of all objects in an image simultaneously. This approach reduces the computational complexity of the algorithm and enables real-time object detection in video streams. By comparing the performance of YOLOv7 with our customized DETR transformer, we can evaluate the effectiveness of each method in various scenarios and determine the optimal approach for specific applications. This enables us to develop customized object detection models that can cater to the unique needs of different industries and use cases
- Efficient-DET: EfficientDet is a state-of-the-art object detection algorithm that achieves high accuracy while being computationally efficient. It is designed to address the trade-off between accuracy and efficiency in object detection tasks, which are crucial for a variety of applications such as autonomous driving, robotics, and surveillance. EfficientDet uses a novel compound scaling method that optimizes the network architecture, input image resolution, and model depth to achieve better accuracy and efficiency simultaneously. It also uses efficient building blocks such as mobile inverted bottleneck convolution (MBConv) and squeeze-and-excitation (SE) modules to reduce the computational cost while maintaining high performance. The EfficientDet models have achieved top performance on the COCO benchmark, which evaluates object detection accuracy and efficiency, and have been widely adopted in real-world applications due to their high efficiency and accuracy.


![arch (1)](https://user-images.githubusercontent.com/78103390/235333519-09e19cef-b927-4caf-895d-f3d714adf0fe.png)

<sub>Our Custom DETR Transformer Backbone</sub>

## Results

- Receiver Operating Characteristic Curve (ROC)

![roc (1)](https://user-images.githubusercontent.com/78103390/235333384-abea82c1-fedf-4632-8444-4846be38241f.png)


- Confusion Matrix

![confusion](https://user-images.githubusercontent.com/78103390/235333379-80290281-7a0e-4e7e-a89d-43bc2a1aaf27.png)

