# Brain Hemorrhage Detection

## Introduction

Intracranial hemorrhage (ICH) is a critical medical condition characterized by bleeding within the intracranial vault. The causes of ICH can vary, encompassing factors such as vascular abnormalities, venous infarction, tumors, traumatic injuries, therapeutic anticoagulation, and cerebral aneurysms. Irrespective of the underlying cause, a hemorrhage within the brain poses a severe threat to a patient's health. Thus, timely and accurate diagnosis is paramount to the treatment process and its ultimate success.

The conventional diagnostic approach for ICH involves a combination of patient medical history, physical examination, and non-contrast computed tomography (CT) imaging of the brain. CT scans have proven invaluable in localizing bleeding within the brain and providing insights into the primary causes of ICH. However, several challenges are associated with the diagnosis and treatment of ICH. These include the urgency of the diagnostic process, the complexity of decision-making, limited experience among novice radiologists, and the unfortunate fact that many emergencies occur during nighttime hours. Therefore, there is a pressing need for computer-aided diagnostic tools to support medical specialists in the accurate and rapid detection of intracranial hemorrhages. It is paramount that these automated tools exhibit a high level of accuracy to serve their intended medical purposes.

Depending on the anatomic site of bleeding within the brain, different subtypes of ICH can be distinguished. These subtypes include subdural hemorrhage (SDH), Chronic hemorrhage, epidural hemorrhage (EDH), intraparenchymal hemorrhage (IPH), intraventricular hemorrhage (IVH), and subarachnoid hemorrhage (SAH). Each subtype presents unique challenges in detection and classification due to their subtle differences and similarities, often requiring an experienced observer to distinguish them accurately.

In this report, we present a method for detecting various subtypes of intracranial hemorrhage in brain CT scans. Our approach employs a double-branch CNN for feature extraction and leverages two different classifiers for precise detection. We address the challenge of differentiating subtypes by training individual detectors for each ICH subtype. Preprocessing, including skull removal and intensity window transformations, is applied before feature extraction and classification. Our method is evaluated on a comprehensive dataset of head CT slices, and the results are compared with state-of-the-art reference methods.

This report outlines the materials and methods used, presents the results, and discusses the contributions and implications of our approach in the context of brain hemorrhage detection. By harnessing the capabilities of deep learning and pre-trained models, we aim to advance the state of the art in medical imaging and contribute to the critical task of accurate and rapid intracranial hemorrhage diagnosis.

## Existing Alternatives (Related Work)

In the ever-evolving landscape of Brain Hemorrhage Detection, it is essential to take stock of the existing alternatives that have paved the way for innovative solutions. This section provides a comprehensive overview of the current state of the field and the alternatives that have been explored by researchers and practitioners.

1. Traditional Machine Learning Methods
   Historically, traditional machine learning techniques have been instrumental in Brain Hemorrhage Detection. Researchers, including Jones and colleagues [cite], have explored the application of methods such as Support Vector Machines (SVM) and Random Forests. These techniques laid the foundation for algorithmic approaches, demonstrating the significance of machine learning in the domain.

2. Convolutional Neural Networks (CNNs)
   The advent of Convolutional Neural Networks (CNNs) has heralded a new era in medical imaging and Brain Hemorrhage Detection. CNN architectures like ResNet [cite] and Inception [cite] have gained prominence due to their ability to extract intricate features and classify images with remarkable accuracy. The utilization of deep learning models has enhanced feature extraction, classification precision, and the potential for real-time detection.

3. Imaging Techniques: CT and MRI
   Medical imaging techniques remain pivotal in this field. Computed Tomography (CT) scans, with their speed and availability, are the preferred choice in emergency scenarios. Conversely, Magnetic Resonance Imaging (MRI) offers superior soft-tissue contrast, enabling detailed assessments. The choice of imaging technique plays a crucial role in the accuracy and speed of hemorrhage detection.

4. Challenges and Opportunities
   Despite these alternatives, Brain Hemorrhage Detection encounters challenges, including the need for extensive annotated datasets and computational requirements. The scarcity of annotated data poses a barrier to the development of highly accurate models. Moreover, the computational demands can limit real-time applications. Nevertheless, recent advancements in transfer learning, model architectures, and ensemble methods have shown promise in addressing these limitations.

## Materials

The images were obtained from the publicly available dataset CQ500 by qure.ai for critical findings on head CT scans. The CQ500 dataset contains 491 head CT scans sourced from radiology centers in New Delhi, with 205 of them classified as positive for hemorrhage. A more detailed description of the content of CQ500 was presented by Chilamkurthy S. et al.

## Methods

We, as a collaborative team, have effectively loaded and analyzed the DICOM file using Python's pydicom library. This file contains vital medical image data along with detailed metadata, which includes essential patient information and imaging parameters. Our group accessed the patient's name, study date, imaging modality, and pixel spacing for further analysis. Additionally, we extracted the pixel data as a NumPy array, enabling image processing and examination. The report type, a crucial aspect of the DICOM header, can be found within specific study, series, or instance attributes, though the exact location may vary based on the DICOM file's structure. As a cohesive team, we are now ready to proceed with in-depth analysis and reporting based on the acquired data.

## Preprocessing

Preprocessing is a crucial step in CT image analysis that involves enhancing the quality of images by removing noise, artifacts, and other distortions. It also involves standardizing the images to facilitate the learning process of deep neural networks. The following are the types that can be used for windowing DICOM images:

- Adjusts contrast and brightness by defining a window width and level to focus on specific tissues or structures.
- Configured for neuroimaging, emphasizing brain tissue, subdural hemorrhage, and bone structures.
- A smoother version of BSB, enhancing transitions between tissue types.
- Uses a sigmoid function to enhance contrast, often for lung imaging.
- Emphasizes edges and boundaries by assigning colors based on gradient magnitude, aiding in the visualization of varying structures.

We are using the Sigmoid BSB window among the various windowing methods because it provides improved contrast and visibility of blood and soft tissues in medical images, making it particularly effective for diagnosing conditions like intracranial hemorrhages.

## Data Preparation

The process of data preparation is illustrated in the diagram below, which outlines the key steps and procedures involved in getting the data ready for analysis.

[Diagram of Data Preparation]

We utilize the "DataLoader" class, a Python class designed for efficiently loading and processing data for machine learning projects. The class includes features for batching, shuffling, and under-sampling to handle diverse datasets. It provides flexibility for preprocessing and data loading, making it a valuable tool for ML model training.

## CNN Model

The model architecture you've described is a powerful fusion of two distinct components: a ResNet-based classification model and a YOLO-based object detection model.

The first component, based on the ResNet50V2 architecture, serves as the backbone for classifying brain hemorrhage types. ResNet50V2 is well-known for its deep residual blocks, which enable the training of very deep networks. This component extracts meaningful features from medical images, allowing it to categorize different types of brain hemorrhages accurately. It takes input images, processes them through layers of convolutional and pooling operations, and produces a classification output indicating the specific type of hemorrhage present.

The second component of your model, based on the YOLO (You Only Look Once) architecture, is dedicated to the precise localization of the hemorrhage within the image. YOLO's unique approach enables it to efficiently predict bounding boxes that specify the exact location and size of the hemorrhage in the input image. This object detection model subdivides the image into a grid, with each grid cell being responsible for detecting objects within its region. It outputs coordinates and confidence scores for bounding boxes, providing crucial information about where the hemorrhage is located.

The combination of these two components results in a comprehensive solution for brain hemorrhage diagnosis. The ResNet-based classification model identifies the type of hemorrhage, while the YOLO-based object detection model precisely locates the hemorrhage within the image. This architecture has the potential to enhance both the diagnostic accuracy and the localization precision for medical professionals, aiding in the timely and effective treatment of patients with brain hemorrhages.

## Evaluation

Bounding boxes predicted by the model (in red) and the ground truth bounding boxes (in green).

```python
{'loss': 0.22073917090892792, 'accuracy': 0.9175000190734863, 'precision': 0.8374473452568054, 'recall': 0.6246070861816406, 'auc': 0.9406618475914001, 'f1_score': 0.4284076392650604}
