# Brain Tumor Detection Project

## Brain Tumor Detection from MRI Images with Deep Learning

### Overview
The project employs MRI images to predict and categorize brain tumor types, utilizing various machine learning models, image preprocessing techniques, and data augmentation.
Developed deep learning models for brain tumor detection using MRI images by employing MLP, CNN, VGG16, EfficientNet B2, and InceptionV3 models. Achieved the best performance using InceptionV3 with an F1 score of 93.55% and an accuracy score of 94.74%. Performed pre-processing using data generators for image processing and augmentation with the ImageDataGenerator.

### Details
#### Project Description
Brain tumors are becoming increasingly prevalent, posing a significant healthcare challenge. This rising occurrence necessitates early detection for effective treatment. Deep Learning can be used to analyze complex data and identify patterns for medical image analysis. It can be leveraged to enhance accuracy, speed, and efficiency in diagnosing diseases. Our project aims to contribute to the development of systems that can identify these fatal diseases in their initial stages. Our objective is to develop a precise deep-learning model for accurate classification of brain MRI images with tumors to improve early detection of brain tumors, leading to better patient outcomes.

### Contributions in the Project Domain
- Synthesizing contrasts using DL to correct distortions in MRI scans.
- Creating and leveraging several DL models like DeepBrainNet to construct accurate classifiers for brain diseases.
- Leveraging probabilistic labels for enhanced image-based diagnostics.
- Vast analysis of the current limitations, benefits, and future prospects in Medical Imaging Diagnosis.

### Project Goals and Benefits
#### Goals
- Develop a deep learning model for accurate brain tumor classification.
- Improve early detection and classification of brain tumors using advanced techniques.

#### Benefits
- **Better Patient Outcomes:** Early detection and accurate classification enable timely intervention and tailored treatment plans.
- **Enhanced Healthcare Efficiency:** Early detection reduces invasive procedures, healthcare costs, and optimizes resource allocation.
- **Empowering Healthcare Providers:** A reliable deep learning model assists in accurate diagnoses, reducing the manual analysis burden.
- **Advancement in Research and Knowledge:** Contributes to the knowledge base in brain tumor detection, fostering collaboration and advancements.

### Deep Learning and Explorations
#### Project Components - Data Set
The image data came from the Brain MRI Images for Brain Tumor Detection dataset. It consists of MRI scans of two classes, stored in separate files:
- **NO (98):** no tumor, encoded as 0
- **YES (155):** tumor, encoded as 1

![Screenshot 1](https://github.com/Uditishah/Brain-Tumor-Detection-/assets/157534932/736d9b68-cb64-4ab8-ab1a-7416ff5af959)
## Project Components - Preprocessing

I set up data generators for image processing and augmentation using the Keras ImageDataGenerator. This technique allowed me to artificially increase the size of the training dataset and improve the generalization ability of my model. This, in turn, helped me address the problem of overfitting and made my model more robust, enabling it to handle different types of images.

I used these data generators to provide augmented images in batches during model training, testing, and validation processes. With this approach, my model was able to learn from a wider range of examples, enabling it to perform well on new, unseen images.

### Deep Learning Models Used + Performance

I employed a range of deep learning models in my project, each serving a specific purpose:

![Screenshot 2](https://github.com/Uditishah/Brain-Tumor-Detection-/assets/157534932/8380ba2a-536c-4f7f-ba79-b588793922f4)

- **MLP (Multi-Layer Perceptron):** This model was widely utilized for both classification and regression tasks, showcasing its versatility in handling various data types.
- **CNN (Convolutional Neural Network):** For image processing tasks, I turned to CNN, a well-established choice known for its effectiveness in handling visual data.
- **VGG16:** This model was specifically chosen for image classification tasks, as it has a proven track record of achieving high accuracy in this domain.
- **EfficientNetB2:** My project benefited from the impressive performance of EfficientNetB2. This member of the EfficientNet family offers improved accuracy and efficiency for image-related training tasks. It utilizes a novel scaling technique that uniformly adjusts network dimensions, including width, depth, and image resolution, resulting in optimal performance without the need for manual tuning.

![Screenshot 2024-07-07 at 10 12 49 PM](https://github.com/Uditishah/Brain-Tumor-Detection-/assets/157534932/855f02de-8dca-4cc6-80a5-7bfd713934a7)

- **InceptionV3:** To further enhance image recognition capabilities, I employed InceptionV3. This advanced model incorporates a variety of building blocks and techniques to achieve outstanding performance. Its architecture features both symmetric and asymmetric building blocks, including convolutional layers, average pooling, max pooling, concatenations, dropouts, and fully connected layers. The extensive use of batch normalization enhances stability and speeds up the training process. Moreover, InceptionV3 employs the softmax function for loss computation, a crucial component in multi-class classification tasks like image recognition.

![Screenshot 2024-07-07 at 10 13 20 PM](https://github.com/Uditishah/Brain-Tumor-Detection-/assets/157534932/52ff1fd5-c8d6-49c6-ac86-f03ba07a8568)

In terms of performance, EfficientNet and InceptionV3 stood out as the top performers in my project, demonstrating their capability to excel in computer vision tasks and image-related challenges.

### Next Steps

For fine-tuning and optimizing the model, I undertook the following steps:

- **Architectural Modifications:** I made adjustments to the model’s architecture, including alterations to hyperparameters and regularization techniques, in order to enhance its performance.
- **Data Augmentation:** I implemented additional data augmentation methods, such as image cropping, to improve the quality of training data and the model’s ability to generalize effectively.
- **Optimization Strategies:** I conducted experiments with various optimization strategies, exploring different learning rates, batch sizes, and regularization techniques. This iterative process aimed to fine-tune the model’s parameters for optimal performance.

In the pursuit of continual refinement and evaluation:

- **Dataset Variation:** To ensure the model’s robustness, I tested it on various datasets and augmented data sources, verifying its consistent and reliable performance across different scenarios.
- **Performance Analysis:** I rigorously analyzed the model’s outputs and systematically made improvements based on the evaluation results, ensuring that it continued to meet the desired performance criteria.

### Documentation

Regarding documentation:

- **Comprehensive Documentation:** I diligently documented the entire development process, capturing the methods employed and the results obtained. This documentation provided transparency and served as a valuable resource for future reference.
- **Ongoing Documentation:** This documentation process was not a one-time effort; I maintained it throughout the further refinement and optimization stages, ensuring that each iteration and improvement was well-documented for a complete record of the project’s evolution.
