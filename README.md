_DEEPFAKE CREATION AND DETECTION_
## Introduction
This project focuses on the creation and detection of deepfakes using Cycle GANs. Deepfakes are synthetic media where a person in an existing image or video is replaced with someone else's likeness. Our goal is to provide an effective solution for both generating and detecting deepfakes to address potential threats to privacy, democracy, and security.

![alt text](
https://github.com/Charan1kh/Deepfake-Creation-Detection/blob/main/github_assets/imgs/Aboutpg.png?raw=true)
#
## Project Goals
Creation of Deepfakes: Utilize Cycle GANs to generate realistic deepfake images and videos.
Detection of Deepfakes: Develop a robust method to distinguish between real and AI-generated fake videos.
Security and Reliability: Ensure the methods are secure, user-friendly, accurate, and reliable.

![alt text](
https://github.com/Charan1kh/Deepfake-Creation-Detection/blob/main/github_assets/imgs/ProposedModel.png?raw=true)
#
## Methods Used
In this project, we employed Generative Adversarial Networks (GANs), particularly Cycle GANs, to create and detect deepfakes. Cycle GANs excel in image-to-image translation without requiring paired training data, ensuring realistic and consistent results. Convolutional Neural Networks (CNNs) were used for both generating and recognizing patterns in images. For deepfake detection, we fine-tuned pre-trained models using transfer learning, enhancing accuracy and efficiency. Data augmentation techniques, such as random cropping and flipping, were implemented to diversify the training data, improving model generalization. Performance metrics like accuracy, precision, recall, and F1 score were used to evaluate our models' effectiveness.
![alt text](https://github.com/Charan1kh/Deepfake-Creation-Detection/blob/main/github_assets/imgs/SystemArchitecture.png?raw=true)
#
## Project Structure
Overall the strucutre should look like this:
![alt text](
https://github.com/Charan1kh/Deepfake-Creation-Detection/blob/main/github_assets/imgs/ProjectStructure.png?raw=true)
#
## Technical Requirements
Software: Python, PyTorch
Hardware: High-performance GPU for training deep learning models
Libraries: CycleGAN, GANs, Autoencoders
Versions Used
Python: 3.8
PyTorch: 1.7
CUDA: 11.0


### - Full dataset: 
link: https://www.kaggle.com/c/deepfake-detection-challenge/data

### - Trained Models:
link: https://drive.google.com/drive/folders/1UX8jXUXyEjhLLZ38tcgOwGsZ6XFSLDJ-

### - PyTorch installation with PIP for CPU
1. pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu
torchaudio===0.8.1 -f
https://download.pytorch.org/whl/torch_stable.html
2. PyTorch installation with PIP for GPU 10.2
pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102
torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

3. PyTorch installation with PIP for GPU 11.1
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111
torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

4. To verify installation
import torch
```print(torch.__version__)```

5. Save the file, then open your command line and change directory to where you saved the file. Then type python versions.py

6. You should then see output like the following: ```1.7.1+cpu```
This confirms that PyTorch is installed correctly and that we are all using the same version.
#
## Results
High Accuracy: Achieved significant accuracy in detecting deepfakes, outperforming existing models.
Realistic Deepfakes: Generated deepfake images and videos that are almost indistinguishable from real ones.
Effective Detection: Developed methods that effectively identify AI-generated fake videos.


![alt text](
https://github.com/Charan1kh/Deepfake-Creation-Detection/blob/main/github_assets/imgs/detectionpg.png?raw=true)
#
## Conclusion
This project successfully demonstrates the creation and detection of deepfakes using Cycle GANs. The developed methods provide a significant step towards mitigating the risks associated with deepfakes. Future work will focus on improving detection accuracy and integrating the solution with social media platforms for real-time deepfake detection.
