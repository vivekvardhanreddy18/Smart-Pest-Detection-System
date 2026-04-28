# Smart-Pest-Detection-System
Smart Pest Detection System
Overview
This project focuses on a Deep Learning-based Smart Pest Detection System designed for sustainable agriculture. It uses a custom Convolutional Neural Network (CNN) to automate the identification of crop-infesting pests, which are responsible for significant productivity losses globally.
Technical Specifications
Model Type: Custom CNN with three convolutional blocks. 
Input Size: Images are resized to $150 \times 150$ pixels. 
Accuracy: Achieved an overall test accuracy of 86.67%.
Pest Classes: Identifies 9 species including Aphids, Armyworm, Beetle, Bollworm, Grasshopper, Mites, Mosquito, Sawfly, and Stem Borer. 
Key Features
Data Augmentation: Includes rotation, shifting, shearing, and flipping to improve model generalization.
Efficiency: Designed for lower computational complexity compared to heavy transfer learning models, making it suitable for edge devices like mobile phones.  Stability: Utilizes Early Stopping to prevent overfitting and Model Checkpointing to save the best-performing weights.  
