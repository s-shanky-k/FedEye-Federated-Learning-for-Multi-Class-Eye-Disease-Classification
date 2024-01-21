# FedEye: Federated Learning for Multi Class Eye DiseaseClassification
In a real-world medical setting, performing accurate and robust medical disease classification requires a large amount of sensitive data. However, most medical institutes have strict rules and
policies when it comes to sharing patient data with external parties. 

To deal with these challenges, this project proposes a robust approach to classify eye diseases in retinal images using federated learning. It aims to collaboratively train a machine learning model without the need for sharing or exchanging raw data across various sites.

## Methodology

The proposed method incorporates federated learning techniques to perform robust eye disease classification on retinal images. ResNet50 and VGG19 architectures are used to train a classification model by making use of transfer learning techniques. Further, the performance of these models is compared against simulated federated learning models based on the ResNet50 and VGG19 architectures. The proposed federated learning model combines the advantages of large computer vision models with distributed learning to ensure the privacy of data.

The project uses the [Flower](https://flower.dev/) federated learning framework and [PyTroch](https://pytorch.org/) to train the models.

## Dataset

The experiments are conducted on the popular benchmark retinal eye disease classification [dataset](https://www.kaggle.com/datasets/kondwani/eye-disease-dataset). This dataset is used since there are multiple classes namely Normal, Diabetic Retinopathy, Cataract and Glaucoma retinal images where each class has approximately 1000 images. There are a total of 4217 images in the dataset across the four classes split uniformly. We split the dataset into training, validation and test set in the ratio 7:2:1 respectively i.e 3036, 759 and 422 images respectively.

## Results

The table provided offers a comprehensive summary of the outcomes derived from various models under distinct training modes.

| S.No | Model | Standard Test Accuracy % | Federated Test Accuracy % |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
1|ResNet fully frozen|88.38|90.52
2|ResNet partially frozen|92.18|95.73
3|VGG fully frozen|88.62|84.59
4|VGG partially frozen|91.70|26

## Repositary Structure

**`Code\federated\`** : Contains jupyter notebooks which consists of code for training the models using federated learning on the dataset. These models were trained using 4217 images. 759 images were used for validation and 422 images were used for testing.

**`Code\standard\`** : Contains jupyter notebooks which consists of code for training the models using the standard methodology on the dataset. These models were trained using 4217 images. 759 images were used for validation and 422 images were used for testing.

**`Documentation\`**: Contains a detailed report of the entire project. The report consists of training details as well.
