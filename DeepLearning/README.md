# DeepLearning Project: Binary Classification of Melanoma using CNN
### This project was developed for the [Deep Learning con applicazioni](https://www.unimi.it/it/corsi/insegnamenti-dei-corsi-di-laurea/2025/deep-learning-con-applicazioni) course of the Master's Degree in Computer Science at the Università degli Studi di Milano.
#### The goal is to develop and evaluate different Convolutional Neural Networks (CNNs) for the task of **binary classification of melanoma (melanoma vs. benign)**.
- **Dataset used**:
the dataset used for the training of the models is the **ISIC Challenge 2018**, you can find it [**here**](https://challenge.isic-archive.com/data/#2018)
- **Presentation**: description and analysis of the project are in [**this**](https://github.com/moroa01/Projects/DeepLearning/presentation.pdf) PDF file

## Models
Three different CNNs were trained and tested:
- **Baseline CNN** (custom simple architecture), [**link**](https://huggingface.co/moro01525/MelanomaClassificationFromScratch)
- **InceptionV3** (transfer learning), [**link**](https://huggingface.co/moro01525/MelanomaClassificationInception/tree/main)
- **EfficientNetB3** (transfer learning), [**link**](https://huggingface.co/moro01525/MelanomaClassificationEfficientNet/tree/main)

## References
- [1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: *Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)*, 2018; https://arxiv.org/abs/1902.03368  
- [2] https://keras.io/api/applications/inceptionv3/
- [3] https://keras.io/api/applications/efficientnet/
