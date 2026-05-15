# NLP project: Hurry up, I'm hungry!
### This is a project for the [Natural Language Processing](https://www.unimi.it/en/education/degree-programme-courses/2024/natural-language-processing-0) course of master's degree in Computer Science at the Università degli Studi di Milano

### Project Overview

* **Project Description**: This project focuses on Natural Language Processing (NLP) and Deep Learning to automate the generation of culinary recipes. The system utilizes a Seq2Seq architecture based on the T5 (Text-to-Text Transfer Transformer) model, fine-tuned on a specific recipes dataset. The goal is to transform a simple list of ingredients into a coherent, step-by-step cooking procedure, evaluating the performance through similarity metrics between the generated text and the original recipes.
* **Starting model**: the model used for this project is the [**T5-small**](https://huggingface.co/google-t5/t5-small)
* **Datasets used**: the datasets used for the training of the model are **RAW_recipes.csv** and **RAW_interactions.csv**, you can find them [**here**](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
* **Documentation**: description and analysis of the project are in [**this**](https://github.com/moroa01/Projects/blob/main/Recipe_Generator/Article.pdf) PDF file
* **Model Deployment**: The final fine-tuned model is available on [**HuggingFace**](https://huggingface.co/moro01525/T5_FineTuning)
