# ASL Recognition Project

This project focuses on recognizing American Sign Language (ASL) gestures using a the ASL Alphabet dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
and a variety of machine learning (ML) models. The goal is to explore the data through visualization, reason through our initial preconceptions, and compare several ML approaches to determine the most effective method for ASL recognition.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Data Visualization](#data-visualization)
- [Preconceptions and Reasoning](#preconceptions-and-reasoning)
- [Machine Learning Models](#machine-learning-models)
- [Getting Started](#getting-started)
- [Results and Evaluation](#results-and-evaluation)


## Project Overview

The project involves:
- **Dataset Selection:** Choosing a dataset that represents various ASL gestures.
- **Data Visualization:** Exploring the dataset through visualizations to understand its distribution, potential biases, and underlying patterns.
- **Hypothesis Formation:** Documenting preconceptions about the data and the challenges in recognizing similar ASL gestures.
- **Model Implementation:** Using multiple machine learning models—including convolutional neural networks (CNNs), support vector machines (SVMs), and ensemble methods—to analyze and classify the ASL images.
- **Evaluation:** Comparing model performance to select the most effective approach for ASL recognition.

## Dataset

The dataset consists of images depicting different ASL signs. Each image is labeled according to the corresponding alphabet or gesture. Detailed information about the dataset, including sample images and class distributions, can be found in the link provided above.

## Project Structure

In this project I included several ML models to test my hypothesis: "By using data augmentation, deep learning and feature extracting, we can improve the issue of ASL Alphabet recognition - similar hand gestures".
The project was devided to 3 main proccesses:
- **Invetigation Phase:** Finding the best possible dataset, thinking of a hypothesis and planning my way of work.
- **Coding Phase:** Writing the ML models and tuning the hyperparameters that will get the most out of the model.
- **Conclusion Phase:** Understanding where I failed and where I found success, looking for a potential answer for the hypothesis.

## Data Visualization

Visual exploration of the dataset was an integral first step. The visualizations include:
- **Class Distribution:** Bar charts showing the frequency of each ASL sign.
- **Sample Images:** A gallery of sample images per category to inspect quality and variability.

These visualizations help uncover potential challenges such as imbalanced classes or high intra-class variability.

## Preconceptions and Reasoning

Before diving into model training, several preconceptions were considered:
- **Image Variability:** ASL gestures may vary widely in appearance due to lighting, background, and signer differences.
- **Model Performance:** Different ML models might have varying levels of performance, especially given the complexity of image data.
- **Feature Discrimination:** Some ASL signs are visually similar, posing a challenge for classifiers.

These insights guided the exploratory data analysis and influenced the selection and tuning of machine learning models.

## Machine Learning Models

Multiple approaches were experimented with:
- **Convolutional Neural Networks (CNNs):** To leverage their strength in capturing spatial hierarchies in image data.
- **Feedforward Neural Network (FFNN):** To present the huge improvement when using neural networks.
- **Logistic Regression:** To give an example of the problems we might face when using the wrong model for the task.

Each model’s architecture, training process, and evaluation metrics are detailed in the scripts under the evaluation and it's own directory.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook (for running the notebooks)
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mmitzy/ASL-Recognition.git
   cd ASL-Recognition
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Results and Evaluation

Each model was evaluated using metrics such as accuracy, precision, recall, and F1-score. Detailed results are documented in the pictures below.
- Model strengths and weaknesses.
- The impact of data pre-processing and augmentation.
- Comparative performance across different classifiers.
