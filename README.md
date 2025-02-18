# ASL Recognition Project

This project focuses on recognizing American Sign Language (ASL) gestures using a custom dataset and a variety of machine learning (ML) models. The goal is to explore the data through visualization, reason through our initial preconceptions, and compare several ML approaches to determine the most effective method for ASL recognition.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Data Visualization](#data-visualization)
- [Preconceptions and Reasoning](#preconceptions-and-reasoning)
- [Machine Learning Models](#machine-learning-models)
- [Getting Started](#getting-started)
- [Results and Evaluation](#results-and-evaluation)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

The project involves:
- **Dataset Selection:** Choosing a dataset that represents various ASL gestures.
- **Data Visualization:** Exploring the dataset through visualizations to understand its distribution, potential biases, and underlying patterns.
- **Hypothesis Formation:** Documenting preconceptions about the data and the challenges in recognizing similar ASL gestures.
- **Model Implementation:** Using multiple machine learning models—including convolutional neural networks (CNNs), support vector machines (SVMs), and ensemble methods—to analyze and classify the ASL images.
- **Evaluation:** Comparing model performance to select the most effective approach for ASL recognition.

## Dataset

The dataset consists of images depicting different ASL signs. Each image is labeled according to the corresponding alphabet or gesture. Detailed information about the dataset, including sample images and class distributions, can be found in the `notebooks/` folder.

*Note: If the dataset is not included directly in the repository, please refer to the instructions in the `data/` folder for downloading and setting up the dataset.*

## Project Structure


## Data Visualization

Visual exploration of the dataset was an integral first step. The visualizations include:
- **Class Distribution:** Bar charts showing the frequency of each ASL sign.
- **Sample Images:** A gallery of sample images per category to inspect quality and variability.
- **Correlation Analysis:** (If applicable) Plots to explore relationships between any extracted features.

These visualizations help uncover potential challenges such as imbalanced classes or high intra-class variability.

## Preconceptions and Reasoning

Before diving into model training, several preconceptions were considered:
- **Image Variability:** ASL gestures may vary widely in appearance due to lighting, background, and signer differences.
- **Model Performance:** Different ML models might have varying levels of performance, especially given the complexity of image data.
- **Feature Discrimination:** Some ASL signs are visually similar, posing a challenge for classifiers.

These insights guided the exploratory data analysis (EDA) and influenced the selection and tuning of machine learning models.

## Machine Learning Models

Multiple approaches were experimented with:
- **Convolutional Neural Networks (CNNs):** To leverage their strength in capturing spatial hierarchies in image data.
- **Support Vector Machines (SVMs):** For baseline classification on feature-extracted representations.
- **Random Forests and Ensemble Methods:** To reduce overfitting and capture complex decision boundaries.
- **Additional Models:** Other classifiers were also considered to provide a comprehensive comparison.

Each model’s architecture, training process, and evaluation metrics are detailed in the scripts under the `src/` directory.

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

