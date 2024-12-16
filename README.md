# Machine Learning Project: Scientific Paper Keyword Prediction

## Description
This project aims to develop a machine learning system capable of predicting relevant keywords (e.g., *"solar wind"*, *"lunar composition"*) by analyzing titles and abstracts of scientific papers. The system performs **multi-label text classification**, meaning each document may be assigned multiple keywords to capture diverse themes or topics.

The annotated dataset for this project is provided by the **NASA ADS/SciX** team and is available through the **HuggingFace repository**. The dataset, known as the **SciX corpus**, comprises:
- **Training Set**: 18,677 documents (titles and abstracts).
- **Test Set**: 3,025 documents (titles and abstracts).

---

## Project Structure
The project is organized as follows:

```
.
├── data/                       # Directory containing datasets
├── doc/                        # Documentation folder
├── README.md                   # Project documentation
├── data_processing.py          # Script containing the utils function
├── hybrid.ipynb                # Notebook for a hybrid model between rule and machine learning model approach
├── model_tfidf.ipynb           # Notebook implementing TF-IDF vectorization for machine learning models
└── rule_prediction.ipynb       # Notebook for rule-based keyword predictions
```

---

## Requirements
To run the project, ensure you have the following dependencies installed:

- Python 3.11+  
You can install all required libraries using:
```bash
pip install -r requirements.txt
```

---

## Dataset
The annotated dataset (SciX corpus) can be accessed via HuggingFace:
https://huggingface.co/datasets/adsabs/SciX_UAT_keywords/tree/main/data
---

## Usage

To observe the results, simply consult the provided notebooks and scroll to the end where the metrics are displayed.

- **Hybrid Notebook**: Performs its metrics tests on the `val` dataset.
- **Model_TFIDF Notebook**: Conducts its metrics tests on the test split of the `train` dataset.
- **Rule_Prediction.ipynb**: Executes its metrics tests directly on the `train` dataset.

---

## Results
The system is evaluated on the test set using multi-label classification metrics such as:
- **Precision**
- **Hamming loss**
- **F1-Score**
