# Content to Topic Classifier

## Overview of Approach
1. Data was downloaded from [kaggle competition](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/data)
2. In order to understand the data, an EDA was made (eda.ipynb).
3. Solution uses a simple TF-IDF vectorizer and cosine similarity to match content items to topics based on their textual fields (title, description, text). 

## Code Access Point
* Start with `src/main.py`.
* Define the content input.


## Metrics

Using the input `TopicPredictionRequest(content=Content(id='c_00002381196d', title='Sumar números de varios dígitos: 48,029+233,930 ', description='Suma 48,029+233,930 mediante el algoritmo estándar.', language='es', kind='video', copyright_holder=None, license=None, text=None))` where obtained the top 5 topic ids using Jaccard Similarity and cosine_similarity. 

When comparing the predicted topics with the correlated topic IDs related to the content ID 'c_00002381196d', not even one ID matched.

Predicted topic ids: ['t_be92ba81bfe2', 't_a3cb4c5904c6', 't_615f00ba4735', 't_565a94d1d001', 't_dc56cdcf0182']
Correlated topic ids with content id: ['t_81be1094dd83', 't_d0edb1c53d90', 't_d66311c2e171', 't_e696cda1adb6', 't_f7f7dbd7d76a']

## What would you have done with more time?
- Experimented with deep learning models (e.g., transformers).
- Hyperparameter tuning of the classification that I implemented, but it doesn't work.
- Ensemble multiple approaches.
- Include semantic search.


## Installation

install [uv](https://docs.astral.sh/uv/guides/install-python/) and run `uv run src/main.py`

or 

```bash
pip install -r requirements.txt
```


## Testing

To test the classifier, run `uv run src/main.py` or `python3 src/main.py` 
