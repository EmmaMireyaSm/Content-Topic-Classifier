from predict_template import TopicPredictor, TopicPredictionRequest
from dacite import from_dict


# Example input for TopicPredictionRequest
example_input = {
    "content": {
        "id": "content_001",
        "title": "Introduction to Fractions",
        "description": "A lesson about basic fractions and their properties.",
        "language": "en",
        "kind": "document",
        "text": "Fractions represent parts of a whole. This lesson covers numerator, denominator, and simple operations.",
        "copyright_holder": "Open Education",
        "license": "CC-BY"
    }
}


request = from_dict(data_class=TopicPredictionRequest, data=example_input)
predictor = TopicPredictor()
topic_ids = predictor.predict(request)
print("Predicted topic ids input 1:", topic_ids)


input_2 = {
    "content": {
        "id": "c_00002381196d",
        "title": "Sumar números de varios dígitos: 48,029+233,930 ",
        "description": "Suma 48,029+233,930 mediante el algoritmo estándar.",
        "language": "es",
        "kind": "video",
    }
}
request = from_dict(data_class=TopicPredictionRequest, data=input_2)
topic_ids = predictor.predict(request)
print("\nPredicted topic ids input 2:", topic_ids)

correlated_topic_ids = predictor._get_correlated_topics("c_00002381196d")
print("\nCorrelated topic ids with content id:", correlated_topic_ids)
