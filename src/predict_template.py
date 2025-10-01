import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


@dataclass
class Content:
    id: Optional[str]
    title: Optional[str]
    description: Optional[str]
    language: Optional[str]
    kind: Optional[str]
    copyright_holder: Optional[str]
    license: Optional[str]
    text: Optional[str] 


@dataclass
class TopicPredictionRequest:
    content: Content


class TopicPredictor:

    def __init__(
        self,
        topics_path="data/learning-equality-curriculum-recommendations/topics.csv",
        content_path="data/learning-equality-curriculum-recommendations/content.csv",
        correlations_path="data/learning-equality-curriculum-recommendations/correlations.csv",
    ):
        # Load topics and content data
        self.topics = pd.read_csv(topics_path)
        self.content = pd.read_csv(content_path)
        self.correlations = pd.read_csv(correlations_path)

        # Preprocess topic text (combine title and description)
        self.topics['text'] = (
            self.topics['title'].fillna('') + ' ' +
            self.topics['description'].fillna('')
        )

        self.content['text_content'] = (
            self.content['title'].fillna('') + ' ' +
            self.content['description'].fillna('') + ' ' +
            self.content['text'].fillna('')
        )

        # TF-IDF
        self.vectorizer = TfidfVectorizer()
        self.topic_tfidf = self.vectorizer.fit_transform(self.topics['text'])
        # # Embeddings
        # self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # self.topic_embeddings = self.embedder.encode(self.topics['text'].tolist(), show_progress_bar=True)

        # Map topic ids to index for lookup
        self.topic_id_to_idx = {tid: i for i, tid in enumerate(self.topics['id'])}

    @staticmethod
    def jaccard_similarity(str1, str2):
        """
        Compute Jaccard Similarity between two strings.
        """
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        if not set1 or not set2:
            return 0.0
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return float(len(intersection)) / len(union)

    def predict(self, request, use_jaccard=False):
        """
        Given a TopicPredictionRequest, return a list of topic ids that match the content.
        If use_jaccard is True, use Jaccard Similarity instead of TF-IDF.
        """
        if request.content.language:
            # Filter topics by language if specified
            lang_filtered_topics = self.topics[self.topics['language'] == request.content.language]
            if not lang_filtered_topics.empty:
                self.topic_tfidf = self.vectorizer.transform(lang_filtered_topics['text'])
                self.topics = lang_filtered_topics
        if request.content.id:
            # If content id is provided, find directly from content dataset
            content_from_id = self.content.loc[self.content['id'] == request.content.id, ['title', 'description', 'text']]
            if not content_from_id.empty:
                content_text = content_from_id.to_string()
            else:
                content_text = (request.content.title or '') + ' ' + (request.content.description or '') + ' ' + (request.content.text or '')
        else:
            # Combine content title and description
            content_text = (request.content.title or '') + ' ' + (request.content.description or '') + ' ' + (request.content.text or '')

        if use_jaccard:
            # Jaccard Similarity
            jaccard_scores = self.topics['text'].apply(lambda topic_text: self.jaccard_similarity(content_text, topic_text))
            top_n = 5
            top_indices = np.argsort(jaccard_scores)[::-1][:top_n]
            recommended_topic_ids = self.topics.iloc[top_indices]['id'].tolist()
            return recommended_topic_ids
        else:
            # TF-IDF similarity
            self.content_vec = self.vectorizer.transform([content_text])
            tfidf_sims = cosine_similarity(self.content_vec, self.topic_tfidf).flatten()
            top_n = 5
            top_indices = np.argsort(tfidf_sims)[::-1][:top_n]
            recommended_topic_ids = self.topics.iloc[top_indices]['id'].tolist()
            return recommended_topic_ids

    def _get_correlated_topics(self, content_id: None):
        """
        Return a list of topic IDs from correlations.
        """
        correlations = self.correlations.copy()
        correlations["content_ids"] = correlations["content_ids"].str.split()
        correlations = correlations.explode("content_ids")
        if content_id:
            correlated = correlations.loc[correlations['content_ids'] == content_id, 'topic_id']
        return correlated.tolist()
