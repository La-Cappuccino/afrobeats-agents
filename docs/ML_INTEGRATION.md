# Afrobeats.no Machine Learning Integration

This document outlines the machine learning components integrated into the Afrobeats.no platform.

## Overview

The platform leverages machine learning for several key features:
- Recommendation systems for DJs, events, and playlists
- Natural language processing for user queries and content analysis
- Trend detection for song popularity and event engagement
- Similarity search for music and DJs

## Recommendation Systems

### DJ Recommendation Engine

The DJ recommendation system uses a hybrid approach combining collaborative filtering and content-based methods.

```python
# ml/recommendation/dj_recommender.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .base_recommender import BaseRecommender

class DJRecommender(BaseRecommender):
    """
    Recommends DJs based on user preferences and past bookings
    """
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.dj_embeddings = None
        self.user_preferences = None
        
    async def load_data(self):
        """Load necessary data from Supabase"""
        # Load DJ embeddings
        dj_response = await self.supabase.table('dj_embeddings').select('dj_id, embedding').execute()
        dj_data = dj_response.data
        
        # Create embedding matrix
        self.dj_ids = [d['dj_id'] for d in dj_data]
        self.dj_embeddings = np.array([d['embedding'] for d in dj_data])
        
        # Load user interactions
        interaction_response = await self.supabase.table('user_interactions').select(
            'user_id, dj_id, interaction_type'
        ).eq('interaction_type', 'book').execute()
        self.user_bookings = pd.DataFrame(interaction_response.data)
        
    async def get_recommendations(self, user_id, preferences, num_recommendations=5):
        """
        Get DJ recommendations for a user
        
        Args:
            user_id: User ID to get recommendations for
            preferences: Dictionary of user preferences (genres, etc.)
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended DJ IDs with scores
        """
        if self.dj_embeddings is None:
            await self.load_data()
        
        # Calculate content-based scores
        user_vector = self._create_user_vector(preferences)
        content_scores = self._calculate_content_scores(user_vector)
        
        # Calculate collaborative scores
        collab_scores = self._calculate_collaborative_scores(user_id)
        
        # Combine scores
        final_scores = 0.7 * content_scores + 0.3 * collab_scores
        
        # Get top DJs
        top_indices = np.argsort(final_scores)[::-1][:num_recommendations]
        recommendations = [
            {"dj_id": self.dj_ids[i], "score": float(final_scores[i])}
            for i in top_indices
        ]
        
        return recommendations
        
    def _create_user_vector(self, preferences):
        """Create a vector representing user preferences"""
        # Implementation depends on the structure of DJ embeddings
        pass
        
    def _calculate_content_scores(self, user_vector):
        """Calculate similarity between user vector and DJ embeddings"""
        return cosine_similarity([user_vector], self.dj_embeddings)[0]
        
    def _calculate_collaborative_scores(self, user_id):
        """Calculate collaborative filtering scores"""
        # Implementation of collaborative filtering
        pass
```

### Playlist Recommendation

The playlist recommendation system uses embeddings of songs and user preferences:

```python
# ml/recommendation/playlist_recommender.py
class PlaylistRecommender:
    # Similar implementation to DJRecommender, but for playlists
    pass
```

## Natural Language Processing

### Intent Classification

The system uses a fine-tuned model to classify user queries into intents:

```python
# ml/nlp/intent_classifier.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class IntentClassifier:
    """
    Classifies user queries into specific intents
    """
    def __init__(self, model_path="afrobeats/intent-classifier"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.intents = [
            "book_dj", "find_event", "get_playlist", "rate_dj", 
            "get_news", "social_media", "get_analytics"
        ]
        
    def classify(self, text):
        """
        Classify text into an intent
        
        Args:
            text: The user query text
            
        Returns:
            The predicted intent and confidence score
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(scores, dim=1).item()
        
        return {
            "intent": self.intents[prediction],
            "confidence": scores[0][prediction].item()
        }
```

### Sentiment Analysis

Sentiment analysis is used for DJ reviews and forum content:

```python
# ml/nlp/sentiment_analyzer.py
class SentimentAnalyzer:
    """
    Analyzes sentiment of text content
    """
    # Implementation using pre-trained model
    pass
```

## Vector Search

The platform uses pgvector in Supabase for similarity search:

```python
# ml/embeddings/song_embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer

class SongEmbedder:
    """
    Generates embeddings for songs
    """
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        
    def generate_embedding(self, song_data):
        """
        Generate embedding for a song
        
        Args:
            song_data: Dictionary with song information
            
        Returns:
            Embedding vector
        """
        # Create a rich text representation of the song
        text = f"{song_data['title']} by {song_data['artist']}. "
        text += f"Album: {song_data['album']}. "
        text += f"Genres: {', '.join(song_data['genre'])}."
        
        # Generate embedding
        embedding = self.model.encode(text)
        
        return embedding.tolist()
    
    async def store_embedding(self, supabase_client, song_id, embedding):
        """
        Store embedding in Supabase
        
        Args:
            supabase_client: Supabase client
            song_id: Song ID
            embedding: Embedding vector
            
        Returns:
            Response from Supabase
        """
        # Store embedding
        response = await supabase_client.table('song_embeddings').upsert({
            'song_id': song_id,
            'embedding': embedding
        }).execute()
        
        return response
```

## Trend Detection

The platform uses time series analysis to detect trends:

```python
# ml/trend_detection/song_trend_detector.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

class SongTrendDetector:
    """
    Detects trends in song popularity
    """
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        
    async def get_song_trend_data(self, song_id, weeks=12):
        """
        Get historical trend data for a song
        
        Args:
            song_id: Song ID
            weeks: Number of weeks of history to get
            
        Returns:
            DataFrame with trend data
        """
        # Get historical ranking data
        response = await self.supabase.table('song_rankings').select(
            'rank, week_number, year'
        ).eq('song_id', song_id).order('year', desc=True).order('week_number', desc=True).limit(weeks).execute()
        
        # Convert to DataFrame
        df = pd.DataFrame(response.data)
        if df.empty:
            return None
            
        # Sort by date (newest to oldest)
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-W' + df['week_number'].astype(str) + '-1', format='%Y-W%W-%w')
        df = df.sort_values('date')
        
        return df
        
    def predict_future_trend(self, trend_data, weeks_ahead=4):
        """
        Predict future trend based on historical data
        
        Args:
            trend_data: DataFrame with historical trend data
            weeks_ahead: Number of weeks to predict ahead
            
        Returns:
            Predicted ranks for future weeks
        """
        if trend_data is None or len(trend_data) < 8:  # Need enough data
            return None
            
        # Fit ARIMA model
        model = ARIMA(trend_data['rank'].values, order=(1, 1, 1))
        model_fit = model.fit()
        
        # Forecast
        forecast = model_fit.forecast(steps=weeks_ahead)
        
        # Create result with dates
        last_date = trend_data['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=weeks_ahead, freq='W')
        
        result = pd.DataFrame({
            'date': future_dates,
            'predicted_rank': forecast
        })
        
        return result
```

## ML Model Management

The platform uses a central model registry:

```python
# ml/model_registry.py
import os
import json
from datetime import datetime

class ModelRegistry:
    """
    Registry for ML models
    """
    def __init__(self, registry_path):
        self.registry_path = registry_path
        self.registry = self._load_registry()
        
    def _load_registry(self):
        """Load the registry from disk"""
        if os.path.exists(os.path.join(self.registry_path, 'registry.json')):
            with open(os.path.join(self.registry_path, 'registry.json'), 'r') as f:
                return json.load(f)
        return {}
        
    def _save_registry(self):
        """Save the registry to disk"""
        os.makedirs(self.registry_path, exist_ok=True)
        with open(os.path.join(self.registry_path, 'registry.json'), 'w') as f:
            json.dump(self.registry, f)
        
    def register_model(self, model_name, model_version, model_path, metrics=None):
        """
        Register a new model
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            model_path: Path to the model files
            metrics: Dictionary of model performance metrics
            
        Returns:
            The registered model details
        """
        if model_name not in self.registry:
            self.registry[model_name] = {}
            
        self.registry[model_name][model_version] = {
            'path': model_path,
            'metrics': metrics or {},
            'registered_at': datetime.now().isoformat()
        }
        
        self._save_registry()
        return self.registry[model_name][model_version]
        
    def get_model(self, model_name, model_version=None):
        """
        Get a model from the registry
        
        Args:
            model_name: Name of the model
            model_version: Version of the model (None for latest)
            
        Returns:
            The model details or None if not found
        """
        if model_name not in self.registry:
            return None
            
        if model_version is None:
            # Get the latest version
            versions = list(self.registry[model_name].keys())
            if not versions:
                return None
            model_version = max(versions)
            
        return self.registry[model_name].get(model_version)
```

## Integration with Agent System

The ML components are integrated with the agent system:

```python
# agents/coordinator_agent.py (excerpt)
from ml.nlp.intent_classifier import IntentClassifier

# Initialize the intent classifier
intent_classifier = IntentClassifier()

def determine_agent(user_query: str) -> str:
    """
    Determine which agent should handle a user query
    
    Args:
        user_query: The user's query
        
    Returns:
        The name of the agent that should handle the query
    """
    # First, use ML to classify the intent
    intent_result = intent_classifier.classify(user_query)
    
    # If confidence is high, use the predicted intent
    if intent_result['confidence'] > 0.85:
        intent = intent_result['intent']
        if intent == 'book_dj':
            return 'dj_booking'
        elif intent == 'find_event':
            return 'event_discovery'
        elif intent == 'get_playlist':
            return 'playlist'
        elif intent == 'rate_dj':
            return 'dj_rating'
        elif intent == 'get_news':
            return 'content'
        elif intent == 'social_media':
            return 'social_media'
        elif intent == 'get_analytics':
            return 'analytics'
    
    # If confidence is low, fall back to LLM-based routing
    # ... (existing LLM-based routing code)
```

## Training Pipeline

The platform includes scripts for training ML models:

```python
# scripts/train_intent_classifier.py
import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from ml.model_registry import ModelRegistry

def main(args):
    # Load dataset
    df = pd.read_csv(args.data_path)
    
    # Split dataset
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=len(df['intent'].unique())
    )
    
    # Tokenize data
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)
    
    train_encodings = train_df.apply(tokenize, axis=1).tolist()
    eval_encodings = eval_df.apply(tokenize, axis=1).tolist()
    
    # Create datasets
    class IntentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
            
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item
        
        def __len__(self):
            return len(self.labels)
    
    train_dataset = IntentDataset(train_encodings, train_df['intent_id'].values)
    eval_dataset = IntentDataset(eval_encodings, eval_df['intent_id'].values)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch'
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Train model
    trainer.train()
    
    # Evaluate model
    eval_results = trainer.evaluate()
    
    # Save model
    model_path = f'./models/intent-classifier-{args.version}'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Register model
    registry = ModelRegistry('./model_registry')
    registry.register_model(
        'intent-classifier',
        args.version,
        model_path,
        metrics=eval_results
    )
    
    print(f"Model trained and saved to {model_path}")
    print(f"Evaluation results: {eval_results}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--version', type=str, required=True, help='Model version')
    args = parser.parse_args()
    
    main(args)
```

## Monitoring and Evaluation

The platform includes tools for monitoring ML model performance:

```python
# ml/monitoring/performance_tracker.py
class ModelPerformanceTracker:
    """
    Tracks performance of ML models in production
    """
    # Implementation
    pass
```