from typing import Dict, Any, List

class ModelConfig:
    """Configuration for different embedding models"""
    
    MODELS = {
        'bge-small': {
            'name': 'BAAI/bge-small-en-v1.5',
            'dimension': 384,
            'batch_size': 32,
            'description': 'Fast, good for testing'
        },
        'bge-large': {
            'name': 'BAAI/bge-large-en-v1.5',
            'dimension': 1024,
            'batch_size': 16,
            'description': 'Better quality, slower'
        },
        'all-mpnet': {
            'name': 'sentence-transformers/all-mpnet-base-v2',
            'dimension': 768,
            'batch_size': 32,
            'description': 'Good general purpose'
        },
        'openai-small': {
            'name': 'text-embedding-3-small',
            'dimension': 1536,
            'batch_size': 100,
            'description': 'OpenAI API, highest quality'
        }
    }
    
    @classmethod
    def get_config(cls, model_key: str) -> Dict[str, Any]:
        """Get configuration for a model"""
        if model_key in cls.MODELS:
            return cls.MODELS[model_key]
        else:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(cls.MODELS.keys())}")
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """List available models"""
        return list(cls.MODELS.keys())