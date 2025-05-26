"""
Configuration settings for AdvancedFinancialML MCP Server
"""
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # API Keys
    alpha_vantage_api_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    quandl_api_key: Optional[str] = Field(None, env="QUANDL_API_KEY")
    newsapi_key: Optional[str] = Field(None, env="NEWSAPI_KEY")

    # Model Configuration
    model_cache_path: str = Field("./models/cache", env="MODEL_CACHE_PATH")
    auto_retrain: bool = Field(True, env="AUTO_RETRAIN")
    retrain_frequency: str = Field("weekly", env="RETRAIN_FREQUENCY")
    performance_threshold: float = Field(0.05, env="PERFORMANCE_THRESHOLD")

    # Kite MCP Integration
    kite_mcp_url: str = Field("http://localhost:8000", env="KITE_MCP_URL")
    kite_timeout: int = Field(30, env="KITE_TIMEOUT")

    # Data Sources
    default_data_source: str = Field("yfinance", env="DEFAULT_DATA_SOURCE")
    backup_data_sources: List[str] = Field(default=["alpha_vantage", "quandl"])

    # Model Defaults
    default_prediction_horizon: int = Field(30, env="DEFAULT_PREDICTION_HORIZON")
    default_confidence_level: float = Field(0.95, env="DEFAULT_CONFIDENCE_LEVEL")
    max_symbols_per_request: int = Field(50, env="MAX_SYMBOLS_PER_REQUEST")

    # Performance Settings
    max_workers: int = Field(4, env="MAX_WORKERS")
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # 1 hour

    # Risk Management
    max_position_size: float = Field(0.1, env="MAX_POSITION_SIZE")  # 10% of portfolio
    default_stop_loss: float = Field(0.05, env="DEFAULT_STOP_LOSS")  # 5%
    default_take_profit: float = Field(0.15, env="DEFAULT_TAKE_PROFIT")  # 15%

    class Config:
        env_file = ".env"
        case_sensitive = False


class ModelConfigs:
    """Model-specific configurations"""

    LSTM_GNN_CONFIG = {
        "auto": {
            "lstm_units_range": (64, 512),
            "gnn_layers_range": (1, 5),
            "learning_rate_range": (1e-5, 1e-2),
            "batch_size_range": (16, 128),
            "epochs_range": (50, 200)
        },
        "simple": {
            "lstm_units": 64,
            "gnn_layers": 2,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 100
        },
        "complex": {
            "lstm_units": 256,
            "gnn_layers": 4,
            "learning_rate": 5e-4,
            "batch_size": 64,
            "epochs": 200
        }
    }

    VOLATILITY_MODELS = {
        "arima": {
            "max_p": 5,
            "max_d": 2,
            "max_q": 5,
            "seasonal": False
        },
        "sarima": {
            "max_p": 3,
            "max_d": 2,
            "max_q": 3,
            "max_P": 2,
            "max_D": 1,
            "max_Q": 2,
            "seasonal_periods": [5, 12, 252]  # weekly, monthly, yearly
        },
        "garch": {
            "max_p": 3,
            "max_q": 3,
            "variants": ["GARCH", "EGARCH", "TGARCH", "FIGARCH"]
        }
    }

    OPTIMIZATION_METHODS = {
        "bayesian": {
            "n_trials": 100,
            "n_startup_trials": 10,
            "sampler": "TPE"
        },
        "grid": {
            "cv_folds": 5,
            "scoring": "neg_mean_squared_error"
        },
        "random": {
            "n_iter": 50,
            "cv_folds": 3
        }
    }


# Global settings instance
settings = Settings()

# Create necessary directories
Path(settings.model_cache_path).mkdir(parents=True, exist_ok=True)
