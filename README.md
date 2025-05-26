# AdvancedFinancialML MCP Server

A comprehensive Financial MCP Server that integrates seamlessly with Kite MCP for sophisticated portfolio analysis and trading signals. Built using the FastMCP framework with advanced machine learning capabilities.

## 🚀 Features

### Core Capabilities
- **LSTM-GNN Hybrid Models**: Advanced stock price prediction combining temporal patterns with cross-asset correlations
- **Dynamic Volatility Modeling**: Comprehensive volatility prediction using ARIMA, SARIMA, GARCH, EGARCH, and TGARCH models
- **Portfolio Intelligence**: Multi-factor analysis combining fundamental, technical, sentiment, and macro-economic factors
- **Advanced Trading Signals**: Risk-adjusted trading recommendations with position sizing and stop-loss/take-profit levels
- **Monte Carlo Simulation**: Portfolio scenario analysis with VaR, CVaR, and stress testing
- **Market Regime Detection**: Real-time market regime identification with probability estimates
- **Automatic Model Management**: Self-optimizing models with hyperparameter tuning and versioning

### Integration Features
- **Seamless Kite MCP Integration**: Direct communication with Kite MCP server for portfolio data
- **Multi-source Data Fetching**: Fallback mechanisms across yfinance, Alpha Vantage, and Quandl
- **Real-time Analysis**: Live market data integration with caching for performance
- **Comprehensive Error Handling**: Graceful degradation and robust error management

## 📋 Prerequisites

- Python 3.8+
- API Keys (optional but recommended):
  - Alpha Vantage API Key
  - Quandl API Key
  - NewsAPI Key

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Financial_mcp_server
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the root directory:
```env
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
QUANDL_API_KEY=your_quandl_key
NEWSAPI_KEY=your_newsapi_key
KITE_MCP_URL=http://localhost:8000
MODEL_CACHE_PATH=./models/cache
AUTO_RETRAIN=true
```

4. **Create necessary directories:**
```bash
mkdir -p models/cache
mkdir -p config
mkdir -p utils
mkdir -p analysis
```

## 🚀 Quick Start

### Running the Server

```bash
python financial_mcp_server.py
```

The server will start and display available tools:
- predict_stock_lstm_gnn_adaptive
- optimize_prediction_models
- predict_volatility_comprehensive
- volatility_model_factory
- analyze_portfolio_comprehensive
- generate_trading_signals_advanced
- simulate_portfolio_scenarios
- configure_financial_models
- market_regime_detector

### Claude Desktop Configuration

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "financial_ml": {
      "command": "python",
      "args": ["/path/to/financial_mcp_server.py"],
      "env": {
        "ALPHA_VANTAGE_API_KEY": "your_key",
        "QUANDL_API_KEY": "your_key",
        "NEWSAPI_KEY": "your_key",
        "MODEL_CACHE_PATH": "/path/to/models",
        "AUTO_RETRAIN": "true",
        "KITE_MCP_URL": "http://localhost:8000"
      }
    }
  }
}
```

## 📊 Tool Documentation

### 1. predict_stock_lstm_gnn_adaptive
Advanced LSTM-GNN stock price prediction with adaptive configuration.

**Parameters:**
- `symbols`: List of stock symbols
- `prediction_horizon`: Days to predict (default: 30)
- `model_complexity`: "auto", "simple", or "complex"
- `correlation_threshold`: Threshold for graph construction (default: 0.7)

**Returns:**
- Price predictions with uncertainty bands
- Model performance metrics
- Confidence scores

### 2. predict_volatility_comprehensive
Comprehensive volatility prediction using multiple models.

**Parameters:**
- `symbols`: List of stock symbols
- `horizon`: Prediction horizon in days (default: 10)
- `model_selection`: "auto" or specific model type
- `seasonality`: Seasonality detection method

**Returns:**
- Volatility forecasts with confidence intervals
- VaR estimates
- Model performance comparison

### 3. analyze_portfolio_comprehensive
Multi-factor portfolio analysis.

**Parameters:**
- `use_kite_portfolio`: Fetch from Kite MCP (default: true)
- `fundamental_analysis`: Include fundamental analysis
- `technical_analysis`: Include technical analysis
- `sentiment_analysis`: Include sentiment analysis
- `macro_analysis`: Include macro-economic analysis

**Returns:**
- Comprehensive portfolio analysis
- Individual stock scores
- Portfolio-level recommendations

### 4. generate_trading_signals_advanced
Advanced trading signal generation with risk management.

**Parameters:**
- `signal_confidence`: Minimum confidence threshold (default: 0.75)
- `risk_tolerance`: "conservative", "moderate", or "aggressive"
- `time_horizon`: "short_term", "medium_term", or "long_term"

**Returns:**
- Buy/sell/hold signals
- Position sizing recommendations
- Stop-loss and take-profit levels

### 5. simulate_portfolio_scenarios
Monte Carlo portfolio simulation.

**Parameters:**
- `portfolio_holdings`: Portfolio data
- `simulation_periods`: Number of simulations (default: 1000)
- `market_scenarios`: Scenarios to test
- `confidence_levels`: VaR confidence levels

**Returns:**
- Scenario analysis results
- Risk metrics (VaR, CVaR, max drawdown)
- Stress testing results

## 🔧 Configuration

### Model Configuration
```python
# Configure automatic model retraining
await configure_financial_models(
    auto_retrain=True,
    retrain_frequency="weekly",
    performance_threshold=0.05,
    model_versioning=True
)
```

### Market Regime Detection
```python
# Detect current market regime
regime_info = await market_regime_detector(
    economic_indicators=["vix", "yield_curve", "credit_spreads"],
    regime_models=["hmm", "markov_switching", "threshold"]
)
```

## 📈 Example Usage

### Complete Portfolio Analysis Workflow

```python
# 1. Analyze portfolio comprehensively
portfolio_analysis = await analyze_portfolio_comprehensive(
    use_kite_portfolio=True,
    fundamental_analysis=True,
    technical_analysis=True,
    sentiment_analysis=True,
    macro_analysis=True
)

# 2. Generate price predictions
predictions = await predict_stock_lstm_gnn_adaptive(
    symbols=["AAPL", "GOOGL", "MSFT"],
    prediction_horizon=30,
    model_complexity="auto"
)

# 3. Predict volatility
volatility = await predict_volatility_comprehensive(
    symbols=["AAPL", "GOOGL", "MSFT"],
    horizon=10,
    model_selection="auto"
)

# 4. Generate trading signals
signals = await generate_trading_signals_advanced(
    signal_confidence=0.75,
    risk_tolerance="moderate",
    time_horizon="medium_term"
)

# 5. Run scenario analysis
scenarios = await simulate_portfolio_scenarios(
    portfolio_holdings=portfolio_data,
    simulation_periods=1000,
    market_scenarios=["bull", "bear", "sideways", "crisis"]
)
```

## 🏗️ Architecture

### Core Components

1. **Data Layer**
   - `utils/data_fetcher.py`: Multi-source data retrieval
   - `utils/kite_integration.py`: Kite MCP communication

2. **Model Layer**
   - `models/lstm_gnn_predictor.py`: LSTM-GNN hybrid models
   - `models/volatility_models.py`: Volatility modeling suite

3. **Analysis Layer**
   - `analysis/portfolio_analyzer.py`: Portfolio intelligence

4. **Configuration**
   - `config/settings.py`: Application settings and model configs

### Data Flow
```
Kite MCP ──┐
           ├─→ Data Fetcher ─→ Models ─→ Analysis ─→ Trading Signals
External APIs ──┘
```

## 🔒 Security & Performance

- **API Rate Limiting**: Built-in rate limiting for external APIs
- **Caching**: Intelligent caching for expensive computations
- **Error Handling**: Comprehensive error handling with graceful degradation
- **Model Versioning**: Automatic model versioning and rollback capabilities
- **Performance Monitoring**: Real-time model performance tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the example usage patterns

## 🔄 Updates

The server includes automatic model retraining and performance monitoring. Models are continuously improved based on market conditions and performance metrics.
