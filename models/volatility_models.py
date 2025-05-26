"""
Comprehensive Volatility Modeling Suite
Supports ARIMA, SARIMA, GARCH, EGARCH, TGARCH with automatic model selection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Time series and volatility modeling
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import scipy.stats as stats

# Model selection and optimization
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
import logging
from datetime import datetime, timedelta
import joblib
import os
from config.settings import settings, ModelConfigs

logger = logging.getLogger(__name__)


class VolatilityModelSelector:
    """Automatic volatility model selection and optimization"""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.model_performance = {}

    def detect_data_characteristics(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze time series characteristics to guide model selection"""
        characteristics = {}

        # Stationarity tests
        adf_stat, adf_pvalue, _, _, adf_critical, _ = adfuller(returns.dropna())
        kpss_stat, kpss_pvalue, _, kpss_critical = kpss(returns.dropna())

        characteristics['is_stationary'] = adf_pvalue < 0.05 and kpss_pvalue > 0.05
        characteristics['adf_pvalue'] = adf_pvalue
        characteristics['kpss_pvalue'] = kpss_pvalue

        # Volatility clustering (ARCH effects)
        try:
            ljung_box = acorr_ljungbox(returns.dropna()**2, lags=10, return_df=True)
            characteristics['arch_effects'] = ljung_box['lb_pvalue'].iloc[-1] < 0.05
        except:
            characteristics['arch_effects'] = False

        # Seasonality detection
        characteristics['seasonality'] = self._detect_seasonality(returns)

        # Asymmetric effects (leverage effect)
        characteristics['asymmetric_effects'] = self._detect_asymmetric_effects(returns)

        # Distribution characteristics
        characteristics['skewness'] = stats.skew(returns.dropna())
        characteristics['kurtosis'] = stats.kurtosis(returns.dropna())
        characteristics['jarque_bera_pvalue'] = stats.jarque_bera(returns.dropna())[1]

        return characteristics

    def _detect_seasonality(self, returns: pd.Series) -> Dict[str, bool]:
        """Detect seasonal patterns in returns"""
        seasonality = {}

        try:
            # Weekly seasonality (5 trading days)
            if len(returns) > 50:
                weekly_autocorr = returns.autocorr(lag=5)
                seasonality['weekly'] = abs(weekly_autocorr) > 0.1
            else:
                seasonality['weekly'] = False

            # Monthly seasonality (approximately 22 trading days)
            if len(returns) > 220:
                monthly_autocorr = returns.autocorr(lag=22)
                seasonality['monthly'] = abs(monthly_autocorr) > 0.1
            else:
                seasonality['monthly'] = False

            # Quarterly seasonality (approximately 66 trading days)
            if len(returns) > 660:
                quarterly_autocorr = returns.autocorr(lag=66)
                seasonality['quarterly'] = abs(quarterly_autocorr) > 0.1
            else:
                seasonality['quarterly'] = False

        except:
            seasonality = {'weekly': False, 'monthly': False, 'quarterly': False}

        return seasonality

    def _detect_asymmetric_effects(self, returns: pd.Series) -> bool:
        """Detect asymmetric volatility effects (leverage effect)"""
        try:
            # Calculate correlation between returns and future squared returns
            returns_clean = returns.dropna()
            if len(returns_clean) < 50:
                return False

            future_vol = returns_clean.shift(-1)**2
            correlation = returns_clean.corr(future_vol)

            # Negative correlation indicates leverage effect
            return correlation < -0.1
        except:
            return False

    def select_optimal_model(
        self,
        returns: pd.Series,
        characteristics: Dict[str, Any] = None
    ) -> str:
        """Select optimal volatility model based on data characteristics"""

        if characteristics is None:
            characteristics = self.detect_data_characteristics(returns)

        # Decision tree for model selection
        if not characteristics['is_stationary']:
            # Non-stationary data - use ARIMA first
            if any(characteristics['seasonality'].values()):
                return 'sarima'
            else:
                return 'arima'

        elif characteristics['arch_effects']:
            # ARCH effects present - use GARCH family
            if characteristics['asymmetric_effects']:
                return 'egarch'  # or 'tgarch'
            else:
                return 'garch'

        else:
            # Stationary without ARCH effects - simple ARIMA
            if any(characteristics['seasonality'].values()):
                return 'sarima'
            else:
                return 'arima'

    def fit_arima_model(
        self,
        returns: pd.Series,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5
    ) -> Tuple[Any, Dict[str, float]]:
        """Fit ARIMA model with automatic order selection"""

        best_aic = np.inf
        best_model = None
        best_order = None

        returns_clean = returns.dropna()

        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(returns_clean, order=(p, d, q))
                        fitted_model = model.fit()

                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                            best_order = (p, d, q)

                    except:
                        continue

        if best_model is None:
            # Fallback to simple ARIMA(1,0,1)
            model = ARIMA(returns_clean, order=(1, 0, 1))
            best_model = model.fit()
            best_order = (1, 0, 1)

        performance = {
            'aic': best_model.aic,
            'bic': best_model.bic,
            'order': best_order
        }

        return best_model, performance

    def fit_sarima_model(
        self,
        returns: pd.Series,
        seasonal_periods: List[int] = [5, 22, 66]
    ) -> Tuple[Any, Dict[str, float]]:
        """Fit SARIMA model with automatic order selection"""

        best_aic = np.inf
        best_model = None
        best_order = None

        returns_clean = returns.dropna()

        # Try different seasonal periods
        for s in seasonal_periods:
            if len(returns_clean) < 4 * s:
                continue

            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        for P in range(2):
                            for D in range(2):
                                for Q in range(2):
                                    try:
                                        model = SARIMAX(
                                            returns_clean,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, s)
                                        )
                                        fitted_model = model.fit(disp=False)

                                        if fitted_model.aic < best_aic:
                                            best_aic = fitted_model.aic
                                            best_model = fitted_model
                                            best_order = ((p, d, q), (P, D, Q, s))

                                    except:
                                        continue

        if best_model is None:
            # Fallback to ARIMA
            return self.fit_arima_model(returns)

        performance = {
            'aic': best_model.aic,
            'bic': best_model.bic,
            'order': best_order
        }

        return best_model, performance

    def fit_garch_model(
        self,
        returns: pd.Series,
        model_type: str = 'GARCH',
        max_p: int = 3,
        max_q: int = 3
    ) -> Tuple[Any, Dict[str, float]]:
        """Fit GARCH family models"""

        best_aic = np.inf
        best_model = None
        best_order = None

        returns_clean = returns.dropna() * 100  # Scale for numerical stability

        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    if model_type == 'GARCH':
                        model = arch_model(
                            returns_clean,
                            vol='GARCH',
                            p=p,
                            q=q,
                            dist='normal'
                        )
                    elif model_type == 'EGARCH':
                        model = arch_model(
                            returns_clean,
                            vol='EGARCH',
                            p=p,
                            q=q,
                            dist='normal'
                        )
                    elif model_type == 'TGARCH':
                        # Use GJR-GARCH (TGARCH) with asymmetric terms
                        model = arch_model(
                            returns_clean,
                            vol='GARCH',
                            p=p,
                            o=p,  # Asymmetric terms for GJR-GARCH
                            q=q,
                            dist='normal'
                        )
                    else:
                        continue

                    fitted_model = model.fit(disp='off')

                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_model = fitted_model
                        best_order = (p, q)

                except:
                    continue

        if best_model is None:
            # Fallback to simple GARCH(1,1)
            model = arch_model(returns_clean, vol='GARCH', p=1, q=1)
            best_model = model.fit(disp='off')
            best_order = (1, 1)

        performance = {
            'aic': best_model.aic,
            'bic': best_model.bic,
            'order': best_order,
            'model_type': model_type
        }

        return best_model, performance

    def compare_models(
        self,
        returns: pd.Series,
        models_to_test: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple volatility models and select the best one"""

        if models_to_test is None:
            models_to_test = ['arima', 'sarima', 'garch', 'egarch', 'tgarch']

        results = {}
        characteristics = self.detect_data_characteristics(returns)

        for model_name in models_to_test:
            try:
                if model_name == 'arima':
                    model, performance = self.fit_arima_model(returns)
                elif model_name == 'sarima':
                    model, performance = self.fit_sarima_model(returns)
                elif model_name == 'garch':
                    model, performance = self.fit_garch_model(returns, 'GARCH')
                elif model_name == 'egarch':
                    model, performance = self.fit_garch_model(returns, 'EGARCH')
                elif model_name == 'tgarch':
                    model, performance = self.fit_garch_model(returns, 'TGARCH')
                else:
                    continue

                results[model_name] = {
                    'model': model,
                    'performance': performance,
                    'fitted': True
                }

            except Exception as e:
                logger.warning(f"Failed to fit {model_name} model: {e}")
                results[model_name] = {
                    'model': None,
                    'performance': {'aic': np.inf},
                    'fitted': False
                }

        # Select best model based on AIC
        best_model_name = min(
            results.keys(),
            key=lambda x: results[x]['performance'].get('aic', np.inf)
        )

        self.best_model = results[best_model_name]['model']
        self.model_performance = results

        return results


# Global volatility model selector
volatility_selector = VolatilityModelSelector()
