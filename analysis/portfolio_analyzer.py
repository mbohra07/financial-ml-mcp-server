"""
Comprehensive Portfolio Analysis Module
Integrates fundamental, technical, sentiment, and macro analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import asyncio
import logging
from datetime import datetime
import yfinance as yf

# Technical analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    import sys
    print("Warning: TA-Lib not available. Using simplified technical indicators.", file=sys.stderr)

# Sentiment analysis
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.kite_integration import kite_client
from utils.data_fetcher import data_fetcher

logger = logging.getLogger(__name__)


class PortfolioAnalyzer:
    """Comprehensive portfolio analysis with multi-factor approach"""

    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.portfolio_data = None
        self.analysis_results = {}

    async def analyze_portfolio_comprehensive(
        self,
        use_kite_portfolio: bool = True,
        fundamental_analysis: bool = True,
        technical_analysis: bool = True,
        sentiment_analysis: bool = True,
        macro_analysis: bool = True,
        portfolio_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive portfolio analysis"""

        try:
            # Get portfolio data
            if use_kite_portfolio:
                portfolio_data = await kite_client.get_portfolio_summary()
            elif portfolio_data is None:
                raise ValueError("Portfolio data must be provided if not using Kite")

            self.portfolio_data = portfolio_data

            # Extract symbols from portfolio
            symbols = self._extract_symbols_from_portfolio(portfolio_data)

            if not symbols:
                return {"error": "No symbols found in portfolio"}

            # Perform different types of analysis concurrently
            analysis_tasks = []

            if fundamental_analysis:
                analysis_tasks.append(self._perform_fundamental_analysis(symbols))

            if technical_analysis:
                analysis_tasks.append(self._perform_technical_analysis(symbols))

            if sentiment_analysis:
                analysis_tasks.append(self._perform_sentiment_analysis(symbols))

            if macro_analysis:
                analysis_tasks.append(self._perform_macro_analysis(symbols))

            # Execute all analyses
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            # Combine results
            combined_results = {
                "portfolio_summary": self._calculate_portfolio_metrics(portfolio_data),
                "symbols_analyzed": symbols,
                "analysis_timestamp": datetime.now().isoformat()
            }

            # Add individual analysis results
            analysis_types = []
            if fundamental_analysis:
                analysis_types.append("fundamental")
            if technical_analysis:
                analysis_types.append("technical")
            if sentiment_analysis:
                analysis_types.append("sentiment")
            if macro_analysis:
                analysis_types.append("macro")

            for i, result in enumerate(analysis_results):
                if not isinstance(result, Exception) and i < len(analysis_types):
                    combined_results[f"{analysis_types[i]}_analysis"] = result
                elif isinstance(result, Exception):
                    logger.error(f"Analysis {i} failed: {result}")

            # Generate overall portfolio score
            combined_results["portfolio_score"] = self._calculate_portfolio_score(combined_results)

            self.analysis_results = combined_results
            return combined_results

        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return {"error": str(e)}

    def _extract_symbols_from_portfolio(self, portfolio_data: Dict[str, Any]) -> List[str]:
        """Extract trading symbols from portfolio data"""
        symbols = set()

        try:
            # From holdings
            if "holdings" in portfolio_data and "holdings" in portfolio_data["holdings"]:
                for holding in portfolio_data["holdings"]["holdings"]:
                    symbol = holding.get("tradingsymbol", "")
                    if symbol:
                        symbols.add(symbol)

            # From positions
            if "positions" in portfolio_data and "net" in portfolio_data["positions"]:
                for position in portfolio_data["positions"]["net"]:
                    symbol = position.get("tradingsymbol", "")
                    if symbol:
                        symbols.add(symbol)

            return list(symbols)

        except Exception as e:
            logger.error(f"Error extracting symbols: {e}")
            return []

    async def _perform_fundamental_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Perform fundamental analysis on portfolio stocks"""

        try:
            # Get fundamental data
            fundamental_data = await data_fetcher.get_fundamental_data(symbols)

            analysis_results = {}

            for symbol, data in fundamental_data.items():
                if not data:
                    continue

                # Calculate fundamental scores
                scores = self._calculate_fundamental_scores(data)

                # Categorize stock
                category = self._categorize_stock(data, scores)

                analysis_results[symbol] = {
                    "fundamental_data": data,
                    "scores": scores,
                    "category": category,
                    "recommendation": self._get_fundamental_recommendation(scores)
                }

            return {
                "individual_analysis": analysis_results,
                "portfolio_fundamental_summary": self._summarize_fundamental_portfolio(analysis_results)
            }

        except Exception as e:
            logger.error(f"Fundamental analysis failed: {e}")
            return {"error": str(e)}

    def _calculate_fundamental_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate fundamental analysis scores"""
        scores = {}

        try:
            # Valuation scores (lower is better for ratios)
            pe_ratio = data.get('pe_ratio', 0)
            pb_ratio = data.get('pb_ratio', 0)
            ps_ratio = data.get('ps_ratio', 0)

            scores['valuation_score'] = self._normalize_valuation_score(pe_ratio, pb_ratio, ps_ratio)

            # Profitability scores (higher is better)
            roe = data.get('roe', 0) or 0
            roa = data.get('roa', 0) or 0
            profit_margin = data.get('profit_margin', 0) or 0

            scores['profitability_score'] = min(100, max(0, (roe * 100 + roa * 100 + profit_margin * 100) / 3))

            # Growth scores
            revenue_growth = data.get('revenue_growth', 0) or 0
            earnings_growth = data.get('earnings_growth', 0) or 0

            scores['growth_score'] = min(100, max(0, (revenue_growth * 100 + earnings_growth * 100) / 2))

            # Financial health scores
            current_ratio = data.get('current_ratio', 0) or 0
            debt_to_equity = data.get('debt_to_equity', 0) or 0

            health_score = min(100, max(0, current_ratio * 25))  # Ideal current ratio ~2-4
            if debt_to_equity > 0:
                health_score -= min(50, debt_to_equity * 10)  # Penalize high debt

            scores['financial_health_score'] = max(0, health_score)

            # Overall fundamental score
            scores['overall_fundamental_score'] = (
                scores['valuation_score'] * 0.3 +
                scores['profitability_score'] * 0.3 +
                scores['growth_score'] * 0.2 +
                scores['financial_health_score'] * 0.2
            )

        except Exception as e:
            logger.error(f"Error calculating fundamental scores: {e}")
            scores = {
                'valuation_score': 50,
                'profitability_score': 50,
                'growth_score': 50,
                'financial_health_score': 50,
                'overall_fundamental_score': 50
            }

        return scores

    def _normalize_valuation_score(self, pe_ratio: float, pb_ratio: float, ps_ratio: float) -> float:
        """Normalize valuation metrics to 0-100 score (higher is better value)"""
        try:
            # Ideal ranges: PE 10-20, PB 1-3, PS 1-5
            pe_score = 100 if pe_ratio == 0 else max(0, 100 - abs(pe_ratio - 15) * 3)
            pb_score = 100 if pb_ratio == 0 else max(0, 100 - abs(pb_ratio - 2) * 20)
            ps_score = 100 if ps_ratio == 0 else max(0, 100 - abs(ps_ratio - 3) * 15)

            return (pe_score + pb_score + ps_score) / 3
        except:
            return 50

    def _categorize_stock(self, data: Dict[str, Any], scores: Dict[str, float]) -> str:
        """Categorize stock based on fundamental characteristics"""
        try:
            pe_ratio = data.get('pe_ratio', 0) or 0
            pb_ratio = data.get('pb_ratio', 0) or 0
            dividend_yield = data.get('dividend_yield', 0) or 0
            market_cap = data.get('market_cap', 0) or 0

            # Size classification
            if market_cap > 100e9:  # > 100B
                size = "Large Cap"
            elif market_cap > 10e9:  # 10B - 100B
                size = "Mid Cap"
            else:
                size = "Small Cap"

            # Style classification
            if pe_ratio > 25 and scores.get('growth_score', 0) > 70:
                style = "Growth"
            elif dividend_yield > 0.03 and pe_ratio < 15:
                style = "Value"
            else:
                style = "Blend"

            return f"{size} {style}"

        except:
            return "Unknown"

    def _get_fundamental_recommendation(self, scores: Dict[str, float]) -> str:
        """Generate recommendation based on fundamental scores"""
        overall_score = scores.get('overall_fundamental_score', 50)

        if overall_score >= 80:
            return "Strong Buy"
        elif overall_score >= 65:
            return "Buy"
        elif overall_score >= 50:
            return "Hold"
        elif overall_score >= 35:
            return "Weak Hold"
        else:
            return "Sell"

    async def _perform_technical_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Perform technical analysis on portfolio stocks"""

        try:
            # Get price data
            price_data = await data_fetcher.get_stock_data(symbols, period="1y", interval="1d")

            analysis_results = {}

            for symbol, data in price_data.items():
                if data is None or data.empty:
                    continue

                # Calculate technical indicators
                indicators = self._calculate_technical_indicators(data)

                # Generate signals
                signals = self._generate_technical_signals(data, indicators)

                # Calculate technical score
                technical_score = self._calculate_technical_score(signals)

                analysis_results[symbol] = {
                    "indicators": indicators,
                    "signals": signals,
                    "technical_score": technical_score,
                    "recommendation": self._get_technical_recommendation(technical_score)
                }

            return {
                "individual_analysis": analysis_results,
                "portfolio_technical_summary": self._summarize_technical_portfolio(analysis_results)
            }

        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {"error": str(e)}

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various technical indicators"""
        indicators = {}

        try:
            close = data['close']
            high = data['high']
            low = data['low']
            volume = data['volume']

            if TALIB_AVAILABLE:
                # Use TA-Lib if available
                close_values = close.values
                high_values = high.values
                low_values = low.values
                volume_values = volume.values

                # Moving averages
                indicators['sma_20'] = talib.SMA(close_values, timeperiod=20)
                indicators['sma_50'] = talib.SMA(close_values, timeperiod=50)
                indicators['ema_12'] = talib.EMA(close_values, timeperiod=12)
                indicators['ema_26'] = talib.EMA(close_values, timeperiod=26)

                # MACD
                macd, macd_signal, macd_hist = talib.MACD(close_values)
                indicators['macd'] = macd
                indicators['macd_signal'] = macd_signal
                indicators['macd_histogram'] = macd_hist

                # RSI
                indicators['rsi'] = talib.RSI(close_values, timeperiod=14)

                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close_values, timeperiod=20)
                indicators['bb_upper'] = bb_upper
                indicators['bb_middle'] = bb_middle
                indicators['bb_lower'] = bb_lower

                # Stochastic
                slowk, slowd = talib.STOCH(high_values, low_values, close_values)
                indicators['stoch_k'] = slowk
                indicators['stoch_d'] = slowd

                # Volume indicators
                indicators['obv'] = talib.OBV(close_values, volume_values)
                indicators['ad'] = talib.AD(high_values, low_values, close_values, volume_values)

                # Volatility
                indicators['atr'] = talib.ATR(high_values, low_values, close_values, timeperiod=14)
            else:
                # Use pandas-based calculations as fallback
                # Moving averages
                indicators['sma_20'] = close.rolling(window=20).mean().values
                indicators['sma_50'] = close.rolling(window=50).mean().values
                indicators['ema_12'] = close.ewm(span=12).mean().values
                indicators['ema_26'] = close.ewm(span=26).mean().values

                # Simple MACD
                ema_12 = close.ewm(span=12).mean()
                ema_26 = close.ewm(span=26).mean()
                macd = ema_12 - ema_26
                macd_signal = macd.ewm(span=9).mean()
                indicators['macd'] = macd.values
                indicators['macd_signal'] = macd_signal.values
                indicators['macd_histogram'] = (macd - macd_signal).values

                # Simple RSI
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                indicators['rsi'] = rsi.values

                # Simple Bollinger Bands
                sma_20 = close.rolling(window=20).mean()
                std_20 = close.rolling(window=20).std()
                indicators['bb_upper'] = (sma_20 + (std_20 * 2)).values
                indicators['bb_middle'] = sma_20.values
                indicators['bb_lower'] = (sma_20 - (std_20 * 2)).values

                # Simple ATR
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                indicators['atr'] = true_range.rolling(window=14).mean().values

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")

        return indicators

    def _generate_technical_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, str]:
        """Generate buy/sell/hold signals from technical indicators"""
        signals = {}

        try:
            current_price = data['close'].iloc[-1]

            # Moving average signals
            if 'sma_20' in indicators and 'sma_50' in indicators:
                sma_20_current = indicators['sma_20'][-1]
                sma_50_current = indicators['sma_50'][-1]

                if current_price > sma_20_current > sma_50_current:
                    signals['ma_signal'] = 'bullish'
                elif current_price < sma_20_current < sma_50_current:
                    signals['ma_signal'] = 'bearish'
                else:
                    signals['ma_signal'] = 'neutral'

            # RSI signals
            if 'rsi' in indicators:
                rsi_current = indicators['rsi'][-1]
                if rsi_current > 70:
                    signals['rsi_signal'] = 'overbought'
                elif rsi_current < 30:
                    signals['rsi_signal'] = 'oversold'
                else:
                    signals['rsi_signal'] = 'neutral'

            # MACD signals
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd_current = indicators['macd'][-1]
                macd_signal_current = indicators['macd_signal'][-1]

                if macd_current > macd_signal_current:
                    signals['macd_signal'] = 'bullish'
                else:
                    signals['macd_signal'] = 'bearish'

            # Bollinger Bands signals
            if all(k in indicators for k in ['bb_upper', 'bb_lower']):
                bb_upper_current = indicators['bb_upper'][-1]
                bb_lower_current = indicators['bb_lower'][-1]

                if current_price > bb_upper_current:
                    signals['bb_signal'] = 'overbought'
                elif current_price < bb_lower_current:
                    signals['bb_signal'] = 'oversold'
                else:
                    signals['bb_signal'] = 'neutral'

        except Exception as e:
            logger.error(f"Error generating technical signals: {e}")

        return signals

    def _calculate_technical_score(self, signals: Dict[str, str]) -> float:
        """Calculate overall technical score from signals"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            total_signals = 0

            signal_weights = {
                'ma_signal': 0.3,
                'rsi_signal': 0.2,
                'macd_signal': 0.25,
                'bb_signal': 0.25
            }

            for signal_name, signal_value in signals.items():
                if signal_name in signal_weights:
                    weight = signal_weights[signal_name]
                    total_signals += weight

                    if signal_value in ['bullish', 'oversold']:
                        bullish_signals += weight
                    elif signal_value in ['bearish', 'overbought']:
                        bearish_signals += weight

            if total_signals == 0:
                return 50

            # Score from 0-100, where 50 is neutral
            score = 50 + (bullish_signals - bearish_signals) / total_signals * 50
            return max(0, min(100, score))

        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 50

    def _get_technical_recommendation(self, technical_score: float) -> str:
        """Generate recommendation based on technical score"""
        if technical_score >= 80:
            return "Strong Buy"
        elif technical_score >= 65:
            return "Buy"
        elif technical_score >= 50:
            return "Hold"
        elif technical_score >= 35:
            return "Weak Hold"
        else:
            return "Sell"

    async def _perform_sentiment_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Perform sentiment analysis using news and social media"""
        try:
            analysis_results = {}

            for symbol in symbols:
                # Get news sentiment
                news_sentiment = await self._get_news_sentiment(symbol)

                # Calculate overall sentiment score
                sentiment_score = self._calculate_sentiment_score(news_sentiment)

                analysis_results[symbol] = {
                    "news_sentiment": news_sentiment,
                    "sentiment_score": sentiment_score,
                    "recommendation": self._get_sentiment_recommendation(sentiment_score)
                }

            return {
                "individual_analysis": analysis_results,
                "portfolio_sentiment_summary": self._summarize_sentiment_portfolio(analysis_results)
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"error": str(e)}

    async def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment for a symbol"""
        try:
            # Use yfinance to get news
            ticker = yf.Ticker(symbol)
            news = ticker.news

            if not news:
                return {"articles": [], "average_sentiment": 0.0}

            sentiments = []
            articles_analyzed = []

            for article in news[:10]:  # Analyze last 10 articles
                title = article.get('title', '')
                summary = article.get('summary', '')

                # Combine title and summary
                text = f"{title} {summary}"

                # TextBlob sentiment
                blob_sentiment = TextBlob(text).sentiment.polarity

                # VADER sentiment
                vader_sentiment = self.sentiment_analyzer.polarity_scores(text)['compound']

                # Average sentiment
                avg_sentiment = (blob_sentiment + vader_sentiment) / 2
                sentiments.append(avg_sentiment)

                articles_analyzed.append({
                    "title": title,
                    "sentiment": avg_sentiment,
                    "published": article.get('providerPublishTime', 0)
                })

            average_sentiment = np.mean(sentiments) if sentiments else 0.0

            return {
                "articles": articles_analyzed,
                "average_sentiment": average_sentiment,
                "sentiment_distribution": {
                    "positive": len([s for s in sentiments if s > 0.1]),
                    "neutral": len([s for s in sentiments if -0.1 <= s <= 0.1]),
                    "negative": len([s for s in sentiments if s < -0.1])
                }
            }

        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return {"articles": [], "average_sentiment": 0.0}

    def _calculate_sentiment_score(self, news_sentiment: Dict[str, Any]) -> float:
        """Calculate sentiment score from 0-100"""
        try:
            avg_sentiment = news_sentiment.get("average_sentiment", 0.0)
            # Convert from -1,1 range to 0-100 range
            score = (avg_sentiment + 1) * 50
            return max(0, min(100, score))
        except:
            return 50

    def _get_sentiment_recommendation(self, sentiment_score: float) -> str:
        """Generate recommendation based on sentiment score"""
        if sentiment_score >= 75:
            return "Very Positive"
        elif sentiment_score >= 60:
            return "Positive"
        elif sentiment_score >= 40:
            return "Neutral"
        elif sentiment_score >= 25:
            return "Negative"
        else:
            return "Very Negative"

    async def _perform_macro_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Perform macro-economic analysis"""
        try:
            # Get sector exposure
            sector_exposure = await self._calculate_sector_exposure(symbols)

            # Get market indicators
            market_indicators = await self._get_market_indicators()

            # Calculate macro risk
            macro_risk = self._calculate_macro_risk(sector_exposure, market_indicators)

            return {
                "sector_exposure": sector_exposure,
                "market_indicators": market_indicators,
                "macro_risk": macro_risk,
                "recommendations": self._get_macro_recommendations(macro_risk)
            }

        except Exception as e:
            logger.error(f"Macro analysis failed: {e}")
            return {"error": str(e)}

    async def _calculate_sector_exposure(self, symbols: List[str]) -> Dict[str, float]:
        """Calculate portfolio sector exposure"""
        try:
            sector_weights = {}
            total_weight = 0

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    sector = info.get('sector', 'Unknown')

                    # For simplicity, assume equal weights
                    # In practice, use actual portfolio weights
                    weight = 1.0 / len(symbols)

                    if sector in sector_weights:
                        sector_weights[sector] += weight
                    else:
                        sector_weights[sector] = weight

                    total_weight += weight

                except Exception as e:
                    logger.warning(f"Could not get sector for {symbol}: {e}")

            # Normalize weights
            if total_weight > 0:
                sector_weights = {k: v/total_weight for k, v in sector_weights.items()}

            return sector_weights

        except Exception as e:
            logger.error(f"Error calculating sector exposure: {e}")
            return {}

    async def _get_market_indicators(self) -> Dict[str, float]:
        """Get key market indicators"""
        try:
            indicators = {}

            # VIX (volatility index)
            try:
                vix = yf.Ticker("^VIX")
                vix_data = vix.history(period="5d")
                if not vix_data.empty:
                    indicators['vix'] = vix_data['Close'].iloc[-1]
            except:
                indicators['vix'] = None

            # 10-year treasury yield
            try:
                treasury = yf.Ticker("^TNX")
                treasury_data = treasury.history(period="5d")
                if not treasury_data.empty:
                    indicators['treasury_10y'] = treasury_data['Close'].iloc[-1]
            except:
                indicators['treasury_10y'] = None

            # Dollar index
            try:
                dxy = yf.Ticker("DX-Y.NYB")
                dxy_data = dxy.history(period="5d")
                if not dxy_data.empty:
                    indicators['dollar_index'] = dxy_data['Close'].iloc[-1]
            except:
                indicators['dollar_index'] = None

            return indicators

        except Exception as e:
            logger.error(f"Error getting market indicators: {e}")
            return {}

    def _calculate_macro_risk(
        self,
        sector_exposure: Dict[str, float],
        market_indicators: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate macro-economic risk factors"""
        try:
            risk_factors = {}

            # Sector concentration risk
            max_sector_weight = max(sector_exposure.values()) if sector_exposure else 0
            risk_factors['concentration_risk'] = min(100, max_sector_weight * 100)

            # Market volatility risk
            vix = market_indicators.get('vix', 20)
            if vix:
                risk_factors['volatility_risk'] = min(100, max(0, (vix - 10) * 5))
            else:
                risk_factors['volatility_risk'] = 50

            # Interest rate risk
            treasury_yield = market_indicators.get('treasury_10y', 3)
            if treasury_yield:
                # Higher rates generally negative for stocks
                risk_factors['interest_rate_risk'] = min(100, max(0, (treasury_yield - 2) * 20))
            else:
                risk_factors['interest_rate_risk'] = 50

            # Overall macro risk
            risk_factors['overall_macro_risk'] = (
                risk_factors['concentration_risk'] * 0.4 +
                risk_factors['volatility_risk'] * 0.4 +
                risk_factors['interest_rate_risk'] * 0.2
            )

            return risk_factors

        except Exception as e:
            logger.error(f"Error calculating macro risk: {e}")
            return {"overall_macro_risk": 50}

    def _get_macro_recommendations(self, macro_risk: Dict[str, Any]) -> List[str]:
        """Generate macro-economic recommendations"""
        recommendations = []

        try:
            overall_risk = macro_risk.get('overall_macro_risk', 50)
            concentration_risk = macro_risk.get('concentration_risk', 50)
            volatility_risk = macro_risk.get('volatility_risk', 50)

            if overall_risk > 70:
                recommendations.append("Consider reducing portfolio risk exposure")

            if concentration_risk > 60:
                recommendations.append("Diversify across more sectors to reduce concentration risk")

            if volatility_risk > 70:
                recommendations.append("High market volatility detected - consider defensive positioning")

            if overall_risk < 30:
                recommendations.append("Low macro risk environment - consider increasing risk exposure")

        except Exception as e:
            logger.error(f"Error generating macro recommendations: {e}")

        return recommendations

    def _calculate_portfolio_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic portfolio metrics"""
        try:
            metrics = {}

            # Extract portfolio value and P&L
            if "holdings" in portfolio_data and "holdings" in portfolio_data["holdings"]:
                total_value = 0
                total_pnl = 0

                for holding in portfolio_data["holdings"]["holdings"]:
                    value = holding.get("last_price", 0) * holding.get("quantity", 0)
                    pnl = holding.get("pnl", 0)

                    total_value += value
                    total_pnl += pnl

                metrics["total_value"] = total_value
                metrics["total_pnl"] = total_pnl
                metrics["total_return_pct"] = (total_pnl / (total_value - total_pnl)) * 100 if total_value > total_pnl else 0

            return metrics

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}

    def _calculate_portfolio_score(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall portfolio score"""
        try:
            scores = {}

            # Extract individual scores
            fundamental_scores = []
            technical_scores = []
            sentiment_scores = []

            if "fundamental_analysis" in analysis_results:
                for symbol_data in analysis_results["fundamental_analysis"]["individual_analysis"].values():
                    if "scores" in symbol_data:
                        fundamental_scores.append(symbol_data["scores"].get("overall_fundamental_score", 50))

            if "technical_analysis" in analysis_results:
                for symbol_data in analysis_results["technical_analysis"]["individual_analysis"].values():
                    technical_scores.append(symbol_data.get("technical_score", 50))

            if "sentiment_analysis" in analysis_results:
                for symbol_data in analysis_results["sentiment_analysis"]["individual_analysis"].values():
                    sentiment_scores.append(symbol_data.get("sentiment_score", 50))

            # Calculate averages
            scores["average_fundamental_score"] = np.mean(fundamental_scores) if fundamental_scores else 50
            scores["average_technical_score"] = np.mean(technical_scores) if technical_scores else 50
            scores["average_sentiment_score"] = np.mean(sentiment_scores) if sentiment_scores else 50

            # Overall portfolio score
            scores["overall_portfolio_score"] = (
                scores["average_fundamental_score"] * 0.4 +
                scores["average_technical_score"] * 0.3 +
                scores["average_sentiment_score"] * 0.3
            )

            return scores

        except Exception as e:
            logger.error(f"Error calculating portfolio score: {e}")
            return {"overall_portfolio_score": 50}

    def _summarize_fundamental_portfolio(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize fundamental analysis for entire portfolio"""
        try:
            categories = {}
            recommendations = {}

            for symbol, data in analysis_results.items():
                category = data.get("category", "Unknown")
                recommendation = data.get("recommendation", "Hold")

                categories[category] = categories.get(category, 0) + 1
                recommendations[recommendation] = recommendations.get(recommendation, 0) + 1

            return {
                "category_distribution": categories,
                "recommendation_distribution": recommendations
            }
        except:
            return {}

    def _summarize_technical_portfolio(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize technical analysis for entire portfolio"""
        try:
            recommendations = {}

            for symbol, data in analysis_results.items():
                recommendation = data.get("recommendation", "Hold")
                recommendations[recommendation] = recommendations.get(recommendation, 0) + 1

            return {"recommendation_distribution": recommendations}
        except:
            return {}

    def _summarize_sentiment_portfolio(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize sentiment analysis for entire portfolio"""
        try:
            recommendations = {}

            for symbol, data in analysis_results.items():
                recommendation = data.get("recommendation", "Neutral")
                recommendations[recommendation] = recommendations.get(recommendation, 0) + 1

            return {"sentiment_distribution": recommendations}
        except:
            return {}


# Global portfolio analyzer instance
portfolio_analyzer = PortfolioAnalyzer()
