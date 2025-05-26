"""
Multi-source data fetcher with fallback mechanisms
Supports yfinance, Alpha Vantage, Quandl, and other data sources
"""
import asyncio
import sys
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    print("Warning: Alpha Vantage not available. Using yfinance only.", file=sys.stderr)
import requests
from config.settings import settings

logger = logging.getLogger(__name__)


class DataFetcher:
    """Multi-source financial data fetcher with intelligent fallback"""

    def __init__(self):
        self.av_ts = None
        self.av_fd = None
        if ALPHA_VANTAGE_AVAILABLE and settings.alpha_vantage_api_key:
            self.av_ts = TimeSeries(key=settings.alpha_vantage_api_key, output_format='pandas')
            self.av_fd = FundamentalData(key=settings.alpha_vantage_api_key, output_format='pandas')

    async def get_stock_data(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch stock price data with fallback sources"""
        results = {}

        for symbol in symbols:
            try:
                # Primary: yfinance
                data = await self._fetch_yfinance_data(symbol, period, interval)
                if data is not None and not data.empty:
                    results[symbol] = data
                    continue

                # Fallback: Alpha Vantage
                if self.av_ts:
                    data = await self._fetch_alpha_vantage_data(symbol, period)
                    if data is not None and not data.empty:
                        results[symbol] = data
                        continue

                logger.warning(f"Could not fetch data for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")

        return results

    async def _fetch_yfinance_data(
        self,
        symbol: str,
        period: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                return None

            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            return data

        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")
            return None

    async def _fetch_alpha_vantage_data(
        self,
        symbol: str,
        period: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage"""
        try:
            if not self.av_ts:
                return None

            data, _ = self.av_ts.get_daily_adjusted(symbol=symbol, outputsize='full')

            if data.empty:
                return None

            # Convert to standard format
            data.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
            data.index.name = 'date'

            # Filter by period
            if period.endswith('y'):
                years = int(period[:-1])
                start_date = datetime.now() - timedelta(days=years*365)
                data = data[data.index >= start_date]

            return data

        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
            return None

    async def get_fundamental_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch fundamental data for symbols"""
        results = {}

        for symbol in symbols:
            try:
                # Try yfinance first
                fundamental_data = await self._fetch_yfinance_fundamentals(symbol)
                if fundamental_data:
                    results[symbol] = fundamental_data
                    continue

                # Fallback to Alpha Vantage
                if self.av_fd:
                    fundamental_data = await self._fetch_alpha_vantage_fundamentals(symbol)
                    if fundamental_data:
                        results[symbol] = fundamental_data
                        continue

                logger.warning(f"Could not fetch fundamental data for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching fundamental data for {symbol}: {e}")

        return results

    async def _fetch_yfinance_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch fundamental data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                return None

            # Extract key fundamental metrics
            fundamentals = {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'peg_ratio': info.get('pegRatio'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'book_value': info.get('bookValue'),
                'enterprise_value': info.get('enterpriseValue'),
                'ebitda': info.get('ebitda'),
                'revenue': info.get('totalRevenue'),
                'gross_profit': info.get('grossProfits'),
                'free_cash_flow': info.get('freeCashflow'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'employees': info.get('fullTimeEmployees'),
                'website': info.get('website'),
                'business_summary': info.get('longBusinessSummary')
            }

            # Remove None values
            fundamentals = {k: v for k, v in fundamentals.items() if v is not None}

            return fundamentals

        except Exception as e:
            logger.error(f"yfinance fundamentals error for {symbol}: {e}")
            return None

    async def _fetch_alpha_vantage_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch fundamental data from Alpha Vantage"""
        try:
            if not self.av_fd:
                return None

            # Get company overview
            overview, _ = self.av_fd.get_company_overview(symbol)

            if overview.empty:
                return None

            # Convert to dictionary
            fundamentals = overview.iloc[0].to_dict()

            # Standardize key names
            key_mapping = {
                'MarketCapitalization': 'market_cap',
                'PERatio': 'pe_ratio',
                'ForwardPE': 'forward_pe',
                'PriceToBookRatio': 'pb_ratio',
                'PriceToSalesRatioTTM': 'ps_ratio',
                'PEGRatio': 'peg_ratio',
                'DebtToEquityRatio': 'debt_to_equity',
                'ReturnOnEquityTTM': 'roe',
                'ReturnOnAssetsTTM': 'roa',
                'ProfitMargin': 'profit_margin',
                'OperatingMarginTTM': 'operating_margin',
                'RevenuePerShareTTM': 'revenue_per_share',
                'QuarterlyRevenueGrowthYOY': 'revenue_growth',
                'QuarterlyEarningsGrowthYOY': 'earnings_growth',
                'CurrentRatio': 'current_ratio',
                'DividendYield': 'dividend_yield',
                'Beta': 'beta',
                'BookValue': 'book_value',
                'EBITDA': 'ebitda',
                'RevenueTTM': 'revenue',
                'GrossProfitTTM': 'gross_profit',
                'Sector': 'sector',
                'Industry': 'industry',
                'Description': 'business_summary'
            }

            standardized = {}
            for old_key, new_key in key_mapping.items():
                if old_key in fundamentals:
                    value = fundamentals[old_key]
                    # Convert string numbers to float
                    if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    standardized[new_key] = value

            return standardized

        except Exception as e:
            logger.error(f"Alpha Vantage fundamentals error for {symbol}: {e}")
            return None

    async def get_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive market data including prices and fundamentals"""
        # Fetch price and fundamental data concurrently
        price_task = self.get_stock_data(symbols)
        fundamental_task = self.get_fundamental_data(symbols)

        price_data, fundamental_data = await asyncio.gather(
            price_task, fundamental_task, return_exceptions=True
        )

        # Combine results
        results = {}
        for symbol in symbols:
            results[symbol] = {
                'price_data': price_data.get(symbol) if isinstance(price_data, dict) else None,
                'fundamental_data': fundamental_data.get(symbol) if isinstance(fundamental_data, dict) else None
            }

        return results


# Global data fetcher instance
data_fetcher = DataFetcher()
