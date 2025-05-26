"""
Kite MCP Server Integration Module
Handles communication with Kite MCP server for portfolio data retrieval
"""
import asyncio
import httpx
import logging
from typing import Dict, List, Optional, Any
from config.settings import settings

logger = logging.getLogger(__name__)


class KiteMCPClient:
    """Client for communicating with Kite MCP Server"""
    
    def __init__(self, base_url: str = None, timeout: int = None):
        self.base_url = base_url or settings.kite_mcp_url
        self.timeout = timeout or settings.kite_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the Kite MCP server"""
        try:
            response = await self.client.post(
                f"{self.base_url}/tools/{tool_name}",
                json={"arguments": arguments}
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Request error calling Kite MCP tool {tool_name}: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling Kite MCP tool {tool_name}: {e}")
            raise
    
    async def get_portfolio_holdings(self) -> Dict[str, Any]:
        """Get current portfolio holdings from Kite"""
        try:
            result = await self.call_tool("get_holdings_npx", {})
            return result.get("data", {})
        except Exception as e:
            logger.error(f"Error fetching portfolio holdings: {e}")
            return {}
    
    async def get_portfolio_positions(self) -> Dict[str, Any]:
        """Get current portfolio positions from Kite"""
        try:
            result = await self.call_tool("get_positions_npx", {})
            return result.get("data", {})
        except Exception as e:
            logger.error(f"Error fetching portfolio positions: {e}")
            return {}
    
    async def get_quotes(self, instruments: List[str]) -> Dict[str, Any]:
        """Get market quotes for instruments"""
        try:
            result = await self.call_tool("get_quotes_npx", {"instruments": instruments})
            return result.get("data", {})
        except Exception as e:
            logger.error(f"Error fetching quotes for {instruments}: {e}")
            return {}
    
    async def get_historical_data(
        self, 
        instrument_token: int, 
        from_date: str, 
        to_date: str, 
        interval: str = "day"
    ) -> Dict[str, Any]:
        """Get historical price data"""
        try:
            result = await self.call_tool("get_historical_data_npx", {
                "instrument_token": instrument_token,
                "from_date": from_date,
                "to_date": to_date,
                "interval": interval
            })
            return result.get("data", {})
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return {}
    
    async def search_instruments(self, query: str) -> List[Dict[str, Any]]:
        """Search for instruments"""
        try:
            result = await self.call_tool("search_instruments_npx", {"query": query})
            return result.get("data", [])
        except Exception as e:
            logger.error(f"Error searching instruments for {query}: {e}")
            return []
    
    async def get_margins(self) -> Dict[str, Any]:
        """Get account margins"""
        try:
            result = await self.call_tool("get_margins_npx", {})
            return result.get("data", {})
        except Exception as e:
            logger.error(f"Error fetching margins: {e}")
            return {}
    
    async def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order through Kite"""
        try:
            result = await self.call_tool("place_order_npx", order_params)
            return result.get("data", {})
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {}
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            # Fetch all portfolio data concurrently
            holdings_task = self.get_portfolio_holdings()
            positions_task = self.get_portfolio_positions()
            margins_task = self.get_margins()
            
            holdings, positions, margins = await asyncio.gather(
                holdings_task, positions_task, margins_task,
                return_exceptions=True
            )
            
            # Extract symbols for quote fetching
            symbols = set()
            if isinstance(holdings, dict) and "holdings" in holdings:
                symbols.update([h.get("tradingsymbol", "") for h in holdings["holdings"]])
            if isinstance(positions, dict) and "net" in positions:
                symbols.update([p.get("tradingsymbol", "") for p in positions["net"]])
            
            # Get current quotes
            quotes = {}
            if symbols:
                instrument_list = [f"NSE:{symbol}" for symbol in symbols if symbol]
                quotes = await self.get_quotes(instrument_list)
            
            return {
                "holdings": holdings if not isinstance(holdings, Exception) else {},
                "positions": positions if not isinstance(positions, Exception) else {},
                "margins": margins if not isinstance(margins, Exception) else {},
                "quotes": quotes,
                "timestamp": asyncio.get_event_loop().time()
            }
        except Exception as e:
            logger.error(f"Error fetching portfolio summary: {e}")
            return {}


# Global client instance
kite_client = KiteMCPClient()
