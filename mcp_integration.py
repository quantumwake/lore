"""MCP (Model Context Protocol) integration."""

import httpx
from typing import Optional, Dict, Any
from rich.console import Console

from config import MCP_SERVER_URL

console = Console()


class MCPClient:
    """Client for interacting with MCP servers."""

    def __init__(self, server_url: str = None):
        self.server_url = server_url or MCP_SERVER_URL
        self.client = httpx.Client(timeout=30.0)

    def get_context(self, query: str) -> Optional[str]:
        """Fetch context from MCP server based on query."""
        try:
            response = self.client.post(
                f"{self.server_url}/context",
                json={"query": query}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("context", "")
        except httpx.HTTPStatusError as e:
            console.print(f"[yellow]MCP server error: {e.response.status_code}[/yellow]")
            return None
        except Exception as e:
            console.print(f"[yellow]Could not fetch MCP context: {e}[/yellow]")
            return None

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """Execute an MCP tool with given parameters."""
        try:
            response = self.client.post(
                f"{self.server_url}/tools/execute",
                json={"tool": tool_name, "params": params}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            console.print(f"[red]MCP tool error: {e.response.status_code}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]Could not execute MCP tool: {e}[/red]")
            return None

    def list_tools(self) -> Optional[list]:
        """List available MCP tools."""
        try:
            response = self.client.get(f"{self.server_url}/tools")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            console.print(f"[yellow]Could not list MCP tools: {e}[/yellow]")
            return None

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()