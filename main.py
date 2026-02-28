# main.py
from core.orchestration.mcp_server import mcp

if __name__ == "__main__":
    # Lancement du serveur MCP
    # Il peut être utilisé par Claude Desktop ou une autre interface
    print("🚀 Model Context Protocol Server is running...")
    mcp.run()