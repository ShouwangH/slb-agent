"""
Run the FastAPI server.

This script starts the SLB Agent API server using uvicorn.

Usage:
    python scripts/run_server.py [--port PORT] [--reload]

Options:
    --port PORT: Port to run the server on (default: 8000)
    --reload: Enable auto-reload for development (default: False)
    --host HOST: Host to bind to (default: 0.0.0.0)

Examples:
    # Run in production mode
    python scripts/run_server.py

    # Run with auto-reload for development
    python scripts/run_server.py --reload

    # Run on custom port
    python scripts/run_server.py --port 8080

Environment Variables:
    OPENAI_API_KEY: Required for real LLM calls (falls back to MockLLMClient if not set)
"""

import argparse
import sys

try:
    import uvicorn
except ImportError:
    print("ERROR: uvicorn is not installed.")
    print("Install dependencies: pip install -e .")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run the SLB Agent API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SLB AGENT API SERVER")
    print("=" * 80)
    print(f"\n🚀 Starting server on http://{args.host}:{args.port}")
    print(f"   Mode: {'Development (auto-reload)' if args.reload else 'Production'}")
    print(f"   Log level: {args.log_level}")
    print(f"\n📚 API Documentation: http://localhost:{args.port}/docs")
    print(f"   Alternative docs: http://localhost:{args.port}/redoc")
    print(f"   OpenAPI spec: http://localhost:{args.port}/openapi.json")
    print(f"\n🏥 Health check: http://localhost:{args.port}/health")
    print(f"\n⚙️  Main endpoint: POST http://localhost:{args.port}/program")
    print("\n" + "=" * 80 + "\n")

    # Run the server
    uvicorn.run(
        "app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
