# APP/app.py
# Fixed main app with proper sub-app mounting AND WebSocket support

import logging
from pathlib import Path
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import socket

# ——— Configuration ———
logger = logging.getLogger("uvicorn.error")
BASE_DIR = Path(__file__).resolve().parent

# ——— Main app instantiation with WebSocket support ———
app = FastAPI(title="FastAPI Project Launcher")

# Add CORS middleware BEFORE mounting sub-apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ——— Mount top-level static/ if present ———
static_dir = BASE_DIR / "static"
if static_dir.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=str(static_dir)),
        name="static",
    )
    logger.info(f"✅ Mounted top-level static at {static_dir}")
else:
    logger.warning(f"⚠️  No static/ directory found at {static_dir}; skipping")

# ——— Health & home endpoints ———
@app.get("/healthz", tags=["health"])
async def health_check():
    return {"status": "ok", "service": "main_app"}

@app.get("/", include_in_schema=False)
async def home():
    index_file = static_dir / "index.html"
    if not index_file.exists():
        logger.error(f"❌ static/index.html not found at {index_file}")
        return {"error": "Index file not found"}
    return FileResponse(str(index_file))

@app.get("/test", tags=["diagnostics"])
async def test():
    return {"message": "Main app working", "mounted_apps": list(sub_apps.keys())}

# ——— Sub-app mounting ———
# Add entries here as "prefix": "module.path:app_variable"
sub_apps = {
    "daily_standup":    "daily_standup.main:app",
    # "weekend_mocktest": "weekend_mocktest.main:app",
    "weekly_interview": "weekly_interview.main:app",
}

# Mount sub-applications with proper error handling
for prefix, import_spec in sub_apps.items():
    module_path, attr = import_spec.split(":")
    try:
        logger.info(f"🔄 Attempting to mount `{prefix}` from {module_path}:{attr}")
        
        # Import the module
        module = __import__(module_path, fromlist=[attr])
        sub_app = getattr(module, attr)
        
        # Verify it's a FastAPI app
        if not hasattr(sub_app, 'routes'):
            raise Exception(f"'{attr}' is not a valid FastAPI application")
        
        # Mount the sub-app
        app.mount(f"/{prefix}", sub_app, name=prefix)
        
        logger.info(f"✅ Successfully mounted `{prefix}` sub-app at /{prefix}")
        logger.info(f"   Available routes: {len(sub_app.routes)} routes")
        
        # Log some example routes for debugging
        for route in sub_app.routes[:3]:  # Show first 3 routes
            if hasattr(route, 'path'):
                logger.info(f"   Route: /{prefix}{route.path}")
        
    except ImportError as e:
        logger.error(f"❌ Import error for `{prefix}`: {e}")
        logger.error(f"   Could not import {module_path}:{attr}")
    except AttributeError as e:
        logger.error(f"❌ Attribute error for `{prefix}`: {e}")
        logger.error(f"   Module {module_path} does not have attribute '{attr}'")
    except Exception as exc:
        logger.error(f"❌ Failed to mount `{prefix}` sub-app: {exc}")
        logger.error(f"   Module: {module_path}, Attribute: {attr}")

# ——— Test endpoint to verify mounting ———
@app.get("/debug/mounted-apps")
async def debug_mounted_apps():
    """Debug endpoint to see what apps are mounted"""
    mounted = {}
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'name'):
            mounted[route.name] = route.path
    return {
        "mounted_routes": mounted,
        "expected_sub_apps": list(sub_apps.keys()),
        "total_routes": len(app.routes)
    }

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't have to be reachable
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    import uvicorn
    local_ip = get_local_ip()
    port = 8060
    
    print(f"🚀 Starting main server at http://{local_ip}:{port}")
    print(f"📋 Expected sub-apps: {list(sub_apps.keys())}")
    print(f"🔗 Daily standup will be at: http://{local_ip}:{port}/daily_standup/")
    print(f"🔌 WebSocket will be at: ws://{local_ip}:{port}/daily_standup/ws/{{session_id}}")
    
    # CRITICAL: Start without SSL and with WebSocket support
    print(f"🔓 Starting with WebSocket support (no SSL)")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
        # No SSL - critical for WebSocket to work
        ws_ping_interval=20,
        ws_ping_timeout=20,
        timeout_keep_alive=30
    )