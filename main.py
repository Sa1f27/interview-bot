# APP/main.py
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ——— Configuration ———
logger = logging.getLogger("uvicorn.error")
BASE_DIR = Path(__file__).resolve().parent

# ——— Main app instantiation ———
app = FastAPI(title="FastAPI Project Launcher")

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
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
async def home():
    index_file = static_dir / "index.html"
    if not index_file.exists():
        logger.error(f"❌ static/index.html not found at {index_file}")
        return {"error": "Index file not found"}
    return FileResponse(str(index_file))

@app.get("/test", tags=["diagnostics"])
async def test():
    return {"message": "Main app working"}

# ——— Sub-app mounting ———
# Add entries here as "prefix": "module.path:app_variable"
sub_apps = {
    # "daily_standup":    "daily_standup.main:app",
    # "weekend_mocktest": "weekend_mocktest.main:app",
    "weekly_interview": "weekly_interview.main:app",
}

for prefix, import_spec in sub_apps.items():
    module_path, attr = import_spec.split(":")
    try:
        module = __import__(module_path, fromlist=[attr])
        sub_app = getattr(module, attr)
        app.mount(f"/{prefix}", sub_app, name=prefix)
        logger.info(f"✅ Mounted `{prefix}` sub-app at /{prefix}")
    except Exception as exc:
        logger.error(f"❌ Failed to mount `{prefix}` sub-app: {exc}")
import socket

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
    print(f"🚀 Starting server at https://{local_ip}:{port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
        ssl_certfile="certs/cert.pem",
        ssl_keyfile="certs/key.pem",
    )
