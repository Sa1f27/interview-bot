# weekend_mocktest/main.py
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .core.config import config
from .core.database import get_db_manager, close_db_manager
from .core.ai_services import get_ai_service, close_ai_service
from .core.utils import cleanup_all
from .api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Mock Test Module starting...")
    
    if config.USE_DUMMY_DATA:
        logger.info("🔧 DUMMY DATA MODE: Using local test data")
    else:
        logger.info("🔗 LIVE DATA MODE: Connecting to databases")
    
    # Validate configuration
    config_validation = config.validate()
    if not config_validation["valid"]:
        raise Exception(f"Configuration invalid: {config_validation['issues']}")
    
    # Initialize services
    try:
        db_manager = get_db_manager()
        if not db_manager:
            raise Exception("Database manager initialization failed")
        
        db_health = db_manager.validate_connection()
        if not db_health["overall"]:
            raise Exception("Database connections failed")
        
        logger.info(f"✅ Database manager initialized ({db_health['mode']})")
        
        ai_service = get_ai_service()
        if not ai_service:
            raise Exception("AI service initialization failed")
        
        if not config.USE_DUMMY_DATA:
            ai_health = ai_service.health_check()
            if ai_health["status"] != "healthy":
                raise Exception("AI service validation failed")
        
        logger.info("✅ AI service initialized")
        logger.info("✅ Mock Test Module ready")
        
    except Exception as e:
        logger.error(f"❌ Mock Test Module initialization failed: {e}")
        raise Exception(f"Failed to initialize: {e}")
    
    yield
    
    # Cleanup
    logger.info("👋 Mock Test Module shutting down...")
    try:
        cleanup_all()
        close_ai_service()
        close_db_manager()
        logger.info("✅ Mock Test Module shutdown completed")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Create FastAPI sub-app (will be mounted by main app.py)
app = FastAPI(
    title="Mock Test Module",
    description="Mock testing sub-module",
    version="5.0.0",
    lifespan=lifespan
)

# Include routes
app.include_router(router)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Mock Test Module - Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail=str(exc))

@app.get("/health")
async def module_health():
    return {
        "status": "healthy", 
        "module": "weekend_mocktest",
        "mode": "dummy_data" if config.USE_DUMMY_DATA else "live_data"
    }
    # weekend_mocktest/main.py - Modular Entry Point
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import modular components
from .core.config import config
from .core.database import get_db_manager, close_db_manager
from .core.ai_services import get_ai_service, close_ai_service
from .core.utils import cleanup_all
from .api.routes import router

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger = logging.getLogger(__name__)

# Get base directory
BASE_DIR = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with dummy data support"""
    logger.info("🚀 Mock Test API - Modular starting...")
    
    # Show configuration mode
    if config.USE_DUMMY_DATA:
        logger.info("🔧 DUMMY DATA MODE: Server unavailable, using local test data")
        logger.info("📝 Set USE_DUMMY_DATA=false in .env when server is back online")
    else:
        logger.info("🔗 LIVE DATA MODE: Connecting to databases and AI services")
    
    # Validate configuration
    config_validation = config.validate()
    if not config_validation["valid"]:
        logger.error(f"❌ Configuration validation failed: {config_validation['issues']}")
        raise Exception(f"Configuration invalid: {config_validation['issues']}")
    
    logger.info(f"✅ Configuration validated (dummy_mode: {config.USE_DUMMY_DATA})")
    
    # Initialize core services
    try:
        # Initialize database manager (handles dummy data internally)
        db_manager = get_db_manager()
        if not db_manager:
            raise Exception("Database manager initialization failed")
        
        # Validate database connections (returns success for dummy mode)
        db_health = db_manager.validate_connection()
        if not db_health["overall"]:
            logger.error(f"❌ Database health check failed: {db_health}")
            raise Exception("Database connections failed")
        
        logger.info(f"✅ Database manager initialized ({db_health['mode']})")
        
        # Initialize AI service (handles dummy data internally)
        ai_service = get_ai_service()
        if not ai_service:
            raise Exception("AI service initialization failed")
        
        if not config.USE_DUMMY_DATA:
            # Validate AI service only in live mode
            ai_health = ai_service.health_check()
            if ai_health["status"] != "healthy":
                logger.error(f"❌ AI service health check failed: {ai_health}")
                raise Exception("AI service validation failed")
            logger.info("✅ AI service initialized and validated")
        else:
            logger.info("✅ AI service initialized (dummy mode)")
        
        # Log system readiness
        logger.info("✅ All core systems ready")
        logger.info(f"📊 Configuration: {config.QUESTIONS_PER_TEST} questions, {config.RECENT_SUMMARIES_COUNT} summaries")
        logger.info(f"⚡ Features: Batch generation, {config.QUESTION_CACHE_DURATION_HOURS}h cache, modular architecture")
        
        if config.USE_DUMMY_DATA:
            logger.info("🔧 Using 7 ML/AI dummy summaries for question generation")
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        raise Exception(f"Failed to initialize system: {e}")
    
    yield
    
    # Cleanup on shutdown
    logger.info("👋 Shutting down...")
    try:
        cleanup_all()
        close_ai_service()
        close_db_manager()
        logger.info("✅ Graceful shutdown completed")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Create FastAPI application
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://192.168.48.201:5173').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Mount static files if frontend directory exists
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    logger.info(f"✅ Static files mounted from {frontend_dir}")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(
        status_code=500,
        detail={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "component": "global_handler"
        }
    )

# Health check at root level for load balancers
@app.get("/health")
async def root_health():
    """Root level health check for load balancers"""
    return {
        "status": "healthy", 
        "service": "mock_test_api",
        "mode": "dummy_data" if config.USE_DUMMY_DATA else "live_data"
    }

# Startup message
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Mock Test API in development mode")
    logger.info(f"🔧 Dummy data mode: {config.USE_DUMMY_DATA}")
    
    uvicorn.run(
        "weekend_mocktest.main:app",
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', '8060')),
        reload=os.getenv('DEBUG_MODE', 'true').lower() == 'true',
        log_level=os.getenv('LOG_LEVEL', 'info').lower()
    )