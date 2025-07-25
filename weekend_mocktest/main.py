# weekend_mocktest/main.py
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import modular components
from .core.config import config
from .core.database import get_db_manager, close_db_manager
from .core.ai_services import get_ai_service, close_ai_service
from .core.utils import cleanup_all
from .api.routes import router

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get base directory
BASE_DIR = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with comprehensive error handling"""
    logger.info("🚀 Mock Test API - Modular starting...")
    
    # Show configuration mode
    if config.USE_DUMMY_DATA:
        logger.info("🔧 DUMMY DATA MODE: Server unavailable, using local test data")
        logger.info("📝 Set USE_DUMMY_DATA=false in .env when server is back online")
    else:
        logger.info("🔗 LIVE DATA MODE: Connecting to databases and AI services")
    
    # Validate configuration
    try:
        config_validation = config.validate()
        if not config_validation["valid"]:
            logger.error(f"❌ Configuration validation failed: {config_validation['issues']}")
            raise Exception(f"Configuration invalid: {config_validation['issues']}")
        
        logger.info(f"✅ Configuration validated (dummy_mode: {config.USE_DUMMY_DATA})")
    except Exception as e:
        logger.error(f"❌ Configuration validation error: {e}")
        raise
    
    # Initialize core services
    try:
        # Initialize database manager (handles dummy data internally)
        logger.info("🔄 Initializing database manager...")
        db_manager = get_db_manager()
        if not db_manager:
            raise Exception("Database manager initialization failed")
        
        # Validate database connections (returns success for dummy mode)
        db_health = db_manager.validate_connection()
        if not db_health["overall"]:
            logger.warning(f"⚠️ Database health check issues: {db_health}")
            if not config.USE_DUMMY_DATA:
                raise Exception("Database connections failed in live mode")
        
        logger.info(f"✅ Database manager initialized ({db_health['mode']})")
        
        # Initialize AI service (handles dummy data internally)
        logger.info("🔄 Initializing AI service...")
        ai_service = get_ai_service()
        if not ai_service:
            raise Exception("AI service initialization failed")
        
        # Validate AI service only in live mode
        ai_health = ai_service.health_check()
        if ai_health["status"] != "healthy" and not config.USE_DUMMY_DATA:
            logger.error(f"❌ AI service health check failed: {ai_health}")
            raise Exception("AI service validation failed in live mode")
        
        logger.info(f"✅ AI service initialized ({ai_health['mode']})")
        
        # Log system readiness
        logger.info("✅ All core systems ready")
        logger.info(f"📊 Configuration: {config.QUESTIONS_PER_TEST} questions, {config.RECENT_SUMMARIES_COUNT} summaries")
        logger.info(f"⚡ Features: Batch generation, {config.QUESTION_CACHE_DURATION_HOURS}h cache, modular architecture")
        
        if config.USE_DUMMY_DATA:
            logger.info("🔧 Using 7 ML/AI dummy summaries for question generation")
            logger.info("🎯 Dummy evaluation with realistic scoring (70% average)")
        
        # Test basic functionality
        try:
            logger.info("🔄 Testing basic functionality...")
            from .services.test_service import get_test_service
            test_service = get_test_service()
            health = test_service.health_check()
            logger.info(f"✅ Test service health: {health['status']}")
        except Exception as e:
            logger.warning(f"⚠️ Test service health check failed: {e}")
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        if config.USE_DUMMY_DATA:
            logger.warning("Continuing with limited functionality in dummy mode")
        else:
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

# Configure CORS with environment variable support
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://192.168.48.201:5173,https://192.168.48.201:8060').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Custom exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "message": str(exc),
            "type": "validation_error"
        }
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    """Handle file not found errors"""
    logger.warning(f"File not found: {exc}")
    return JSONResponse(
        status_code=404,
        content={
            "error": "Resource Not Found",
            "message": "The requested resource was not found",
            "type": "not_found_error"
        }
    )

@app.exception_handler(ConnectionError)
async def connection_error_handler(request: Request, exc: ConnectionError):
    """Handle connection errors"""
    logger.error(f"Connection error: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service Unavailable",
            "message": "External service temporarily unavailable",
            "type": "connection_error",
            "suggestion": "Please try again later"
        }
    )

@app.exception_handler(TimeoutError)
async def timeout_error_handler(request: Request, exc: TimeoutError):
    """Handle timeout errors"""
    logger.error(f"Timeout error: {exc}")
    return JSONResponse(
        status_code=504,
        content={
            "error": "Request Timeout",
            "message": "The request took too long to process",
            "type": "timeout_error",
            "suggestion": "Please try again with a simpler request"
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Don't expose internal details in production
    if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
        detail = {
            "error": "Internal Server Error",
            "message": str(exc),
            "type": type(exc).__name__,
            "component": "global_handler"
        }
    else:
        detail = {
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "type": "server_error"
        }
    
    return JSONResponse(
        status_code=500,
        content=detail
    )

# Health check at root level for load balancers
@app.get("/health")
async def root_health():
    """Root level health check for load balancers"""
    try:
        # Quick health check of core services
        health_status = {
            "status": "healthy",
            "service": "mock_test_api",
            "mode": "dummy_data" if config.USE_DUMMY_DATA else "live_data",
            "timestamp": os.getenv('API_VERSION', config.API_VERSION)
        }
        
        # Add component health checks
        try:
            from .services.test_service import get_test_service
            test_service = get_test_service()
            test_health = test_service.health_check()
            health_status["test_service"] = test_health["status"]
        except Exception as e:
            health_status["test_service"] = "error"
            logger.warning(f"Test service health check failed: {e}")
        
        try:
            ai_service = get_ai_service()
            ai_health = ai_service.health_check()
            health_status["ai_service"] = ai_health["status"]
        except Exception as e:
            health_status["ai_service"] = "error"
            logger.warning(f"AI service health check failed: {e}")
        
        try:
            db_manager = get_db_manager()
            db_health = db_manager.validate_connection()
            health_status["database"] = "healthy" if db_health["overall"] else "degraded"
        except Exception as e:
            health_status["database"] = "error"
            logger.warning(f"Database health check failed: {e}")
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "mock_test_api",
                "error": str(e)
            }
        )

# API information endpoint
@app.get("/info")
async def api_info():
    """API information and capabilities"""
    return {
        "name": config.API_TITLE,
        "version": config.API_VERSION,
        "description": config.API_DESCRIPTION,
        "mode": "dummy_data" if config.USE_DUMMY_DATA else "live_data",
        "features": {
            "question_generation": True,
            "batch_processing": True,
            "caching": True,
            "pdf_export": True,
            "dummy_fallback": True
        },
        "configuration": {
            "questions_per_test": config.QUESTIONS_PER_TEST,
            "dev_time_limit": config.DEV_TIME_LIMIT,
            "non_dev_time_limit": config.NON_DEV_TIME_LIMIT,
            "cache_duration_hours": config.QUESTION_CACHE_DURATION_HOURS
        },
        "endpoints": {
            "start_test": "POST /api/test/start",
            "submit_answer": "POST /api/test/submit", 
            "get_results": "GET /api/test/results/{test_id}",
            "download_pdf": "GET /api/test/pdf/{test_id}",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

# Development startup message
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8060'))
    debug_mode = os.getenv('DEBUG_MODE', 'true').lower() == 'true'
    
    logger.info("🚀 Starting Mock Test API in development mode")
    logger.info(f"🔧 Dummy data mode: {config.USE_DUMMY_DATA}")
    logger.info(f"🌐 Server: http://{host}:{port}")
    logger.info(f"📚 Docs: http://{host}:{port}/docs")
    logger.info(f"🔍 Debug mode: {debug_mode}")
    
    uvicorn.run(
        "weekend_mocktest.main:app",
        host=host,
        port=port,
        reload=debug_mode,
        log_level=os.getenv('LOG_LEVEL', 'info').lower(),
        access_log=debug_mode
    )