# weekend_mocktest/api/routes.py
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import io

from ..services.test_service import get_test_service
from ..services.pdf_service import get_pdf_service
from ..core.utils import DateTimeUtils

logger = logging.getLogger(__name__)

router = APIRouter()
test_service = get_test_service()
pdf_service = get_pdf_service()

@router.get("/")
async def home():
    """Home endpoint"""
    return {
        "service": "Mock Test API",
        "version": "5.0.0",
        "status": "operational"
    }

@router.get("/api/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": DateTimeUtils.get_current_timestamp()
    }

@router.post("/weekend_mocktest/api/test/start")
async def start_test(request_data: dict):
    """Start test - Frontend compatible"""
    try:
        user_type = request_data.get("user_type", "dev")
        
        if user_type == "developer":
            user_type = "dev"
        elif user_type == "non-developer": 
            user_type = "non_dev"
        
        test_response = await test_service.start_test(user_type)
        
        return {
            "testId": test_response.test_id,
            "test_id": test_response.test_id,
            "sessionId": f"session_{test_response.test_id[:8]}",
            "session_id": f"session_{test_response.test_id[:8]}",
            "userType": test_response.user_type,
            "user_type": test_response.user_type,
            "totalQuestions": test_response.total_questions,
            "total_questions": test_response.total_questions,
            "timeLimit": test_response.time_limit,
            "time_limit": test_response.time_limit,
            "questionNumber": test_response.question_number,
            "question_number": test_response.question_number,
            "questionHtml": test_response.question_html,
            "question_html": test_response.question_html,
            "options": test_response.options,
            "raw": {
                "test_id": test_response.test_id,
                "session_id": f"session_{test_response.test_id[:8]}",
                "user_type": test_response.user_type,
                "total_questions": test_response.total_questions,
                "time_limit": test_response.time_limit,
                "question_number": test_response.question_number,
                "question_html": test_response.question_html,
                "options": test_response.options
            }
        }
        
    except Exception as e:
        logger.error(f"Test start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/weekend_mocktest/api/test/submit")
async def submit_answer(request_data: dict):
    """Submit answer - Frontend compatible"""
    try:
        test_id = request_data.get("test_id")
        question_number = request_data.get("question_number")
        answer = request_data.get("answer", "")
        
        if not test_id or not question_number:
            raise ValueError("test_id and question_number are required")
        
        response = await test_service.submit_answer(test_id, question_number, answer)
        
        if response.test_completed:
            return {
                "testCompleted": True,
                "test_completed": True,
                "score": response.score,
                "totalQuestions": response.total_questions,
                "total_questions": response.total_questions,
                "analytics": response.analytics
            }
        else:
            next_q = response.next_question
            return {
                "testCompleted": False,
                "test_completed": False,
                "nextQuestion": {
                    "questionNumber": next_q.question_number,
                    "question_number": next_q.question_number,
                    "totalQuestions": next_q.total_questions,
                    "total_questions": next_q.total_questions,
                    "questionHtml": next_q.question_html,
                    "question_html": next_q.question_html,
                    "options": next_q.options,
                    "timeLimit": next_q.time_limit,
                    "time_limit": next_q.time_limit
                }
            }
        
    except Exception as e:
        logger.error(f"Answer submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/weekend_mocktest/api/test/results/{test_id}")
async def get_test_results(test_id: str):
    """Get test results - Frontend compatible"""
    try:
        results = await test_service.get_test_results(test_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Test results not found")
        
        return {
            "testId": test_id,
            "test_id": test_id,
            "score": results["score"],
            "totalQuestions": results["total_questions"],
            "total_questions": results["total_questions"],
            "scorePercentage": results.get("score_percentage", 0),
            "analytics": results["analytics"],
            "timestamp": results["timestamp"],
            "pdfAvailable": True,
            "pdf_available": True
        }
        
    except Exception as e:
        logger.error(f"Error fetching results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/weekend_mocktest/api/test/pdf/{test_id}")
async def download_pdf(test_id: str):
    """Download PDF - Frontend compatible"""
    try:
        pdf_bytes = await pdf_service.generate_test_results_pdf(test_id)
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=test_results_{test_id}.pdf"}
        )
        
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/tests")
async def get_all_tests():
    """Get all test results"""
    try:
        results = await test_service.get_all_tests()
        return {
            "count": len(results),
            "results": results,
            "timestamp": DateTimeUtils.get_current_timestamp()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/students")
async def get_students():
    """Get students list"""
    try:
        students = await test_service.get_students()
        return {
            "count": len(students),
            "students": students
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/students/{student_id}/tests")
async def get_student_tests(student_id: str):
    """Get student tests"""
    try:
        tests = await test_service.get_student_tests(student_id)
        return {"count": len(tests), "tests": tests}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/cleanup")
async def cleanup_resources():
    """Cleanup expired tests"""
    try:
        result = test_service.cleanup_expired_tests()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))