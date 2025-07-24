# weekend_mocktest/core/dummy_data.py
import time
import random
from typing import List, Dict, Any

# Dummy summaries for ML/AI topics when database is unavailable
DUMMY_SUMMARIES = [
    {
        "_id": "dummy_1",
        "summary": """1. Machine Learning Operations (MLOps) represents a critical discipline that bridges the gap between machine learning model development and production deployment, encompassing the entire lifecycle from data preparation to model monitoring.

2. Data pipeline automation forms the backbone of MLOps, involving automated data ingestion, cleaning, validation, and feature engineering processes that ensure consistent and reliable data flow for model training and inference.

3. Model versioning and experiment tracking enable teams to maintain reproducibility, compare different model iterations, and roll back to previous versions when necessary, using tools like MLflow, DVC, or custom versioning systems.

4. Continuous integration and continuous deployment (CI/CD) for ML involves automated testing of data quality, model performance validation, and seamless deployment of models to production environments.

5. Infrastructure as Code (IaC) principles apply to ML environments, allowing teams to define, version, and replicate computing resources, container configurations, and deployment specifications programmatically.

6. Model monitoring and observability in production include tracking data drift, model performance degradation, prediction latency, and business metrics to ensure models continue to deliver value over time.

7. Feature stores provide centralized repositories for storing, versioning, and serving features across different ML projects, promoting reusability and consistency in feature engineering.

8. A/B testing frameworks for ML models enable safe deployment strategies, allowing teams to compare model performance in production environments before full rollout.""",
        "timestamp": time.time() - 3600,
        "date": "2024-01-15",
        "session_id": "session_ml_001"
    },
    {
        "_id": "dummy_2", 
        "summary": """1. Deep Learning architectures have revolutionized artificial intelligence by enabling automatic feature extraction from raw data through multiple layers of neural networks, eliminating the need for manual feature engineering in many applications.

2. Convolutional Neural Networks (CNNs) excel in computer vision tasks by using convolution operations, pooling layers, and hierarchical feature maps to detect patterns, edges, and complex visual structures in images.

3. Recurrent Neural Networks (RNNs) and their variants like LSTM and GRU are designed to handle sequential data by maintaining hidden states that capture temporal dependencies and context across time steps.

4. Transformer architectures, introduced with attention mechanisms, have become the foundation for modern natural language processing, enabling models to focus on relevant parts of input sequences without recurrent connections.

5. Transfer learning allows pre-trained models to be adapted for new tasks with minimal additional training, leveraging knowledge gained from large datasets to improve performance on smaller, domain-specific datasets.

6. Regularization techniques such as dropout, batch normalization, and weight decay help prevent overfitting in deep networks by reducing model complexity and improving generalization to unseen data.

7. Optimization algorithms like Adam, RMSprop, and SGD with momentum are crucial for training deep networks efficiently, each with different strategies for updating weights based on gradient information.

8. Hardware acceleration through GPUs, TPUs, and specialized AI chips has made training large deep learning models feasible, enabling breakthrough applications in image recognition, natural language processing, and game playing.""",
        "timestamp": time.time() - 7200,
        "date": "2024-01-14", 
        "session_id": "session_dl_002"
    },
    {
        "_id": "dummy_3",
        "summary": """1. Natural Language Processing (NLP) encompasses computational techniques for analyzing, understanding, and generating human language, combining linguistics, computer science, and machine learning to enable human-computer interaction through text and speech.

2. Text preprocessing forms the foundation of NLP pipelines, involving tokenization, stemming, lemmatization, stop word removal, and normalization to convert raw text into structured formats suitable for machine learning algorithms.

3. Word embeddings like Word2Vec, GloVe, and FastText transform words into dense vector representations that capture semantic relationships, enabling machines to understand word similarities and contexts mathematically.

4. Named Entity Recognition (NER) identifies and classifies entities such as persons, organizations, locations, and dates within text, serving as a crucial component for information extraction and knowledge graph construction.

5. Sentiment analysis determines the emotional tone or opinion expressed in text, using classification algorithms to categorize content as positive, negative, or neutral for applications in social media monitoring and customer feedback analysis.

6. Language models, from n-grams to transformer-based architectures like GPT and BERT, learn to predict and generate text by understanding patterns, grammar, and semantics from large text corpora.

7. Machine translation systems translate text between languages using neural networks, attention mechanisms, and encoder-decoder architectures to preserve meaning while adapting to target language structures and cultural contexts.

8. Question answering systems combine reading comprehension, information retrieval, and natural language generation to provide accurate responses to user queries from structured or unstructured knowledge sources.""",
        "timestamp": time.time() - 10800,
        "date": "2024-01-13",
        "session_id": "session_nlp_003"
    },
    {
        "_id": "dummy_4",
        "summary": """1. Computer Vision enables machines to interpret and understand visual information from images and videos, combining image processing, pattern recognition, and machine learning to replicate human visual perception capabilities.

2. Image preprocessing techniques including resizing, normalization, augmentation, and noise reduction prepare visual data for analysis, ensuring consistent input formats and improving model robustness through data diversity.

3. Feature extraction methods detect edges, corners, textures, and shapes using filters, gradients, and descriptors like SIFT, HOG, and LBP to represent visual content in mathematical formats suitable for analysis.

4. Object detection algorithms like YOLO, R-CNN, and SSD locate and classify multiple objects within images by combining classification and localization tasks through bounding box regression and confidence scoring.

5. Image segmentation partitions images into meaningful regions or objects using techniques like semantic segmentation, instance segmentation, and panoptic segmentation for detailed scene understanding and analysis.

6. Facial recognition systems identify and verify individuals by analyzing facial features, landmarks, and biometric patterns using deep learning models trained on large face datasets with privacy and ethical considerations.

7. Optical Character Recognition (OCR) converts text within images into machine-readable format, enabling document digitization, automated data entry, and text extraction from photographs and scanned documents.

8. Medical imaging analysis applies computer vision to X-rays, MRIs, CT scans, and other medical images for disease detection, diagnosis assistance, and treatment planning, improving healthcare outcomes through AI-powered insights.""",
        "timestamp": time.time() - 14400,
        "date": "2024-01-12",
        "session_id": "session_cv_004"
    },
    {
        "_id": "dummy_5",
        "summary": """1. Reinforcement Learning (RL) enables agents to learn optimal behaviors through interaction with environments, using trial-and-error exploration and reward signals to develop strategies for complex decision-making problems.

2. Markov Decision Processes (MDPs) provide the mathematical framework for RL, defining states, actions, transition probabilities, and rewards that characterize the environment and enable optimal policy computation.

3. Q-Learning and Deep Q-Networks (DQN) learn action-value functions that estimate the expected future rewards for taking specific actions in given states, enabling agents to make optimal decisions without explicit environment models.

4. Policy gradient methods like REINFORCE, Actor-Critic, and PPO directly optimize the agent's policy by estimating gradients of expected rewards, allowing for continuous action spaces and more stable learning.

5. Exploration strategies balance the trade-off between exploiting known good actions and exploring potentially better alternatives using epsilon-greedy, Upper Confidence Bound (UCB), and curiosity-driven approaches.

6. Multi-agent reinforcement learning addresses scenarios where multiple agents interact simultaneously, requiring coordination, competition, or cooperation strategies to achieve individual or collective objectives.

7. Transfer learning in RL enables agents to leverage knowledge from previous tasks to accelerate learning in new environments, reducing training time and sample complexity through shared representations and experiences.

8. Real-world applications of RL span autonomous vehicles, game playing, robotics, recommendation systems, and resource allocation, demonstrating the versatility of learning-based decision making in complex domains.""",
        "timestamp": time.time() - 18000,
        "date": "2024-01-11",
        "session_id": "session_rl_005"
    },
    {
        "_id": "dummy_6",
        "summary": """1. AI Ethics and Responsible AI development address the moral implications, biases, and societal impacts of artificial intelligence systems, ensuring fairness, transparency, accountability, and human-centered design principles in AI applications.

2. Algorithmic bias occurs when AI systems produce discriminatory outcomes based on race, gender, age, or other protected characteristics, often stemming from biased training data, flawed algorithms, or inadequate testing across diverse populations.

3. Explainable AI (XAI) develops techniques to make AI decision-making processes interpretable and understandable to humans, using methods like LIME, SHAP, and attention visualization to build trust and enable effective human-AI collaboration.

4. Privacy-preserving machine learning employs techniques like differential privacy, federated learning, and homomorphic encryption to protect sensitive personal information while still enabling valuable insights and model training.

5. AI governance frameworks establish guidelines, regulations, and oversight mechanisms for AI development and deployment, addressing safety standards, testing requirements, and accountability measures across industries and jurisdictions.

6. Human-in-the-loop systems maintain human oversight and control in AI decision-making processes, particularly for high-stakes applications like healthcare, criminal justice, and autonomous vehicles where human judgment remains crucial.

7. Robustness and safety testing evaluate AI systems' behavior under adversarial attacks, edge cases, and unexpected inputs to prevent failures that could cause harm or unintended consequences in real-world deployments.

8. Sustainable AI considers the environmental impact of large-scale model training and inference, promoting energy-efficient algorithms, green computing practices, and carbon footprint reduction in AI research and development.""",
        "timestamp": time.time() - 21600,
        "date": "2024-01-10",
        "session_id": "session_ethics_006"
    },
    {
        "_id": "dummy_7",
        "summary": """1. AI in Healthcare transforms medical practice through diagnostic assistance, drug discovery, personalized treatment plans, and predictive analytics, improving patient outcomes while reducing costs and enhancing healthcare accessibility worldwide.

2. Medical image analysis uses deep learning to detect diseases in radiology scans, pathology slides, and ophthalmology images, often achieving diagnostic accuracy comparable to or exceeding human specialists in specific domains.

3. Drug discovery and development leverage AI to identify potential therapeutic compounds, predict molecular properties, optimize drug formulations, and accelerate clinical trial design, reducing the time and cost of bringing new medicines to market.

4. Electronic Health Record (EHR) analysis applies natural language processing and machine learning to extract insights from clinical notes, predict patient risks, and support clinical decision-making through automated pattern recognition.

5. Precision medicine uses AI to analyze genomic data, biomarkers, and patient characteristics to tailor treatments to individual patients, moving away from one-size-fits-all approaches toward personalized therapeutic strategies.

6. Robotic surgery systems integrate AI for enhanced precision, tremor reduction, and real-time guidance during minimally invasive procedures, improving surgical outcomes and reducing recovery times for patients.

7. Mental health applications employ chatbots, sentiment analysis, and behavioral pattern recognition to provide accessible mental health support, early intervention, and monitoring for depression, anxiety, and other conditions.

8. Healthcare operations optimization uses AI for resource allocation, staff scheduling, inventory management, and patient flow prediction, improving hospital efficiency and reducing wait times while maintaining quality of care.""",
        "timestamp": time.time() - 25200,
        "date": "2024-01-09",
        "session_id": "session_health_007"
    }
]

# Dummy student data for SQL fallback
DUMMY_STUDENTS = [
    {"ID": 1001, "First_Name": "John", "Last_Name": "Doe"},
    {"ID": 1002, "First_Name": "Jane", "Last_Name": "Smith"}, 
    {"ID": 1003, "First_Name": "Alice", "Last_Name": "Johnson"},
    {"ID": 1004, "First_Name": "Bob", "Last_Name": "Wilson"},
    {"ID": 1005, "First_Name": "Carol", "Last_Name": "Brown"}
]

DUMMY_SESSIONS = [
    {"Session_ID": "session_001"},
    {"Session_ID": "session_002"},
    {"Session_ID": "session_003"},
    {"Session_ID": "session_004"},
    {"Session_ID": "session_005"}
]

class DummyDataService:
    """Service for providing dummy data when database is unavailable"""
    
    def __init__(self):
        self.summaries = DUMMY_SUMMARIES
        self.students = DUMMY_STUDENTS
        self.sessions = DUMMY_SESSIONS
    
    def get_recent_summaries(self, limit: int = 7) -> List[Dict[str, Any]]:
        """Get recent dummy summaries"""
        # Sort by timestamp descending and limit
        sorted_summaries = sorted(
            self.summaries, 
            key=lambda x: x["timestamp"], 
            reverse=True
        )
        return sorted_summaries[:limit]
    
    def get_random_student_info(self) -> tuple:
        """Get random student info for test results"""
        student = random.choice(self.students)
        session = random.choice(self.sessions)
        
        return (
            student["ID"],
            student["First_Name"], 
            student["Last_Name"],
            session["Session_ID"]
        )
    
    def get_all_students(self) -> List[Dict[str, Any]]:
        """Get all dummy students"""
        return [
            {
                "Student_ID": student["ID"],
                "name": f"{student['First_Name']} {student['Last_Name']}"
            }
            for student in self.students
        ]
    
    def validate_dummy_data(self) -> bool:
        """Validate dummy data integrity"""
        try:
            # Check summaries
            if not self.summaries or len(self.summaries) < 5:
                return False
            
            # Check each summary has required fields
            for summary in self.summaries:
                required_fields = ["_id", "summary", "timestamp", "date", "session_id"]
                if not all(field in summary for field in required_fields):
                    return False
                
                # Check summary content has numbered points
                if not any(f"{i}." in summary["summary"] for i in range(1, 6)):
                    return False
            
            # Check students
            if not self.students or len(self.students) < 3:
                return False
            
            return True
            
        except Exception:
            return False

# Global dummy data service instance
dummy_data_service = DummyDataService()

def get_dummy_data_service() -> DummyDataService:
    """Get dummy data service instance"""
    return dummy_data_service