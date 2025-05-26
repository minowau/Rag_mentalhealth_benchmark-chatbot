import os
import json
import sqlite3
import datetime
import re
import openai
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import numpy as np
import os





if 'OPENAI_API_KEY' not in os.environ:
    raise ValueError("Please set the OPENAI_API_KEY environment variable with your OpenAI API key.")
# Configuration and Setup
class ConversationRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class CrisisLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRISIS = "crisis"

@dataclass
class Message:
    role: ConversationRole
    content: str
    timestamp: datetime.datetime
    emotion_score: Optional[float] = None
    crisis_score: Optional[float] = None

@dataclass
class Assessment:
    assessment_type: str
    questions: List[str]
    responses: List[int]
    score: int
    interpretation: str
    date: datetime.datetime

class MentalHealthChatbot:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.db_path = "mental_health_chatbot.db"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.init_database()
        self.load_resources()
        
        # Crisis keywords for detection
        self.crisis_keywords = [
            "suicide", "kill myself", "end it all", "want to die", "hurt myself",
            "self harm", "cutting", "overdose", "jump off", "hanging", "gun",
            "pills", "bridge", "worthless", "hopeless", "can't go on","die"
        ]
        
        # System prompt for mental health conversations
        self.system_prompt = """You are a compassionate mental health support chatbot. Your role is to:

1. Provide empathetic, non-judgmental support
2. Listen actively and validate feelings
3. Offer coping strategies and resources
4. Encourage professional help when appropriate
5. Maintain clear boundaries - you are not a therapist

IMPORTANT SAFETY GUIDELINES:
- If you detect suicidal ideation or crisis, immediately provide crisis resources
- Never diagnose mental health conditions
- Always encourage professional help for serious concerns
- Maintain confidentiality and respect privacy
- Be supportive but not overly clinical

Respond with warmth, empathy, and genuine care. Keep responses concise but meaningful."""

    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                preferences TEXT
            )
        ''')
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                emotion_score REAL,
                crisis_score REAL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')
        
        # Assessments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                assessment_type TEXT NOT NULL,
                questions TEXT,
                responses TEXT,
                score INTEGER,
                interpretation TEXT,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Resources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                url TEXT,
                phone TEXT,
                available_24_7 BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_resources(self):
        """Load mental health resources into database"""
        resources = [
            {
                "category": "crisis",
                "title": "National Suicide Prevention Lifeline",
                "description": "24/7 crisis support for people in distress",
                "url": "https://suicidepreventionlifeline.org/",
                "phone": "988",
                "available_24_7": True
            },
            {
                "category": "crisis",
                "title": "Crisis Text Line",
                "description": "24/7 support via text message",
                "url": "https://www.crisistextline.org/",
                "phone": "Text HOME to 741741",
                "available_24_7": True
            },
            {
                "category": "therapy",
                "title": "Psychology Today",
                "description": "Find mental health professionals",
                "url": "https://www.psychologytoday.com/",
                "phone": None,
                "available_24_7": False
            },
            {
                "category": "anxiety",
                "title": "Anxiety and Depression Association",
                "description": "Resources for anxiety and depression",
                "url": "https://adaa.org/",
                "phone": None,
                "available_24_7": False
            },
            {
                "category": "depression",
                "title": "National Alliance on Mental Illness",
                "description": "Mental health advocacy and support",
                "url": "https://www.nami.org/",
                "phone": "1-800-950-6264",
                "available_24_7": False
            }
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if resources already exist
        cursor.execute("SELECT COUNT(*) FROM resources")
        if cursor.fetchone()[0] == 0:
            for resource in resources:
                cursor.execute('''
                    INSERT INTO resources (category, title, description, url, phone, available_24_7)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    resource["category"], resource["title"], resource["description"],
                    resource["url"], resource["phone"], resource["available_24_7"]
                ))
        
        conn.commit()
        conn.close()
        
    def detect_crisis(self, text: str) -> Tuple[CrisisLevel, float]:
        """Detect crisis level in user message"""
        text_lower = text.lower()
        crisis_score = 0.0
        
        # Check for explicit crisis keywords
        for keyword in self.crisis_keywords:
            if keyword in text_lower:
                crisis_score += 0.3
                
        # Check for patterns that might indicate crisis
        crisis_patterns = [
            r"i want to (die|kill myself|end it)",
            r"(nobody|no one) (cares|loves me)",
            r"i (can't|cannot) (take it|go on)",
            r"life (isn't|is not) worth",
            r"better off (dead|without me)"
        ]
        
        for pattern in crisis_patterns:
            if re.search(pattern, text_lower):
                crisis_score += 0.4
                
        # Determine crisis level
        if crisis_score >= 0.7:
            return CrisisLevel.CRISIS, crisis_score
        elif crisis_score >= 0.4:
            return CrisisLevel.HIGH, crisis_score
        elif crisis_score >= 0.2:
            return CrisisLevel.MODERATE, crisis_score
        else:
            return CrisisLevel.LOW, crisis_score
            
    def get_crisis_response(self, crisis_level: CrisisLevel) -> str:
        """Get appropriate crisis response based on level"""
        if crisis_level == CrisisLevel.CRISIS:
            return """I'm very concerned about what you're going through right now. Your safety is the most important thing.

🚨 **IMMEDIATE HELP AVAILABLE:**
• **Call 988** - National Suicide Prevention Lifeline (24/7)
• **Text HOME to 741741** - Crisis Text Line (24/7)
• **Call 911** if you're in immediate danger

You don't have to face this alone. These trained professionals are there to help you right now. Please reach out to them - your life has value and things can get better."""

        elif crisis_level == CrisisLevel.HIGH:
            return """I can hear that you're going through a really difficult time. It takes courage to share these feelings.

**Support Resources:**
• National Suicide Prevention Lifeline: **988** (24/7)
• Crisis Text Line: **Text HOME to 741741** (24/7)

Would you like to talk about what's been making things so hard lately? I'm here to listen, and these professional resources are available whenever you need them."""

        elif crisis_level == CrisisLevel.MODERATE:
            return """It sounds like you're dealing with some heavy feelings right now. I want you to know that reaching out shows strength.

If things feel overwhelming, remember that help is available:
• National Suicide Prevention Lifeline: **988**
• Crisis Text Line: **Text HOME to 741741**

What's been weighing on your mind lately? Sometimes talking through difficult feelings can help."""

        return None # type: ignore
        
    def analyze_emotion(self, text: str) -> float:
        """Simple emotion analysis (negative sentiment score)"""
        negative_words = [
            "sad", "depressed", "anxious", "worried", "scared", "angry", "frustrated",
            "hopeless", "worthless", "lonely", "tired", "exhausted", "overwhelmed",
            "stressed", "pain", "hurt", "crying", "tears", "dark", "empty"
        ]
        
        positive_words = [
            "happy", "good", "better", "hopeful", "grateful", "thankful", "joy",
            "excited", "calm", "peaceful", "strong", "confident", "loved", "supported","excellent"
        ]
        
        text_lower = text.lower()
        negative_count = sum(1 for word in negative_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        # Return emotion score (-1 to 1, where -1 is very negative, 1 is very positive)
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
            
        emotion_score = (positive_count - negative_count) / max(total_words, 1)
        return max(-1.0, min(1.0, emotion_score * 10))  # Scale and clamp
        
    def generate_response(self, messages: List[Message], user_id: int = None) -> str: # type: ignore
        """Generate response using OpenAI API"""
        try:
            # Check for crisis in the latest message
            latest_message = messages[-1].content
            crisis_level, crisis_score = self.detect_crisis(latest_message)
            
            if crisis_level in [CrisisLevel.CRISIS, CrisisLevel.HIGH]:
                return self.get_crisis_response(crisis_level)
            
            # Prepare messages for OpenAI API
            api_messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history (limit to last 10 messages to manage context)
            for message in messages[-10:]:
                api_messages.append({
                    "role": message.role.value,
                    "content": message.content
                })
            
            # Generate response
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=api_messages, # type: ignore
                max_tokens=300,
                temperature=0.7,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            assistant_response = response.choices[0].message.content
            
            # Add crisis resources if moderate crisis detected
            if crisis_level == CrisisLevel.MODERATE:
                crisis_addendum = self.get_crisis_response(crisis_level)
                assistant_response += f"\n\n{crisis_addendum}" # type: ignore
            
            return assistant_response # type: ignore
            
        except Exception as e:
            return f"I apologize, but I'm having trouble responding right now. If you're in crisis, please call 988 (Suicide Prevention Lifeline) or text HOME to 741741. Error: {str(e)}"
            
    def save_conversation(self, user_id: int, messages: List[Message]) -> int:
        """Save conversation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create conversation
        cursor.execute('''
            INSERT INTO conversations (user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        ''', (user_id, f"Conversation {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", 
              datetime.datetime.now(), datetime.datetime.now()))
        
        conversation_id = cursor.lastrowid
        
        # Save messages
        for message in messages:
            cursor.execute('''
                INSERT INTO messages (conversation_id, role, content, timestamp, emotion_score, crisis_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (conversation_id, message.role.value, message.content, 
                  message.timestamp, message.emotion_score, message.crisis_score))
        
        conn.commit()
        conn.close()
        return conversation_id # type: ignore
        
    def get_phq9_questions(self) -> List[str]:
        """PHQ-9 Depression Screening Questions"""
        return [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself or that you are a failure or have let yourself or your family down",
            "Trouble concentrating on things, such as reading the newspaper or watching television",
            "Moving or speaking so slowly that other people could have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual",
            "Thoughts that you would be better off dead, or of hurting yourself in some way"
        ]
        
    def get_gad7_questions(self) -> List[str]:
        """GAD-7 Anxiety Screening Questions"""
        return [
            "Feeling nervous, anxious, or on edge",
            "Not being able to stop or control worrying",
            "Worrying too much about different things",
            "Trouble relaxing",
            "Being so restless that it is hard to sit still",
            "Becoming easily annoyed or irritable",
            "Feeling afraid, as if something awful might happen"
        ]
        
    def calculate_phq9_score(self, responses: List[int]) -> Tuple[int, str]:
        """Calculate PHQ-9 score and interpretation"""
        total_score = sum(responses)
        
        if total_score <= 4:
            interpretation = "Minimal depression"
        elif total_score <= 9:
            interpretation = "Mild depression"
        elif total_score <= 14:
            interpretation = "Moderate depression"
        elif total_score <= 19:
            interpretation = "Moderately severe depression"
        else:
            interpretation = "Severe depression"
            
        return total_score, interpretation
        
    def calculate_gad7_score(self, responses: List[int]) -> Tuple[int, str]:
        """Calculate GAD-7 score and interpretation"""
        total_score = sum(responses)
        
        if total_score <= 4:
            interpretation = "Minimal anxiety"
        elif total_score <= 9:
            interpretation = "Mild anxiety"
        elif total_score <= 14:
            interpretation = "Moderate anxiety"
        else:
            interpretation = "Severe anxiety"
            
        return total_score, interpretation
        
    def save_assessment(self, user_id: int, assessment: Assessment):
        """Save assessment results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO assessments (user_id, assessment_type, questions, responses, score, interpretation, date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            assessment.assessment_type,
            json.dumps(assessment.questions),
            json.dumps(assessment.responses),
            assessment.score,
            assessment.interpretation,
            assessment.date
        ))
        
        conn.commit()
        conn.close()
        
    def get_user_assessments(self, user_id: int) -> List[Assessment]:
        """Get user's assessment history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT assessment_type, questions, responses, score, interpretation, date
            FROM assessments 
            WHERE user_id = ? 
            ORDER BY date DESC
        ''', (user_id,))
        
        assessments = []
        for row in cursor.fetchall():
            assessment = Assessment(
                assessment_type=row[0],
                questions=json.loads(row[1]),
                responses=json.loads(row[2]),
                score=row[3],
                interpretation=row[4],
                date=datetime.datetime.fromisoformat(row[5])
            )
            assessments.append(assessment)
            
        conn.close()
        return assessments
        
    def get_resources_by_category(self, category: str) -> List[Dict]:
        """Get resources by category"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, description, url, phone, available_24_7
            FROM resources 
            WHERE category = ?
        ''', (category,))
        
        resources = []
        for row in cursor.fetchall():
            resources.append({
                "title": row[0],
                "description": row[1],
                "url": row[2],
                "phone": row[3],
                "available_24_7": bool(row[4])
            })
            
        conn.close()
        return resources

# Streamlit UI Implementation
def main():
    st.set_page_config(
        page_title="Mental Health Support Chatbot",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Mental Health Support Chatbot")
    st.markdown("*A compassionate AI companion for mental health support*")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        api_key =  os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            st.stop()
        st.session_state.chatbot = MentalHealthChatbot(api_key) # type: ignore
        
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 1  # Default user for demo
        
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Tools & Resources")
        
        tool = st.selectbox(
            "Choose a tool:",
            ["Chat", "Depression Screening (PHQ-9)", "Anxiety Screening (GAD-7)", 
             "Assessment History", "Crisis Resources", "Mental Health Resources"]
        )
        
        if st.button("🔄 New Conversation"):
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("---")
        st.markdown("### 🚨 Crisis Resources")
        st.markdown("**Suicide Prevention Lifeline**")
        st.markdown("📞 **988** (24/7)")
        st.markdown("**Crisis Text Line**")
        st.markdown("📱 **Text HOME to 741741**")
        
    # Main content area
    if tool == "Chat":
        show_chat_interface()
    elif tool == "Depression Screening (PHQ-9)":
        show_phq9_assessment()
    elif tool == "Anxiety Screening (GAD-7)":
        show_gad7_assessment()
    elif tool == "Assessment History":
        show_assessment_history()
    elif tool == "Crisis Resources":
        show_crisis_resources()
    elif tool == "Mental Health Resources":
        show_mental_health_resources()

def show_chat_interface():
    """Display chat interface"""
    st.header("💬 Chat with Your Mental Health Support Bot")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message.role.value):
            st.write(message.content)
            if message.emotion_score is not None:
                emotion_text = "😊 Positive" if message.emotion_score > 0 else "😔 Negative" if message.emotion_score < 0 else "😐 Neutral"
                st.caption(f"Emotion: {emotion_text}")
    
    # Chat input
    if prompt := st.chat_input("How are you feeling today?"):
        # Add user message
        user_message = Message(
            role=ConversationRole.USER,
            content=prompt,
            timestamp=datetime.datetime.now()
        )
        
        # Analyze emotion and crisis level
        user_message.emotion_score = st.session_state.chatbot.analyze_emotion(prompt)
        _, user_message.crisis_score = st.session_state.chatbot.detect_crisis(prompt)
        
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
            if user_message.emotion_score is not None:
                emotion_text = "😊 Positive" if user_message.emotion_score > 0 else "😔 Negative" if user_message.emotion_score < 0 else "😐 Neutral"
                st.caption(f"Emotion: {emotion_text}")
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.generate_response(
                    st.session_state.messages,
                    st.session_state.user_id
                )
                
                assistant_message = Message(
                    role=ConversationRole.ASSISTANT,
                    content=response,
                    timestamp=datetime.datetime.now()
                )
                
                st.session_state.messages.append(assistant_message)
                st.write(response)

def show_phq9_assessment():
    """Display PHQ-9 depression screening"""
    st.header("📋 PHQ-9 Depression Screening")
    st.markdown("*Patient Health Questionnaire - 9 items*")
    
    st.info("Over the last 2 weeks, how often have you been bothered by any of the following problems?")
    
    questions = st.session_state.chatbot.get_phq9_questions()
    
    with st.form("phq9_form"):
        st.markdown("**Rate each item:** 0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day")
        
        responses = []
        for i, question in enumerate(questions):
            response = st.select_slider(
                f"{i+1}. {question}",
                options=[0, 1, 2, 3],
                format_func=lambda x: ["Not at all", "Several days", "More than half the days", "Nearly every day"][x],
                key=f"phq9_{i}"
            )
            responses.append(response)
        
        submitted = st.form_submit_button("Calculate Score")
        
        if submitted:
            score, interpretation = st.session_state.chatbot.calculate_phq9_score(responses)
            
            # Save assessment
            assessment = Assessment(
                assessment_type="PHQ-9",
                questions=questions,
                responses=responses,
                score=score,
                interpretation=interpretation,
                date=datetime.datetime.now()
            )
            
            st.session_state.chatbot.save_assessment(st.session_state.user_id, assessment)
            
            # Display results
            st.success(f"**Assessment Complete!**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("PHQ-9 Score", score, delta=None)
            
            with col2:
                st.metric("Interpretation", interpretation)
            
            # Provide recommendations
            if score >= 15:
                st.error("**Important:** Your score indicates moderately severe to severe depression. Please consider speaking with a mental health professional.")
                st.markdown("### 🆘 Immediate Resources:")
                st.markdown("- **National Suicide Prevention Lifeline:** 988")
                st.markdown("- **Crisis Text Line:** Text HOME to 741741")
            elif score >= 10:
                st.warning("Your score suggests moderate depression. Consider reaching out to a mental health professional for support.")
            elif score >= 5:
                st.info("Your score indicates mild depression. Self-care strategies and monitoring your symptoms may be helpful.")
            else:
                st.success("Your score suggests minimal depression symptoms. Continue taking care of your mental health!")
            
            # Show score interpretation chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "PHQ-9 Depression Score"},
                gauge = {
                    'axis': {'range': [None, 27]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 4], 'color': "lightgreen"},
                        {'range': [5, 9], 'color': "yellow"},
                        {'range': [10, 14], 'color': "orange"},
                        {'range': [15, 19], 'color': "red"},
                        {'range': [20, 27], 'color': "darkred"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_gad7_assessment():
    """Display GAD-7 anxiety screening"""
    st.header("📋 GAD-7 Anxiety Screening")
    st.markdown("*Generalized Anxiety Disorder - 7 items*")
    
    st.info("Over the last 2 weeks, how often have you been bothered by the following problems?")
    
    questions = st.session_state.chatbot.get_gad7_questions()
    
    with st.form("gad7_form"):
        st.markdown("**Rate each item:** 0 = Not at all, 1 = Several days, 2 = More than half the days, 3 = Nearly every day")
        
        responses = []
        for i, question in enumerate(questions):
            response = st.select_slider(
                f"{i+1}. {question}",
                options=[0, 1, 2, 3],
                format_func=lambda x: ["Not at all", "Several days", "More than half the days", "Nearly every day"][x],
                key=f"gad7_{i}"
            )
            responses.append(response)
        
        submitted = st.form_submit_button("Calculate Score")
        
        if submitted:
            score, interpretation = st.session_state.chatbot.calculate_gad7_score(responses)
            
            # Save assessment
            assessment = Assessment(
                assessment_type="GAD-7",
                questions=questions,
                responses=responses,
                score=score,
                interpretation=interpretation,
                date=datetime.datetime.now()
            )
            
            st.session_state.chatbot.save_assessment(st.session_state.user_id, assessment)
            
            # Display results
            st.success(f"**Assessment Complete!**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("GAD-7 Score", score, delta=None)
            
            with col2:
                st.metric("Interpretation", interpretation)
            
            # Provide recommendations
            if score >= 15:
                st.error("**Important:** Your score indicates severe anxiety. Please consider speaking with a mental health professional.")
                st.markdown("### 🆘 Resources:")
                st.markdown("- **National Suicide Prevention Lifeline:** 988")
                st.markdown("- **Crisis Text Line:** Text HOME to 741741")
            elif score >= 10:
                st.warning("Your score suggests moderate anxiety. Consider reaching out to a mental health professional for support.")
            elif score >= 5:
                st.info("Your score indicates mild anxiety. Relaxation techniques and stress management may be helpful.")
            else:
                st.success("Your score suggests minimal anxiety symptoms. Keep up the good work with your mental health!")
            
            # Show score interpretation chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "GAD-7 Anxiety Score"},
                gauge = {
                    'axis': {'range': [None, 21]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 4], 'color': "lightgreen"},
                        {'range': [5, 9], 'color': "yellow"},
                        {'range': [10, 14], 'color': "orange"},
                        {'range': [15, 21], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_assessment_history():
    """Display user's assessment history"""
    st.header("📊 Assessment History")
    
    assessments = st.session_state.chatbot.get_user_assessments(st.session_state.user_id)
    
    if not assessments:
        st.info("No assessments completed yet. Try taking a PHQ-9 or GAD-7 screening!")
        return
    
    # Create tabs for different assessment types
    phq9_assessments = [a for a in assessments if a.assessment_type == "PHQ-9"]
    gad7_assessments = [a for a in assessments if a.assessment_type == "GAD-7"]
    
    tab1, tab2 = st.tabs(["PHQ-9 (Depression)", "GAD-7 (Anxiety)"])
    
    with tab1:
        if phq9_assessments:
            st.subheader("Depression Screening History")
            
            # Create trend chart
            dates = [a.date for a in phq9_assessments]
            scores = [a.score for a in phq9_assessments]
            
            fig = px.line(
                x=dates, y=scores,
                title="PHQ-9 Score Trend",
                labels={'x': 'Date', 'y': 'PHQ-9 Score'},
                markers=True
            )
            fig.add_hline(y=5, line_dash="dash", line_color="yellow", annotation_text="Mild Depression")
            fig.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="Moderate Depression")
            fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Moderately Severe Depression")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Assessment table
            st.subheader("Detailed History")
            phq9_df = pd.DataFrame([
                {
                    "Date": a.date.strftime("%Y-%m-%d %H:%M"),
                    "Score": a.score,
                    "Interpretation": a.interpretation
                }
                for a in phq9_assessments
            ])
            st.dataframe(phq9_df, use_container_width=True)
        else:
            st.info("No PHQ-9 assessments completed yet.")
    
    with tab2:
        if gad7_assessments:
            st.subheader("Anxiety Screening History")
            
            # Create trend chart
            dates = [a.date for a in gad7_assessments]
            scores = [a.score for a in gad7_assessments]
            
            fig = px.line(
                x=dates, y=scores,
                title="GAD-7 Score Trend",
                labels={'x': 'Date', 'y': 'GAD-7 Score'},
                markers=True
            )
            fig.add_hline(y=5, line_dash="dash", line_color="yellow", annotation_text="Mild Anxiety")
            fig.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="Moderate Anxiety")
            fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Severe Anxiety")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Assessment table
            st.subheader("Detailed History")
            gad7_df = pd.DataFrame([
                {
                    "Date": a.date.strftime("%Y-%m-%d %H:%M"),
                    "Score": a.score,
                    "Interpretation": a.interpretation
                }
                for a in gad7_assessments
            ])
            st.dataframe(gad7_df, use_container_width=True)
        else:
            st.info("No GAD-7 assessments completed yet.")

def show_crisis_resources():
    """Display crisis intervention resources"""
    st.header("🚨 Crisis Resources")
    st.markdown("**If you or someone you know is in immediate danger, call 911.**")
    
    resources = st.session_state.chatbot.get_resources_by_category("crisis")
    
    for resource in resources:
        with st.expander(f"📞 {resource['title']}", expanded=True):
            st.markdown(f"**Description:** {resource['description']}")
            
            if resource['phone']:
                st.markdown(f"**Phone:** {resource['phone']}")
            
            if resource['url']:
                st.markdown(f"**Website:** [{resource['url']}]({resource['url']})")
            
            if resource['available_24_7']:
                st.success("✅ Available 24/7")
            
            st.markdown("---")
    
    st.markdown("### 🛡️ Additional Safety Tips")
    st.markdown("""
    - **Remove means of self-harm** from your immediate environment
    - **Stay with someone** you trust or ask someone to stay with you
    - **Go to your nearest emergency room** if you feel unsafe
    - **Call a trusted friend, family member, or mental health professional**
    - **Use the Crisis Text Line** if you prefer texting: Text HOME to 741741
    """)
    
    st.markdown("### 🤝 How to Help Someone in Crisis")
    st.markdown("""
    - **Listen without judgment** and take their concerns seriously
    - **Ask directly** if they are thinking about suicide
    - **Stay with them** and help them connect with professional help
    - **Remove potential means of harm** if possible and safe to do so
    - **Follow up** to show you care and check on their wellbeing
    """)

def show_mental_health_resources():
    """Display general mental health resources"""
    st.header("🌟 Mental Health Resources")
    
    categories = ["therapy", "anxiety", "depression"]
    
    for category in categories:
        resources = st.session_state.chatbot.get_resources_by_category(category)
        
        if resources:
            st.subheader(f"📚 {category.title()} Resources")
            
            for resource in resources:
                with st.expander(f"{resource['title']}"):
                    st.markdown(f"**Description:** {resource['description']}")
                    
                    if resource['phone']:
                        st.markdown(f"**Phone:** {resource['phone']}")
                    
                    if resource['url']:
                        st.markdown(f"**Website:** [{resource['url']}]({resource['url']})")
                    
                    if resource['available_24_7']:
                        st.success("✅ Available 24/7")
    
    st.markdown("---")
    st.markdown("### 💡 Self-Care Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Daily Habits:**
        - Maintain a regular sleep schedule
        - Exercise regularly (even light walking helps)
        - Eat nutritious meals
        - Practice mindfulness or meditation
        - Stay connected with supportive people
        """)
    
    with col2:
        st.markdown("""
        **Coping Strategies:**
        - Deep breathing exercises
        - Progressive muscle relaxation
        - Journaling or writing
        - Creative activities (art, music, crafts)
        - Spending time in nature
        """)
    
    st.markdown("### 📱 Mental Health Apps")
    st.markdown("""
    - **Headspace**: Meditation and mindfulness
    - **Calm**: Sleep stories and relaxation
    - **Daylio**: Mood tracking
    - **Sanvello**: Anxiety and mood tracking
    - **PTSD Coach**: For trauma recovery
    """)

# Additional utility functions
def create_mood_chart(user_id: int):
    """Create mood tracking visualization"""
    # This would integrate with a mood tracking feature
    # For now, we'll create a sample chart
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    moods = np.random.normal(5, 2, 30)  # Sample mood data
    
    fig = px.line(
        x=dates, y=moods,
        title="Mood Tracking Over Time",
        labels={'x': 'Date', 'y': 'Mood (1-10 scale)'},
        markers=True
    )
    fig.update_layout(height=400)
    return fig

def export_conversation_history():
    """Export conversation to text file"""
    if not st.session_state.messages:
        return None
    
    conversation_text = f"Mental Health Chatbot Conversation Export\n"
    conversation_text += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    conversation_text += "="*50 + "\n\n"
    
    for message in st.session_state.messages:
        role = "You" if message.role == ConversationRole.USER else "Bot"
        conversation_text += f"{role} ({message.timestamp.strftime('%H:%M')}):\n"
        conversation_text += f"{message.content}\n\n"
    
    return conversation_text

# Main execution
if __name__ == "__main__":
    main()