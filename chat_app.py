"""
Bias Heist - US Visa Approval "Officer" Chatbot (rule-based, no AI/LLM)

A scripted conversational flow that presents itself like a visa officer
interview. Under the hood it's a fixed state machine walking through a
question list - the 8 real parameters PLUS 3 decoy questions that are
logged but never passed to the model. This makes naive "change one field"
testing harder, since participants have to run full conversations and
figure out which of the ~11 questions actually matter.

No AI/LLM is used anywhere - purely deterministic scripted dialogue.
"""

import streamlit as st
import numpy as np
import joblib
import time
import random

st.set_page_config(page_title="Visa Officer", page_icon="🛂")

@st.cache_resource
def load_model():
    return joblib.load("visa_model.joblib")

bundle = load_model()
model = bundle["model"]
encoder = bundle["encoder"]

# ---------------------------------------------------------------------
# Question script
# Each question has: key, officer's line, input widget type + options,
# and whether it's a REAL field (feeds the model) or a DECOY (ignored).
# ---------------------------------------------------------------------

QUESTIONS = [
    {
        "key": "greeting_purpose",
        "text": "Good morning. I'll be conducting your visa interview today. "
                "Let's start simple - what is your name?",
        "widget": "text",
        "real": False,
        "field": "name",
    },
    {
        "key": "age",
        "text": "Thank you. Could you tell me your age?",
        "widget": "number",
        "min": 18, "max": 80, "default": 30,
        "real": True,
        "field": "age",
    },
    {
        "key": "purpose",
        "text": "what is the purpose of your visit to the United States?",
        "widget": "select",
        "options": ["travel", "business", "education", "medical"],
        "real": True,
        "field": "purpose_of_visit",
    },
    {
        "key": "relatives_us",
        "text": "Do you have any friends or relatives currently residing in the United States?",
        "widget": "select",
        "options": ["no", "yes"],
        "real": False,
        "field": None,
    },
    {
        "key": "income",
        "text": "Let's talk finances. What is your annual income, in INR?",
        "widget": "number",
        "min": 80000, "max": 15000000, "default": 600000, "step": 10000,
        "real": True,
        "field": "annual_income_inr",
    },
    {
        "key": "balance",
        "text": "And what is your current bank balance, in INR?",
        "widget": "number",
        "min": 20000, "max": 20000000, "default": 300000, "step": 10000,
        "real": True,
        "field": "bank_balance_inr",
    },
    {
        "key": "pets",
        "text": "Do you have any pets traveling with you?",
        "widget": "select",
        "options": ["no", "yes"],
        "real": False,
        "field": None,
    },
    {
        "key": "education",
        "text": "What is your highest level of education?",
        "widget": "select",
        "options": ["high_school", "bachelors", "masters", "phd"],
        "real": True,
        "field": "education_level",
    },
    {
        "key": "english",
        "text": "How would you rate your own English communication ability?",
        "widget": "select",
        "options": ["basic", "good", "fluent"],
        "real": True,
        "field": "english_level",
    },
    {
        "key": "prior_approvals",
        "text": "How many times have you previously been approved for a US visa?",
        "widget": "number",
        "min": 0, "max": 10, "default": 0,
        "real": True,
        "field": "prior_approvals",
    },
    {
        "key": "credit_score",
        "text": "Could you share your current credit score (CIBIL score)?",
        "widget": "number",
        "min": 300, "max": 900, "default": 700,
        "real": False,
        "field": None,
    },
    {
        "key": "legal",
        "text": "Finally - do you have any pending legal charges against you, anywhere?",
        "widget": "select",
        "options": ["no", "yes"],
        "real": True,
        "field": "legal_charges",
    },
]

TOTAL_Q = len(QUESTIONS)

# ---------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------

def get_question_text(q):
    text = q["text"]
    if q["widget"] == "select":
        text += f" (Please choose from: {', '.join(q['options'])})"
    elif q["widget"] == "number":
        text += f" (Please enter a number between {q.get('min', 0)} and {q.get('max', 100000000)})"
    return text

if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.chat_log = []
    st.session_state.decided = False

if "shuffled_questions" not in st.session_state:
    first_two = QUESTIONS[:2]
    rest = QUESTIONS[2:]
    random.shuffle(rest)
    st.session_state.shuffled_questions = first_two + rest
    st.session_state.chat_log.append(("assistant", get_question_text(st.session_state.shuffled_questions[0])))

st.title("🛂 Visa Officer Interview")
st.caption("Answer the officer's questions. A decision will be given at the end of the interview.")

# Render past chat log
for role, msg in st.session_state.chat_log:
    with st.chat_message(role):
        st.write(msg)

# ---------------------------------------------------------------------
# Ask current question / show result
# ---------------------------------------------------------------------

if st.session_state.step < TOTAL_Q and not st.session_state.decided:
    q = st.session_state.shuffled_questions[st.session_state.step]

    if user_input := st.chat_input("Type your answer here..."):
        st.session_state.chat_log.append(("user", user_input))
        
        valid = False
        parsed_answer = None
        
        if q["widget"] == "text":
            if user_input.strip():
                valid = True
                parsed_answer = user_input.strip()
        elif q["widget"] == "select":
            for opt in q["options"]:
                if opt.lower() in user_input.lower():
                    valid = True
                    parsed_answer = opt
                    break
        else: # number
            import re
            nums = re.findall(r'\d+', user_input.replace(',', ''))
            if nums:
                parsed_answer = int(nums[0])
                if q.get("min", 0) <= parsed_answer <= q.get("max", 100000000):
                    valid = True
        
        if valid:
            if q["real"]:
                st.session_state.answers[q["field"]] = parsed_answer
            st.session_state.step += 1
            if st.session_state.step < TOTAL_Q:
                next_q = st.session_state.shuffled_questions[st.session_state.step]
                st.session_state.chat_log.append(("assistant", get_question_text(next_q)))
            st.rerun()
        else:
            if q["widget"] == "select":
                err = f"Please choose from: {', '.join(q['options'])}."
            elif q["widget"] == "number":
                err = f"Please enter a valid number between {q.get('min', 0)} and {q.get('max', 100000000)}."
            else:
                err = "Please enter a valid response."
            st.session_state.chat_log.append(("assistant", err))
            st.rerun()

elif not st.session_state.decided:
    with st.chat_message("assistant"):
        st.write("Thank you. That concludes the interview. Let me review your application...")
    time.sleep(0.6)
    st.session_state.decided = True
    st.rerun()

else:
    a = st.session_state.answers
    row_cat = encoder.transform(
        [[a["education_level"], a["purpose_of_visit"], a["english_level"], a["legal_charges"]]]
    )
    row = np.hstack([[[a["age"], a["annual_income_inr"], a["bank_balance_inr"], a["prior_approvals"]]], row_cat])
    proba = model.predict_proba(row)[0, 1]
    approved = proba >= 0.65 

    with st.chat_message("assistant"):
        if approved:
            st.success(f"✅ Your visa application has been **APPROVED**.")
        else:
            st.error(f"❌ Your visa application has been **REJECTED**.")

    if st.button("Start a new interview"):
        st.session_state.step = 0
        st.session_state.answers = {}
        st.session_state.chat_log = []
        st.session_state.decided = False
        
        first_two = QUESTIONS[:2]
        rest = QUESTIONS[2:]
        random.shuffle(rest)
        st.session_state.shuffled_questions = first_two + rest
        st.session_state.chat_log.append(("assistant", get_question_text(st.session_state.shuffled_questions[0])))
        st.rerun()
