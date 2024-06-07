import os 
from constants import openai_key
import streamlit as st 
from st_audiorec import st_audiorec
from transformers import AutoModel, AutoTokenizer
import torch

os.environ["OPENAI_API_KEY"]=openai_key

from openai import OpenAI
client = OpenAI(api_key=openai_key)

total_score=[]

def get_response(prompt):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
    {
      "role": "user",
      "content": prompt
    }
      ]
    ,
    max_tokens=150,
)
    return response.choices[0].message.content

def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    return transcript

def calculate_score(user_answer, correct_answer):
    with st.spinner("Evaluating your answer.."):    
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_length = 151  
        user_encoded = tokenizer(user_answer, padding="max_length", truncation=True, return_tensors="pt")
        correct_encoded = tokenizer(correct_answer, padding="max_length", truncation=True, return_tensors="pt")

        user_embedding = model(**user_encoded)[0][0]
        correct_embedding = model(**correct_encoded)[0][0]

        similarity = (user_embedding * correct_embedding).sum() / (
            torch.linalg.norm(user_embedding) * torch.linalg.norm(correct_embedding)
        )
        score = similarity.item() * 100
        return score
##initialize our streamlit app

st.set_page_config(page_title="Generative AI Quiz")
st.title("Generative AI Quiz")
st.header("Let's test your understanding of Generative AI")
st.divider()

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'score' not in st.session_state:
    st.session_state['score'] = [] 

ask_question=st.button("Ask the question")
st.write("Provide your answer by either recording or typing in the text box and click 'submit'")

footer_container = st.container()
with footer_container:
    audio_bytes = st_audiorec()

input=st.text_input("Your Answer: ",key="input")
submit=st.button('Submit')

if ask_question:
    response=get_response(f'''
                          Ask me a question about generative AI topics such as 
                          algorithms, applications and technological impacts
                          ensuring that the question is logically sequenced based on previous questions asked by ai in
                          {st.session_state['chat_history']} if there are any .

''')
    # Add user query and response to session state chat history
    st.subheader("The Question is")
    with st.chat_message("ai"):
        st.write(response)
        st.session_state['chat_history'].append(("ai", response))
    

if input and submit: 
    user_answer= input.strip()
    response=get_response( f''' 
            Check if my answer is correct 
            Question: refer to the last question asked by ai in {st.session_state['chat_history']}
            Answer: {input}
            
            ''')
    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("user", input))
    correct_answer=get_response(f'''
                                 answer to the last question asked by ai in {st.session_state['chat_history']}
                                                                             ''')
    st.session_state['score'].append(("score", calculate_score(user_answer, correct_answer)))
    st.subheader("Review of your answer:")
    with st.chat_message("ai"):
        st.write(response)
        st.session_state['chat_history'].append(("ai", response))
    correct_answer=get_response(f'''
                                 answer to the last question asked by ai in {st.session_state['chat_history']}
                                                                             ''')
                
if audio_bytes and submit:
     with st.spinner("Please wait a moment..."):
        # Write the audio bytes to a temporary file
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)

        # Convert the audio to text using the speech_to_text function
        transcript = speech_to_text(webm_file_path)
        if transcript:
            user_answer = transcript.strip()
            response=get_response( f''' 
            Check if my answer is correct 
            Question: refer to the last question asked by ai in {st.session_state['chat_history']}
            Answer: {transcript}
            
            ''')
            # Add user query and response to session state chat history
            
            st.session_state['chat_history'].append(("user", transcript))
            st.subheader("Your answer: ")
            with st.chat_message("user"):
                st.write(transcript)
            correct_answer=get_response(f'''
                                 answer to the last question asked by ai in {st.session_state['chat_history']}
                                                                             ''')
            st.session_state['score'].append(("score", calculate_score(user_answer, correct_answer)))
            st.subheader("Review of your answer:")
            with st.chat_message("ai"):
                st.write(response)
                st.session_state['chat_history'].append(("ai", response))
total_score=[]            
if len(st.session_state['score'])>0 :
    st.subheader("Check your score: ") 
    printscore=st.button("Show score")
    if printscore:
        with st.spinner("Calculating Score..."):
            for score,question_score in st.session_state['score']:
                    total_score.append(question_score)
                    final_score = sum(total_score)/len(total_score)
                    st.write(f"Your Score: {final_score:.2f}/100")
st.divider()
st.subheader("Chat History :")   
for role, text in st.session_state['chat_history']:
    with st.chat_message(role):
        st.write(f"{text}")
        

