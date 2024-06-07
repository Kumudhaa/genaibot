# Gen-AI Q&A App
Interactive Generative AI Assessment

System Overview :

This system simulates an interactive assessment of a user's knowledge about Generative AI. 
The user can respond both in audio format and text format, the AI model will analyze and evaluate the user's responses.

Interactive AI Model :

The system utilizes a large language model (LLM) specifically OpenAI GPT-3.5.
The LLM is prompted to ask relevant and informative questions related to Generative AI topics.
The LLM is further prompted to compare the user's answer and the correct answer to return a feedback to the user.

Maintaining Conversational Flow :

The model keeps track of the conversation history and uses that information to ensure the questions it asks are logically connected to the previous interactions.

Speech Recognition :

The system transcribes audio responses into text using the OpenAI Whisper model.
Whisper enables accurate transcription and performs well on various audio qualities, including background noise.

Response Analysis and Evaluation System :

The system utilizes sentence transformers for the response analysis functionality. 
Sentence transformers are a type of neural network model trained to embed sentences into numerical representations that capture their semantic meaning.
The scores are calculated by capturing the semantic similarity between the user's answer the correct answer.

User interface :

Streamlit is used as the framework for building the UI and deploying the application.
The UI prioritizes clarity and ease of use. It avoids overwhelming the user with unnecessary elements.
The chat history and feedback section promote a conversational feel, making the assessment feel more interactive.

 


