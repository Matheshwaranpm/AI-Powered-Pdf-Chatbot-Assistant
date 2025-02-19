# *AI-Powered PDF Chatbot Assistant*  

An AI-driven chatbot that allows users to upload PDF documents and ask questions about their content. The system processes the PDF, extracts text, retrieves relevant information using embeddings, and generates human-like responses using Microsoft's *Phi-2 model*.  

---

## *📌 Table of Contents*  
- [Introduction](#introduction)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Technologies Used](#technologies-used)  
- [API Endpoints](#api-endpoints)  
- [Troubleshooting](#troubleshooting)  
- [Future Enhancements](#future-enhancements)  
- [Contributing](#contributing)  
- [License](#license)  

---

## *📖 Introduction*  
This chatbot is designed to *analyze PDFs* and provide intelligent responses to user queries. It utilizes *NLP, machine learning, and AI* to enhance user experience by retrieving and summarizing relevant document content.  

---

## *✨ Features*  
✅ *PDF Text Extraction* – Extracts text from uploaded PDF documents.  
✅ *AI-Powered Responses* – Uses Microsoft's *Phi-2 model* to generate meaningful answers.  
✅ *Contextual Search* – Finds the most relevant sections in the PDF using *text embeddings and cosine similarity*.  
✅ *Web API* – Built using *Flask* for easy integration with front-end applications.  

---

## *⚙ Installation*  

Follow these steps to install and run the project on your local machine.  

### *1️⃣ Clone the Repository*  
bash
git clone https://github.com/your-username/pdf-chatbot-assistant.git
cd pdf-chatbot-assistant


### *2️⃣ Create a Virtual Environment (Optional but Recommended)*  
bash
python -m venv venv  
source venv/bin/activate  # On macOS/Linux  
venv\Scripts\activate  # On Windows  


### *3️⃣ Install Required Dependencies*  
bash
pip install -r requirements.txt


### *4️⃣ Download and Set Up the AI Model*  
python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

This ensures the required *Phi-2 model* is downloaded and ready for use.  

### *5️⃣ Run the Application*  
bash
python server.py

The API will start running at *http://127.0.0.1:8000/*  

---

## *🚀 Usage*  

### *Uploading a PDF*  
1. Access the web interface or use an API tool (e.g., *Postman*).  
2. Send a *POST request* to /process-document with the PDF file.  
3. The document will be processed, and text will be extracted for further queries.  

### *Asking a Question*  
- Send a *POST request* to /process-message with your query.  
- The chatbot retrieves the most relevant PDF content and generates an AI-powered response.  

---

## *📂 Project Structure*  

pdf-chatbot-assistant/
│── worker.py            # Handles PDF processing, embeddings, and AI responses  
│── server.py            # Flask API for user interaction  
│── requirements.txt     # List of dependencies  
│── uploads/             # Directory for storing uploaded PDFs  
│── README.md            # Project documentation  


---

## *🛠 Technologies Used*  
- *Python 3.x* – Core programming language  
- *Transformers* – AI model processing (Phi-2)  
- *Sentence Transformers* – Embedding generation for similarity matching  
- *Torch* – Optimized deep learning execution  
- *Flask* – Web framework for API development  
- *pdfplumber* – PDF text extraction  

---

## *📡 API Endpoints*  

| Endpoint               | Method | Description |  
|------------------------|--------|-------------|  
| /process-document    | POST   | Uploads and processes a PDF |  
| /process-message     | POST   | Accepts a user query and returns an AI-generated response |  

---

## *🛑 Troubleshooting*  
- *Model Loading Issues*: Ensure you have downloaded the microsoft/phi-2 model.  
- *PDF Processing Errors*: Make sure the PDF has selectable text (scanned images may not work).  
- *Slow Response Time: Use **GPU acceleration* for faster AI processing (torch_dtype=torch.float16, device_map="auto").  

---

## *🚀 Future Enhancements*  
- *Summarization Feature* – Generate concise document summaries.  
- *Multilingual Support* – Expand AI responses to multiple languages.  
- *Web-Based UI* – Develop an interactive frontend for better user experience.  

---

## *🤝 Contributing*  
Want to contribute? Fork the repo, make changes, and submit a pull request!  

---

## *📜 License*  
This project is open-source and available under the *MIT License*.  

---

