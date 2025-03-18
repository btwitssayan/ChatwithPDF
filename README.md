# PDF Chat Assistant

**PDF Chat Assistant** is an interactive web application that transforms your PDF documents into a dynamic conversational experience. By leveraging cutting-edge language models and advanced document retrieval techniques, users can upload PDFs and engage in natural language queries to extract insightful information directly from the document content.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Features
- **Interactive Chat Interface:** Engage in a seamless conversation with your PDF content.
- **Document Upload and Preview:** Easily upload PDFs and view previews of document pages.
- **Context-Aware Retrieval:** Uses a history-aware retriever chain to formulate contextually relevant queries.
- **Advanced Language Modeling:** Powered by state-of-the-art models (e.g., Qwen-2.5-32b) and HuggingFace embeddings.
- **Modular and Extensible:** Built with a modular architecture using LangChain components for efficient document processing and retrieval.
- **Customizable Configuration:** Simple API key and session management to tailor the chat experience.

## Technologies Used
- **Streamlit:** For rapid web application development and user interface rendering.
- **LangChain:** Orchestrates document processing and conversational chains, including:
  - `combine_documents`
  - `retrieval`
  - `history_aware_retriever`
- **LangChain Community Components:** Enhances chat functionalities with custom message histories.
- **FAISS:** Utilized for vector storage and efficient similarity searches.
- **HuggingFace Embeddings:** Implements the `all-MiniLM-L6-v2` model for high-quality text embeddings.
- **PyPDFLoader:** For efficient PDF document loading and processing.
- **dotenv:** Manages environment variables securely.
- **Custom Utilities:** Includes PDF-to-image conversion for document preview.

## Installation
Follow these steps to set up the project locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/pdf-chat-assistant.git
   cd pdf-chat-assistant
   ```
2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables:**
   - Create a `.env` file in the root directory.
   - Add your HuggingFace token and any other necessary variables:
     ```bash
     HF_TOKEN=your_huggingface_token_here
     ```
##Configuration
  - API Key: The application requires a GROQ AI API key for model integration. Input your API key in the sidebar once the app is running.
  - Session Management: Utilize the session ID feature to handle multiple conversations independently.

##Usage
  1. Run the Application:
     ```bash
     streamlit run app.py
     ```
  2. Enter API Key: Provide your GROQ AI API key in the sidebar.
  3. Upload PDFs: Select and upload one or more PDF documents.
  4. Start Chatting: Ask questions about your PDFs using the chat interface. The application will retrieve context from the documents and respond with accurate, concise answers.
  5. Clear Chat History: Use the clear button to reset the conversation if needed.

##Project Structure
  ```bash
    pdf-chat-assistant/
  ├── app.py                  # Main Streamlit application script
  ├── utils.py                # Utility functions (e.g., PDF to image conversion)
  ├── requirements.txt        # List of project dependencies
  ├── .env                    # Environment variables (not tracked by version control)
  └── README.md               # Project documentation
  ```
##License
This project is licensed under the MIT License. See the LICENSE file for more details.
  




