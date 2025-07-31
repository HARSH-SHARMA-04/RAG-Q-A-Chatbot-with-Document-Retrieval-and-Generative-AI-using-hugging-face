# RAG-Q-A-Chatbot-with-Document-Retrieval-and-Generative-AI-using-hugging-face
This repo implements a RAG Q&amp;A chatbot using document retrieval and generative AI. It retrieves relevant documents from a loan dataset and generates responses using Falcon-RW-1B hosted on Hugging Face. It uses Sentence-Transformers and FAISS for efficient document retrieval and context-aware answer generation.

# RAG Q&A Chatbot using Document Retrieval and Generative AI

This project implements a **Retrieval-Augmented Generation (RAG)** Q&A chatbot using document retrieval and generative AI for intelligent response generation. The core idea is to first retrieve relevant documents based on the query and then generate a response using a **lightweight language model** from **Hugging Face**.

### Project Overview:
The chatbot is powered by two main components:
1. **Semantic Embeddings for Document Retrieval** (using **sentence-transformers** from Hugging Face).
2. **Generative AI Model for Answer Generation** (using **Falcon-RW-1B** model from Hugging Face).

The Q&A system makes use of a **Loan Approval Dataset** from Kaggle to answer questions related to loan approval features. 

### Resources:
- **Dataset**: [Loan Approval Dataset](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction?select=Training+Dataset.csv)
  
### Key Technologies Used:
1. **Hugging Face**: 
   - **Transformers Library**: To use and fine-tune large language models like Falcon-RW-1B for text generation.
   - **Sentence-Transformers**: For converting text documents and queries into vector embeddings for semantic search.
   - **FAISS**: A library used for efficient similarity search on high-dimensional vectors.

2. **Python Libraries**:
   - **pandas**: For data manipulation and cleaning.
   - **faiss-cpu**: For vector-based document retrieval.
   - **transformers**: To load and use generative language models from Hugging Face.

### Setup & Installation:

#### Step 1: Install Required Libraries
Make sure to install the required libraries using `pip`:

```bash
!pip install faiss-cpu sentence-transformers transformers pandas -q
Step 2: Mount Google Drive and Load the Dataset
You can store the dataset on Google Drive and load it into your code using the following commands:

python
Copy
Edit
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
csv_path = '/content/drive/MyDrive/ML LAB/Training Dataset.csv'
df = pd.read_csv(csv_path)
Step 3: Convert Dataset Rows into Documents
Each row of the dataset is converted into a readable document to be embedded and retrieved based on semantic similarity:

python
Copy
Edit
documents = []
for _, row in df.iterrows():
    doc = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    documents.append(doc)
Step 4: Load SentenceTransformer Model for Embeddings
Using the MiniLM model to convert documents and queries into vector embeddings:

python
Copy
Edit
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
Step 5: Embed Documents and Create FAISS Index
The documents are embedded using the MiniLM model and stored in a FAISS index for fast retrieval:

python
Copy
Edit
doc_embeddings = embedder.encode(documents, show_progress_bar=True)
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))
Step 6: Retrieve Relevant Documents for Query
Given a query, the system will retrieve the most relevant documents using semantic similarity:

python
Copy
Edit
def retrieve_relevant_docs(query, k=3):
    query_vec = embedder.encode([query])
    ...
Step 7: Load and Use Falcon-RW-1B Model for Answer Generation
The Falcon-RW-1B model is used to generate responses based on the retrieved documents:

python
Copy
Edit
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
Step 8: Generate Answer Using the Model
The query and context are tokenized, and the response is generated using the model:

python
Copy
Edit
inputs = tokenizer(prompt, return_tensors="pt", ...)
outputs = model.generate(...)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
Step 9: RAG Pipeline (Retrieval + Generation)
The core function of the chatbot:

python
Copy
Edit
def rag_chatbot(question):
    retrieved_docs = retrieve_relevant_docs(question)
    context = "\n\n".join(retrieved_docs)
    return generate_answer_with_falcon(context, question)
Step 10: Ask and Answer Questions
Now you can ask questions related to loan approval and the chatbot will provide answers based on the retrieved documents:

python
Copy
Edit
query = "What features affect loan approval?"
answer = rag_chatbot(query)
Hugging Face’s Role in the Code:
Sentence-Transformers: Converts documents and queries into semantic vector embeddings for efficient retrieval.

Transformers Library: Loads the Falcon-RW-1B model to generate intelligent responses based on the query and context.

Results:
This chatbot retrieves the most relevant documents from the dataset based on the user query and generates a response by using a generative AI model.

Use Cases:
Q&A Chatbots: Automating customer service or information retrieval.

Document Search Systems: Use document retrieval systems for intelligent searches.

Loan Approval Prediction: Predict loan approval features based on data.

Conclusion:
This project demonstrates the power of combining document retrieval (RAG) and generative AI models for intelligent Q&A systems. Hugging Face plays a pivotal role in providing state-of-the-art models and tools for seamless integration.

Acknowledgments:
Hugging Face for their transformers library and models.

Kaggle for providing the Loan Approval Dataset.

License:
This project is licensed under the MIT License
