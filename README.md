
# Chat with your documents: ultimate RAG system

This repository demonstrates how to build a Retrieved Augmented Generation (RAG) system in Python. The RAG system allows you to ask queries related to your documents, which can be in various formats such as PDF, DOC, PPTX. The system provides relevant answers based on the context of your documents. Additionally, the application presents retrieved pieces of the document used as context for the AI-generated response. This information includes the source name, page number, and title of the relevant section. If your question is not directly related to the loaded document, the AI will notify you that it doesn't have enough context to provide an answer.



The repo deployed in Streamlit and contains three versions of the app:
* Version 1: Supports only the PDF file format. It uses PyPDF2 to extract raw text from PDF files. However, it does not provide additional information about the source name or page number of the retrieved documents.

* Version 2: Supports only the PDF file format. It utilizes LLM Sherpa for parsing PDFs, allowing extraction of sections, headings, and paragraphs. It also incorporates content-aware chunking and provides additional information about the retrieved documents, including the source name, page number, and section title.

* Version 3: Supports multiple file formats, including PDF, DOC, PPT, and HTML. It preprocesses PDFs using Unstructured.io, enabling extraction of tables, images, and headers. The system intelligently chunks the documents by title and provides additional information about the retrieved documents, such as the source name and page number.
# How RAG Works
![](https://miro.medium.com/v2/resize:fit:1200/1*kSkeaXRvRzbJ9SrFZaMoOg.png)
RAG consists of two main components:

* Retrieval: This component retrieves relevant documents from a large corpus based on the input query.
* Generation: This component generates text using the retrieved documents and the input query as context.
The retrieval component uses a dense retriever to find the most relevant documents for the query. The generation component then uses a transformer-based language model to generate text based on the retrieved documents and the query. 

# Advanced Retrieval with query expansion and cross-encoder
 ![](https://miro.medium.com/v2/resize:fit:1400/1*i0BXxkKW1IKVghHtcmKqQQ.png)
App provides two-step retriever
* first step: multi query retrieval --> system expands user's query by looking at different perspectives and paraphrasing the query; then retrieves possible relevant documents
* second step: reranking the documents using cross-encoder model
## Acknowledgements

Usefull resources
* [Advanced Retrieval for AI with Chroma - Deeplearning.AI course][1]
* [Preprocessing Unstructured Data for LLM Applications - Deeplearning.AI course][2]

Inspired by following projects
* [MultiPDF Chat App by Alejandro AO][3]

[1]: https://learn.deeplearning.ai/courses/advanced-retrieval-for-ai/lesson/1/introduction 
[2]: https://learn.deeplearning.ai/courses/preprocessing-unstructured-data-for-llm-applications/lesson/1/introduction 
[3]: https://github.com/alejandro-ao/ask-multiple-pdfs/tree/main 

## Demo
![IMAGE ALT TEXT HERE](https://github.com/NurNazaR/Chat_with_documents/blob/main/img/image%20copy%202.png?raw=true)
![IMAGE ALT TEXT HERE](https://github.com/NurNazaR/Chat_with_documents/blob/main/img/image%20copy.png?raw=true)
![IMAGE ALT TEXT HERE](https://github.com/NurNazaR/Chat_with_documents/blob/main/img/image.png?raw=true)


## Run Locally

Clone the project

```bash
  git clone https://github.com/NurNazaR/Chat_with_documents.git 
```

Go to the project directory

```bash
  cd Chat_with_documents
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the application 

```bash
  streamlit run app_v1_simple_chunking_method.py 
```

```bash
  streamlit run app_v2_llmsherpa_method.py
```

```bash
  streamlit run app_v3_any_file_format.py
```


## License

[MIT](https://choosealicense.com/licenses/mit/)

