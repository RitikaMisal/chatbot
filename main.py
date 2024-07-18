
# ???>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>running



# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import PlainTextResponse
# import PyPDF2
# import io
# import uvicorn
# from pydantic import BaseModel
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import logging

# app = FastAPI()
# uploaded_pdfs = {}

# gpt2_model_name = "gpt2"
# gpt2_model = None
# gpt2_tokenizer = None

# try:
#     from transformers import GPT2Tokenizer, GPT2LMHeadModel

#     gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
#     gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

# except Exception as e:
#     logging.error(f"Error loading GPT-2 model: {str(e)}")

# chunk_size = 50 

# class QuestionRequest(BaseModel):
#     pdf_id: int
#     question: str

# @app.post("/upload")
# async def upload_pdf(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         reader = PyPDF2.PdfReader(io.BytesIO(contents))

#         text_chunks = []
#         for page in range(len(reader.pages)):
#             page_text = reader.pages[page].extract_text()
#             for i in range(0, len(page_text), chunk_size):
#                 text_chunks.append(page_text[i:i+chunk_size])

#         pdf_id = len(uploaded_pdfs) + 1
#         uploaded_pdfs[pdf_id] = text_chunks

#         return {"message": "PDF uploaded successfully", "pdf_id": pdf_id}

#     except PyPDF2.utils.PdfReadError as e:
#         raise HTTPException(status_code=400, detail="Invalid PDF file")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.post("/pdf")
# async def get_pdf_data_and_answer(question_request: QuestionRequest):
#     try:
#         pdf_id = question_request.pdf_id
#         question = question_request.question

#         if gpt2_model is None or gpt2_tokenizer is None:
#             raise HTTPException(status_code=500, detail="GPT-2 model not loaded")

#         if pdf_id not in uploaded_pdfs:
#             raise HTTPException(status_code=404, detail="PDF not found")

        
#         pdf_text_chunks = uploaded_pdfs[pdf_id]
#         full_text = "".join(pdf_text_chunks)
#         context = full_text[:2000]


#         max_input_length = len(gpt2_tokenizer.encode(question + context))
#         max_length = min(max_input_length + 100, 512)
#         inputs = gpt2_tokenizer(question, context, return_tensors="pt")
#         outputs = gpt2_model.generate(**inputs, max_length=max_length)

#         answer = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return PlainTextResponse(answer)

#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>running but not response properly>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Perfect Running>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>..


# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import PlainTextResponse
# import PyPDF2
# import io
# import uvicorn
# from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# import torch

# app = FastAPI()
# uploaded_pdfs = {}

# model_name = "deepset/roberta-base-squad2" 

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# chunk_size = 250  

# class QuestionRequest(BaseModel):
#     pdf_id: int
#     question: str

# @app.post("/upload")
# async def upload_pdf(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         reader = PyPDF2.PdfReader(io.BytesIO(contents))

#         text_chunks = []
#         for page in range(len(reader.pages)):
#             page_text = reader.pages[page].extract_text()
#             for i in range(0, len(page_text), chunk_size):
#                 text_chunks.append(page_text[i:i+chunk_size])

#         pdf_id = len(uploaded_pdfs) + 1
#         uploaded_pdfs[pdf_id] = {
#             'chunks': text_chunks,
#             'full_text': "".join(text_chunks)
#         }

#         return {"message": "PDF uploaded successfully", "pdf_id": pdf_id}

#     except PyPDF2.utils.PdfReadError as e:
#         raise HTTPException(status_code=400, detail="Invalid PDF file")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.post("/pdf")
# async def get_pdf_data_and_answer(question_request: QuestionRequest):
#     try:
#         pdf_id = question_request.pdf_id
#         question = question_request.question

#         if pdf_id not in uploaded_pdfs:
#             raise HTTPException(status_code=404, detail="PDF not found")

        
#         pdf_data = uploaded_pdfs[pdf_id]
#         full_text = pdf_data['full_text']
#         text_chunks = pdf_data['chunks']

    
#         context = full_text[:2000]

    
#         inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
#         input_ids = inputs["input_ids"].tolist()[0]


#         with torch.no_grad():
#             outputs = model(**inputs)
#             answer_start = torch.argmax(outputs.start_logits)
#             answer_end = torch.argmax(outputs.end_logits) + 1
#             answer = tokenizer.decode(input_ids[answer_start:answer_end])

#         return PlainTextResponse(answer)

#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Perfect Running>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
import PyPDF2
import io
import uvicorn
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

app = FastAPI()
uploaded_pdfs = {}

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

chunk_size = 250 

class QuestionRequest(BaseModel):
    pdf_id: int
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        reader = PyPDF2.PdfReader(io.BytesIO(contents))

        text_chunks = []
        for page in range(len(reader.pages)):
            page_text = reader.pages[page].extract_text()
            for i in range(0, len(page_text), chunk_size):
                text_chunks.append(page_text[i:i+chunk_size])

        pdf_id = len(uploaded_pdfs) + 1
        uploaded_pdfs[pdf_id] = {
            'chunks': text_chunks,
            'full_text': "".join(text_chunks)
        }

        return {"message": "PDF uploaded successfully", "pdf_id": pdf_id}

    except PyPDF2.utils.PdfReadError as e:
        raise HTTPException(status_code=400, detail="Invalid PDF file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/pdf")
async def get_pdf_data_and_answer(question_request: QuestionRequest):
    try:
        pdf_id = question_request.pdf_id
        question = question_request.question

        if pdf_id not in uploaded_pdfs:
            raise HTTPException(status_code=404, detail="PDF not found")

        pdf_data = uploaded_pdfs[pdf_id]
        full_text = pdf_data['full_text']
        text_chunks = pdf_data['chunks']

       
        context = full_text[:2000]

      
        inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

       
        with torch.no_grad():
            outputs = model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = tokenizer.decode(input_ids[answer_start:answer_end])

      
        answer_lines = answer.split('\n')[:3] 
        formatted_answer = '\n'.join(answer_lines)

        return PlainTextResponse(formatted_answer)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import PlainTextResponse
# from pydantic import BaseModel
# import PyPDF2
# import io
# import uvicorn
# import torch
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# app = FastAPI()
# uploaded_pdfs = {}

# model_name = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# chunk_size = 250

# class QuestionRequest(BaseModel):
#     pdf_id: int
#     question: str

# @app.post("/upload")
# async def upload_pdf(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         reader = PyPDF2.PdfReader(io.BytesIO(contents))

#         text_chunks = []
#         for page in range(len(reader.pages)):
#             page_text = reader.pages[page].extract_text()
#             for i in range(0, len(page_text), chunk_size):
#                 text_chunks.append(page_text[i:i+chunk_size])

#         pdf_id = len(uploaded_pdfs) + 1
#         uploaded_pdfs[pdf_id] = {
#             'chunks': text_chunks,
#             'full_text': "".join(text_chunks)
#         }

#         return {"message": "PDF uploaded successfully", "pdf_id": pdf_id}

#     except PyPDF2.utils.PdfReadError as e:
#         raise HTTPException(status_code=400, detail="Invalid PDF file")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.post("/pdf")
# async def get_pdf_data_and_answer(question_request: QuestionRequest):
#     try:
#         pdf_id = question_request.pdf_id
#         question = question_request.question

#         if pdf_id not in uploaded_pdfs:
#             raise HTTPException(status_code=404, detail="PDF not found")

#         pdf_data = uploaded_pdfs[pdf_id]
#         full_text = pdf_data['full_text']
#         text_chunks = pdf_data['chunks']

#         context = full_text[:2000]

#         inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
#         input_ids = inputs["input_ids"].tolist()[0]

#         with torch.no_grad():
#             outputs = model(**inputs)
#             answer_start = torch.argmax(outputs.start_logits)
#             answer_end = torch.argmax(outputs.end_logits) + 1
#             answer = tokenizer.decode(input_ids[answer_start:answer_end])

#         answer_lines = answer.split('\n')[:3]  
#         formatted_answer = '\n'.join(answer_lines)

#         return PlainTextResponse(formatted_answer)

#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import PlainTextResponse
# from pydantic import BaseModel
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering, GPT2LMHeadModel, GPT2Tokenizer
# import PyPDF2
# import io
# import torch
# from vectordb.client import VectorDatabase

# app = FastAPI()
# uploaded_pdfs = {}
# db = VectorDatabase("localhost", 19530)  # Adjust hostname and port as per your VectorDB setup

# # Load Question Answering Model
# qa_model_name = "deepset/roberta-base-squad2"
# qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
# qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# # Load LocalAI Language Model (LLM)
# llm_model_name = "gpt2"
# llm_tokenizer = GPT2Tokenizer.from_pretrained(llm_model_name)
# llm_model = GPT2LMHeadModel.from_pretrained(llm_model_name)

# chunk_size = 250

# class QuestionRequest(BaseModel):
#     pdf_id: int
#     question: str

# @app.post("/upload")
# async def upload_pdf(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         reader = PyPDF2.PdfReader(io.BytesIO(contents))

#         text_chunks = []
#         for page in range(len(reader.pages)):
#             page_text = reader.pages[page].extract_text()
#             for i in range(0, len(page_text), chunk_size):
#                 text_chunks.append(page_text[i:i+chunk_size])

#         pdf_id = len(uploaded_pdfs) + 1
#         uploaded_pdfs[pdf_id] = {
#             'chunks': text_chunks,
#             'full_text': "".join(text_chunks)
#         }

#         # Store full text in VectorDB
#         db.put_vector(str(pdf_id), "".join(text_chunks))

#         return {"message": "PDF uploaded successfully", "pdf_id": pdf_id}

#     except PyPDF2.utils.PdfReadError as e:
#         raise HTTPException(status_code=400, detail="Invalid PDF file")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.post("/pdf")
# async def get_pdf_data_and_answer(question_request: QuestionRequest):
#     try:
#         pdf_id = question_request.pdf_id
#         question = question_request.question

#         # Retrieve full text from VectorDB
#         if str(pdf_id) not in uploaded_pdfs:
#             raise HTTPException(status_code=404, detail="PDF not found")

#         pdf_data = uploaded_pdfs[str(pdf_id)]
#         full_text = pdf_data['full_text']
#         text_chunks = pdf_data['chunks']

#         context = full_text[:2000]

#         # Question Answering with QA Model
#         qa_inputs = qa_tokenizer.encode_plus(question, context, return_tensors="pt")
#         qa_input_ids = qa_inputs["input_ids"].tolist()[0]

#         with torch.no_grad():
#             qa_outputs = qa_model(**qa_inputs)
#             answer_start = torch.argmax(qa_outputs.start_logits)
#             answer_end = torch.argmax(qa_outputs.end_logits) + 1
#             answer = qa_tokenizer.decode(qa_input_ids[answer_start:answer_end])

#         answer_lines = answer.split('\n')[:3]  
#         formatted_answer = '\n'.join(answer_lines)

#         # Text Generation with LLM
#         llm_input_ids = llm_tokenizer.encode(question, return_tensors="pt")
#         llm_output = llm_model.generate(llm_input_ids, max_length=100, num_return_sequences=1)
#         generated_text = llm_tokenizer.decode(llm_output[0], skip_special_tokens=True)

#         return {"qa_answer": formatted_answer, "llm_generated_text": generated_text}

#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
