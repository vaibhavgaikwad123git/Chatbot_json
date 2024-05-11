import fitz
import re
from langchain.docstore.document import Document
import os

class ContentLoader:
    def strip_consecutive_newlines(self,text: str) -> str:
        return re.sub(r"\s*\n\s*", "\n", text)    
    
    def get_docs(self):
        print("Warning: not implemented")
        return None
        
class PDFLoader(ContentLoader):
   
    def __init__(self, file_name):
        self.pdf_file_name = file_name

    def get_docs(self):
        base_file_name = os.path.basename(self.pdf_file_name)
        file = open(self.pdf_file_name,"rb")
        pdf = fitz.open(stream=file.read(),filetype="pdf")
        pdf_docs = []
        for i, page in enumerate(pdf):
            text = page.get_text(sort=True)
            doc = Document(page_content=self.strip_consecutive_newlines(text.strip()))
            doc.metadata["page"] = i + 1
            doc.metadata["source"] = f"{base_file_name} p-{i+1}"
            pdf_docs.append(doc)
        return pdf_docs

def save_text_file(content:str,fn:str):
    with open(fn, 'w', encoding="utf-8") as outfile:
        outfile.write(content)

def combine_docs_and_save_text_file(docs,fn):
    all_text_list = []
    for doc in docs:
        all_text_list.append(doc.page_content)
    all_text = "\n".join(all_text_list)
    save_text_file(all_text,fn)

def get_docs_from_folder(folder_name:str):
    all_docs = []
    for file in os.listdir(folder_name):
     # check the files which are end with specific extension
        if file.endswith(".pdf"):
            # print path name of selected files
            loader = PDFLoader(os.path.join(folder_name, file))
            all_docs.extend(loader.get_docs())
    return all_docs