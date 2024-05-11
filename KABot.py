import streamlit as st

from llm_utils import get_kb_vector_db
from llm_utils import select_llm
from llm_utils import get_response_openai
import calendar;
import time;
import os

st.set_page_config(page_title="Industry DOT AI - Knowledge Assistant for Windchill",layout="wide",page_icon=":brain:")

#side bar
st.sidebar.image("data/wipro-logo-small.png")
st.sidebar.title("Industry DOT AI")

hide_streamlit_style = """
           <style>
           #MainMenu {visibility: hidden;}
           footer {visibility: hidden;}
           .stDeployButton {visibility: hidden;}
           </style>
           """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("💬 AI Knowledge Assistant")
st.caption("🚀 Powered by Wipro Industry DOT AI with Windchill Knowledge Base")


st.divider()

from os import walk

# f = []
# layer = 1
# w = walk("/plmbot-vol")
# for (dirpath, dirnames, filenames) in w:
#     if layer <= 4:
#         for fn in filenames:        
#             f.append(dirpath+"/"+fn)
#         for fold in dirnames:        
#             f.append(dirpath+"/"+fold)        
#         layer += 1
#     else:
#         break
#print(f)

select_llm("OpenAI")
t1 = calendar.timegm(time.gmtime())
db = get_kb_vector_db()
t2 = calendar.timegm(time.gmtime())
print("Time taken to vecstore "+str(t2-t1))

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello Peter, How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    print("Initiated query to openai for prompt "+ prompt)
    t3 = calendar.timegm(time.gmtime())
    ans = get_response_openai(db,prompt)
    t4 = calendar.timegm(time.gmtime())
    print("Time taken to get response from open-ai "+str(t4-t3))
    #response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    #msg = response.choices[0].message.content
    msg = ans["output_text"]
    print("AI response  "+ msg)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)