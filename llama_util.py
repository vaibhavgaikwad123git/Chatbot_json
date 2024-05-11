# Import boto3 library
import boto3
import json
import os

def get_template_for_spec():
		return {"prompt": "Below is an instruction that describes a task, paired with an input that provides further context. "
		"Write a response that appropriately completes the request.\n\n"
		"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n",
		"completion": " {response}",
        }
def get_template_for_chat():
		return {"prompt": "create a final answer to the given question using the provided document excerpts"
           "(given in no particular order) as sources. ALWAYS include a \"SOURCES\" section in your answer "
            " citing only the minimal set of sources needed to answer the question. "
            "If you are unable to answer the question, simply state that you do not have enough information"
            " to answer the question and leave the SOURCES section empty. Use only the provided documents and"
            " do not attempt to fabricate an answer.\n"
            "### Question:\n{question}\n\n"
		"### Sources:\n{sources}\n\n",
		"completion": " {response}"
        }


def get_prediction(inputs:str):
    # Create a SageMaker runtime client
    client = boto3.client('sagemaker-runtime',region_name="us-east-1")
    content_type = 'application/json'
    input_output_demarkation_key = "\n\n### Response:\n"
    payload = {
        "inputs":inputs 
            + input_output_demarkation_key,
        "parameters": {"max_new_tokens": 1000},
        }
    data_json = json.dumps(payload)
    # Encode the JSON string as bytes
    data_bytes = data_json.encode()
    print("Prompt : " +inputs)
    print("Resonse :\n")
    custom_attribute = 'Accept:application/json;ContentType:application/json;accept_eula=true'
    response = client.invoke_endpoint(
        EndpointName=os.getenv("SAGEMAKER_ENDPOINT"),
        ContentType=content_type,
        Body=data_bytes,
        CustomAttributes=custom_attribute
        )
    result = json.loads(response['Body'].read().decode())
    return result[0]['generation']
	


def predict_and_print_text(instruction:str,context:str):
    print( get_prediction(instruction,str))
	
	
def getsa(sysreq:str,streq:str,pd:str):    
    try: 
        input = get_template_for_spec()["prompt"].format(
            instruction="Generate system architecture specification for {sysreq}".format(sysreq=sysreq),
            context="The system requirement {sysreq} is to satisfy the stake holder requirement {streq} for" 
                           "a {pd}".format(pd=pd,streq=streq,sysreq=sysreq)
        )
        resp = get_prediction(input)
        resp=  "<br>".join(l for l in resp.splitlines() if l)
        return resp

    except Exception as e:
        print(e)

def chat(source_content,query):
    input = get_template_for_chat()["prompt"].format(
            question=query,
            sources=source_content
        )
    resp = get_prediction(input)
    return resp    