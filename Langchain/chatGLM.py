from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
import torch
import csv
import traceback
import json
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Optional

class ChatGLM3(LLM):
    max_token: int = 8192
    do_sample: bool = True
    temperature: float = 1.5
    top_p = 0.8
    tokenizer: object = None
    model: object = None
    history: List = []
    tool_names: List = []
    has_search: bool = False

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM3"

    def load_model(self, model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=model_config, trust_remote_code=True, device_map="auto").eval()
        self.model = self.model.to("cuda")

    def _call(self, prompt: str, history: List = [], stop: Optional[List[str]] = ["<|user|>"]):
        query = prompt

        response, self.history = self.model.chat(
            self.tokenizer,
            query,
            history=self.history,
            do_sample=self.do_sample,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        history.append((prompt, response))
        return response

llm = ChatGLM3()
# llm.load_model(model_name_or_path="/data/lzl/ChatGLM3-main/models")
llm.load_model(model_name_or_path="/root/autodl-tmp/ChatGLM3-main/models")
prompt1 = "You are now a text sentiment data generator. Please refer to the following example to generate an English" \
          " movie or TV show review text. Assign a label (pos or neg) that matches the sentiment of the text. " \
          "Please ensure that the content is original and not extracted from existing datasets:" \
          "{{\"text\":\"{text}\",\"label\":\"{label}\"}}"
generating_chain = LLMChain(
  llm = llm,
  prompt = PromptTemplate.from_template(prompt1),
  output_key = "generator"
)
prompt2 = "Adjust the format and content of the {generator} data according to the following requirements:" \
          "1.The content of the 'text' must adhere to the JSON format specification. Remember to escape quotes if used." \
          "2.Responses must only contain English."
modifying_chain = LLMChain(
  llm = llm,
  prompt = PromptTemplate.from_template(prompt2),
  output_key = "modified"
)
overall_chain = SequentialChain(
  chains = [generating_chain, modifying_chain],
  input_variables = ["text", "label"],
  output_variables = ["generator", "modified"],
  verbose = True
)

# 打开CSV文件
with open('IMDB_Dataset.csv', encoding='utf-8', newline='') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    i = 0
    index = 0
    json_data = []
    # 逐行读取数据
    for row in csv_reader:
        print(f"当前GPU内存使用情况：{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        # 在此处处理每行的数据
        t = row['review']
        l = row['sentiment']
        response = overall_chain({"text": t, "label": l})
        index += 1
        print(f"response{index}:", response["modified"])
        try:
            tmp = []
            tmp.append(json.loads(response["modified"]))
            json_data.extend(tmp)
            with open('GLM_Dataset.csv', 'a', newline='', encoding="utf-8") as f:
                csv_writer = csv.writer(f)
                # 写入数据
                for row in json_data:
                    try:
                        csv_writer.writerow([row['text'], row['label']])
                        i += 1
                        print(f"已写入{i}条数据")
                    except:
                        pass
                json_data.clear()
        except Exception:
                print(traceback.print_exc())