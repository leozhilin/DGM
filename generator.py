import csv
import json
import traceback

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("../ChatGLM3-main/models", trust_remote_code=True)
model = AutoModel.from_pretrained("../ChatGLM3-main/models", trust_remote_code=True).half().cuda()
model = model.eval()

# 打开CSV文件
with open('IMDB_Dataset.csv', encoding='utf-8', newline='', errors='ignore') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        i = 0
        index = 0
        json_data = []
        history = []
        # 逐行读取数据
        for row in csv_reader:
                # 在此处处理每行的数据
                text = row['review']
                label = row['sentiment']
                query = f"You are now a text sentiment data generator, here is a reference example:" \
                        f"{{\"text\":\"{text}\",\"label\":\"{label}\"}}" \
                        f"The requirements for generating data are as follows:" \
                        f"1.The text should be an English review about a movie or TV show, and the label should match the sentiment tendency of the text (pos or neg)." \
                        f"2.You only need to generate one piece of data, and the content of the 'text' field should comply with the JSON format. Remember to escape quotes if you want to use them." \
                        f"3.The data format should completely match the example provided." \
                        f"4.Your response should be in English." \
                        f"5.Try to maintain a difference of over 30% compared to the data generated previously"
                response, history = model.chat(tokenizer, query, temperature=1.0, history = history)
                #控制历史记录的长度，以免爆显存
                if len(history) > 6:
                        history.pop(0)
                        history.pop(1)



                index += 1
                # print(f"text{index}:", text)
                print(f"response{index}:", response)
                try:
                        tmp = []
                        tmp.append(json.loads(response))
                        json_data.extend(tmp)
                        i += 1
                        print(f"已生成{i}条数据")
                        # print("json:", json_data[-1])
                        with open('GLM_Dataset.csv', 'a', newline='', encoding="utf-8") as f:
                                csv_writer = csv.writer(f)
                                # 写入数据
                                for row in json_data:
                                        try:
                                                csv_writer.writerow([row['text'], row['label']])
                                        except:
                                                pass
                                json_data.clear()

                except Exception:
                        print(traceback.print_exc())

