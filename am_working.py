# %%
from transformers import pipeline
pipe = pipeline("text-classification", model="roberta-base-openai-detector")
print(pipe("Hello world! Is this content AI-generated?"))  # [{'label': 'Real', 'score': 0.8036582469940186}]
# %%
#load in training_set_rel3.tsv
import pandas as pd
import numpy as np
import openai
import os
import dotenv
import requests
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv('training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')
df.head()
# %%
#remove all but the essay_set 1,2,7,8
df = df[df['essay_set'].isin([1,2,7,8])]
#return total number of rows
df.shape

# %%
# reindex the dataframe
df = df.reset_index(drop=True)

# %%
# print the first 5 rows of the dataframe
for i in range(5):
    print(df['essay'][i])
    print('---------------------')

# %%
native_prompts = {
    1:'More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.',
    2: 'Write an essay in which you explain how the author builds an argument to persuade his/her audience that authors claim. In your essay, analyze how the author uses one or more of the features listed above (or features of your own choice) to strengthen the logic and persuasiveness of his/her argument. Be sure that your analysis focuses on the most relevant features of the passage.',
    7: 'Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining. Do only one of the following: write a story about a time when you were patient OR write a story about a time when someone you know was patient OR write a story in your own way about patience.',
    8: 'We all understand the benefits of laughter. For example, someone once said, “Laughter is the shortest distance between two people.” Many other people believe that laughter is an important part of any relationship. Tell a true story in which laughter was one element or part.',
}

non_native_prompts = {
    1: 'Do you agree or disagree with the following statement? It is better to have broad knowledge of many academic subjects than to specialize in one specific subject. Use specific reasons and examples to support your answer.',
    2: 'Do you agree or disagree with the following statement? Young people enjoy life more than older people do. Use specific reasons and examples to support your answer.',
    3: 'Do you agree or disagree with the following statement? Young people nowadays do not give enough time to helping their communities. Use specific reasons and examples to support your answer.',
    4: 'Do you agree or disagree with the following statement? Most advertisements make products seem much better than they really are. Use specific reasons and examples to support your answer.',
    5: 'Do you agree or disagree with the following statement? In twenty years, there will be fewer cars in use than there are today. Use reasons and examples to support your answer.',
    6: 'Do you agree or disagree with the following statement? The best way to travel is in a group led by a tour guide. Use reasons and examples to support your answer.',
    7: 'Do you agree or disagree with the following statement? It is more important for students to understand ideas and concepts than it is for them to learn facts. Use reasons and examples to support your answer.',
    8: 'Do you agree or disagree with the following statement? Successful people try new things and take risks rather than only doing what they already know how to do well. Use reasons and examples to support your answer.'
}

# %%

class GPTZeroAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.gptzero.me/v2/predict'
    def text_predict(self, document):
        url = f'{self.base_url}/text'
        headers = {
            'accept': 'application/json',
            'X-Api-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        data = {
            'document': document
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()
    def file_predict(self, file_path):
        url = f'{self.base_url}/files'
        headers = {
            'accept': 'application/json',
            'X-Api-Key': self.api_key
        }
        files = {
            'files': (os.path.basename(file_path), open(file_path, 'rb'))
        }
        response = requests.post(url, headers=headers, files=files)
        return response.json()
# use openai with 3.5 

# %%
def completion_3_5(prompt): 
  comp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Write a 250 word essay responding to this prompt: " + prompt
         }
    ],
    temperature=0.7
  )
  return comp.choices[0].message.content

def completion_4(prompt):
  comp = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": "Write a 250 word essay responding to this prompt: " + prompt
          }
    ],
    temperature=0.7
  )
  return comp.choices[0].message.content

def completion_3(prompt):
  comp = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Write a 250 word essay responding to this prompt: " + prompt,
    temperature=0.7,
    max_tokens=250
  )
  return comp.choices[0].text

# %%
# rewrite the above two for loops more efficiently
output = []
for prompt in native_prompts.values():
  for model in ['completion_3', 'completion_4', 'completion_3_5']:
    entry = {}
    entry['prompt'] = prompt
    entry['completion'] = eval(model)(prompt)
    entry['native'] = True
    output.append(entry)

for prompt in non_native_prompts.values():
  for model in ['completion_3', 'completion_4', 'completion_3_5']:
    entry = {}
    entry['prompt'] = prompt
    entry['completion'] = eval(model)(prompt)
    entry['native'] = False
    output.append(entry)

# %%

# save output to json
import json
with open('gpt_generated.json', 'w+') as outfile:
    json.dump(output, outfile, indent=4)
    
# %%
len(output)



# %%
gptzero = GPTZeroAPI('9b8298a6976e4a6e992c33904b472b60')
# %%
response = gptzero.text_predict('Write a 250 word essay responding to this prompt: ' + native_prompts[1])

print(response['documents'][0]['average_generated_prob'])

# %%
test_set = []
for i in range(len(output)):
  entry = {}
  entry['native'] = output[i]['native']
  entry['generated'] = True
  entry['prompt'] = output[i]['prompt']
  average_prob = gptzero.text_predict(output[i]['completion'])['documents'][0]['average_generated_prob']
  entry['average_probability'] = average_prob
  test_set.append(entry)


# %%
#print out the average probability for each prompt
for i in range(len(test_set)):
  print(test_set[i]['average_probability'])

  

# %%

# for each .txt file in the directory, load it in and store it in a list

non_native_list = []

for filename in os.listdir('drive-download-20230601T034313Z-001'):
  with open('drive-download-20230601T034313Z-001/' + filename, 'r') as f:
    non_native_list.append(f.read())
    

# %%
len(non_native_list)
# %%

for essay in non_native_list:
  entry = {}
  entry['native'] = False
  entry['generated'] = False
  average_prob = gptzero.text_predict(essay)['documents'][0]['average_generated_prob']
  entry['average_probability'] = average_prob
  test_set.append(entry)
# %%

for i in range(25):
  entry = {}
  entry['native'] = False
  entry['generated'] = False
  average_prob = gptzero.text_predict(df['essay'][i])['documents'][0]['average_generated_prob']
  entry['average_probability'] = average_prob
  test_set.append(entry)
# %%


# %%
results = []
# %%
#count the number of correct predictions

threshold = 0.99999999
for i in range(len(test_set)):
  if test_set[i]['average_probability'] > threshold:
    test_set[i]['predicted_generated'] = True
  else:
    test_set[i]['predicted_generated'] = False

correct = 0
false_positives = 0
false_negatives = 0
for i in range(len(test_set)):
  if test_set[i]['predicted_generated'] == test_set[i]['generated']:
    correct += 1
  elif test_set[i]['predicted_generated'] == True and test_set[i]['generated'] == False:
    false_positives += 1
  elif test_set[i]['predicted_generated'] == False and test_set[i]['generated'] == True:
    false_negatives += 1

result_entry = {}
result_entry['threshold'] = threshold
result_entry['correct'] = correct
result_entry['false_positives'] = false_positives
result_entry['false_negatives'] = false_negatives
if result_entry not in results:
  results.append(result_entry)

print(len(test_set))
print(correct)
print(false_positives)
print(false_negatives)

# %%

# save results to json
import json
with open('test_gpt_results.json', 'w+') as outfile:
    json.dump(results, outfile, indent=4)
# %%
