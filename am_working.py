# %%
from transformers import pipeline
pipe = pipeline("text-classification", model="roberta-base-openai-detector")
print(pipe("Hello world! Is this content AI-generated?"))  # [{'label': 'Real', 'score': 0.8036582469940186}]
# %%
#load in training_set_rel3.tsv
import pandas as pd
import numpy as np

df = pd.read_csv('training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')
df.head()
# %%
#remove all but the essay_set 1,2,7,8
df = df[df['essay_set'].isin([1,2,7,8])]
#return total number of rows
df.shape

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
import openai
import os
import dotenv
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

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

## set up openai



# %%

output = {}

for prompt in native_prompts.values():
  entry = {}
  entry['prompt'] = prompt
  entry['completion'] = completion_3(prompt)
  entry['native'] = True
  output.append(entry)

  entry = {}
  entry['prompt'] = prompt
  entry['completion'] = completion_4(prompt)
  entry['native'] = True
  output.append(entry)

  entry = {}
  entry['prompt'] = prompt
  entry['completion'] = completion_3_5(prompt)
  entry['native'] = True
  output.append(entry)

for prompt in non_native_prompts.values():
  entry = {}
  entry['prompt'] = prompt
  entry['completion'] = completion_3(prompt)
  entry['native'] = False
  output.append(entry)

  entry = {}
  entry['prompt'] = prompt
  entry['completion'] = completion_4(prompt)
  entry['native'] = False
  output.append(entry)

  entry = {}
  entry['prompt'] = prompt
  entry['completion'] = completion_3_5(prompt)
  entry['native'] = False
  output.append(entry)

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


