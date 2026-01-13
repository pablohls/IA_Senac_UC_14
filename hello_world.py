from transformers import pipeline
classifier = pipeline('sentiment-analysis')
text = 'Hugging Face is a fantastic library for AI!'
result = classifier(text)
print(f'Text: {text}')
print(f'Result: {result}')
