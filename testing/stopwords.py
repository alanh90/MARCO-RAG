from nltk.corpus import stopwords
text = "This is a sample sentence with some stop words."
stop_words = set(stopwords.words('english'))
words = text.split()
filtered_words = [word for word in words if word.lower() not in stop_words]
print(filtered_words)
# Output: ['This', 'sample', 'sentence', 'stop', 'words.']