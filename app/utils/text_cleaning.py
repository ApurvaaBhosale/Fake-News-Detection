import string

def clean_text(text):
    # Ensure input is string
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Add more cleaning steps here if needed
    
    return text
