```python
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extract text from a local PDF file."""
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Path to the local PDF file
pdf_path = 'acura_mdx_manual.pdf'

# Extract text from the PDF
manual_text = extract_text_from_pdf(pdf_path)
print("PDF text extracted.")
```

    PDF text extracted.
    


```python
import openai

def generate_embeddings(text):
    """Generate embeddings using OpenAI's latest API."""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",  # The embedding model
        input=text
    )
    
    # Access the embeddings correctly using response.data
    embeddings = response.data[0].embedding
    return embeddings
```


```python
def search_relevant_content(query, manual_text):
    """Generate query embeddings and perform similarity search (mocking RAG)."""
    query_embedding = generate_embeddings(query)
    
    # Here you would perform the similarity search using FAISS or a similar tool
    # For now, we'll return a mocked section of the manual text
    return manual_text[:1000]  # Mock returning the first part of the text

# Example query:
query = "How do I reset the oil change light on a 2022 Acura MDX?"
relevant_content = search_relevant_content(query, manual_text)
print("Relevant content found:", relevant_content)
```

    Relevant content found: 2022 MDX Owner’s ManualEvent Data Recorders
    This vehicle is equipped with an event data recorder (EDR).  
    The main purpose of an EDR is to record, in certain crash or near 
    crash-like situations, such as an air bag deployment or hitting a road obstacle, data that will assist in understanding how a vehicle’s 
    systems performed. The EDR is designed to record data related 
    to vehicle dynamics and safety systems for a short period of 
    time, typically 30 seconds or less. The EDR in this vehicle is 
    designed to record such data as:
    •How various systems in your  vehicle were operating;
    •Whether or not the driver and passenger safety belts were 
    buckled/fastened;
    •How far (if at all) the driver was depressing the accelerator 
    and/or brake pedal; and,
    •How fast the vehicle was traveling.22 ACURA MDX-31TYA6011.book  0 ページ  ２０２１年１２月１４日　火曜日　午前１０時４２分These data can help provide a better understanding of the 
    circumstances in which crashes and injuries occur. NOTE: EDR data are recorded by your vehic
    


```python
import os
from openai import OpenAI

# Instantiate the client using environment variable for API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Check if the API key is loaded properly
if not client.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
```


```python
def query_openai_with_rag(query, relevant_content):
    """Query OpenAI API using RAG and gpt-4o-mini in chat format."""
    prompt = f"Based on the owner's manual, answer the following question:\n\nManual Section: {relevant_content}\n\nQuestion: {query}\n\nAnswer:"
    
    # Call the new chat completion API method
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Chat-based model
        messages=[{"role": "user", "content": prompt}],  # Pass the prompt in chat message format
        max_tokens=100,
        temperature=0.7
    )
    
    # Convert response to dictionary format if needed
    return response.model_dump()['choices'][0]['message']['content'].strip()

# Example query
query = "How do I reset the oil change light on a 2022 Acura MDX?"
relevant_content = "This is a section of the manual..."  # Mock content
response = query_openai_with_rag(query, relevant_content)
print(f"Query: {query}")
print(f"Response: {response}\n")

# Example query
query = "When does the Auto high beam mode turn ON?"
relevant_content = "This is a section of the manual..."  # Mock content
response = query_openai_with_rag(query, relevant_content)
print(f"Query: {query}")
print(f"Response: {response}")
```

    Query: How do I reset the oil change light on a 2022 Acura MDX?
    Response: To reset the oil change light on a 2022 Acura MDX, follow these steps:
    
    1. Turn on the ignition without starting the engine. This can be done by pressing the start button twice without pressing the brake pedal.
    2. Use the steering wheel controls to navigate to the "Settings" menu on the display.
    3. Select "Maintenance."
    4. Choose "Oil Life" from the options.
    5. Select "Reset" and confirm the action when prompted.
    
    This should reset the oil
    
    Query: When does the Auto high beam mode turn ON?
    Response: The Auto high beam mode typically turns ON when the vehicle is in low-light conditions, such as at night or in dark environments, and when no other vehicles are detected in front of you. It automatically switches from high beams to low beams when it detects the headlights of oncoming vehicles or taillights of vehicles in front of you. Always refer to the specific owner's manual for your vehicle for precise details and conditions.
    
