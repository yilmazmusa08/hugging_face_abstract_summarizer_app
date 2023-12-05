# https://huggingface.co/spaces/yilmazmusa-ml/abstract_summarizer

# Here are the imports
import warnings
import pdfplumber
import torch
from transformers import pipeline, AutoProcessor, AutoModel
import numpy as np
import gradio as gr
from io import BytesIO
from scipy.io.wavfile import write as write_wav
warnings.filterwarnings("ignore")


# Here is the code
def extract_abstract(uploaded_file):
    pdf_bytes = BytesIO(uploaded_file)
    with pdfplumber.open(pdf_bytes) as pdf:
        abstract = ""
        # Iterate through each page
        for page in pdf.pages:
            text = page.extract_text(x_tolerance = 1, y_tolerance = 1) # these parameters are set 1 in order to get spaces between words and lines
            # Search for the "Abstract" keyword
            if "abstract" in text.lower():
                # Found the "Abstract" keyword
                start_index = text.lower().find("abstract") # find the "abstract" title as starter index
                end_index = text.lower().find("introduction") # find the "introduction" title as end index
                abstract = text[start_index:end_index]
                break
    print(abstract)
    return abstract

def process_summary(summary):
    # Split the summary by the first period
    summary = summary[0]["summary_text"]
    sentences = summary.split('.', 1)
    if len(sentences) > 0:
        # Retrieve the first part before the period
        processed_summary = sentences[0].strip() + "."
        # Replace "-" with an empty string
        processed_summary = processed_summary.replace("-", "")
        return processed_summary

# Function for summarization and audio conversion
def summarize_and_convert_to_audio(pdf_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Move models and related tensors to CUDA device if available
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small").to(device)
    
    # Extract abstract
    abstract_text = extract_abstract(pdf_file)

    if not abstract_text:
        return "No 'abstract' section found in the uploaded PDF. Please upload a different PDF."
    
    # Summarize the abstract
    summarization_pipeline = pipeline(task='summarization', model='knkarthick/MEETING_SUMMARY', min_length=15, max_length=30)
    summarized_text = summarization_pipeline(abstract_text)
    one_sentence_summary = process_summary(summarized_text)

    print(one_sentence_summary)
    
    # Text-to-audio conversion
    inputs = processor(
        text=[one_sentence_summary],
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    speech_values = model.generate(**inputs, do_sample=True)
    sampling_rate = model.generation_config.sample_rate
    
    # Convert speech values to audio data
    audio_data = speech_values.cpu().numpy().squeeze()

    # Convert audio data to bytes
    with BytesIO() as buffer:
        write_wav(buffer, sampling_rate, audio_data.astype(np.float32))
        audio_bytes = buffer.getvalue()
    
    return audio_bytes, one_sentence_summary


# Create a Gradio interface
iface = gr.Interface(
    fn=summarize_and_convert_to_audio,
    inputs=gr.UploadButton(label="Upload PDF", type="binary", file_types=["pdf"]),  # Set to accept only PDF files
    outputs=[gr.Audio(label="Audio"), gr.Textbox(label="Message")],
    title="PDF Abstract Summarizer",
    description="""
    This application is supposed to summarize the 'abstract' section of a PDF file and convert the summarization into a speech.
    Please make sure you upload a PDF file with the 'abstract' section for application to work.
    Note: If you get an error while processing the file please refresh your browser and try again.
    """
)

iface.launch()