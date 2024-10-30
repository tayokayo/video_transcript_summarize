import os
import signal
import psutil
import gc
import logging
from typing import List, Generator
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API key is set
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")

# Timeout handling
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")

# Set the timeout (e.g., 300 seconds = 5 minutes)
TIMEOUT = 300

signal.signal(signal.SIGALRM, timeout_handler)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Current memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def clear_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")
    except ImportError:
        pass  # If torch is not installed, do nothing

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total: 
        print()

def initialize_llm():
    """Initialize and return an OpenAI LLM instance."""
    return OpenAI(model="gpt-4", temperature=0.75, max_tokens=1500)

def initialize_embedding_model():
    """Initialize and return an OpenAIEmbedding instance."""
    return OpenAIEmbedding()

def stream_transcript(file_path: str, chunk_size: int = 1000) -> Generator[str, None, None]:
    """Stream the transcript file in chunks."""
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

def create_index_from_documents(documents: List[Document]) -> VectorStoreIndex:
    """Create an index from a list of documents."""
    Settings.llm = initialize_llm()
    Settings.embed_model = initialize_embedding_model()
    return VectorStoreIndex.from_documents(documents)

def generate_content(index: VectorStoreIndex, query: str) -> str:
    """Generate content using the provided index and query."""
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response.response

def save_output(content: str, filename: str):
    """Save the generated content to a file."""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Content saved to {filename}")
    except Exception as e:
        print(f"Error saving content to {filename}: {e}")

def is_long_transcript(transcript_path: str, threshold_words: int = 1000) -> bool:
    """Determine if a transcript is considered 'long' based on word count."""
    try:
        with open(transcript_path, 'r', encoding='utf-8') as file:
            word_count = len(file.read().split())
        return word_count > threshold_words
    except Exception as e:
        print(f"Error reading transcript: {e}")
        return False

def process_transcript(transcript_path: str, video_name: str):
    """Process a transcript file to generate summary, key points, and haikus."""
    print(f"Starting to process transcript: {transcript_path}")
    log_memory_usage()
    
    try:
        signal.alarm(TIMEOUT)
        
        output_dir = 'social_con/processed_transcripts'
        os.makedirs(output_dir, exist_ok=True)

        if is_long_transcript(transcript_path):
            print("Transcript is long. Processing in chunks...")
            process_long_transcript(transcript_path, video_name)
        else:
            print("Transcript is not long. Processing as a whole...")
            documents = SimpleDirectoryReader(input_files=[transcript_path]).load_data()
            index = create_index_from_documents(documents)
            
            # Updated summary prompt
            summary_prompt = "Generate a 300-word summary of the transcript in a conversational, introspective tone. Frame the content as a calm reflection amidst chaos, using rhetorical questions and personal observations. Begin with 'You know, I've been thinking about...' and weave the key ideas together as a thoughtful exploration, not as instructions."
            summary = generate_content(index, summary_prompt)
            save_output(summary, os.path.join(output_dir, f"summary_{video_name}.txt"))

            # Updated key points prompt
            key_points_prompt = "Distill the transcript into 3 key points. Each point should be a single sentence that captures a core idea in a reflective, grounded manner. Frame these as insights rather than instructions, and consider using relevant emojis to highlight each point. Aim for a tone that's both introspective and reassuring."
            key_points = generate_content(index, key_points_prompt)
            save_output(key_points, os.path.join(output_dir, f"keypoints_{video_name}.txt"))

            # Haiku prompt
            haikus = generate_content(index, "Based on the themes or content of the transcript, generate 3 haikus.")
            save_output(haikus, os.path.join(output_dir, f"haikus_{video_name}.txt"))

        signal.alarm(0)  # Disable the alarm
        print("Transcript processing completed successfully.")
        log_memory_usage()
        return True  # Indicate successful completion
    except TimeoutException:
        print("Transcript processing timed out. Try processing a shorter transcript or increasing the timeout.")
    except Exception as e:
        print(f"An error occurred during transcript processing: {str(e)}")
        print("Transcript processing failed.")
    
    return False  # Indicate failed completion

def process_long_transcript(transcript_path: str, video_name: str, chunk_size: int = 1000):
    """Process a long transcript file by chunking it and then generating content."""
    Settings.llm = initialize_llm()
    Settings.embed_model = initialize_embedding_model()

    # Initialize the index
    index = VectorStoreIndex([])

    # Set up the ingestion pipeline
    node_parser = SimpleNodeParser.from_defaults()
    pipeline = IngestionPipeline(transformations=[node_parser])

    # Process the transcript in chunks
    for i, chunk in enumerate(stream_transcript(transcript_path, chunk_size)):
        print(f"Processing chunk {i+1}")
        print_progress_bar(i + 1, i + 2, prefix='Progress:', suffix='Complete', length=50)
        log_memory_usage()
        
        doc = Document(text=chunk, id_=f"chunk_{i}")
        nodes = pipeline.run(documents=[doc])
        index.insert_nodes(nodes)

        # Clear memory
        gc.collect()
        clear_gpu_memory()

    # Generate content using the complete index
    output_dir = 'social_con/processed_transcripts'
    os.makedirs(output_dir, exist_ok=True)

    # Updated summary prompt
    summary_prompt = "Generate a 300-word summary of the transcript in a conversational, introspective tone. Frame the content as a calm reflection amidst chaos, using rhetorical questions and personal observations. Begin with 'You know, I've been thinking about...' and weave the key ideas together as a thoughtful exploration, not as instructions."
    summary = generate_content(index, summary_prompt)
    save_output(summary, os.path.join(output_dir, f"summary_{video_name}.txt"))

    # Updated key points prompt
    key_points_prompt = "Distill the transcript into 3 key points. Each point should be a single sentence that captures a core idea in a reflective, grounded manner. Frame these as insights rather than instructions, and consider using relevant emojis to highlight each point. Aim for a tone that's both introspective and reassuring."
    key_points = generate_content(index, key_points_prompt)
    save_output(key_points, os.path.join(output_dir, f"keypoints_{video_name}.txt"))

    haikus = generate_content(index, "Based on the themes or content of the transcript, generate 3 haikus.")
    save_output(haikus, os.path.join(output_dir, f"haikus_{video_name}.txt"))

    print("\nLong transcript processing completed.")
    log_memory_usage()

logging.basicConfig(filename='api_usage.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def log_api_call(model: str, n_input_tokens: int, n_output_tokens: int):
    logging.info(f"API Call - Model: {model}, Input Tokens: {n_input_tokens}, Output Tokens: {n_output_tokens}")

if __name__ == "__main__":
    print("This module is not meant to be run directly.")