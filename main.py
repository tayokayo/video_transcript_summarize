import os
from youtube_transcript_api import YouTubeTranscriptApi
import sys
from llm_processor import process_transcript

def get_video_id(url):
    if "youtube.com" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be" in url:
        video_id = url.split("/")[-1]
    else:
        raise ValueError("Invalid YouTube URL")
    return video_id

def get_transcript(video_url):
    try:
        video_id = get_video_id(video_url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([entry['text'] for entry in transcript])
        return full_transcript
    except Exception as e:
        print(f"An error occurred while fetching the transcript: {e}")
        return None

def main():
    print("Starting main function")
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    else:
        video_url = input("Please enter a YouTube video URL: ")
    
    print(f"Video URL: {video_url}")
    video_name = input("Please enter a name for this video: ")
    print(f"Video name: {video_name}")
    
    print("Fetching transcript...")
    transcript = get_transcript(video_url)
    
    if transcript:
        print(f"Transcript fetched. Length: {len(transcript)} characters")
        
        # Create 'transcripts' folder if it doesn't exist
        os.makedirs('transcripts', exist_ok=True)
        
        # Save transcript with video name
        file_path = os.path.join('transcripts', f'{video_name}.txt')
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"\nTranscript has been saved to '{file_path}'")

        # Process the transcript
        print("\nProcessing transcript...")
        success = process_transcript(file_path, video_name)
        if success:
            print("Transcript processing completed successfully.")
        else:
            print("Transcript processing failed. Please check the error messages above.")
    else:
        print("Failed to retrieve the transcript.")
    
    print("Main function completed")

if __name__ == "__main__":
    main()