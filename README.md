# YouTube Transcript Analyzer

This project automatically fetches YouTube video transcripts and generates summaries, key points, and creative content using LLMs (Large Language Models).

## Features

- Fetches transcripts from YouTube videos
- Processes transcripts using GPT-4
- Generates:
  - Comprehensive summaries
  - Key points with emoji highlights
  - Themed haikus
- Handles both short and long transcripts efficiently
- Memory usage monitoring and optimization

## Prerequisites

- Python 3.8+
- OpenAI API key
- YouTube Data API access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/youtube-transcript-analyzer.git
cd youtube-transcript-analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the script with a YouTube URL:
```bash
python main.py https://www.youtube.com/watch?v=video_id
```

Or run it without arguments to enter the URL interactively:
```bash
python main.py
```

## Output

The program creates two main directories:
- `transcripts/`: Contains raw transcripts
- `social_con/processed_transcripts/`: Contains generated content including:
  - Summaries
  - Key points
  - Haikus

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.