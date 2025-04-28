"""
This script processes and cleans lyrics data from JSON files for various artists.
Author: Junhao Yu
Date: 2025-04-27

Command: python parent_dir/preprocess.py --data_dir="" --save_dir=""
"""
# NOTE: The processed lyrics may not be 100% clean, mainly due to the repetitive lyrics problem. If you have time, you can dive into the saved json and try to fix the process script.
import os
import json
import re
from typing import Dict, Any, List
import argparse  # Add argparse for command-line arguments

def clean_lyrics_content(lyrics: str, song_name: str) -> str:
    """
    Cleans the lyrics content by performing the following steps:
    1. Removes content before "Lyrics".
    2. Removes names after tags like [Intro], [Verse].
    3. Extracts content after the second occurrence of the same tag (if multiple tags exist).
    4. Preserves line breaks before tags and removes empty lines.
    5. Filters lyrics based on length and unique word count.
    """
    # 1. Remove content before "Lyrics"
    if "Lyrics" not in lyrics:
        print(f"Lyrics not found in song: {song_name}")
        return lyrics
    lyrics_content = lyrics.split("Lyrics", 1)[1].strip()

    # 2. Remove names after tags like [Intro], [Verse]
    lyrics_content = re.sub(r'\[(.*?)(:.*?)\]', r'[\1]', lyrics_content)


    # 3. Extracts content between the first and second occurrence of the same tag.
    lyrics_content = extract_first_tag_content(lyrics_content, song_name)


    # 4. Preserve line breaks before tags and remove empty lines
    lyrics_content = preserve_line_breaks(lyrics_content)

    # 5. Filter lyrics based on length and unique word count
    if not filter_lyrics(lyrics_content, song_name):
        print(f"Lyrics for song {song_name} did not pass the filtering criteria.")
        return ""
    
    return lyrics_content


def extract_first_tag_content(lyrics_content: str, song_name: str) -> str:
    """
    Removes duplicate sections of lyrics and keeps only the first occurrence.
    Handles both overall repetition and duplicate sections.
    """
    # Remove irrelevant content (e.g., advertisements)
    lyrics_content = re.sub(r'See .*?Get tickets as low as \$\d+', '', lyrics_content)

    # Detect and remove overall repetition
    mid_point = len(lyrics_content) // 2
    first_half = lyrics_content[:mid_point].strip()
    second_half = lyrics_content[mid_point:].strip()
    if first_half == second_half:
        print(f"Overall repetition detected for song {song_name}. Keeping only the first half.")
        lyrics_content = first_half

    # Split lyrics by tags (e.g., [Chorus], [Verse 1])
    sections = re.split(r'(\[[^\[\]]+\])', lyrics_content)
    if len(sections) < 3:
        print(f"No tags or insufficient content found for song {song_name}. Keeping all content.")
        return lyrics_content.strip()

    # Reconstruct sections into pairs of [tag, content]
    structured_sections = []
    for i in range(1, len(sections), 2):
        tag = sections[i].strip()
        content = sections[i + 1].strip() if i + 1 < len(sections) else ""
        structured_sections.append((tag, content))

    # Detect and remove duplicate sections
    seen_sections = set()
    unique_sections = []
    for tag, content in structured_sections:
        section_key = f"{tag}:{content}"
        if section_key not in seen_sections:
            unique_sections.append(f"{tag}\n{content}")
            seen_sections.add(section_key)
        else:
            print(f"Duplicate section detected: {tag} for song {song_name}. Removing duplicate.")

    # Reconstruct the cleaned lyrics
    cleaned_lyrics = "\n\n".join(unique_sections)
    return cleaned_lyrics.strip()


def preserve_line_breaks(lyrics_content: str) -> str:
    """
    Preserves line breaks before tags and removes empty lines.
    """
    # Simplified logic to handle line breaks and empty lines
    return '\n'.join(
        f"\n{line.strip()}" if line.strip().startswith('[') and not line.strip().endswith('[Intro]') else line.strip()
        for line in lyrics_content.splitlines() if line.strip()
    ).strip()


def filter_lyrics(lyrics_content: str, song_name: str) -> bool:
    """
    Filters lyrics based on two criteria:
    1. Minimum length of lyrics (e.g., 50 characters).
    2. Minimum number of unique words (e.g., 10 unique words).

    Args:
        lyrics_content: The cleaned lyrics content.
        song_name: The name of the song (for logging purposes).

    Returns:
        True if the lyrics pass the filtering criteria, False otherwise.
    """
    min_length = 1024
    min_unique_words = 100
    
    # Check length
    if len(lyrics_content) < min_length:
        print(f"Lyrics for song {song_name} are too short (length: {len(lyrics_content)}).")
        return False

    # Check unique word count
    words = re.findall(r'\b\w+\b', lyrics_content.lower())
    unique_words = set(words)
    if len(unique_words) < min_unique_words:
        print(f"Lyrics for song {song_name} have too few unique words (count: {len(unique_words)}).")
        return False

    return True


def process_lyrics(lyrics: str, song_name: str) -> str:
    """
    Processes the lyrics string by calling the cleaning function.
    """
    return clean_lyrics_content(lyrics, song_name)


def process_artist_data(data_dir: str, artist: str, debug: bool = False) -> None:
    """
    Processes the JSON data for a specific artist and saves the cleaned data.

    Args:
        data_dir: Path to the data directory.
        artist: Name of the artist.
        debug: Whether to enable debug mode (print additional information).
    """
    artist_dir = os.path.join(data_dir, artist)
    meta_path = os.path.join(artist_dir, "meta.json")
    save_meta_path = os.path.join(artist_dir, "meta_1.json")

    # Read the JSON file
    with open(meta_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # Process lyrics for each song
    filtered_data = {}
    for song_id, song_data in data.items():
        if not isinstance(song_data, dict) or 'lyrics' not in song_data:
            continue
        
        song_name = song_data.get('song_name', song_id)
        lyrics = song_data['lyrics']
        processed_lyrics = process_lyrics(lyrics, song_name)
        
        # Only include songs that pass the filtering criteria
        if processed_lyrics:
            song_data['lyrics'] = processed_lyrics
            filtered_data[song_id] = song_data

    # Save the filtered and processed data
    with open(save_meta_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)


def extract_lyrics_into_dict_from_one_artist(data_dir: str, artist: str) -> List[Dict[str, str]]:
    """
    Extracts lyrics from the JSON files of a specific artist and saves them into a dictionary.

    Args:
        data_dir: Path to the data directory.
        artist: Name of the artist.
    """
    artist_dir = os.path.join(data_dir, artist)
    meta_path = os.path.join(artist_dir, "meta_1.json")
    extracted_data = []

    # Read the JSON file
    with open(meta_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # Process lyrics for each song
    for song_id, song_data in data.items():
        if not isinstance(song_data, dict) or 'lyrics' not in song_data:
            continue
        
        processed_lyrics = song_data['lyrics']
        extracted_data.append({"text": processed_lyrics})

    return extracted_data

def main():
    """
    Main function to:
    1. Iterate through the data directory and process artist data.
    2. Extract lyrics into a dictionary and save them into a JSON file.
    """
    parser = argparse.ArgumentParser(description="Preprocess lyrics data")
    parser.add_argument("--data_dir", type=str, default="json", help="Directory containing artist data")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save the processed data.json file")
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)  # Ensure save_dir exists

    extracted_datas = []
    for artist in sorted(os.listdir(data_dir)):
        process_artist_data(data_dir, artist)
        extracted_data = extract_lyrics_into_dict_from_one_artist(data_dir, artist)
        if isinstance(extracted_data, list):
            extracted_datas.extend(extracted_data)

    save_path = os.path.join(save_dir, "data.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(extracted_datas, f, indent=4, ensure_ascii=False)
    print(f"Processed data saved to {save_path}")


if __name__ == "__main__":
    main()