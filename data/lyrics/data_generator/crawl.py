"""
This script crawls lyrics from Genius API for specified artists and saves them in a structured format.
Author: Junhao Yu
Date: 2023-04-27 21:36

Running command: python parent_dir/crawl.py --user_token='' ...
"""
import lyricsgenius
import os
import json
import shutil
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
import argparse  # Added for command-line argument parsing


def get_tgt_artists_name(file_path):
    """Read target artist names from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def update_meta_dict(meta_json_dict, idx, artist_name, song_name, language, lyrics):
    """Update the metadata dictionary with song details."""
    meta_json_dict[idx] = {
        "artist_name": artist_name,
        "song_name": song_name,
        "language": language,
        "lyrics": lyrics
    }


def get_language(json_path):
    """Extract the language from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data["language"]


def get_song_list(data_dir, tgt_artist):
    """Retrieve a list of songs already present in the dataset."""
    song_list = []
    for artist in os.listdir(data_dir):
        if artist not in tgt_artist:
            artist_dir = os.path.join(data_dir, artist)
            if os.path.exists(artist_dir):
                meta_json_path = os.path.join(artist_dir, "meta.json")
                with open(meta_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for song_data in data.values():
                    if isinstance(song_data, dict):
                        song_list.append(song_data['song_name'])
    return song_list


def setup_artist_directory(data_dir, artist_name):
    """Prepare the directory structure for an artist."""
    artist_dir = os.path.join(data_dir, artist_name, "meta")
    if os.path.exists(artist_dir):
        shutil.rmtree(artist_dir)
        logging.info(f"{artist_dir} exists, removed.")
    os.makedirs(artist_dir, exist_ok=True)
    logging.info(f"{artist_dir} is created.")
    return artist_dir


def process_artist(genius, artist_name, data_dir, song_list, max_songs=50):
    """Process a single artist to fetch and save song data."""
    artist_dir = setup_artist_directory(data_dir, artist_name)
    meta_json_path = os.path.join(data_dir, artist_name, "meta.json")
    meta_json_dict = {}
    origin_cwd = os.getcwd()
    os.chdir(origin_cwd)

    artist = genius.search_artist(artist_name, max_songs=max_songs, include_features=False, sort="popularity")
    songs = artist.songs
    pb_songs = tqdm(songs, desc=f"Processing songs for {artist_name}")

    counts = 0
    for idx, song in enumerate(pb_songs):
        if counts >= 25:
            break
        song_name = song.title

        if song_name not in song_list:
            counts += 1
            song_list.append(song_name)
            pb_songs.set_postfix_str(song.title)

            # Use full path for saving the JSON file
            os.chdir(artist_dir) # NOTE: For windows path system compatibility
            save_json_path = os.path.join(f"{idx}.json")
            # Save lyrics directly to the specified path
            song.save_lyrics(save_json_path)

            # Extract language and update metadata
            song_language = get_language(save_json_path)

            os.chdir(origin_cwd)
            update_meta_dict(meta_json_dict, idx, artist_name, song_name, song_language, song.lyrics)

    meta_json_dict['counts'] = counts
    with open(meta_json_path, "w", encoding="utf-8") as json_file:
        json.dump(meta_json_dict, json_file, indent=4)


def process_artist_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing."""
    return process_artist(*args)


def main():
    """Main function to orchestrate the lyrics crawling."""
    parser = argparse.ArgumentParser(description="Lyrics Crawler")
    parser.add_argument("--user_token", type=str, required=True, help="Genius API user token")
    parser.add_argument("--data_dir", type=str, default="json", help="Directory to save song data")
    parser.add_argument("--max_songs", type=int, default=40, help="Maximum number of songs to fetch per artist")
    parser.add_argument("--artist_file", type=str, default="artist.txt", help="File containing artist names (one per line)")
    parser.add_argument("--artists", type=str, nargs="*", help="List of artist names provided via command line")
    args = parser.parse_args()

    # Create a Genius client
    genius = lyricsgenius.Genius(args.user_token, timeout=20, retries=3)

    # Determine the source of artist names
    if args.artists:
        artists_name = args.artists  # Use command-line input if provided
        if not isinstance(artists_name, list):
            artists_name = [artists_name]
        logging.info(f"Using artist names from command-line input: {artists_name}")
    else:
        artists_name = get_tgt_artists_name(args.artist_file)  # Fallback to file input
        logging.info(f"Using artist names from file: {args.artist_file}")

    # Get the list of existing songs
    song_list = get_song_list(args.data_dir, artists_name)
    logging.info(f"Found {len(song_list)} existing songs.")

    # Prepare arguments for multiprocessing
    args_list = [(genius, artist_name, args.data_dir, song_list, args.max_songs) for artist_name in artists_name]

    # Use multiprocessing to process artists in parallel
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(process_artist_wrapper, args_list), total=len(artists_name), desc="Processing artists"))


if __name__ == "__main__":
    main()