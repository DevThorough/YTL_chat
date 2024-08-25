import json
from dotenv import load_dotenv
import os
import subprocess
from googleapiclient.discovery import build
import re

load_dotenv()
api_key = os.getenv("API_KEY")
YOUTUBE_API_KEY = MY_YOUTUBE_API_KEY

def search_channel(channel_handle):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    # Search for the channel by @handle
    search_response = youtube.channels().list(
        forHandle=channel_handle,
        part='id,snippet,statistics',
        maxResults=1
    ).execute()
    
    if not search_response['items'][0]:
        print("No channel found with that handle.")
        return
    
    # search_response = youtube.search().list(
    #     q=channel_name,
    #     type='channel',
    #     part='snippet',
    #     maxResults=10  # Get multiple channels for user confirmation
    # ).execute()
    
    item = search_response['items'][0]

    channel = {
            'channelId': item['id'],
            'handle': channel_handle,
            'channelTitle': item['snippet']['title'],
            'description': item['snippet']['description'],
            'thumbnail': item['snippet']['thumbnails']['default']['url'],
            'subscriberCount': item['statistics'].get('subscriberCount', 'N/A'),
            'videoCount': item['statistics'].get('videoCount', 'N/A')
        }
    
    # for item in search_response['items']:
    #     channel_id = item['id']['channelId']
        
    #     # Get channel statistics
    #     channel_response = youtube.channels().list(
    #         id=channel_id,
    #         part='statistics'
    #     ).execute()
        
    #     statistics = channel_response['items'][0]['statistics']
        
    #     channels.append({
    #         'channelId': channel_id,
    #         'title': item['snippet']['title'],
    #         'description': item['snippet']['description'],
    #         'thumbnail': item['snippet']['thumbnails']['default']['url'],
    #         'subscriberCount': statistics.get('subscriberCount', 'N/A'),
    #         'videoCount': statistics.get('videoCount', 'N/A')
    #     })
    
    return channel

def get_channel_videos(channel_id):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    
    # # Search for the channel by name
    # search_response = youtube.search().list(
    #     q=channel_name,
    #     type='channel',
    #     part='id',
    #     maxResults=1
    # ).execute()
    # channel_id = search_response['items'][0]['id']['channelId']

    # Get all video IDs from the channel
    video_objects = []
    next_page_token = None
    
    while True:
        playlist_response = youtube.playlistItems().list(
            playlistId=f'UU{channel_id[2:]}',
            part='contentDetails,snippet',
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        for item in playlist_response['items']:
            videoId = item['contentDetails']['videoId']
            stats = youtube.videos().list(
                id=videoId,
                part="statistics",
            ).execute()['items'][0]['statistics']
            obj = {
                'videoId': videoId,
                'title': item['snippet']['title'],
                'publishedAt': item['snippet']['publishedAt'],
                'stats': stats
            }
            video_objects.append(obj)
        
        #video_ids_and_titles.extend([[item['contentDetails']['videoId'],item['snippet']['title']] for item in playlist_response['items']])
        
        next_page_token = playlist_response.get('nextPageToken')
        if not next_page_token:
            break
    
    return video_objects

def download_captions(video_id):
    command = [
        'yt-dlp',
        '--skip-download',
        '--write-auto-sub',
        '--sub-lang', 'en',
        '--output', f'{video_id}.%(ext)s',
        f'https://www.youtube.com/watch?v={video_id}'
    ]
    subprocess.run(command, check=True)
    return f'{video_id}.en.vtt'

def parse_vtt(file_path, videoObj):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    captions = []
    last_caption = ""
    for line in lines:
        if '-->' in line:
            time_range = line.strip()
        elif line.strip() and not line.startswith(('WEBVTT', 'NOTE')):
            # Remove HTML-like tags
            cleaned_line = re.sub(r'<[^>]+>', '', line.strip())
            # Avoid adding repeated lines
            if cleaned_line != last_caption:
                captions.append(cleaned_line)
                last_caption = cleaned_line
    #system_message = "You are an expert at understanding the unique story of each video and the unique story that is presented over all videos. You are an assistant for deeply understanding and querying information about a specific youtuber."
    #prompt = f'This Youtuber handle is {handle} and the Video Title is {title}'
    #formattedPrompt = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n {system_message} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n {prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    captionString = " ".join(captions)
    videoObj['captions'] = captionString
    return videoObj
    #return {'prompt': formattedPrompt, 'completion': captionString}

#llama 3 Instruct Format
#<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n {prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
#llama 3 Instruct Format with System Message
#<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n {system_message} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n {prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
#formattedString = f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n You are an expert at understanding the unique story of each video and the unique story that is presented over all videos. You are an assistant for deeply understanding and querying information about a specific youtuber. This Youtuber is: {handle} <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n Video Title: {title}. Video captions: {" ".join(captions)} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

def main():
    channel_handle = input('Enter YouTube channel handle(begins with @): ')
    # channels = search_channel(channel_handle)
    
    # if not channels:
    #     print("No channels found.")
    #     return
    
    # # Display channels for user confirmation
    # for i, channel in enumerate(channels):
    #     print(f"{i + 1}. {channel['title']}\n   Description: {channel['description']}\n   Thumbnail: {channel['thumbnail']}\n   Subscribers: {channel['subscriberCount']}\n   Videos: {channel['videoCount']}\n")
    
    channel = search_channel(channel_handle)
    
    if not channel:
        print("No channel found.")
        return
    
    # Display channel for user confirmation
    print(f"{channel['channelTitle']}\n   Description: {channel['description']}\n   Thumbnail: {channel['thumbnail']}\n   Subscribers: {channel['subscriberCount']}\n   Videos: {channel['videoCount']}\n")

    choice = int(input('Is this the correct channel? (1/0): '))
    if choice != 1:
        return

    video_objects = get_channel_videos(channel['channelId'])
    output_dir = f"{channel['channelTitle']}"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, 'allCaptions.json')
    
    with open(output_file, 'w') as f:
        allCaptionsObj = {
            "channelInformation": channel
        }
        for obj in video_objects:
            video_id = obj['videoId']
            try:
                vtt_file = download_captions(video_id)
                captions_data = parse_vtt(vtt_file, obj)
                keys_to_remove = {'captions', 'stats'}
                modified_captions_data = {k: v for k, v in captions_data.items() if k not in keys_to_remove}
                allCaptionsObj[video_id] = modified_captions_data
                chunkObj = {
                    video_id: captions_data
                }

                # Write to individual JSON file for each video
                individual_file = os.path.join(output_dir, f'{video_id}.json')
                with open(individual_file, 'w') as video_f:
                    json.dump(chunkObj, video_f, indent=4)
                
                os.remove(vtt_file)  # Clean up the .vtt file after processing
            except Exception as e:
                print(f'Error processing video {video_id}: {e}')

        main_document = {}
        main_document[f"{output_dir}_MAIN_DOC"] = allCaptionsObj
        json.dump(main_document, f, indent=4)
    
    print(f'Captions saved to {output_file} and individual JSON files in {output_dir}.')

if __name__ == '__main__':
    main()