#!/usr/bin/env python3
"""
Standalone Result Video Processor for Smash Bros

This script processes result screen videos through Gemini API and saves match stats to the database.
Can be used to reprocess videos or process manually captured result screens.

Usage:
    python process_result_video.py result_screen_video.mp4
    python process_result_video.py result_screen_video.mp4 --slowdown 5
    python process_result_video.py result_screen_video.mp4 --force-save
"""

import argparse
import datetime
import os
import sys
import time
import logging
from typing import List, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

class PlayerStats(BaseModel):
    is_online_match: bool
    smash_character: str
    player_name: str
    is_cpu: bool
    total_kos: int
    total_falls: int
    total_sds: int
    has_won: bool

# Initialize Gemini client
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    gemini_client = genai.Client(api_key=gemini_api_key)
    gemini_model = "gemini-2.5-pro-preview-06-05"
except Exception as e:
    print(f"Warning: Failed to initialize Gemini client: {e}")
    gemini_client = None

# Initialize Supabase client
try:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not found in environment variables")
    
    supabase_client: Client = create_client(supabase_url, supabase_key)
except Exception as e:
    print(f"Warning: Failed to initialize Supabase client: {e}")
    supabase_client = None

class ResultVideoProcessor:
    def __init__(self, video_path: str, slowdown_factor: int = 5, force_save: bool = False):
        self.video_path = video_path
        self.slowdown_factor = slowdown_factor
        self.force_save = force_save
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging to file and console"""
        # Use a fixed log filename that gets overwritten each time
        log_filename = "result_processor.log"
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.video_path)
        log_filepath = os.path.join(log_dir, log_filename)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filepath, mode='w'),  # 'w' mode overwrites the file
                logging.StreamHandler()
            ],
            force=True  # Force reconfiguration if already configured
        )
        
        # Suppress verbose HTTP logging from Google API client
        logging.getLogger('google.auth.transport.requests').setLevel(logging.WARNING)
        logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Result Video Processor started - Log file: {log_filename}")
        self.logger.info(f"Processing video: {self.video_path}")
        self.logger.info(f"Slowdown factor: {self.slowdown_factor}")
        self.logger.info(f"Force save: {self.force_save}")
    
    def get_match_stats(self) -> Optional[List[PlayerStats]]:
        """Extract player stats from result screen video using Gemini API"""
        if not gemini_client:
            self.logger.error("Gemini client not available")
            return None
        
        if not os.path.exists(self.video_path):
            self.logger.error(f"Video file not found: {self.video_path}")
            return None
        
        try:
            self.logger.info(f"Processing result screen video: {self.video_path}")
            
            # First use ffmpeg to slow down the video
            final_video_filepath = "./temp_processed_video.mp4"
            ffmpeg_cmd = f"ffmpeg -y -an -i \"{self.video_path}\" -vf \"setpts={self.slowdown_factor}*PTS\" \"{final_video_filepath}\" -loglevel quiet"
            
            self.logger.info(f"Slowing down video by factor of {self.slowdown_factor}")
            if os.system(ffmpeg_cmd) != 0:
                self.logger.error("Failed to process video with ffmpeg")
                return None
            
            # Upload file to Gemini
            self.logger.info("Uploading video to Gemini API...")
            file = gemini_client.files.upload(file=final_video_filepath)
            
            # Wait for file to be processed
            self.logger.info("Waiting for video to be processed by Gemini...")
            while True:
                file_info = gemini_client.files.get(name=file.name)
                if file_info.state == "ACTIVE":
                    break
                time.sleep(1)
            
            # Prepare content for Gemini
            contents = [    
                file,
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text="""Here is a video recording of the results screen of a super smash bros ultimate match.

Output the following information about the game's results as valid json following this schema (where it's a list of json objects -- one for each player in the match):

```
[
{
\"is_online_match\" : boolean,
\"smash_character\" : string,
\"player_name\" : string,
\"is_cpu\" : boolean,
\"total_kos\" : int,
\"total_falls\" : int,
\"total_sds\" : int,
\"has_won\" : boolean
},
...
]
```

keep the following in mind:

- the total number of KOs is an integer number located to the right of the label, and cannot be null. if you can't see a number next to the \"KOs\" label, then instead, the KO's are counted by counting the number of mini character icons shown under the \"KOs\" section of the character card
- total number of falls is an integer number located to the right of the label, and cannot be null. if you can't see a number next to the \"Falls\" label, then instead, the falls are counted by counting the number of mini character icons shown under the \"Falls\" section of the character card
- total number of SDs is an integer number located to the right of the label, and cannot be null. if you can't see a number next to the \"SDs\" label, then instead, the SD's are counted by counting the number of mini character icons shown under the \"SDs\" section of the character card
- \"has_won\" denotes whether or not the character won (labeled with a gold-colored number 1 at the top right of the player card. if there is no such number ranking on the top right, then the character did not win; for \"no contest\" matches, no character wins)
- \"is_online_match\" There are likely to be 2 players in the match. If you see "onlineacc" as one of the player names, then return true, otherwise it is an offline match. If the player name is not "onlineacc" or "offlineacc", return false.
- If is_cpu is false, then it's impossible to have only 1 player in the match. Really make sure that you have identified all the players in the match.
"""),
                    ],
                ),
            ]

            self.logger.info("Analyzing video with Gemini API...")
            response = gemini_client.models.generate_content(
                model=gemini_model,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=list[PlayerStats],
                ),
                contents=contents,
            )
            
            # Clean up uploaded file
            gemini_client.files.delete(name=file.name)
            
            # Clean up temporary video file
            try:
                os.remove(final_video_filepath)
            except:
                pass
            
            self.logger.info(f"Successfully extracted stats for {len(response.parsed)} players")
            
            # Log the extracted stats
            for i, stat in enumerate(response.parsed):
                self.logger.info(f"Player {i+1}: {stat.player_name} ({stat.smash_character}) - KOs: {stat.total_kos}, Falls: {stat.total_falls}, SDs: {stat.total_sds}, Won: {stat.has_won}")
            
            return response.parsed
            
        except Exception as e:
            self.logger.error(f"Error extracting match stats: {e}")
            return None
    
    def update_elo(self, rating_a: float, rating_b: float, winner: str, k: int = 32) -> tuple[int, int]:
        """Update ELO ratings after a match"""
        # Expected scores (logistic curve)
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a

        # Actual scores
        if winner.upper() == 'A':
            score_a, score_b = 1.0, 0.0
        elif winner.upper() == 'B':
            score_a, score_b = 0.0, 1.0
        elif winner.lower() in ('draw', 'tie', 'd'):
            score_a = score_b = 0.5
        else:
            raise ValueError("winner must be 'A', 'B', or 'draw'")

        # Rating updates
        new_rating_a = rating_a + k * (score_a - expected_a)
        new_rating_b = rating_b + k * (score_b - expected_b)

        # Return as integers
        return round(new_rating_a), round(new_rating_b)
    
    def get_player(self, player_name: str) -> Optional[dict]:
        """Get or create a player in the database"""
        if not supabase_client:
            return None
        
        try:
            response = (
                supabase_client.table("players")
                .upsert({"name": player_name}, on_conflict="name")
                .execute()
            )
            return response.data[0]
        except Exception as e:
            self.logger.error(f"Error getting/creating player {player_name}: {e}")
            return None
    
    def update_player_elo(self, player_id: str, elo: int):
        """Update a player's ELO in the database"""
        if not supabase_client:
            return
        
        try:
            supabase_client.table("players").update({"elo": elo}).eq("id", player_id).execute()
        except Exception as e:
            self.logger.error(f"Error updating player ELO: {e}")
    
    def create_match(self) -> Optional[int]:
        """Create a new match in the database"""
        if not supabase_client:
            return None
        
        try:
            response = (
                supabase_client.table("matches")
                .insert({})
                .execute()
            )
            return response.data[0]['id']
        except Exception as e:
            self.logger.error(f"Error creating match: {e}")
            return None
    
    def save_match_stats(self, stats: List[PlayerStats]) -> bool:
        """Save match stats to the database"""
        if not supabase_client:
            self.logger.error("Supabase client not available")
            return False
        
        try:
            # Check if match should be skipped
            if not self.force_save:
                # Skip no contest matches
                match_is_no_contest = all(not stat.has_won for stat in stats)
                if match_is_no_contest:
                    self.logger.warning("Match is a no contest, skipping database save")
                    return False
                
                # Skip matches with CPU players
                match_has_cpu = any(stat.is_cpu for stat in stats)
                if match_has_cpu:
                    self.logger.warning("Match has CPU players, skipping database save")
                    return False
                
                # Skip matches with unknown players
                match_has_unknown_players = False
                for stat in stats:
                    if re.match(r"^Player \d+$", stat.player_name) or re.match(r"^P\d+$", stat.player_name) or re.match(r"^P \d+$", stat.player_name):
                        match_has_unknown_players = True
                        break
                
                if match_has_unknown_players:
                    self.logger.warning("Match has unknown players (Player 1,2,3,etc.), skipping database save")
                    return False
                
                # Skip online matches
                if stats[0].is_online_match:
                    self.logger.warning("Match is online, skipping database save")
                    return False
            
            # Create match
            match_id = self.create_match()
            if match_id is None:
                return False
            
            players = []
            winners = []
            
            self.logger.info(f"Saving match stats to database (Match ID: {match_id})")
            
            for stat in stats:
                player = self.get_player(stat.player_name)
                if player is None:
                    continue
                
                # Save match participant
                response = (
                    supabase_client.table("match_participants")
                    .insert({
                        "player": player['id'], 
                        "smash_character": stat.smash_character.upper(),
                        "is_cpu": stat.is_cpu,
                        "total_kos": stat.total_kos,
                        "total_falls": stat.total_falls,
                        "total_sds": stat.total_sds,
                        "has_won": stat.has_won,
                        "match_id": match_id,
                    })
                    .execute()
                )
                
                players.append({
                    "id": player['id'], 
                    "elo": player['elo'], 
                    "name": player['name'], 
                    "character": stat.smash_character,
                    "has_won": stat.has_won,
                    "kos": stat.total_kos,
                    "falls": stat.total_falls,
                    "sds": stat.total_sds
                })
                
                if stat.has_won:
                    winners.append(player['name'])
            
            # Print match results
            self.logger.info("=" * 60)
            self.logger.info("MATCH RESULTS")
            self.logger.info("=" * 60)
            
            if winners:
                self.logger.info(f"🏆 Winner(s): {', '.join(winners)}")
            else:
                self.logger.info("🤝 No Contest")
            
            self.logger.info("Player Stats:")
            for player in players:
                status = "🏆 WINNER" if player['has_won'] else ""
                self.logger.info(f"  {player['name']} ({player['character']}) - KOs: {player['kos']}, Falls: {player['falls']}, SDs: {player['sds']} {status}")
            
            # Update ELO ratings for 1v1 matches
            if len(stats) == 2:
                self.logger.info("1v1 Match detected - Updating ELO ratings:")
                
                old_elo_1 = players[0]['elo']
                old_elo_2 = players[1]['elo']
                
                winner_index = 1 if players[0]['has_won'] else 2
                new_elo_1, new_elo_2 = self.update_elo(old_elo_1, old_elo_2, 'A' if winner_index == 1 else 'B')
                
                self.update_player_elo(players[0]['id'], new_elo_1)
                self.update_player_elo(players[1]['id'], new_elo_2)
                
                # Print ELO changes
                elo_change_1 = new_elo_1 - old_elo_1
                elo_change_2 = new_elo_2 - old_elo_2
                
                self.logger.info(f"  {players[0]['name']}: {old_elo_1} → {new_elo_1} ({elo_change_1:+d})")
                self.logger.info(f"  {players[1]['name']}: {old_elo_2} → {new_elo_2} ({elo_change_2:+d})")
            
            self.logger.info("=" * 60)
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving match stats: {e}")
            return False
    
    def process(self):
        """Main processing function"""
        self.logger.info("Starting result video processing...")
        
        # Extract match stats
        match_stats = self.get_match_stats()
        
        if not match_stats:
            self.logger.error("Failed to extract match stats")
            return False
        
        # Save to database
        success = self.save_match_stats(match_stats)
        
        if success:
            self.logger.info("Successfully processed and saved match results")
        else:
            self.logger.warning("Match results extracted but not saved to database")
        
        return success

def main():
    parser = argparse.ArgumentParser(description='Process Smash Bros result screen videos')
    parser.add_argument('video_path', type=str, help='Path to the result screen video file')
    parser.add_argument('--slowdown', type=int, default=5, help='Video slowdown factor (default: 5)')
    parser.add_argument('--force-save', action='store_true', help='Force save even if match has CPU/unknown players/is online')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Create processor and run
    processor = ResultVideoProcessor(args.video_path, args.slowdown, args.force_save)
    success = processor.process()
    
    if success:
        print("✅ Processing completed successfully!")
        sys.exit(0)
    else:
        print("❌ Processing failed or match was skipped")
        sys.exit(1)

if __name__ == "__main__":
    main()