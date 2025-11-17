#!/usr/bin/env python3
"""
Shared ELO utilities for Smash Bros Leaderboard

This module contains all ELO calculation functions.
"""

import datetime
from typing import Dict, Tuple
from supabase import Client


def calculate_top_ten_played_for_player(player_id: str, supabase_client: Client) -> int:
    """
    Calculate how many original top 10 players this specific player has faced.
    Used for incremental updates after each match.
    
    Args:
        player_id: The player to calculate top_ten_played for
        supabase_client: Supabase client instance
    
    Returns:
        int: Number of original top 10 players this player has faced
    """
    try:
        # Get original top 10 player IDs (ranked players with highest ELO)
        players_response = supabase_client.table("players").select("id, elo, top_ten_played").execute()
        players_data = players_response.data
        
        # Filter ranked players (top_ten_played >= 3) and get top 10 by ELO
        ranked_players = [p for p in players_data if p.get('top_ten_played', 0) >= 3]
        ranked_players.sort(key=lambda x: x['elo'], reverse=True)
        original_top_ten_ids = {p['id'] for p in ranked_players[:10]}
        
        if not original_top_ten_ids:
            return 0
        
        # Get all 1v1 matches this player participated in
        participants_response = supabase_client.table("match_participants")\
            .select("match_id")\
            .eq("player", player_id)\
            .execute()
        
        if not participants_response.data:
            return 0
        
        match_ids = [p['match_id'] for p in participants_response.data]
        
        # Get all participants for these matches to find opponents
        all_participants_response = supabase_client.table("match_participants")\
            .select("match_id, player")\
            .in_("match_id", match_ids)\
            .execute()
        
        # Group by match_id and find 1v1 matches where player faced top 10
        match_participants = {}
        for participant in all_participants_response.data:
            match_id = participant['match_id']
            if match_id not in match_participants:
                match_participants[match_id] = []
            match_participants[match_id].append(participant['player'])
        
        # Find unique top 10 opponents in 1v1 matches
        top_ten_opponents = set()
        for match_id, players in match_participants.items():
            if len(players) == 2 and player_id in players:
                # Find the opponent
                opponent_id = players[0] if players[1] == player_id else players[1]
                if opponent_id in original_top_ten_ids:
                    top_ten_opponents.add(opponent_id)
        
        return len(top_ten_opponents)
        
    except Exception as e:
        print(f"Error calculating top_ten_played for player {player_id}: {e}")
        return 0


def update_player_top_ten_played(player_id: str, top_ten_played: int, supabase_client: Client) -> bool:
    """
    Update a player's top_ten_played count in the database.
    
    Args:
        player_id: The player to update
        top_ten_played: New top_ten_played count
        supabase_client: Supabase client instance
    
    Returns:
        bool: True if update succeeded, False otherwise
    """
    try:
        supabase_client.table("players")\
            .update({"top_ten_played": top_ten_played})\
            .eq("id", player_id)\
            .execute()
        return True
    except Exception as e:
        print(f"Error updating top_ten_played for player {player_id}: {e}")
        return False


def check_if_player_becomes_ranked(player_id: str, supabase_client: Client) -> Tuple[bool, int]:
    """
    Check if a player becomes ranked (top_ten_played >= 3) after updating their top_ten_played.
    
    Args:
        player_id: The player to check
        supabase_client: Supabase client instance
    
    Returns:
        Tuple[bool, int]: (became_ranked, new_top_ten_played_count)
    """
    try:
        # Get player's current ranking status
        player_response = supabase_client.table("players")\
            .select("top_ten_played")\
            .eq("id", player_id)\
            .execute()
        
        if not player_response.data:
            return False, 0
        
        old_top_ten_played = player_response.data[0].get('top_ten_played', 0)
        was_ranked = old_top_ten_played >= 3
        
        # Calculate new top_ten_played count
        new_top_ten_played = calculate_top_ten_played_for_player(player_id, supabase_client)
        
        # Update the database with new count
        update_player_top_ten_played(player_id, new_top_ten_played, supabase_client)
        
        # Check if player became ranked
        is_now_ranked = new_top_ten_played >= 3
        became_ranked = not was_ranked and is_now_ranked
        
        return became_ranked, new_top_ten_played
        
    except Exception as e:
        print(f"Error checking if player {player_id} becomes ranked: {e}")
        return False, 0


def recalculate_all_matches_for_player(player_id: str, supabase_client: Client) -> bool:
    """
    Recalculate ELO for all historical matches involving a newly ranked player.
    This ensures their match history is properly processed with ELO updates.
    
    Args:
        player_id: The newly ranked player
        supabase_client: Supabase client instance
    
    Returns:
        bool: True if recalculation succeeded, False otherwise
    """
    try:
        print(f"Recalculating all matches for newly ranked player {player_id}")
        
        # Get all matches this player participated in, chronologically
        participants_response = supabase_client.table("match_participants")\
            .select("match_id")\
            .eq("player", player_id)\
            .execute()
        
        if not participants_response.data:
            return True
        
        match_ids = [p['match_id'] for p in participants_response.data]
        
        # Get match details with dates
        matches_response = supabase_client.table("matches")\
            .select("id, created_at")\
            .in_("id", match_ids)\
            .eq("archived", False)\
            .order("created_at", desc=False)\
            .execute()
        
        if not matches_response.data:
            return True
        
        # Get all current player data for ranking checks
        all_players_response = supabase_client.table("players").select("id, elo, top_ten_played").execute()
        players_data = {p['id']: p for p in all_players_response.data}
        
        processed_matches = 0
        elo_updates = 0
        
        for match in matches_response.data:
            match_id = match['id']
            
            # Get participants for this match
            match_participants_response = supabase_client.table("match_participants")\
                .select("player, has_won")\
                .eq("match_id", match_id)\
                .execute()
            
            participants = match_participants_response.data
            
            if len(participants) != 2:
                continue
            
            player_ids = [p['player'] for p in participants]
            winners = [p['player'] for p in participants if p['has_won']]
            
            if len(winners) != 1:
                continue
            
            # Check if both players are now ranked
            player1_id, player2_id = player_ids[0], player_ids[1]
            player1_data = players_data.get(player1_id)
            player2_data = players_data.get(player2_id)
            
            if (not player1_data or not player2_data or 
                player1_data.get('top_ten_played', 0) < 3 or 
                player2_data.get('top_ten_played', 0) < 3):
                continue
            
            # Get current ELOs
            rating_a = player1_data['elo']
            rating_b = player2_data['elo']
            winner_id = winners[0]
            winner = 'A' if winner_id == player1_id else 'B'
            
            # Calculate new ELOs using streaming function
            new_elo_a, new_elo_b = calculate_elo_update_for_streaming(
                rating_a, rating_b, winner, player1_id, player2_id, supabase_client
            )
            
            # Update ELOs in database if they changed
            if new_elo_a != rating_a:
                supabase_client.table("players")\
                    .update({"elo": new_elo_a})\
                    .eq("id", player1_id)\
                    .execute()
                players_data[player1_id]['elo'] = new_elo_a
                
            if new_elo_b != rating_b:
                supabase_client.table("players")\
                    .update({"elo": new_elo_b})\
                    .eq("id", player2_id)\
                    .execute()
                players_data[player2_id]['elo'] = new_elo_b
            
            if new_elo_a != rating_a or new_elo_b != rating_b:
                elo_updates += 1
            
            processed_matches += 1
        
        print(f"Processed {processed_matches} matches for player {player_id}, {elo_updates} ELO updates applied")
        return True
        
    except Exception as e:
        print(f"Error recalculating matches for player {player_id}: {e}")
        return False




def update_elo(rating_a: float, rating_b: float, winner: str, k: int = 32) -> tuple[int, int]:
    """
    Return the new (rating_a, rating_b) after one game.

    Parameters
    ----------
    rating_a : current Elo for Player A
    rating_b : current Elo for Player B
    winner   : 'A', 'B', or 'draw'
    k        : K-factor (default 32)

    Returns
    -------
    tuple[int, int] : New ratings as integers for (Player A, Player B)
    """
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




def calculate_elo_update_for_streaming(rating_a: float, rating_b: float, winner: str,
                                     player_a_id: str, player_b_id: str,
                                     supabase_client: Client, k: int = 32) -> Tuple[int, int]:
    """
    High-level function to calculate ELO update for streaming (real-time) processing.
    Only processes ELO updates if both players are ranked (top_ten_played >= 3).
    Also handles dynamic ranking progression for unranked players.
    
    Returns:
        tuple[int, int]: (new_elo_a, new_elo_b)
    """
    # Check if both players are ranked before processing ELO updates
    players_response = supabase_client.table("players").select("id, elo, top_ten_played").execute()
    players_data = {p['id']: p for p in players_response.data}
    
    player_a_data = players_data.get(player_a_id)
    player_b_data = players_data.get(player_b_id)
    
    # First, update top_ten_played for any unranked players and check if they become ranked
    newly_ranked_players = []
    
    if player_a_data and player_a_data.get('top_ten_played', 0) < 3:
        became_ranked, new_count = check_if_player_becomes_ranked(player_a_id, supabase_client)
        if became_ranked:
            newly_ranked_players.append(player_a_id)
            player_a_data['top_ten_played'] = new_count
            print(f"Player {player_a_id} became ranked! top_ten_played: {new_count}")
    
    if player_b_data and player_b_data.get('top_ten_played', 0) < 3:
        became_ranked, new_count = check_if_player_becomes_ranked(player_b_id, supabase_client)
        if became_ranked:
            newly_ranked_players.append(player_b_id)
            player_b_data['top_ten_played'] = new_count
            print(f"Player {player_b_id} became ranked! top_ten_played: {new_count}")
    
    # Trigger full batch recompute if any player became ranked
    # This is necessary because newly ranked players affect everyone's historical ELOs
    if newly_ranked_players:
        print(f"Triggering full ELO recompute due to newly ranked players: {newly_ranked_players}")
        # Import and run the batch recompute function
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from recompute_all_player_elos import recompute_all_player_elos
            recompute_all_player_elos()
        except Exception as e:
            print(f"Error during full recompute: {e}")
            # Fall back to individual recalculation if batch fails
            for player_id in newly_ranked_players:
                recalculate_all_matches_for_player(player_id, supabase_client)
    
    # Return original ELOs if either player is not found or not ranked
    if (not player_a_data or not player_b_data or 
        player_a_data.get('top_ten_played', 0) < 3 or 
        player_b_data.get('top_ten_played', 0) < 3):
        return int(rating_a), int(rating_b)
    
    # Use normal ELO calculation
    new_elo_a, new_elo_b = update_elo(rating_a, rating_b, winner, k)
    return new_elo_a, new_elo_b