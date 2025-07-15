# Advanced ELO System for Smash Bros Leaderboard

This document describes the advanced ELO rating system implemented for the Smash Bros leaderboard, which includes sophisticated constraints to ensure fair and competitive rankings.

## Overview

The advanced ELO system addresses three main issues with traditional ELO systems:

1. **Ranking Constraints**: Prevents players from ranking above others they haven't legitimately beaten
2. **Daily Matchup Limits**: Prevents ELO farming by limiting games that count between the same players
3. **Inactivity Penalties**: Encourages active participation by penalizing inactive players

## Key Features

### 1. Ranking Constraints

**Problem**: In traditional ELO, Player A could theoretically rank above Player B without ever facing them, just by beating weaker opponents.

**Solution**: Players cannot rank above others unless they have:

- Beaten that player directly, OR
- Beaten someone who has beaten that player (transitive victory)

**Implementation**: When a player's ELO would increase above another player's ELO, the system checks:

- Has the player ever beaten the higher-ranked player?
- Has the player beaten anyone who has beaten the higher-ranked player?
- If neither condition is met, the player's ELO is capped at 1 point below the higher-ranked player

### 2. Daily Matchup Limits

**Problem**: Players could "farm" ELO by playing the same opponent repeatedly in one day.

**Solution**: Players can only maintain a maximum 2-game lead against any specific opponent per day.

**Implementation**:

- Tracks daily head-to-head records between all players
- If a match would result in a game lead > 2, the match doesn't count for ELO
- A "2-game lead" means the difference between wins and losses is 2 or less
- Example: If Player A beats Player B 3 times and Player B beats Player A 1 time, the lead is 2 (3-1=2)
- The next win by Player A wouldn't count for ELO

### 3. Inactivity Penalties

**Problem**: Players could achieve high rankings and then stop playing to maintain their position.

**Solution**: Players lose 15 ELO per week of inactivity.

**Implementation**:

- Tracks when each player last played a match
- Players who haven't played in 7+ days lose 15 ELO per week
- ELO has a minimum floor of 800 to prevent excessive penalties
- Can be run manually or scheduled as a cron job

## Usage

### Basic Integration

The advanced ELO system is automatically used when processing matches:

```python
# In capture_card_processor.py
processor = SmashBrosProcessor()
# The processor automatically uses AdvancedEloSystem if available
```

### Manual ELO Management

Use the `elo_manager.py` script for manual ELO operations:

```bash
# Show current rankings
python elo_manager.py --rankings

# Show detailed player stats
python elo_manager.py --player-stats "PlayerName"

# Run inactivity penalties
python elo_manager.py --inactivity-penalties

# Show today's match statistics
python elo_manager.py --today

# Show specific date statistics
python elo_manager.py --daily-stats "2024-01-15"
```

### Running Inactivity Penalties

You can run inactivity penalties in several ways:

1. **Using the main processor**:

   ```bash
   python capture_card_processor.py --run-inactivity-penalties
   ```

2. **Using the ELO manager**:

   ```bash
   python elo_manager.py --inactivity-penalties
   ```

3. **Scheduled execution** (recommended):
   Add to crontab to run weekly:
   ```bash
   # Run every Sunday at 2 AM
   0 2 * * 0 cd /path/to/project && python elo_manager.py --inactivity-penalties
   ```

## Database Schema Requirements

The advanced ELO system requires the following database tables:

### `players` table

- `id` (UUID, primary key)
- `name` (string, unique)
- `elo` (integer, default: 1200)
- `created_at` (timestamp)

### `matches` table

- `id` (UUID, primary key)
- `created_at` (timestamp)

### `match_participants` table

- `id` (UUID, primary key)
- `match_id` (UUID, foreign key to matches)
- `player` (UUID, foreign key to players)
- `smash_character` (string)
- `is_cpu` (boolean)
- `total_kos` (integer)
- `total_falls` (integer)
- `total_sds` (integer)
- `has_won` (boolean)

## Configuration

The advanced ELO system can be configured by modifying the `AdvancedEloSystem` class:

```python
class AdvancedEloSystem:
    def __init__(self, supabase_client: Client):
        self.k_factor = 32                    # Standard ELO K-factor
        self.daily_matchup_limit = 2          # Max game lead per day
        self.inactivity_penalty = 15          # ELO lost per week
        self.inactivity_threshold_days = 7    # Days before penalty
```

## Examples

### Example 1: Ranking Constraint

- Player A (ELO: 1400) beats Player C (ELO: 1300)
- Player B (ELO: 1450) has never faced Player A
- Player A's ELO would normally go to 1415
- But Player A has never beaten Player B or anyone who beat Player B
- So Player A's ELO is capped at 1449 (1 below Player B)

### Example 2: Daily Matchup Limit

- Player A vs Player B on Monday:
  - Game 1: A wins (Lead: A+1)
  - Game 2: A wins (Lead: A+2) ✓ Counts for ELO
  - Game 3: A wins (Lead: A+3) ✗ Doesn't count for ELO
  - Game 4: B wins (Lead: A+2) ✓ Counts for ELO
  - Game 5: B wins (Lead: A+1) ✓ Counts for ELO

### Example 3: Inactivity Penalty

- Player A (ELO: 1500) last played on January 1st
- On January 8th (7 days later): ELO becomes 1485 (-15)
- On January 15th (14 days later): ELO becomes 1470 (-15)
- Player A plays on January 16th: Penalties stop

## Monitoring and Debugging

### Check Daily Limits

```python
# Check if a match would count for ELO
elo_system = AdvancedEloSystem(supabase_client)
would_count = elo_system.should_count_match_for_elo(player1_id, player2_id, winner_id)
```

### View Head-to-Head Records

```python
# Get daily record
daily_record = elo_system.get_daily_head_to_head_record(player1_id, player2_id)
print(f"Today's lead: {daily_record['game_lead']}")

# Get all-time record
all_time_record = elo_system.get_all_time_matchup_record(player1_id, player2_id)
print(f"All-time: {all_time_record['player1_wins']}-{all_time_record['player2_wins']}")
```

### Check Ranking Constraints

```python
# Check what ELO would be allowed
proposed_elo = 1500
actual_elo = elo_system.get_ranking_constraint_elo(player_id, proposed_elo)
if actual_elo != proposed_elo:
    print(f"ELO capped due to ranking constraints: {proposed_elo} → {actual_elo}")
```

## Troubleshooting

### Common Issues

1. **Matches not counting for ELO**: Check if daily matchup limit is reached
2. **ELO not increasing as expected**: Check for ranking constraints
3. **Inactivity penalties not working**: Ensure the script is running and database is accessible

### Debugging Commands

```bash
# Show detailed player stats including constraints
python elo_manager.py --player-stats "PlayerName"

# Show today's match activity
python elo_manager.py --today

# Show full rankings to identify constraint issues
python elo_manager.py --rankings --limit 50
```

## Future Enhancements

Potential improvements to consider:

1. **Seasonal ELO decay**: Gradual ELO reduction over time
2. **Tournament mode**: Special ELO rules for tournament matches
3. **Character-specific ELO**: Separate rankings per character
4. **Streak bonuses**: Additional ELO for win streaks
5. **Placement matches**: Special rules for new players

## Performance Considerations

- The system performs multiple database queries per match
- Consider caching for frequently accessed data
- Monitor database performance with large datasets
- Consider indexing on player, match_id, and created_at columns
