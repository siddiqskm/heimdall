# heimdall/adapt/config.py

MAX_BIAS: float = 0.25     # absolute cap per label
DECAY: float = 0.98        # decay applied every turn
REWARD: float = 0.02       # reinforce when action worked
PENALTY: float = 0.05      # penalize when action failed
