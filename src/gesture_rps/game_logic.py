WINMAP = {"rock": "scissors", "paper": "rock", "scissors": "paper"}

def adjudicate(player: str, ai: str) -> str:
    """
    Return one of: 'win' | 'lose' | 'tie' | 'invalid'
    """
    valid = {"rock", "paper", "scissors"}
    if player not in valid or ai not in valid:
        return "invalid"
    if player == ai:
        return "tie"
    return "win" if WINMAP[player] == ai else "lose"
