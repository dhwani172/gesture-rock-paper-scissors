from gesture_rps.game_logic import adjudicate

def test_adjudicate_rules():
    assert adjudicate("rock", "scissors") == "win"
    assert adjudicate("rock", "paper") == "lose"
    assert adjudicate("rock", "rock") == "tie"
    assert adjudicate("invalid", "rock") == "invalid"