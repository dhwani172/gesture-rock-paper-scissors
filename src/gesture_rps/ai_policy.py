from collections import Counter, deque
import random

MOVES = ["rock", "paper", "scissors"]
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
LOSES_TO = {v: k for k, v in BEATS.items()}  # inverse

class AIPolicy:
    def choose(self, history: "deque[str]"):
        raise NotImplementedError
    def learn(self, history: "deque[str]"):
        # optional; used by Markov
        pass

class RandomPolicy(AIPolicy):
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
    def choose(self, history):
        return random.choice(MOVES)

class AdaptiveFrequencyPolicy(AIPolicy):
    """Predict player's next move by their most frequent past move; counter it."""
    def __init__(self):
        pass
    def choose(self, history):
        if not history:
            return random.choice(MOVES)
        counts = Counter(history)
        predicted_player = max(counts.items(), key=lambda kv: kv[1])[0]
        # counter the predicted player's move
        return LOSES_TO[predicted_player]

class MarkovPolicy(AIPolicy):
    """
    First-order Markov: P(next | last). We learn transitions as you play.
    If not enough data, fallback to frequency, then random.
    """
    def __init__(self):
        # transitions[last][next] = count
        self.transitions = {m: Counter() for m in MOVES}

    def learn(self, history: "deque[str]"):
        if len(history) >= 2:
            prev, cur = history[-2], history[-1]
            if prev in MOVES and cur in MOVES:
                self.transitions[prev][cur] += 1

    def choose(self, history):
        if history:
            last = history[-1]
            row = self.transitions.get(last, Counter())
            if row:
                # predict player's next as argmax row; then counter it
                predicted_player = max(row.items(), key=lambda kv: kv[1])[0]
                return LOSES_TO[predicted_player]
        # fallback: frequency → random
        if history:
            counts = Counter(history)
            predicted_player = max(counts.items(), key=lambda kv: kv[1])[0]
            return LOSES_TO[predicted_player]
        return random.choice(MOVES)
