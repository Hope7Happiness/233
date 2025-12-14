from game import State, Action, GameManager, rollout
import random

def random_policy(state: State) -> Action:
    """Random policy for testing"""
    print('state:', state)
    return random.choice(list(Action))

if __name__ == "__main__":
    game = GameManager()
    # score = rollout(random_policy, render=False, render_delay=0.5)
    score = rollout(random_policy, render=True, render_delay=0.1)
    print(f"Final score with random policy: {score}")