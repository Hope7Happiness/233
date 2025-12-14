import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from game import State, Action, GameManager, rollout
import time, os

class Q_network(nn.Module):
    def __init__(self, input_len=9, num_actions=4, hidden_dim=128, num_vocab=20):
        super(Q_network, self).__init__()
        self.emb = nn.Embedding(num_vocab, hidden_dim)
        self.fc1 = nn.Linear(input_len * hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, x):
        B, L = x.shape
        x = self.emb(x)  # B x L x D
        x = x.view(B, -1)  # B x (L*D)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    
def explore_step(q_network, batch_state, eps=0.1):
    if random.random() < eps:
        return random.choice(list(Action))
    else:
        with torch.no_grad():
            q_values = q_network(batch_state)
            return [Action(torch.argmax(q_values, dim=1)[i].item()) for i in range(batch_state.size(0))]

def batch_rollout(
            policy_fn: Callable[State, Action], 
            batch_size: int,
        ) -> int:
    """Run a complete game using the given policy"""
    # game = GameManager()
    games = [GameManager() for _ in range(batch_size)]
    
    def init_worker():
        seed = int(time.time()) ^ os.getpid()
        random.seed(seed)
    
    def simulate_step(game: GameManager):
        step_count = 0
        history = []
        while not game.is_game_terminated():
            state = game.get_state()
            action = policy_fn(state)
            
            next_state, reward, done = game.step(action)
            history.append((state, action, reward, next_state, done))
            step_count += 1
        return history, game.score
    
    # use parallel processing to run multiple games
    from multiprocessing import Pool
    with Pool(initializer=init_worker) as pool:
        all_histories, all_scores = zip(*pool.map(simulate_step, games))
    return all_histories, all_scores

def train(num_steps, batch_size, update_target_per, update_buffer_per):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = Q_network().to(device)
    target_net = Q_network().to(device)
    target_net.eval()
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(10000)
    
    for step in range(num_steps):
        if step % update_target_per == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if step % update_buffer_per == 0:
            new_histories, new_scores = batch_rollout(
                lambda s: explore_step(policy_net, torch.tensor(s['grid'], dtype=torch.long).unsqueeze(0).to(device)), 
                batch_size
            )
            print(f'Rollout at step {step}, scores statistics: min {min(new_scores)}, max {max(new_scores)}, avg {sum(new_scores)/len(new_scores)}')
            for history in new_histories:
                for (s, a, r, s_next, d) in history:
                    s_tensor = torch.tensor(s['grid'], dtype=torch.long, device=device).reshape(-1)
                    s_next_tensor = torch.tensor(s_next['grid'], dtype=torch.long, device=device).reshape(-1)
                    replay_buffer.push(s_tensor, torch.tensor(a.value, device=device), torch.tensor(r, dtype=torch.float, device=device), s_next_tensor, torch.tensor(d, dtype=torch.float, device=device))
        
        if len(replay_buffer) < batch_size:
            print(f'size of replay buffer: {len(replay_buffer)} < batch size {batch_size}, waiting to fill...')
            continue
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(batch_size)
        q_values = policy_net(batch_state)
        q_s_a = q_values[torch.arange(batch_size), batch_action]
        with torch.no_grad():
            q_next_values = target_net(batch_next_state)
            q_next_max = q_next_values.max(dim=-1).values
            q_target = batch_reward + (1 - batch_done) * 0.99 * q_next_max
            
        loss = F.mse_loss(q_s_a, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
if __name__ == "__main__":
    train(num_steps=1000, batch_size=16, update_target_per=50, update_buffer_per=10)