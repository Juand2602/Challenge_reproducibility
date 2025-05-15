import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from gym.wrappers import AtariPreprocessing, FrameStack
import imageio
import multiprocessing as mp

# Política convolucional para Pong
class CNNPolicy(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.n)
        )

    def forward(self, x):
        x = x / 255.0  # Normalizar imagen
        return self.fc(self.conv(x))

    def act(self, obs):
        with torch.no_grad():
            obs = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0)
            logits = self.forward(obs)
            return torch.argmax(logits, dim=1).item()

def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    pointer = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[pointer:pointer + numel].view_as(p))
        pointer += numel

def make_env():
    env = gym.make("ALE/Pong-v5", frameskip=1)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    env = FrameStack(env, 4)
    return env

# Función auxiliar para evaluar una perturbación (para multiprocessing)
def evaluate_candidate(args):
    env_fn, base_params, noise_i, sigma = args
    env = env_fn()
    obs = env.reset()[0]
    model = CNNPolicy(env.action_space)
    perturbed = base_params + sigma * noise_i
    set_flat_params(model, perturbed)
    total_reward = 0
    done = False
    while not done:
        action = model.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    env.close()
    return total_reward

# Algoritmo Evolution Strategies con paralelización
def evolution_strategy(env_fn, model, iterations=50, pop_size=20, sigma=0.02, alpha=0.03):
    theta = get_flat_params(model)
    reward_history = []

    for iter in range(iterations):
        noise = [torch.randn_like(theta) for _ in range(pop_size)]
        args_list = [(env_fn, theta, eps, sigma) for eps in noise]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            rewards = pool.map(evaluate_candidate, args_list)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        A = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        gradient = sum([a * eps for a, eps in zip(A, noise)]) / pop_size
        theta += alpha * gradient

        # Evaluación del modelo actual sin ruido
        test_reward = evaluate_candidate((env_fn, theta, torch.zeros_like(theta), 0.0))
        reward_history.append(test_reward)
        print(f"Iteración {iter + 1}: Recompensa promedio = {test_reward:.2f}")

    set_flat_params(model, theta)
    return model, reward_history

# Grabar video del agente entrenado
def record_video(env_fn, policy, path="C:\\Users\\Usuario\\Downloads\\pong_es_parallel.mp4"):
    env = env_fn()
    obs = env.reset()[0]
    frames = []
    done = False
    while not done:
        frame = env.render()
        frames.append(frame)
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    imageio.mimsave(path, frames, fps=30)
    print(f"🎥 Video guardado en {path}")

# Graficar recompensas por iteración
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.title("Recompensa por iteración (Pong)")
    plt.xlabel("Iteración")
    plt.ylabel("Recompensa")
    plt.grid(True)
    plt.savefig("rewards_pong_para.png")
    plt.show()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Requerido en Windows
    env_fn = make_env
    dummy_env = env_fn()
    model = CNNPolicy(dummy_env.action_space)
    trained_model, rewards = evolution_strategy(env_fn, model, iterations=100)
    plot_rewards(rewards)
    record_video(env_fn, trained_model)