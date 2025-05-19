import gym
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
from gym.wrappers import AtariPreprocessing, FrameStack
import multiprocessing as mp

# --- Definición de la política MLP simple ---
class Policy:
    def __init__(self, input_size, hidden_size, output_size):
        self.shapes = [(input_size, hidden_size), (hidden_size,),
                       (hidden_size, hidden_size), (hidden_size,),
                       (hidden_size, output_size), (output_size,)]
        self.params = self._init_params()

    def _init_params(self):
        return [np.random.randn(*shape) * 0.05 for shape in self.shapes]

    def set_params(self, params):
        self.params = params

    def get_params(self):
        return self.params

    def forward(self, x):
        x = np.array(x, dtype=np.float32).flatten()  # Asegura vector plano
        w1, b1, w2, b2, w3, b3 = self.params
        x = np.tanh(x @ w1 + b1)
        x = np.tanh(x @ w2 + b2)
        return x @ w3 + b3  # No tanh final, logits para acción discreta

    def act(self, obs):
        logits = self.forward(obs)
        return np.argmax(logits)  # Acción discreta

# --- Wrappers para entorno Pong ---
def make_env():
    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    env = FrameStack(env, 4)
    return env

# --- Evaluación de un candidato ---
def evaluate_candidate(args):
    env_name, base_params, noise_i, sigma = args
    env = make_env()
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    policy = Policy(np.prod(obs_shape), 64, n_actions)
    new_params = [p + sigma * n for p, n in zip(base_params, noise_i)]
    policy.set_params(new_params)

    obs = env.reset()[0]
    total_reward = 0
    done = False
    step_count = 0
    while not done:
        if step_count < 10000:
            action = env.action_space.sample()
        else:
            action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        if reward > 0:
            reward *= 50
        total_reward += reward
        done = terminated or truncated
        step_count += 1
    env.close()
    return total_reward

# --- Algoritmo Evolution Strategies secuencial ---
def evolution_strategy(env_name, pop_size=40, sigma=0.1, alpha=0.03, iterations=100):
    obs_shape = make_env().observation_space.shape
    n_actions = make_env().action_space.n
    policy = Policy(np.prod(obs_shape), 64, n_actions)
    reward_history = []

    for iteration in range(iterations):
        params = policy.get_params()
        noise = [[np.random.randn(*p.shape) for p in params] for _ in range(pop_size)]
        args_list = [(env_name, params, eps, sigma) for eps in noise]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            rewards = pool.map(evaluate_candidate, args_list)

        rewards = np.array(rewards)
        A = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        for i in range(len(params)):
            update = sum(A[j] * noise[j][i] for j in range(pop_size))
            params[i] += alpha / (pop_size * sigma) * update
        policy.set_params(params)
        # Evaluación del modelo actual
        test_reward = evaluate_candidate((env_name, params, [np.zeros_like(p) for p in params], 0.0))
        reward_history.append(test_reward)
        print(f"Iteración {iteration + 1}/{iterations}: Recompensa promedio = {test_reward:.2f}")
    return policy, reward_history

# --- Evaluación de la política entrenada ---
def evaluate_policy(policy, env, num_episodes=10):
    rewards = []
    for _ in range(num_episodes):
        obs = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return rewards

# --- Grabar video ---
def record_video(policy, env, path="es_pong.mp4", max_frames=500):
    obs = env.reset()[0]
    frames = []
    done = False
    count = 0
    step_count = 0
    while not done and count < max_frames:
        frame = env.render()
        frames.append(frame)
        if step_count < 10000:
            action = env.action_space.sample()
        else:
            action = policy.act(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        count += 1
        step_count += 1
    env.close()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=30)
    print(f"Video guardado en {path}")

# --- Main ---
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Requerido en Windows
    env_name = "ALE/Pong-v5"
    policy, rewards = evolution_strategy(env_name, pop_size=50, sigma=0.1, alpha=0.03, iterations=1000)
    # Evaluar desempeño
    eval_rewards = evaluate_policy(policy, make_env())
    avg_reward = sum(eval_rewards) / len(eval_rewards)
    print(f"Recompensa promedio en evaluación: {avg_reward:.2f}")

    # Graficar recompensas
    plt.plot(rewards, marker='o')
    plt.title("Recompensa por iteración (ES Pong)")
    plt.xlabel("Iteración")
    plt.ylabel("Recompensa total")
    plt.grid(True)
    plt.savefig("rewards_es_pong.png")
    plt.show()

    # Grabar video
    record_video(policy, make_env(), path="C:/Users/Franchesco/Desktop/challenge/results/es_pong.mp4")