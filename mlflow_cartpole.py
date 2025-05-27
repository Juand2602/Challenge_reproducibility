#!/usr/bin/env python3
import os
import gym
import numpy as np
import imageio
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Política MLP para entornos discretos
class Policy:
    def __init__(self, input_size, hidden_size, output_size):
        self.shapes = [
            (input_size, hidden_size),
            (hidden_size,),
            (hidden_size, output_size),
            (output_size,)
        ]
        self.params = [np.random.randn(*s) * 0.1 for s in self.shapes]

    def set_params(self, params):
        self.params = params

    def get_params(self):
        return self.params

    def forward(self, x):
        x = np.array(x, dtype=np.float32)
        w1, b1, w2, b2 = self.params
        h = np.tanh(x @ w1 + b1)
        logits = h @ w2 + b2
        return logits

    def act(self, obs):
        logits = self.forward(obs)
        probs = np.exp(logits - np.max(logits))  # para estabilidad numérica
        probs /= np.sum(probs)
        return int(np.argmax(probs))

# Evaluación sin render (solo retornos)
def evaluate_noise(noise_i, env_name, sigma, base_params):
    env = gym.make(env_name)
    policy = Policy(env.observation_space.shape[0], 32, env.action_space.n)
    policy.set_params([p + sigma * n for p, n in zip(base_params, noise_i)])
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    env.close()
    return total_reward

def save_results(policy, rewards, path="/home/hadoop/MLFLOW_taller/results"):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "params.npy"), policy.get_params())
    np.save(os.path.join(path, "rewards.npy"), rewards)
    print(f"✅ Resultados guardados en {path}")

def plot_rewards(rewards, path="/home/hadoop/MLFLOW_taller/rewards_plot.png"):
    plt.figure()
    plt.plot(rewards, marker='o')
    plt.xlabel("Iteración")
    plt.ylabel("Recompensa promedio")
    plt.title("Entrenamiento ES - CartPole")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"📈 Gráfica guardada en {path}")

def record_video(env_name, policy, video_path="/home/hadoop/MLFLOW_taller/cartpole_video.mp4"):
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []
    done = False
    while not done:
        frame = env.render()
        frames.append(frame)
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    imageio.mimsave(video_path, frames, fps=30)
    print(f"🎥 Video guardado en {video_path}")

def evolution_strategy(env_name='CartPole-v1', pop_size=50, sigma=0.1, alpha=0.01, iterations=5):
    mlflow.log_param("env_name", env_name)
    mlflow.log_param("pop_size", pop_size)
    mlflow.log_param("sigma", sigma)
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("iterations", iterations)

    env = gym.make(env_name)
    policy = Policy(env.observation_space.shape[0], 32, env.action_space.n)
    env.close()

    reward_history = []

    for iteration in range(iterations):
        base_params = policy.get_params()
        noise = [[np.random.randn(*p.shape) for p in base_params] for _ in range(pop_size)]
        rewards = np.array([evaluate_noise(noise_i, env_name, sigma, base_params) for noise_i in noise])

        A = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

        for i in range(len(base_params)):
            update = sum(A[j] * noise[j][i] for j in range(pop_size))
            base_params[i] += alpha / (pop_size * sigma) * update

        policy.set_params(base_params)

        # Evaluación sin ruido para medir progreso
        test_reward = evaluate_noise([np.zeros_like(p) for p in base_params], env_name, 0.0, base_params)
        reward_history.append(test_reward)

        mlflow.log_metric("test_reward", test_reward, step=iteration)
        print(f"Iteración {iteration + 1}/{iterations}: Recompensa promedio = {test_reward:.2f}")

    save_results(policy, reward_history)
    plot_rewards(reward_history)
    record_video(env_name, policy)

    # Log artifacts con MLflow
    mlflow.log_artifact("/home/hadoop/MLFLOW_taller/results/params.npy")
    mlflow.log_artifact("/home/hadoop/MLFLOW_taller/results/rewards.npy")
    mlflow.log_artifact("/home/hadoop/MLFLOW_taller/rewards_plot.png")
    mlflow.log_artifact("/home/hadoop/MLFLOW_taller/cartpole_video.mp4")

    return policy, reward_history

if __name__ == "__main__":
    mlflow.set_experiment("ES_CartPole")

    with mlflow.start_run():
        policy, rewards = evolution_strategy(
            env_name="CartPole-v1",
            pop_size=50,
            sigma=0.1,
            alpha=0.01,
            iterations=5
        )

        mlflow.sklearn.log_model(policy, "model", registered_model_name="ES_CartPole_Model")
        client = MlflowClient()
        # Asumiendo que la versión 1 ya existe; si no, maneja la versión dinámicamente
        client.transition_model_version_stage(name="ES_CartPole_Model", version=1, stage="Staging")
