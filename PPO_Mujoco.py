import gym
from stable_baselines3 import PPO
import torch
import matplotlib.pyplot as plt
import imageio
import os

def evaluate_policy(model, env, num_episodes=20):
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return rewards

def record_video(model, env, path="C:/Users/Franchesco/Desktop/challenge/results/ppo.mp4", max_frames=1000):
    obs, _ = env.reset()
    frames = []
    done = False
    count = 0
    while not done and count < max_frames:
        frame = env.render()
        frames.append(frame)
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        count += 1
    env.close()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=30)
    print(f"Video guardado en {path}")

if __name__ == "__main__":
    env = gym.make("HalfCheetah-v4", render_mode="rgb_array")

    # Forzar el uso de CPU
    device = "cpu"
    print(f"Usando dispositivo: {device}")

    # Crear modelo PPO en CPU
    model = PPO("MlpPolicy", env, verbose=1, device=device)

    # Entrenar (5 millones de pasos como ejemplo)
    model.learn(total_timesteps=5_000_000)

    # Guardar modelo entrenado
    model.save("ppo_halfcheetah_cpu")

    # Evaluar
    rewards = evaluate_policy(model, env, num_episodes=20)
    avg_reward = sum(rewards) / len(rewards)
    print(f"Recompensa promedio en evaluación: {avg_reward:.2f}")

    # Graficar recompensas
    plt.plot(rewards, marker='o')
    plt.title("Recompensa por episodio (evaluación PPO)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa total")
    plt.grid(True)
    plt.show()

    # Grabar video
    record_video(model, gym.make("HalfCheetah-v4", render_mode="rgb_array"))