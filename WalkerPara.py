import gym
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import multiprocessing as mp

# Política MLP simple
class Policy:
    def __init__(self, input_size, hidden_size, output_size):
        self.shapes = [(input_size, hidden_size), (hidden_size,),
                       (hidden_size, hidden_size), (hidden_size,),
                       (hidden_size, output_size), (output_size,)]
        self.params = self._init_params()

    def _init_params(self):
        return [np.random.randn(*shape) * 0.1 for shape in self.shapes]

    def set_params(self, params):
        self.params = params

    def get_params(self):
        return self.params

    def forward(self, x):
        x = np.array(x, dtype=np.float32)
        w1, b1, w2, b2, w3, b3 = self.params
        x = np.tanh(x @ w1 + b1)
        x = np.tanh(x @ w2 + b2)
        return np.tanh(x @ w3 + b3)

    def act(self, obs):
        obs = np.array(obs, dtype=np.float32)
        return self.forward(obs)


# Evaluación individual (para multiprocessing)
def evaluate_candidate(args):
    env_name, base_params, noise_i, sigma = args
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = Policy(obs_dim, 32, act_dim)

    new_params = [p + sigma * n for p, n in zip(base_params, noise_i)]
    policy.set_params(new_params)

    total_reward = 0
    obs, _ = env.reset()
    done = False
    initial_x = env.unwrapped.data.qpos[0]
    initial_z = env.unwrapped.data.qpos[1]
    last_leg_positions = None
    
    # Secuencia de impulsos alternados para iniciar el patrón de caminata
    def apply_leg_impulse(leg_index, strength=0.5):
        action = np.zeros(act_dim)
        action[leg_index] = strength  # Flexión de la pierna
        action[leg_index + 1] = strength * 0.6  # Movimiento hacia adelante
        return action

    # Primero movemos la pierna izquierda
    for _ in range(3):
        obs, _, _, _, _ = env.step(apply_leg_impulse(2, 0.4))
    
    # Pequeña pausa para estabilidad
    obs, _, _, _, _ = env.step(np.zeros(act_dim))
    
    # Luego la pierna derecha
    for _ in range(3):
        obs, _, _, _, _ = env.step(apply_leg_impulse(4, 0.4))
    
    # Otra pequeña pausa
    obs, _, _, _, _ = env.step(np.zeros(act_dim))
    
    last_x = env.unwrapped.data.qpos[0]
    steps_without_progress = 0
    last_leg_phase = 0
    base_height = initial_z
    
    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        current_x = env.unwrapped.data.qpos[0]
        current_z = env.unwrapped.data.qpos[1]
        velocity = current_x - last_x
        
        # Obtener posiciones de las piernas
        leg_positions = env.unwrapped.data.qpos[2:8]
        
        # Recompensa base por mantenerse en pie y no caerse
        survival_reward = 1.0 if not done else 0.0
        
        # Recompensas por movimiento horizontal (más suaves)
        forward_reward = max(0, velocity * 2.0)  # Solo recompensamos movimiento hacia adelante
        distance_reward = max(0, (current_x - initial_x))  # Recompensa acumulativa por distancia
        
        # Penalización suave por altura excesiva
        height_diff = current_z - base_height
        height_penalty = -max(0, abs(height_diff) - 0.1) * 1.5
        
        # Sistema mejorado de recompensa por movimiento de piernas
        leg_movement_reward = 0
        if last_leg_positions is not None:
            leg_diff = leg_positions - last_leg_positions
            
            # Calcular movimiento de cada pierna
            left_leg_moving = abs(leg_diff[0]) + abs(leg_diff[1])
            right_leg_moving = abs(leg_diff[2]) + abs(leg_diff[3])
            
            # Recompensa por movimiento alternado
            if left_leg_moving > 0.02 and right_leg_moving < 0.01:
                if last_leg_phase == 2:  # Si la última vez se movió la pierna derecha
                    leg_movement_reward += 2.0  # Bonus grande por alternancia correcta
                last_leg_phase = 1
            elif right_leg_moving > 0.02 and left_leg_moving < 0.01:
                if last_leg_phase == 1:  # Si la última vez se movió la pierna izquierda
                    leg_movement_reward += 2.0  # Bonus grande por alternancia correcta
                last_leg_phase = 2
            
            # Penalización más suave por movimiento simultáneo
            if left_leg_moving > 0.02 and right_leg_moving > 0.02:
                leg_movement_reward -= 0.5
        
        # Penalización por quedarse quieto (más suave)
        if abs(velocity) < 0.01:
            steps_without_progress += 1
            if steps_without_progress > 30:  # Más tolerancia antes de penalizar
                forward_reward -= 0.5
        else:
            steps_without_progress = 0
        
        # Combinación más equilibrada de recompensas
        modified_reward = (
            survival_reward +  # Recompensa base por sobrevivir
            forward_reward +   # Recompensa por velocidad
            distance_reward +  # Recompensa por distancia total
            height_penalty +   # Penalización suave por saltos
            leg_movement_reward * 1.5  # Aumentamos la importancia del movimiento alternado
        )
        
        done = terminated or truncated
        total_reward += modified_reward
        last_x = current_x
        last_leg_positions = leg_positions.copy()

    env.close()
    return total_reward


# Algoritmo ES con paralelización
def evolution_strategy(env_name='HalfCheetah-v4', pop_size=100, sigma=0.1, alpha=0.01, iterations=200):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = Policy(obs_dim, 32, act_dim)
    reward_history = []

    for iteration in range(iterations):
        params = policy.get_params()
        noise = [[np.random.randn(*p.shape) for p in params] for _ in range(pop_size)]

        # Preparar argumentos para evaluación paralela
        args_list = [(env_name, params, eps, sigma) for eps in noise]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            rewards = pool.map(evaluate_candidate, args_list)

        rewards = np.array(rewards)
        A = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

        # Actualizar parámetros
        for i in range(len(params)):
            update = sum(A[j] * noise[j][i] for j in range(pop_size))
            params[i] += alpha / (pop_size * sigma) * update

        policy.set_params(params)

        # Evaluación del modelo actual
        test_reward = evaluate_candidate((env_name, params, [np.zeros_like(p) for p in params], 0.0))
        reward_history.append(test_reward)
        print(f"Iteración {iteration + 1}/{iterations}: Recompensa promedio = {test_reward:.2f}")

    env.close()
    return policy, reward_history


def record_video(env_name, policy, video_path="C:\\Users\\Franchesco\\Desktop\\challenge\\results\\HopperPara.mp4"):
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


def save_results(policy, rewards, path="C:\\Users\\Franchesco\\Desktop\\challenge\\results"):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "paramsHooperPara.npy"), policy.get_params())
    np.save(os.path.join(path, "rewardsHopperPara.npy"), rewards)
    print(f"✅ Resultados guardados en carpeta: {path}")


def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Iteración")
    plt.ylabel("Recompensa promedio")
    plt.title("Entrenamiento ES en MuJoCo")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("HopperPara.png")
    plt.show()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Requerido en Windows
    env_name = "Walker2d-v4"  # Puedes cambiar por HalfCheetah-v4, Walker2d-v4, etc.
    policy, rewards = evolution_strategy(env_name=env_name, iterations=50)
    save_results(policy, rewards)
    plot_rewards(rewards)
    record_video(env_name, policy)