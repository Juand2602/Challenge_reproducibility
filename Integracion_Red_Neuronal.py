import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import concurrent.futures
import time

# --- Funciones Auxiliares Esenciales ---
def compute_centered_ranks(x):
    """Calcula rangos y los escala a [-0.5, 0.5]."""
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    y = ranks.astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

# --- Red Neuronal de Política (MLP Simple) ---
class SimpleMLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=32, is_continuous=False, action_scale=None, action_bias=None):
        super(SimpleMLPPolicy, self).__init__()
        self.is_continuous = is_continuous
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, act_dim)

        if is_continuous:
            self.action_scale = torch.tensor(action_scale if action_scale is not None else 1.0, dtype=torch.float32)
            self.action_bias = torch.tensor(action_bias if action_bias is not None else 0.0, dtype=torch.float32)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output_logits = self.fc_out(x)
        if self.is_continuous:
            action_scaled = torch.tanh(output_logits)
            action_scaled = action_scaled * self.action_scale.to(x.device) + self.action_bias.to(x.device)
            return action_scaled
        return output_logits

    def get_action(self, observation, device="cpu"):
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        observation = observation.to(device)
        self.to(device)
        with torch.no_grad():
            output = self.forward(observation)
            if self.is_continuous:
                action = output.cpu().numpy().flatten()
            else:
                action = torch.argmax(output, dim=1).cpu().item()
        return action

    def get_flattened_parameters(self):
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_flattened_parameters(self, flat_params):
        offset = 0
        for param in self.parameters():
            param_shape = param.shape
            param_size = np.prod(param_shape)
            param_data_np = flat_params[offset : offset + param_size].reshape(param_shape)
            param.data = torch.from_numpy(param_data_np.astype(np.float32)).to(param.device)
            offset += param_size

# --- Agente de Estrategias Evolutivas (ES) ---
class EvolutionStrategyAgent:
    def __init__(self, policy_network, env_name, learning_rate=0.01,
                 noise_std_dev=0.1, population_size=50, weight_decay_coeff=0.005,
                 fitness_shaping=True, seed=42, device="cpu"):
        self.policy_network = policy_network
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.noise_std_dev = noise_std_dev
        self.population_size = population_size
        if self.population_size % 2 != 0: # Asegurar par para antithetic sampling
            self.population_size += 1
            print(f"Tamaño de población ajustado a {self.population_size} para ser par.")
        self.weight_decay_coeff = weight_decay_coeff
        self.use_fitness_shaping = fitness_shaping
        self.seed = seed
        self.device = device
        self.policy_network.to(self.device)
        self.num_params = len(self.policy_network.get_flattened_parameters())
        self.rng = np.random.default_rng(seed)
        print(f"Agente ES inicializado con {self.num_params} parámetros en '{self.env_name}'.")

    def _evaluate_fitness(self, parameters_to_evaluate, eval_seed):
        eval_env = gym.make(self.env_name) # Crear nueva instancia para thread-safety
        # Es crucial sembrar el entorno para CADA episodio de evaluación si quieres reproducibilidad
        # en la evaluación de un individuo específico.
        obs, _ = eval_env.reset(seed=eval_seed) 
        
        # Crear una copia de la red para esta evaluación o asegurar que set_parameters sea seguro
        # Para este ejemplo, asumimos que set_flattened_parameters en la red principal es seguro
        # si get_action no modifica el estado interno de la red más allá de la inferencia.
        # Una solución más robusta para paralelismo de procesos sería que cada worker
        # tuviera su propia instancia de la red.
        current_main_params = self.policy_network.get_flattened_parameters() # Guardar params originales
        self.policy_network.set_flattened_parameters(parameters_to_evaluate) # Cargar params del individuo

        total_reward = 0.0
        terminated, truncated = False, False
        # Obtener max_episode_steps del entorno si está disponible, o usar un default
        max_steps = getattr(eval_env, '_max_episode_steps', 
                            getattr(eval_env.spec, 'max_episode_steps', 500) if eval_env.spec else 500)

        for _step in range(max_steps):
            action = self.policy_network.get_action(obs, device=self.device)
            try:
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            except Exception as e:
                print(f"Error en env.step con acción {action}: {e}")
                total_reward = -float('inf') # Penalizar error
                break
        
        self.policy_network.set_flattened_parameters(current_main_params) # Restaurar params originales
        eval_env.close()
        return total_reward

    def evaluate_population(self):
        current_params_original = self.policy_network.get_flattened_parameters()
        all_fitnesses = np.zeros(self.population_size)
        all_perturbations = np.zeros((self.population_size, self.num_params))
        num_base_perturbations = self.population_size // 2
        base_epsilons = self.rng.standard_normal((num_base_perturbations, self.num_params))

        tasks = []
        for i in range(num_base_perturbations):
            epsilon = base_epsilons[i]
            # Semilla para el episodio de evaluación: se basa en la semilla del agente y el índice
            # para asegurar que diferentes perturbaciones (y sus antitéticas) tengan diferentes
            # secuencias de episodios si el entorno es estocástico.
            eval_seed_base = self.seed + self.current_generation * self.population_size 
            
            tasks.append({'params': (current_params_original + self.noise_std_dev * epsilon).copy(), 
                          'idx': 2 * i, 'epsilon': epsilon, 'eval_seed': eval_seed_base + 2*i})
            tasks.append({'params': (current_params_original - self.noise_std_dev * epsilon).copy(), 
                          'idx': 2 * i + 1, 'epsilon': -epsilon, 'eval_seed': eval_seed_base + 2*i + 1})
        
        # Usar ThreadPoolExecutor para paralelizar las evaluaciones en hilos
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.population_size, 10)) as executor: # Limitar workers
            future_to_task_info = {
                executor.submit(self._evaluate_fitness, task['params'], task['eval_seed']): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task_info):
                task_info = future_to_task_info[future]
                idx, epsilon_for_this_eval = task_info['idx'], task_info['epsilon']
                all_perturbations[idx, :] = epsilon_for_this_eval
                try:
                    all_fitnesses[idx] = future.result()
                except Exception: # Capturar cualquier excepción durante la evaluación
                    all_fitnesses[idx] = -np.inf 
        return all_fitnesses, all_perturbations

    def update_policy(self, all_fitnesses, all_perturbations):
        if self.use_fitness_shaping:
            shaped_fitnesses = compute_centered_ranks(all_fitnesses)
        else:
            # Normalizar fitness crudos puede ayudar si no se usa shaping
            mean_fit = np.mean(all_fitnesses)
            std_fit = np.std(all_fitnesses) + 1e-8 # Evitar división por cero
            shaped_fitnesses = (all_fitnesses - mean_fit) / std_fit
            
        weighted_sum_perturbations = np.dot(shaped_fitnesses, all_perturbations)
        es_gradient_estimate = (1 / (self.population_size * self.noise_std_dev)) * weighted_sum_perturbations
        current_params = self.policy_network.get_flattened_parameters()
        update_step = self.learning_rate * es_gradient_estimate
        weight_decay_term = self.learning_rate * self.weight_decay_coeff * current_params
        new_params = current_params + update_step - weight_decay_term
        self.policy_network.set_flattened_parameters(new_params)

    def train(self, num_generations, print_every=10):
        total_start_time = time.time()
        self.current_generation = 0 # Para sembrado de evaluación
        for generation in range(num_generations):
            self.current_generation = generation
            gen_start_time = time.time()
            all_fitnesses, all_perturbations = self.evaluate_population()
            self.update_policy(all_fitnesses, all_perturbations)

            if (generation + 1) % print_every == 0 or generation == 0:
                mean_fitness = np.mean(all_fitnesses[np.isfinite(all_fitnesses)]) # Ignorar -inf si los hubo
                std_fitness = np.std(all_fitnesses[np.isfinite(all_fitnesses)])
                max_fitness = np.max(all_fitnesses[np.isfinite(all_fitnesses)]) if np.any(np.isfinite(all_fitnesses)) else -np.inf
                gen_elapsed_time = time.time() - gen_start_time
                print(f"Gen: {generation + 1}/{num_generations}, "
                      f"Fit (Mean±Std): {mean_fitness:.2f}±{std_fitness:.2f}, "
                      f"Max: {max_fitness:.2f}, T_gen: {gen_elapsed_time:.1f}s")
        print(f"Entrenamiento completado en {time.time() - total_start_time:.2f}s.")

# --- Punto de Entrada Principal del Script ---
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {DEVICE}")

    # --- Configuración del Experimento ---
    # Cambia ENV_NAME para probar diferentes entornos
    # ENV_NAME_TO_RUN = "CartPole-v1"
    ENV_NAME_TO_RUN = "Pendulum-v1"

    # Hiperparámetros (ajusta estos valores)
    NUM_TRAINING_GENERATIONS = 100
    POPULATION_SIZE = 50
    LEARNING_RATE = 0.05
    NOISE_STD_DEV = 0.1
    WEIGHT_DECAY = 0.001
    RANDOM_SEED = 42
    PRINT_INTERVAL = 5
    HIDDEN_LAYER_DIM = 32 # Para la SimpleMLPPolicy

    # Ajustes específicos por entorno (ejemplos)
    if ENV_NAME_TO_RUN == "CartPole-v1":
        LEARNING_RATE = 0.1
        NOISE_STD_DEV = 0.1
        NUM_TRAINING_GENERATIONS = 50
        HIDDEN_LAYER_DIM = 32
    elif ENV_NAME_TO_RUN == "Pendulum-v1":
        LEARNING_RATE = 0.02
        NOISE_STD_DEV = 0.05
        NUM_TRAINING_GENERATIONS = 200 # Pendulum necesita más entrenamiento
        POPULATION_SIZE = 64
        HIDDEN_LAYER_DIM = 64 # Red un poco más grande para Pendulum

    # --- Preparación del Entorno y la Red ---
    temp_env = gym.make(ENV_NAME_TO_RUN)
    obs_dim = temp_env.observation_space.shape[0]
    act_scale, act_bias, is_cont = None, None, False
    if isinstance(temp_env.action_space, gym.spaces.Discrete):
        act_dim = temp_env.action_space.n
    elif isinstance(temp_env.action_space, gym.spaces.Box):
        act_dim = temp_env.action_space.shape[0]
        is_cont = True
        act_scale = temp_env.action_space.high # Usar el límite superior como escala si tanh produce [-1,1]
        # Para Pendulum: action_space.high es [2.], action_space.low es [-2.]
        # Si la red produce [-1,1] con tanh, multiplicar por 2.0.
        act_scale = temp_env.action_space.high[0] if act_dim == 1 else temp_env.action_space.high
        act_bias = 0.0 # Asumiendo que el rango es simétrico alrededor de 0 después de tanh
    temp_env.close()
    
    print(f"Entorno: {ENV_NAME_TO_RUN}, Obs dim: {obs_dim}, Act dim: {act_dim}, Continuo: {is_cont}")

    policy_network_instance = SimpleMLPPolicy(
        obs_dim, act_dim, hidden_dim=HIDDEN_LAYER_DIM, 
        is_continuous=is_cont, action_scale=act_scale, action_bias=act_bias
    ).to(DEVICE)

    # --- Creación y Entrenamiento del Agente ---
    es_agent_instance = EvolutionStrategyAgent(
        policy_network=policy_network_instance,
        env_name=ENV_NAME_TO_RUN,
        learning_rate=LEARNING_RATE,
        noise_std_dev=NOISE_STD_DEV,
        population_size=POPULATION_SIZE,
        weight_decay_coeff=WEIGHT_DECAY,
        fitness_shaping=True,
        seed=RANDOM_SEED,
        device=DEVICE
    )
    es_agent_instance.train(num_generations=NUM_TRAINING_GENERATIONS, print_every=PRINT_INTERVAL)

    # --- Evaluación Final (Opcional pero Recomendado) ---
    print("\nEvaluando política final entrenada...")
    final_rewards_list = []
    num_final_eval_episodes = 10
    final_trained_params = es_agent_instance.policy_network.get_flattened_parameters().copy()
    for i in range(num_final_eval_episodes):
        eval_run_seed = RANDOM_SEED + NUM_TRAINING_GENERATIONS + i # Semillas diferentes para cada eval
        reward = es_agent_instance._evaluate_fitness(final_trained_params, eval_seed=eval_run_seed)
        final_rewards_list.append(reward)
        print(f"  Evaluación final {i+1}/{num_final_eval_episodes}: Recompensa = {reward:.2f}")
    
    if final_rewards_list:
        mean_final_reward = np.mean(final_rewards_list)
        std_final_reward = np.std(final_rewards_list)
        print(f"Recompensa promedio final en {num_final_eval_episodes} episodios: {mean_final_reward:.2f} ± {std_final_reward:.2f}")
