import numpy as np
import gymnasium as gym
import concurrent.futures
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Funciones Auxiliares (del código anterior de Fase 1) ---
def compute_ranks(x):
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

# --- Implementación de la Red Neuronal de Política con PyTorch ---
class SimpleMLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64, is_continuous=False, action_scale=None, action_bias=None):
        super(SimpleMLPPolicy, self).__init__()
        self.is_continuous = is_continuous
        self.act_dim = act_dim

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, act_dim)

        # Para acciones continuas, podemos necesitar escalar la salida
        if is_continuous:
            self.action_scale = torch.tensor(action_scale if action_scale is not None else 1.0, dtype=torch.float32)
            self.action_bias = torch.tensor(action_bias if action_bias is not None else 0.0, dtype=torch.float32)
        
        # Inicialización de pesos (opcional, pero puede ayudar)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0) # Inicializar biases a cero

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Usar ReLU como activación
        x = F.relu(self.fc2(x))
        output_logits = self.fc_out(x)

        if self.is_continuous:
            # Para acciones continuas, a menudo se usa tanh y luego se escala
            action_scaled = torch.tanh(output_logits)
            action_scaled = action_scaled * self.action_scale.to(x.device) + self.action_bias.to(x.device)
            return action_scaled
        else:
            # Para acciones discretas, la salida son logits
            return output_logits

    def get_action(self, observation, device="cpu"):
        """
        Obtiene una acción de la política para una observación dada.
        Para ES, usualmente tomamos la acción de forma determinista basada en los parámetros actuales.
        """
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        
        # Asegurar que la observación tenga la forma (batch_size, obs_dim)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        
        observation = observation.to(device)
        self.to(device) # Asegurar que la red esté en el dispositivo correcto

        with torch.no_grad(): # No necesitamos gradientes durante la actuación
            output = self.forward(observation)
            if self.is_continuous:
                action = output.cpu().numpy().flatten()
            else:
                # Para acciones discretas, tomar argmax de los logits
                action = torch.argmax(output, dim=1).cpu().item()
        return action

    def get_flattened_parameters(self):
        """Obtiene los parámetros de la red como un vector 1D NumPy."""
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def set_flattened_parameters(self, flat_params):
        """Establece los parámetros de la red desde un vector 1D NumPy."""
        offset = 0
        for param in self.parameters():
            param_shape = param.shape
            param_size = np.prod(param_shape)
            # Asegurar que el tipo de dato coincida
            param_data_np = flat_params[offset : offset + param_size].reshape(param_shape)
            param.data = torch.from_numpy(param_data_np.astype(np.float32)).to(param.device)
            offset += param_size

# --- Clase del Agente ES (Modificada para usar la red real) ---
class EvolutionStrategyAgent:
    def __init__(self, policy_network, env_name,
                 learning_rate=0.01,
                 noise_std_dev=0.02,
                 population_size=30,
                 weight_decay_coeff=0.005,
                 fitness_shaping=True,
                 seed=42,
                 device="cpu"): # Añadir dispositivo
        self.policy_network = policy_network # Ahora es una instancia de SimpleMLPPolicy
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.noise_std_dev = noise_std_dev
        self.population_size = population_size
        self.weight_decay_coeff = weight_decay_coeff
        self.use_fitness_shaping = fitness_shaping
        self.seed = seed
        self.device = device # Guardar el dispositivo

        self.policy_network.to(self.device) # Mover la red al dispositivo especificado

        if self.population_size % 2 != 0:
            print(f"Advertencia: population_size ({self.population_size}) no es par. Incrementando a {self.population_size + 1}.")
            self.population_size += 1

        # Obtener el número de parámetros de la red real
        self.num_params = len(self.policy_network.get_flattened_parameters())
        print(f"Número de parámetros en la política: {self.num_params}")

        self.rng = np.random.default_rng(seed)

    def _evaluate_fitness(self, parameters_to_evaluate, eval_seed):
        """
        Evalúa el fitness (retorno) de un conjunto de parámetros dado en el entorno.
        Se pasa una copia de los parámetros para evitar problemas de concurrencia.
        """
        # Crear una instancia del entorno para esta evaluación
        eval_env = gym.make(self.env_name)
        eval_env.reset(seed=eval_seed) # Usar semilla para la evaluación del episodio

        # Crear una copia temporal de la red o simplemente establecer sus parámetros
        # Para evitar problemas con el estado de la red principal si se usa en paralelo,
        # es más seguro que cada "evaluación" tenga su propia instancia o
        # al menos que la configuración de parámetros sea atómica para esa evaluación.
        # Aquí, estamos estableciendo los parámetros en la red del agente,
        # lo cual está bien para ThreadPoolExecutor si la red no tiene estado interno complejo
        # más allá de sus pesos, y get_action es thread-safe.
        self.policy_network.set_flattened_parameters(parameters_to_evaluate)

        observation, info = eval_env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False
        
        max_steps_per_episode = getattr(eval_env, '_max_episode_steps', 500) # Límite de pasos
        current_steps = 0

        while not terminated and not truncated and current_steps < max_steps_per_episode:
            action = self.policy_network.get_action(observation, device=self.device)
            
            # Si la acción es un array NumPy (para continuo) y el espacio de acción espera un float
            if isinstance(action, np.ndarray) and eval_env.action_space.shape == ():
                 action_to_step = action.item() # Tomar el escalar
            elif isinstance(action, np.ndarray) and len(action) == 1 and eval_env.action_space.shape == ():
                 action_to_step = action[0]
            else:
                 action_to_step = action

            try:
                observation, reward, terminated, truncated, info = eval_env.step(action_to_step)
                total_reward += reward
                current_steps += 1
            except Exception as e:
                print(f"Error durante env.step con acción {action_to_step}: {e}")
                terminated = True # Terminar el episodio si hay un error
                total_reward = -float('inf') # Penalizar fuertemente

        eval_env.close()
        return total_reward

    def evaluate_population(self):
        current_params_original = self.policy_network.get_flattened_parameters()
        all_fitnesses = np.zeros(self.population_size)
        all_perturbations = np.zeros((self.population_size, self.num_params))

        num_base_perturbations = self.population_size // 2
        base_perturbations = self.rng.standard_normal((num_base_perturbations, self.num_params))

        # Preparar argumentos para los workers del pool
        tasks = []
        for i in range(num_base_perturbations):
            epsilon = base_perturbations[i]
            
            # Perturbación positiva
            idx_pos = 2 * i
            params_pos = current_params_original + self.noise_std_dev * epsilon
            # Pasar una copia de los parámetros para evitar modificaciones concurrentes
            tasks.append({'params': params_pos.copy(), 'idx': idx_pos, 'epsilon': epsilon, 'eval_seed': self.seed + idx_pos})
            
            # Perturbación negativa (antithetic)
            idx_neg = 2 * i + 1
            params_neg = current_params_original - self.noise_std_dev * epsilon
            tasks.append({'params': params_neg.copy(), 'idx': idx_neg, 'epsilon': -epsilon, 'eval_seed': self.seed + idx_neg})

        # Usar ThreadPoolExecutor para paralelizar las evaluaciones
        # Para una paralelización más robusta entre procesos, se usaría multiprocessing.Pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.population_size) as executor:
            # Mapear la función de evaluación a las tareas
            future_to_task_info = {
                executor.submit(self._evaluate_fitness, task['params'], task['eval_seed']): task
                for task in tasks
            }

            for future in concurrent.futures.as_completed(future_to_task_info):
                task_info = future_to_task_info[future]
                idx = task_info['idx']
                epsilon_for_this_eval = task_info['epsilon']
                all_perturbations[idx, :] = epsilon_for_this_eval # Guardar la perturbación usada

                try:
                    fitness = future.result()
                    all_fitnesses[idx] = fitness
                except Exception as exc:
                    print(f'Evaluación (idx {idx}) generó una excepción: {exc}')
                    all_fitnesses[idx] = -np.inf # Penalizar fallos

        return all_fitnesses, all_perturbations

    def update_policy(self, all_fitnesses, all_perturbations):
        if self.use_fitness_shaping:
            shaped_fitnesses = compute_centered_ranks(all_fitnesses)
        else:
            shaped_fitnesses = all_fitnesses

        weighted_sum_perturbations = np.dot(shaped_fitnesses, all_perturbations)
        es_gradient_estimate = (1 / (self.population_size * self.noise_std_dev)) * weighted_sum_perturbations

        current_params = self.policy_network.get_flattened_parameters()
        update_step = self.learning_rate * es_gradient_estimate
        weight_decay_term = self.learning_rate * self.weight_decay_coeff * current_params
        new_params = current_params + update_step - weight_decay_term
        self.policy_network.set_flattened_parameters(new_params)

    def train(self, num_generations, print_every=10):
        total_start_time = time.time()
        for generation in range(num_generations):
            gen_start_time = time.time()
            all_fitnesses, all_perturbations = self.evaluate_population()
            self.update_policy(all_fitnesses, all_perturbations)

            if (generation + 1) % print_every == 0 or generation == 0:
                mean_fitness = np.mean(all_fitnesses)
                std_fitness = np.std(all_fitnesses)
                max_fitness = np.max(all_fitnesses)
                min_fitness = np.min(all_fitnesses)
                gen_elapsed_time = time.time() - gen_start_time
                total_elapsed_time = time.time() - total_start_time
                print(f"Gen: {generation + 1}/{num_generations}, "
                      f"Fit (Mean±Std): {mean_fitness:.2f}±{std_fitness:.2f}, "
                      f"Min: {min_fitness:.2f}, Max: {max_fitness:.2f}, "
                      f"T_gen: {gen_elapsed_time:.1f}s, T_total: {total_elapsed_time:.1f}s")
        print(f"Entrenamiento completado en {time.time() - total_start_time:.2f} segundos.")

# --- Ejemplo de Uso con la Red Real ---
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {DEVICE}")

    # 1. Configurar el entorno
    # ENV_NAME = "CartPole-v1" # Acciones discretas
    ENV_NAME = "Pendulum-v1" # Acciones continuas
    
    temp_env = gym.make(ENV_NAME)
    obs_dim = temp_env.observation_space.shape[0]
    
    action_scale = None
    action_bias = None
    if isinstance(temp_env.action_space, gym.spaces.Discrete):
        act_dim = temp_env.action_space.n
        is_continuous_action = False
    elif isinstance(temp_env.action_space, gym.spaces.Box):
        act_dim = temp_env.action_space.shape[0]
        is_continuous_action = True
        # Para Pendulum, la acción está en [-2, 2]
        action_scale = torch.tensor(temp_env.action_space.high, dtype=torch.float32)
        # El bias no es necesario si la red usa tanh (salida en [-1,1]) y luego escalamos por action_scale
        # Si el rango no es simétrico, action_bias = (high+low)/2 y action_scale = (high-low)/2
        # Para Pendulum, high es [2.], low es [-2.]. Entonces scale es 2.0, bias es 0.0.
        action_scale = temp_env.action_space.high[0] # Asumiendo que es un escalar
        action_bias = 0.0 # Ya que tanh produce [-1,1] y queremos [-scale, scale]
    else:
        raise NotImplementedError(f"Espacio de acción no soportado: {type(temp_env.action_space)}")
    temp_env.close()

    print(f"Entorno: {ENV_NAME}, Obs dim: {obs_dim}, Act dim: {act_dim}, Continuo: {is_continuous_action}")

    # 2. Crear la instancia de la red real
    policy_net_instance = SimpleMLPPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=32, # Red más pequeña para pruebas rápidas
        is_continuous=is_continuous_action,
        action_scale=action_scale,
        action_bias=action_bias
    ).to(DEVICE)

    # 3. Crear el agente ES
    es_agent = EvolutionStrategyAgent(
        policy_network=policy_net_instance,
        env_name=ENV_NAME,
        learning_rate=0.05,      # Puede necesitar ajuste
        noise_std_dev=0.01,       # Puede necesitar ajuste
        population_size=30,      # Probar 30-100. Debe ser par.
        weight_decay_coeff=0.001, # Puede necesitar ajuste
        fitness_shaping=True,
        seed=42,
        device=DEVICE
    )

    # 4. Entrenar al agente
    NUM_GENERATIONS = 50  # Aumentar para problemas más difíciles
    if ENV_NAME == "CartPole-v1":
        NUM_GENERATIONS = 50
        es_agent.learning_rate = 0.1
        es_agent.noise_std_dev = 0.1
    elif ENV_NAME == "Pendulum-v1":
        NUM_GENERATIONS = 50
        es_agent.learning_rate = 0.02 # Pendulum es sensible a LR
        es_agent.noise_std_dev = 0.05
        es_agent.population_size = 64


    es_agent.train(num_generations=NUM_GENERATIONS, print_every=5)

    # (Opcional) Evaluar la política final entrenada
    print("\nEvaluando política final entrenada múltiples veces...")
    final_rewards = []
    num_eval_episodes = 10
    for i in range(num_eval_episodes):
        # Pasar una copia de los parámetros finales para la evaluación
        final_params_copy = es_agent.policy_network.get_flattened_parameters().copy()
        reward = es_agent._evaluate_fitness(final_params_copy, eval_seed=1000+i)
        final_rewards.append(reward)
        print(f"Evaluación {i+1}/{num_eval_episodes}: Recompensa = {reward:.2f}")
    print(f"Recompensa promedio final en {num_eval_episodes} episodios: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}")

