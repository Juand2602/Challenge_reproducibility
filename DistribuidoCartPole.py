#!/usr/bin/env python3
import os
import gym
import numpy as np
import imageio
from pyspark import SparkContext, SparkConf
import argparse
import pyarrow.fs as pafs

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
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return int(np.argmax(probs))

# Evaluación sin render (solo retornos)
def evaluate_noise(args_tuple):
    noise_i, env_name, sigma, base_params = args_tuple
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

# Entrenamiento distribuido con Spark (sin video)
def train(args):
    sc = SparkContext(conf=SparkConf().setAppName("ES-CartPole-Train").setMaster(args.master))
    env = gym.make(args.env)
    policy = Policy(env.observation_space.shape[0], 32, env.action_space.n)
    env.close()

    for it in range(args.iterations):
        base_params = policy.get_params()
        noise = [[np.random.randn(*p.shape) for p in base_params] for _ in range(args.pop_size)]
        paired = [(n, args.env, args.sigma, base_params) for n in noise]
        rdd = sc.parallelize(paired, numSlices=args.pop_size)
        rewards = np.array(rdd.map(evaluate_noise).collect())
        A = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        for i in range(len(base_params)):
            base_params[i] += (args.alpha / (args.pop_size * args.sigma)) * sum(A[j] * noise[j][i] for j in range(args.pop_size))
        policy.set_params(base_params)
        print(f"Iter {it+1}/{args.iterations} done")

    # Guardar params en HDFS usando pyarrow
    tmp_path = "/tmp/params.npy"
    np.save(tmp_path, policy.get_params())
    fs = pafs.HadoopFileSystem('default')
    with open(tmp_path, 'rb') as f_local, fs.open_output_stream(args.hdfs_params) as f_hdfs:
        f_hdfs.write(f_local.read())
    sc.stop()
    print(f"✅ Training complete. Params in {args.hdfs_params}")

# Grabación de video local y subida a HDFS usando RecordVideo wrapper
def record_video(args):
    print("🚀 Iniciando generación de video...", flush=True)

    fs = pafs.HadoopFileSystem('default')
    with fs.open_input_file(args.hdfs_params) as f_hdfs:
        import io
        data = f_hdfs.read()
        params = np.load(io.BytesIO(data), allow_pickle=True).tolist()

    import tempfile
    from gym.wrappers import RecordVideo

    video_dir = tempfile.mkdtemp(prefix="videos_")
    base_env = gym.make(args.env, render_mode="rgb_array")
    env = RecordVideo(base_env, video_folder=video_dir, name_prefix="cartpole", episode_trigger=lambda x: True)
    policy = Policy(base_env.observation_space.shape[0], 32, base_env.action_space.n)
    policy.set_params(params)

    obs, _ = env.reset()
    done = False
    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()

    import glob
    files = glob.glob(os.path.join(video_dir, "cartpole*mp4"))
    if not files:
        raise FileNotFoundError("No se encontró el video en " + video_dir)
    local_video = files[-1]

    with open(local_video, 'rb') as f_loc, fs.open_output_stream(args.video) as f_vid:
        f_vid.write(f_loc.read())
    print(f"🎥 Video saved at {args.video}")

# Main con parsing de argumentos y dispatch
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--env", type=str, required=True)
    parser_train.add_argument("--iterations", type=int, default=10)
    parser_train.add_argument("--pop_size", type=int, default=50)
    parser_train.add_argument("--sigma", type=float, default=0.1)
    parser_train.add_argument("--alpha", type=float, default=0.01)
    parser_train.add_argument("--master", type=str, default="local[*]")
    parser_train.add_argument("--hdfs_params", type=str, required=True)

    parser_video = subparsers.add_parser("video")
    parser_video.add_argument("--env", type=str, required=True)
    parser_video.add_argument("--hdfs_params", type=str, required=True)
    parser_video.add_argument("--video", type=str, required=True)

    args = parser.parse_args()
    print(f"[DEBUG] Argumentos: {args}", flush=True)

    if args.command == "train":
        train(args)
    elif args.command == "video":
        record_video(args)
    else:
        print("❌ Comando no reconocido. Usa 'train' o 'video'.", flush=True)
