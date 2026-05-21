# generate_samples.py
import numpy as np
import os

def make_sample_data():
    os.makedirs("data", exist_ok=True)
    
    # 模拟用户定义：生成 3 个 episode，分别有不同长度和关节数
    episodes_config = {
        "episode_001": {"steps": 120, "joints": 6},
        "episode_002": {"steps": 80,  "joints": 12},  # 比如模拟一个更多自由度的灵巧手
        "episode_003": {"steps": 150, "joints": 6}
    }
    
    for ep_name, config in episodes_config.items():
        t = config["steps"]
        j = config["joints"]
        
        # 1. 模拟关节角度 (逐渐握紧的趋势)
        time_seq = np.linspace(0, np.pi, t)
        joint_angles = np.array([np.sin(time_seq + i) for i in range(j)]).T
        # 随机加一点噪声
        joint_angles += np.random.normal(0, 0.02, joint_angles.shape)
        
        # 2. 模拟图片数组 (形状为 T, H, W, C)
        # 这里用一些渐变色的纯色块矩阵代替摄像头画面
        images = np.zeros((t, 240, 320, 3), dtype=np.uint8)
        for i in range(t):
            images[i, :, :, 0] = int(255 * (i / t))  # R通道渐变
            images[i, :, :, 1] = 100                 # G通道固定
            images[i, :, :, 2] = 200                 # B通道固定
            
        # 3. 核心：保存为开源社区通用的 npz 压缩格式
        out_path = os.path.join("data", f"{ep_name}.npz")
        np.savez_compressed(out_path, joint_angles=joint_angles, images=images)
        print(f"成功创建模板样例数据: {out_path} (帧数: {t}, 关节数: {j})")

if __name__ == "__main__":
    make_sample_data()