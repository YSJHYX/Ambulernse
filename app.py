import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import glob
from PIL import Image
import altair as alt

st.set_page_config(page_title="EmboScrubber Template", layout="wide")

# ==========================================
# 1. Data Loader
# ==========================================
DATA_DIR = "data"  # Define the root directory for data storage

@st.cache_data
def load_real_episode(ep_name):
    """
   Read the.npz file with the specified name. 
   If the user uses other formats (such as HDF5/ROSbag), only the internal parsing logic of this function needs to be modified.
    """
    file_path = os.path.join(DATA_DIR, f"{ep_name}.npz")
    data = np.load(file_path, allow_pickle=True)
    
    # Extract the joint Angle array, and the shape is expected to be: (Timesteps, Num_Joints)
    joints = data['joint_angles'] 
    
    # Extract the Image sequence. The shape is expected to be: (Timesteps, Height, Width, 3) or the PIL Image list
    images_raw = data['images']
    
    # Convert NumPy array images to Streamlit readable PIL Image format
    images = []
    for img_arr in images_raw:
        # If it is float32 (0 to 1) when saved, convert it to uint8 (0 to 255)
        if img_arr.dtype == np.float32 or img_arr.max() <= 1.0:
            img_arr = (img_arr * 255).astype(np.uint8)
        images.append(Image.fromarray(img_arr))
        
    return joints, images

# Scan all.npz files under the data folder as an Episode list
def sync_episodes():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    # Match all.npz files under the data/ path
    search_path = os.path.join(DATA_DIR, "*.npz")
    files = glob.glob(search_path)
    # Extract file name as episode ID
    episodes = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return sorted(episodes)

# ==========================================
# 2. UI Setup
# ==========================================
st.title("Ambulense: VLA trajectory data cleaning compass")

# Obtain all the data in the current folder
available_episodes = sync_episodes()

if 'annotations' not in st.session_state:
    st.session_state.annotations = {}

# Directory check and user guidance
if not available_episodes:
    st.warning(f"未在 `{DATA_DIR}/` 文件夹中检测到任何数据！")
    st.info("请先运行 `python generate_samples.py` 生成测试数据，或将您自己的 `.npz` 文件放入 `data/` 目录中。")
    st.stop()

# Sidebar: Dynamic navigation
with st.sidebar:
    st.header("数据集导航")
    selected_ep = st.selectbox(
        f"检测到 {len(available_episodes)} 个轨迹，请选择:", 
        available_episodes
    )
    
    try:
        joints_data, image_seq = load_real_episode(selected_ep)
        num_joints = joints_data.shape[1]
        all_joint_names = [f"Joint_{i}" for i in range(num_joints)]
    except Exception as e:
        st.error(f"加载文件 {selected_ep} 失败，请检查格式是否合规。报错信息: {e}")
        st.stop()
    
    st.divider()
    st.header("视图过滤")
    selected_joints = st.multiselect(
        "选择要显示的关节:",
        options=all_joint_names,
        default=all_joint_names
    )
    
    st.divider()
    st.header("数据打标")
    quality = st.radio("轨迹质量", ["🟢 优秀 (Keep)", "🟡 瑕疵 (Needs Trim)", "🔴 废弃 (Discard)"])
    notes = st.text_input("备注信息", placeholder="例如：抓取时滑脱")
    
    if st.button("保存标记", use_container_width=True):
        st.session_state.annotations[selected_ep] = {"quality": quality, "notes": notes}
        st.success(f"{selected_ep} 标记已保存！")
        
    if st.button("导出清洗结果 (JSON)", type="primary", use_container_width=True):
        output_file = "clean_dataset_meta.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.annotations, f, ensure_ascii=False, indent=2)
        st.balloons()
        st.success(f"已导出至 {output_file}")

# ==========================================
# 3. Main interface: Multimodal visualization
# ==========================================
timesteps = len(joints_data)

st.subheader(f"正在查看: `{selected_ep}` (总帧数: {timesteps})")

# Time axis slider (dynamically determines max_value based on the true length of the data)
current_step = st.slider("时间轴 (Timestep)", min_value=0, max_value=timesteps-1, value=0)

col1, col2 = st.columns([1, 1.5])

with col1:
    st.write("**摄像头视角**")
    st.image(image_seq[current_step], caption=f"Frame: {current_step}", use_column_width=True)
    
    # Abnormal speed detection
    if current_step > 0:
        diffs = np.abs(joints_data[current_step] - joints_data[current_step-1])
        if np.any(diffs > 0.5):  # The threshold can be dynamically adjusted according to the actual dexterous hand
            st.error("Warning!警告：当前帧检测到关节速度突变（可能存在抖动/丢包）！")

with col2:
    st.write("**灵巧手关节角度 (Joint Angles)**")
    
    df_joints = pd.DataFrame(joints_data, columns=all_joint_names)
    df_joints['timestep'] = df_joints.index
    df_melt = df_joints.melt('timestep', var_name='Joint', value_name='Angle')
    
    if not selected_joints:
        st.warning("请在左侧侧边栏至少选择一个关节进行显示。")
    else:
        df_filtered = df_melt[df_melt['Joint'].isin(selected_joints)]
        
        lines = alt.Chart(df_filtered).mark_line().encode(
            x=alt.X('timestep:Q', title='时间步 (Timestep)'),
            y=alt.Y('Angle:Q', title='关节角度 (Radians)', scale=alt.Scale(zero=False)),
            color='Joint:N'
        )
        
        rule = alt.Chart(pd.DataFrame({'timestep': [current_step]})).mark_rule(
            color='red', strokeDash=[5, 5], strokeWidth=2
        ).encode(x='timestep:Q')
        
        st.altair_chart(lines + rule, use_container_width=True)
    
    if selected_joints:
        status_strs = [f"{j}: {joints_data[current_step][all_joint_names.index(j)]:.2f}" for j in selected_joints]
        st.info(f"**当前状态** (t={current_step}): \n" + ", ".join(status_strs))