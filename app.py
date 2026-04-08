import streamlit as st
import os
import shutil
import tempfile
import json
from main import run_pipeline

st.set_page_config(page_title="Multi-Object Tracking", page_icon="🏃", layout="wide")

st.title("🏃 Multi-Object Tracking & Detection Pipeline")
st.markdown("""
Welcome to the live demo of the Predusk AI Assessment pipeline! 
Upload a video featuring multiple people (e.g. sports, crowds, or walking) to detect and uniquely track each person across frames using **YOLOv8** and **BoT-SORT**.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a Video (mp4, avi, mov)", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    st.subheader("Original Video")
    st.video(uploaded_file)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("### Model Options")
        conf_thresh = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.35, step=0.05)
        # Adding a warning since they are deploying on CPU
        st.info("💡 Note: Running via Streamlit Community Cloud uses a CPU. To prevent timeouts, it's recommended to test with a short video snippet (10-20 seconds).")
        run_btn = st.button("🚀 Run Tracking Pipeline", type="primary", use_container_width=True)

    with col2:
        if run_btn:
            with st.spinner("Processing video frame-by-frame... This may take a few minutes on a CPU server."):
                # 1. Setup temporary directory for processing
                temp_dir = tempfile.mkdtemp()
                input_path = os.path.join(temp_dir, "input.mp4")
                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(output_dir, exist_ok=True)
                output_video_path = os.path.join(output_dir, "annotated_video.mp4")
                
                # 2. Save uploaded file
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                # 3. Run the pipeline logic imported from main.py
                try:
                    run_pipeline(
                        video_path=input_path,
                        output_video=output_video_path,
                        model_name="yolov8n.pt",
                        tracker_config="botsort.yaml",
                        confidence=conf_thresh,
                        target_class=0, # Person
                        trail_length=40,
                        skip_frames=0
                    )
                    
                    st.success("✅ Processing Complete!")
                    
                    st.subheader("Annotated Output")
                    
                    # 4. Convert MP4 codec for HTML5 browser compatibility
                    # OpenCV creates mp4v codec by default, which some browsers refuse to play.
                    # We use system ffmpeg to convert to H264 (x264) codec
                    browser_video_path = os.path.join(output_dir, "browser_annotated.mp4")
                    exit_code = os.system(f"ffmpeg -y -i {output_video_path} -vcodec libx264 {browser_video_path}")
                    
                    # 5. Display video
                    if exit_code == 0 and os.path.exists(browser_video_path):
                        with open(browser_video_path, 'rb') as vf:
                            st.video(vf.read())
                    else:
                        st.warning("Could not encode to H.264. Falling back to default output codec.")
                        with open(output_video_path, 'rb') as vf:
                            st.video(vf.read())

                    # 6. Display Analytics
                    st.markdown("---")
                    st.subheader("📊 Pipeline Analytics")
                    
                    heat_path = os.path.join(output_dir, "heatmap.png")
                    traj_path = os.path.join(output_dir, "trajectory_map.png")
                    count_path = os.path.join(output_dir, "count_over_time.png")
                    
                    img_c1, img_c2, img_c3 = st.columns(3)
                    if os.path.exists(traj_path): img_c1.image(traj_path, caption="Trajectory Map", use_container_width=True)
                    if os.path.exists(heat_path): img_c2.image(heat_path, caption="Movement Heatmap", use_container_width=True)
                    if os.path.exists(count_path): img_c3.image(count_path, caption="Object Count Temporal Chart", use_container_width=True)
                    
                    json_path = os.path.join(output_dir, "analytics.json")
                    if os.path.exists(json_path):
                        with open(json_path, "r") as f:
                            stats = json.load(f)
                        st.json(stats)
                        
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                
                finally:
                    # Cleanup is handled by system temp routines, but we can be explicit
                    # shutil.rmtree(temp_dir, ignore_errors=True)
                    pass
