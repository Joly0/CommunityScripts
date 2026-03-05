import os
import sys
# Set DeepFace home directory
os.environ["DEEPFACE_HOME"] = "."
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logs
# Add the plugins directory to sys.path
plugins_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if plugins_dir not in sys.path:
    sys.path.insert(0, plugins_dir)


from stashapi.stashapp import StashInterface


try:
    from models.data_manager import DataManager
    from web.interface import WebInterface
except ImportError as e:
    print(f"Error importing modules: {e}")
    input("Ensure you have installed the required dependencies. Press Enter to exit.")



def main():
    """Main entry point for the application"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import gradio as gr
    import uvicorn

    # Initialize data manager
    data_manager = DataManager(
        voy_root_folder=os.path.abspath(os.path.join(os.path.dirname(__file__),"../voy_db")),
    )

    # Initialize web interface and get the Gradio demo
    web_interface = WebInterface(data_manager, default_threshold=0.5)
    demo = web_interface.create_demo()

    # Create FastAPI app with CORS middleware
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Gradio app inside FastAPI
    app = gr.mount_gradio_app(app, demo, path="/")

    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
