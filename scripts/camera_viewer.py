import mujoco
import mujoco.viewer
import cv2
import numpy as np

# Load your scene
model = mujoco.MjModel.from_xml_path("/home/drake/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 480, 640)

def main():
    while True:
        # Note: In a real setup, you'd sync 'data' with the simulator 
        # For now, this just opens the renderer to check the angle
        renderer.update_scene(data, camera="d435i_rgb")
        pixels = renderer.render()
        
        # Display via OpenCV
        frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.imshow("G1 Realsense Feed", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()

