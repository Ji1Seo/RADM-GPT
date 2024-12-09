# Robotmaster Camera Calling
# Image preprocessing, Depth Pro

from robomaster import robot, camera
import time
import cv2
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from depth_pro import create_model_and_transforms

def estimate_depth(model, transform, image, device, focal_length_px):
    jpeg_image = Image.open(BytesIO(image))
    image_tensor = transform(jpeg_image).unsqueeze(0)
    focal_length_px_tensor = torch.tensor(focal_length_px, device=device)

    with torch.no_grad():
        depth_result = model.infer(image_tensor.to(device), f_px=focal_length_px_tensor)
    depth_map = depth_result.get("depth").cpu().numpy().squeeze()
    return depth_map

def save_depth_image(image, depth_map, image_file_idx):
    inverse_depth = 1 / depth_map
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu)
    depth_map_colored = cv2.applyColorMap((inverse_depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    concatenated_image = np.hstack((image, depth_map_colored))
    file_name = f"{image_file_idx}.jpg"
    result_image_path = f"./Image_file/{file_name}"
    cv2.imwrite(result_image_path, concatenated_image)

def camera_capture(ep_camera):

    time.sleep(1) # Image Stabilization Time
    image_array = ep_camera.read_cv2_image(strategy="newest")
    _, jpeg_buffer = cv2.imencode('.jpg', image_array)
    return jpeg_buffer.tobytes()

def camera_main(file_idx, ep_camera):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    model, transform = create_model_and_transforms(device=device, precision=torch.half) # Precision
    model = model.eval()
    focal_length = 368
    image = camera_capture(ep_camera)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("image processing")
    depth_map = estimate_depth(model, transform, image_rgb, device, focal_length)
    save_depth_image(image, depth_map, file_idx)
