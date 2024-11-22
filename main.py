import open3d as o3d
import matplotlib.pyplot as plt
import os
import numpy as np
from transformers import pipeline
import torch 
from diffusers import StableDiffusionPipeline

DATA_PATH = "data"

def visualize_point_cloud():
    try:
        # Load and visualize point cloud
        point_cloud_path = os.path.join(DATA_PATH, "Goatskull.ply")
        point_cloud = o3d.io.read_point_cloud(point_cloud_path)
        
        '''
        # Set visualization parameters
        vis_params = {
            'window_name': 'Point Cloud Viewer',
            'width': 1920,
            'height': 1080,
            'left': 50,
            'top': 50,
            'point_show_normal': False,
            'mesh_show_wireframe': False,
            'mesh_show_back_face': False
        }
        '''
        # Visualize
        o3d.visualization.draw_geometries([point_cloud])
        
        print(np.asarray(point_cloud.points).shape)
        print(np.asarray(point_cloud.colors).shape)

        return point_cloud
    except Exception as e:
        print(f"Error visualizing point cloud: {e}")
        return None

def generate_text(prompt):
    try:
        # Initialize text generator
        generator = pipeline('text2text-generation', 
                           model="facebook/bart-large-cnn",
                           device=0 if torch.cuda.is_available() else -1)
        
        # Generate text
        result = generator(prompt)
        return result[0]['generated_text']
    except Exception as e:
        print(f"Error generating text: {e}")
        return None
    
if __name__ == "__main__":
    # First visualize point cloud
    point_cloud = visualize_point_cloud()
    
    if point_cloud is not None:
        # Then generate text
        prompt = "Adicionar luzes neon ao ambiente noturno."
        generated_text = generate_text(prompt)
        if generated_text:
            print(f"Generated instructions: {generated_text}")


    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.to("cuda")
    result = pipe("sunset with vibrant colors")
    result.images[0].save("output_texture.png")

    texture = o3d.io.read_image("output_texture.png")
    mesh = o3d.io.read_triangle_mesh("input_model.obj")
    mesh.textures = [texture]
    o3d.visualization.draw_geometries([mesh])
