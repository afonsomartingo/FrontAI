import open3d as o3d
import matplotlib.pyplot as plt
import os
import numpy as np
from transformers import pipeline
import torch 

DATA_PATH = "data"

def visualize_mesh():
    try:
        # Load and visualize 3D mesh
        mesh_path = os.path.join(DATA_PATH, "mesh.obj")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        # Check if the mesh is successfully loaded
        if mesh.is_empty():
            raise ValueError("Mesh could not be loaded. Check the file path and format.")
        
        '''
        # Set visualization parameters
        vis_params = {
            'window_name': 'Mesh Viewer',
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
        o3d.visualization.draw_geometries([mesh])
        
        print(np.asarray(mesh.vertices).shape)
        print(np.asarray(mesh.triangles).shape)

        return mesh
    except Exception as e:
        print(f"Error visualizing mesh: {e}")
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
    # First visualize mesh
    mesh = visualize_mesh()
    
    if mesh is not None:
        # Then generate text
        prompt = "Adicionar luzes neon ao ambiente noturno."
        generated_text = generate_text(prompt)
        if generated_text:
            print(f"Generated instructions: {generated_text}")
