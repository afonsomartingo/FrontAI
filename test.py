import open3d as o3d
import os
from PIL import Image
import numpy as np
from transformers import pipeline
import torch
from diffusers import StableDiffusionPipeline

# Caminho base do projeto
DATA_PATH = "C:/Users/Afonso/Downloads/FrontAI/data/Testm_textured_mesh_obj"

def visualize_mesh_with_textures(obj_file="mesh.obj"):
    try:
        # Caminho completo para o arquivo .obj
        obj_path = os.path.join(DATA_PATH, obj_file)

        # Carregar o modelo .obj com texturas
        mesh = o3d.io.read_triangle_mesh(obj_path)

        # Verifique se a mesh tem normas de vértices, se não, calcule
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # Verifique se a mesh tem texturas
        if not mesh.textures:
            print("Aviso: Nenhuma textura foi carregada na mesh.")
        else:
            print(f"Texturas carregadas: {len(mesh.textures)}")

        # Visualizar o modelo com as texturas carregadas
        o3d.visualization.draw_geometries([mesh])
        print(f"Número de vértices: {len(mesh.vertices)}")
        print(f"Número de faces: {len(mesh.triangles)}")

        return mesh
    except Exception as e:
        print(f"Erro ao visualizar o modelo com texturas: {e}")
        return None


# Função para gerar nova textura usando Stable Diffusion
def generate_texture(prompt, output_file="new_texture.png"):
    try:
        # Inicializar o pipeline de Stable Diffusion
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Gerar textura
        result = pipe(prompt)
        result.images[0].save(os.path.join(DATA_PATH, output_file))
        print(f"Nova textura gerada salva em: {output_file}")
        return output_file
    except Exception as e:
        print(f"Erro ao gerar nova textura: {e}")
        return None

# Função para aplicar a nova textura como overlay sobre a original
def apply_texture_overlay(original_texture_file, new_texture_file, output_file="overlay_texture.png"):
    try:
        # Carregar as duas texturas
        original_texture = Image.open(original_texture_file).convert("RGBA")
        new_texture = Image.open(new_texture_file).convert("RGBA")
        
        # Redimensionar as texturas para garantir que tenham o mesmo tamanho
        new_texture = new_texture.resize(original_texture.size, Image.ANTIALIAS)

        # Combinar as texturas como overlay (exemplo simples de multiplicação)
        combined_texture = Image.blend(original_texture, new_texture, alpha=0.5)
        
        # Salvar o resultado do overlay
        combined_texture.save(output_file)
        print(f"Overlay de texturas gerado em: {output_file}")
        return output_file
    except Exception as e:
        print(f"Erro ao aplicar overlay de texturas: {e}")
        return None

def apply_new_texture_to_mesh(mesh, new_texture_file):
    try:
        # Carregar a nova textura
        new_texture = o3d.io.read_image(new_texture_file)
        
        # Certifique-se de que a mesh tenha uma lista de texturas
        if not mesh.textures:
            mesh.textures = [new_texture]
        else:
            mesh.textures[0] = new_texture  # Substitui a primeira textura, ou adicione lógica para múltiplas texturas

        # Visualizar o modelo com a nova textura
        o3d.visualization.draw_geometries([mesh])
        print("Nova textura aplicada ao modelo com sucesso!")
    except Exception as e:
        print(f"Erro ao aplicar nova textura: {e}")


def update_mtl_file(mtl_file, texture_file, material_name="material0000"):
    try:
        # Abrir o arquivo .mtl
        with open(mtl_file, "r") as f:
            lines = f.readlines()

        # Substituir a referência da textura
        updated_lines = []
        for line in lines:
            if line.startswith(f"map_Kd") and material_name in line:
                updated_lines.append(f"map_Kd {texture_file}\n")
            else:
                updated_lines.append(line)

        # Escrever o arquivo .mtl atualizado
        with open(mtl_file, "w") as f:
            f.writelines(updated_lines)

        print(f"Arquivo .mtl atualizado com a nova textura: {texture_file}")
    except Exception as e:
        print(f"Erro ao atualizar o arquivo .mtl: {e}")


# Fluxo principal
if __name__ == "__main__":
    # Etapa 1: Carregar e visualizar o modelo original com texturas
    print("Visualizando o modelo original...")
    mesh = visualize_mesh_with_textures()  # Carrega a mesh com a textura original

    if mesh:
        # Etapa 2: Gerar nova textura com base em um prompt
        prompt = "Textura de pedra escura com iluminação suave."
        new_texture_file = generate_texture(prompt, output_file="new_texture.png")

        if new_texture_file:
            # Etapa 3: Aplicar overlay de texturas (mesclar a textura original com a nova)
            original_texture_file = os.path.join(DATA_PATH, "textures", "mesh_material0000_map_Kd.png")
            overlay_texture_file = apply_texture_overlay(original_texture_file, new_texture_file, output_file="overlay_texture.png")

            if overlay_texture_file:
                # Etapa 4: Atualizar o arquivo .mtl com a nova textura
                update_mtl_file(mtl_file="mesh.mtl", texture_file="overlay_texture.png", material_name="material0000")

                # Etapa 5: Visualizar o modelo com a nova textura aplicada
                print("Visualizando o modelo com a nova textura...")
                apply_new_texture_to_mesh(mesh, overlay_texture_file)
