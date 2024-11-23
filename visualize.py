import trimesh
import numpy as np

# Load the OBJ file with the MTL material (automatically handles material mapping)
mesh = trimesh.load_mesh('data/Testm_textured_mesh_obj/mesh.obj')

# If the loaded object is a Scene, we get the first mesh in the scene
if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump()[0]  # Extract the first mesh from the scene (if there are multiple)

# Ensure the mesh was loaded properly
if mesh.is_empty:
    print("Mesh is empty!")
else:
    print(f"Mesh loaded successfully with {len(mesh.faces)} faces and {len(mesh.vertices)} vertices.")

# Check if the mesh is watertight (no holes in the mesh)
if mesh.is_watertight:
    print("The mesh is watertight.")
else:
    print("The mesh is not watertight. It might have holes or disconnected faces.")

# Scale the mesh to a visible size (if needed)
mesh.apply_scale(10)  # Adjust scale factor as necessary

# Apply a translation to ensure the mesh is in a viewable position
mesh.apply_translation([0, 0, 0])  # Adjust translation as needed

# Load textures from the MTL file (if available)
material = mesh.visual.material
if material:
    print(f"Material: {material}")
    # Check for diffuse texture map
    if hasattr(material, 'map_Kd'):
        print(f"Diffuse texture map: {material.map_Kd}")
    else:
        print("No diffuse texture map (map_Kd) found.")

# Setup the scene (if working with multiple meshes or adding background)
scene = trimesh.Scene(mesh)

# Automatically adjust the camera to fit the mesh
scene.show(camera={'near': 0.1, 'far': 100.0})

# Optionally, apply rotation if the mesh is not oriented correctly
# Create a 4x4 identity rotation matrix
rotation_matrix = np.eye(4)
mesh.apply_transform(rotation_matrix)  # Apply the rotation

# Check the bounding box to ensure the mesh fits in the view
bbox = mesh.bounding_box
print(f"Bounding box: {bbox.bounds}")

# Display the scene
scene.show()