import random
import open3d as o3d
import os

# OBJ Processing and Point Cloud Generation
def read_obj(filename):
    vertices = []
    faces = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split()

            if line.startswith("v "):
                vertices.append(tuple(map(float, parts[1:4])))
            elif line.startswith("f "):
                face = tuple(map(int, [p.split("/")[0] for p in parts[1:4]]))
                faces.append(face)

    return vertices, faces

def sample_points_on_triangle(v1, v2, v3, num_samples):
    points = []
    for _ in range(num_samples):
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * v1[i] + (t-s)*v2[i] + (1-t)*v3[i]
        points.append((f(0), f(1), f(2)))
    return points

def generate_point_cloud(vertices, faces, total_points=131072):
    triangle_areas = []
    for face in faces:
        a, b, c = [vertices[i-1] for i in face]
        ab = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
        ac = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
        cross_product = (ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0])
        area = 0.5 * (cross_product[0]**2 + cross_product[1]**2 + cross_product[2]**2)**0.5
        triangle_areas.append(area)

    total_area = sum(triangle_areas)
    points = []
    for i, face in enumerate(faces):
        a, b, c = [vertices[j-1] for j in face]
        num_samples = int(triangle_areas[i] / total_area * total_points)
        points.extend(sample_points_on_triangle(a, b, c, num_samples))

    while len(points) < total_points:
        face = random.choice(faces)
        a, b, c = [vertices[j-1] for j in face]
        points.extend(sample_points_on_triangle(a, b, c, 1))

    return points

# Point Cloud Visualization with Open3D
def load_point_cloud_from_points(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def visualize_point_cloud(point_cloud):
    o3d.visualization.draw_geometries([point_cloud])
    
def list_subdirectories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def save_view(vis):
    image = vis.capture_screen_float_buffer(False)
    o3d.io.write_image("point_cloud_view.png", image)
    return False

if __name__ == "__main__":
    directory = "../pybullet-URDF-models/urdf_models/models"
    subdirectories = list_subdirectories(directory)
    input_filename = os.path.join(directory, random.choice(subdirectories), "textured.obj")
    print("loaded object: ", input_filename)
    
    vertices, faces = read_obj(input_filename)
    point_cloud_data = generate_point_cloud(vertices, faces)

    pc = load_point_cloud_from_points(point_cloud_data)
    visualize_point_cloud(pc)
    # o3d.visualization.draw_geometries_with_custom_animation([pc], save_view)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pc)
    # vis.capture_screen_image("point_cloud_view.png", do_render=True)
    # vis.destroy_window()
