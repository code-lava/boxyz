import open3d as o3d
import numpy as np


def get_camera_parameters_interactive(pcb):
    """
    Get the camera parameters from the Open3D window. The user can adjust the view in the window and when closed,
   the function will return the corresponding camera parameters.
   Args:
        pcb: o3d.geometry.Geometry: Open3D geometry (PointCloud, TriangleMesh, etc.) to visualize
   Returns:
        PinholeCameraParameters: o3d.camera.PinholeCameraParameters: containing extrinsic and intrinsic matrices
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Adjust View - Close Window When Done')
    vis.add_geometry(pcb)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 2.0

    vis.run()

    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()

    vis.destroy_window()
    extrinsic = camera_params.extrinsic
    intrinsic = camera_params.intrinsic

    rotation_matrix = extrinsic[:3, :3]
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    if sy > 1e-6:  # !not at singularity
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    angles_deg = np.degrees([x, y, z])

    print("Camera Extrinsic Matrix:")
    print(extrinsic)
    print("\nCamera Position:", -rotation_matrix.T @ extrinsic[:3, 3])
    print("\nCamera Angles (degrees):")
    print(f"  Pitch (X): {angles_deg[0]:.2f}°")
    print(f"  Yaw (Y): {angles_deg[1]:.2f}°")
    print(f"  Roll (Z): {angles_deg[2]:.2f}°")

    return camera_params