import open3d as o3d
import numpy as np
import copy
import math
from tqdm import tqdm
import pyvista as pv
import time

class scapula():
    def __init__(self, file_name):
        self.mesh_pv = pv.read(file_name)
        n_cell = self.mesh_pv.n_cells
        triangles = []
        for i in range(n_cell):
            triangles.append(self.mesh_pv.cell_point_ids(i))
        triangles = np.array(triangles)

        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(np.array(self.mesh_pv.points))
        self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
        self.mesh.compute_vertex_normals()
        self.guide_mesh = 0

        self.pcd = o3d.geometry.PointCloud()
        V_mesh = np.array(self.mesh.vertices)
        self.pcd.points = o3d.utility.Vector3dVector(V_mesh)

        self.change = []

    def select_points2(self, picked_id_pcd):
        a = self.pcd.points
        self.p1 = a[picked_id_pcd[0]]
        self.p2 = a[picked_id_pcd[1]]
        self.p3 = a[picked_id_pcd[2]]
        self.id = picked_id_pcd
    
    def select_points1(self):
        def pick_points(pcd):
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window()
            vis.add_geometry(pcd)
            vis.add_geometry(pcd)
            vis.run()
            vis.destroy_window()
            return vis.get_picked_points()
        value = self.pcd.points
        picked_id_pcd = pick_points(self.pcd)
        self.p1 = value[picked_id_pcd[0]]
        self.p2 = value[picked_id_pcd[1]]
        self.p3 = value[picked_id_pcd[2]]
        self.id = picked_id_pcd

    def computer_circle(self):
        def find_center(p1, p2, p3):
            x1 = p1[0];y1 = p1[1];z1 = p1[2]
            x2 = p2[0];y2 = p2[1];z2 = p2[2]
            x3 = p3[0];y3 = p3[1];z3 = p3[2]
            a1 = (y1*z2 - y2*z1 - y1*z3 + y3*z1 + y2*z3 - y3*z2)
            b1 = -(x1*z2 - x2*z1 - x1*z3 + x3*z1 + x2*z3 - x3*z2)
            c1 = (x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2)
            d1 = -(x1*y2*z3 - x1*y3*z2 - x2*y1*z3 + x2*y3*z1 + x3*y1*z2 - x3*y2*z1)
            a2 = 2 * (x2 - x1)
            b2 = 2 * (y2 - y1)
            c2 = 2 * (z2 - z1)
            d2 = x1*x1 + y1*y1 + z1*z1 - x2*x2 - y2*y2 - z2*z2
            a3 = 2 * (x3 - x1)
            b3 = 2 * (y3 - y1)
            c3 = 2 * (z3 - z1)
            d3 = x1*x1 + y1*y1 + z1*z1 - x3*x3 - y3*y3 - z3*z3
            x = -(b1*c2*d3 - b1*c3*d2 - b2*c1*d3 + b2*c3*d1 + b3*c1*d2 - b3*c2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
            y = (a1*c2*d3 - a1*c3*d2 - a2*c1*d3 + a2*c3*d1 + a3*c1*d2 - a3*c2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
            z = -(a1*b2*d3 - a1*b3*d2 - a2*b1*d3 + a2*b3*d1 + a3*b1*d2 - a3*b2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
            return x, y, z

        p1 = self.p1; p2 = self.p2; p3 = self.p3
        x, y, z = find_center(p1, p2, p3)
        r_circle = np.sqrt((p1[0] - x)**2 + (p1[1] - y)**2 + (p1[2] - z)**2)
        
        self.center = [x, y, z]
        self.r = r_circle

    def move_center_to_O(self):
        def change_mesh(mesh_first, x, y, z):
            a = [-x, -y, -z]
            mesh_second = copy.deepcopy(mesh_first).translate(tuple(a))
            mesh_second.compute_vertex_normals()
            return mesh_second
        x = self.center[0]; y = self.center[1]; z = self.center[2]
        self.mesh = change_mesh(self.mesh, x, y, z)

        self.change.append(['translate', (x, y, z)])

    def find_vector(self, filename, d):
        def find_normal_vector(p1, p2, p3):
            x1 = p1[0];y1 = p1[1];z1 = p1[2]
            x2 = p2[0];y2 = p2[1];z2 = p2[2]
            x3 = p3[0];y3 = p3[1];z3 = p3[2]
            a = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
            b = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1)
            c = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            return [a, b, c]

        def find_dis(point, mesh):
            mesh2 = copy.deepcopy(mesh)
            mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            scene = o3d.t.geometry.RaycastingScene()
            _ = scene.add_triangles(mesh2)
            query_point = o3d.core.Tensor([point], dtype=o3d.core.Dtype.Float32)
            return scene.compute_signed_distance(query_point)

        def amount_point(normal_vector, mesh_second):
            length = 0.1
            j = 0
            for i in range(100):
                vector_point = normal_vector * (length * i)
                if find_dis(vector_point, mesh_second) < 0:
                    j = j + 1
            return j

        def dis(x, y):
            return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)

        def find_angle(p1, p2, p3):
            l1 = dis(p1, p2); l2 = dis(p2, p3); l3 = dis(p1, p3)
            if l1 * l2==0:
                print ('出现错误', p1, p2, p3)
            return math.acos((l1 ** 2 + l2 ** 2 - l3 ** 2) / (2 * l1 * l2))/np.pi

        def rotate_mesh(normal_vector):
            point_coordinate = [0, 0, 0]
            # 向量OB，也就是法向量
            vector_ob = [normal_vector[0], normal_vector[1], normal_vector[2]]

            # 法向量与z轴的夹角
            theta = find_angle(vector_ob, [0, 0, 0], [0, 0, 1])

            # 第一次旋转
            vector_ob2 = [0, np.sin(np.pi * theta), np.cos(np.pi * theta)]
            alpha = find_angle(vector_ob, [0, 0,np.cos(np.pi * theta)], vector_ob2)
            if vector_ob[0] < 0:
                alpha = - alpha
            
            R = self.mesh.get_rotation_matrix_from_xyz((0, 0, np.pi * alpha))
            mesh_third = copy.deepcopy(self.mesh)
            mesh_third.rotate(R, center=point_coordinate)

            self.change.append(['rotate', self.mesh.get_rotation_matrix_from_xyz((0, 0, - np.pi * alpha))])

            # 第二次旋转
            R = self.mesh.get_rotation_matrix_from_xyz((np.pi * theta, 0, 0))
            mesh_fourth = copy.deepcopy(mesh_third)
            mesh_fourth.rotate(R, center=point_coordinate)

            self.change.append(['rotate', self.mesh.get_rotation_matrix_from_xyz((- np.pi * theta, 0, 0))])
            return mesh_fourth

        def rotate_mesh2(normal_vector, mesh):
            point_coordinate = (0, 0, 0)
            # 向量OB，也就是法向量
            vector_ob = [normal_vector[0], normal_vector[1], normal_vector[2]]
            # print (vector_ob)

            # 法向量与z轴的夹角
            mesh_second = copy.deepcopy(mesh)
            theta = find_angle(vector_ob, [0, 0, 0], [0, 1, 0])
            R = mesh_second.get_rotation_matrix_from_xyz((0, 0, theta * np.pi))
            mesh_third = copy.deepcopy(mesh)
            mesh_third.rotate(R, center=point_coordinate)

            self.change.append(['rotate', mesh_second.get_rotation_matrix_from_xyz((0, 0,  - theta * np.pi))])
            return mesh_third

        def change_cylinder(mesh_cylinder1, up):
            point_coordinate = [0, 0, 0]
            a = [0, 0, 0] - up / 2
            # print('中心（前）', mesh_cylinder1.get_center())
            mesh_cylinder2 = copy.deepcopy(mesh_cylinder1).translate(tuple(a))
            mesh_cylinder2.compute_vertex_normals()
            # print('中心（中）', mesh_cylinder2.get_center())
            theta1 = find_angle(mesh_cylinder2.get_center(), [0, 0, 0], [0, 0, 1])
            R = mesh_cylinder2.get_rotation_matrix_from_xyz((0, np.pi * theta1, 0))
            mesh_cylinder = copy.deepcopy(mesh_cylinder2)
            mesh_cylinder.rotate(R, center=point_coordinate)
            # print('中心（后）', mesh_cylinder.get_center())
            return mesh_cylinder

        def change_cylinder2(mesh_cylinder1):
            point_coordinate = [0, 0, 0]
            a = [0, 0, 0]
            # print('中心（前）', mesh_cylinder1.get_center())
            mesh_cylinder2 = copy.deepcopy(mesh_cylinder1).translate(tuple(a))
            mesh_cylinder2.compute_vertex_normals()
            # print('中心（中）', mesh_cylinder2.get_center())
            theta1 = find_angle(mesh_cylinder2.get_center(), [0, 0, 0], [0, 0, 1])
            R = mesh_cylinder2.get_rotation_matrix_from_xyz((np.pi * theta1, 0, 0))
            mesh_cylinder = copy.deepcopy(mesh_cylinder2)
            mesh_cylinder.rotate(R, center=point_coordinate)
            # print('中心（后）', mesh_cylinder.get_center())
            return mesh_cylinder

        p1 = self.p1; p2 = self.p2; p3 = self.p3
        normal_vector_zero = find_normal_vector(p1, p2, p3)
        normal_vector_module = (normal_vector_zero[0] **2 + normal_vector_zero[1] **2 + normal_vector_zero[2] **2) **0.5
        normal_vector = (np.asarray(normal_vector_zero)) / normal_vector_module
        normal_vector_back = normal_vector * (-1)

        numeber =  amount_point(normal_vector, self.mesh)
        numeber_back = amount_point(normal_vector_back, self.mesh)
        if numeber_back > numeber:
            normal_vector = normal_vector_back

        self.mesh = rotate_mesh(normal_vector)
        # print (normal_vector)

        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 100)
        self.mesh_frame.compute_vertex_normals()

        p1 = np.array(self.mesh.vertices)[self.id[0]]
        vector2 = np.array(p1) / ((p1[0] **2 + p1[1] **2 + p1[2] **2) **0.5)
        self.mesh = rotate_mesh2(vector2, self.mesh)

        self.cylinder10 = o3d.geometry.TriangleMesh.create_cylinder(radius=d/2,
                                                          height=50)
        self.cylinder10 = change_cylinder(self.cylinder10, np.asarray(self.cylinder10.vertices)[0] - np.asarray(self.cylinder10.vertices)[1])
        print ('圆柱顶点', np.asarray(self.cylinder10.vertices)[0], np.asarray(self.cylinder10.vertices)[1])

        self.cylinder101pv = pv.read(filename)
        print (self.cylinder101pv)
        n_cell = self.cylinder101pv.n_cells
        triangles = []
        for i in range(n_cell):
            triangles.append(self.cylinder101pv.cell_point_ids(i))
        triangles = np.array(triangles)
        # print (triangles.shape)

        self.cylinder101 = o3d.geometry.TriangleMesh()
        self.cylinder101.vertices = o3d.utility.Vector3dVector(np.array(self.cylinder101pv.points))
        self.cylinder101.triangles = o3d.utility.Vector3iVector(triangles)
        self.cylinder101.compute_vertex_normals()
        self.cylinder101 = change_cylinder2(self.cylinder101)

        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 100)
        self.mesh_frame.compute_vertex_normals()

        # o3d.visualization.draw_geometries([self.cylinder101, self.mesh, self.mesh_frame])

    def find_nail(self, theta1 = 5/8, theta2 = 20/20, num_point = 400):
        def dis(x, y):
            return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)

        def find_dis2(point):
            query_point = o3d.core.Tensor([point], dtype=o3d.core.Dtype.Float32)
            return scene.compute_signed_distance(query_point)

        mesh = self.mesh; point_coordinate = (0, 0, 0)
        mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh2)

        # 1.设定步长，角度1是1°，角度2是18°；已在函数传递中完成

        # 2.初始化记录器
        location = [0, [], []] # 长度，点的位置，圆柱的位置

        # 3.开始穷举
        p = []; know = []
        for i in range(int(5/theta1)):
            for j in range(int(20/theta2)):
                p.append([i, j])
        
        for z in tqdm(p):
            i = z[0]; j = z[1]
                
                # 3.1.得出当前需要计算的圆柱位置，并将位于初始位置的圆柱旋转到那里
            theta_y = 10 + theta1 * i; theta_z = theta2 * j - 10
            R = self.cylinder10.get_rotation_matrix_from_xyz((0, theta_z * np.pi / 180, 0))
            mesh_cylinderchange1 = copy.deepcopy(self.cylinder10)
            mesh_cylinderchange1.rotate(R, center=point_coordinate)
            R = self.cylinder10.get_rotation_matrix_from_xyz((theta_y * np.pi / 180, 0, 0))
            mesh_cylinderchange = copy.deepcopy(mesh_cylinderchange1)
            mesh_cylinderchange.rotate(R, center=point_coordinate)

                # 3.2.对当前圆柱位置进行判定，计算算法为：对于圆柱的每一个点，沿着x轴正负方向各走200个单位长度，如果有一侧全部在模型外侧，则这个点在模型外侧。找到在模型外侧且离圆心最近的钉子上的点。
            dis_origin = 100
            pcd2 = mesh_cylinderchange.sample_points_uniformly(number_of_points=num_point)
            point = np.asarray(pcd2.points)
            point_dis_coordinate = np.array([dis(point[k], point_coordinate) for k in range(num_point)])
                
            for k in range(num_point):
                if (point_dis_coordinate[k] >= dis_origin) or (point_dis_coordinate[k] <= 5):
                    continue

                judge1 = -1; judge2 = -1
                position_x = np.arange(0, 40, 0.1) + point[k][0]
                position_x = position_x.reshape(-1, 1)
                position_y = np.repeat(point[k][1], 400).reshape(-1, 1)
                position_z = np.repeat(point[k][2], 400).reshape(-1, 1)
                position = np.concatenate((position_x, position_y, position_z),axis=1)
                dis2 = find_dis2(position)
                dis2 = dis2.reshape(-1)
                if (dis2>=0).all():
                    judge1 = 1
                    
                position_x = np.arange(-40, 0, 0.1) + point[k][0]
                position_x = position_x.reshape(-1, 1)
                position_y = np.repeat(point[k][1], 400).reshape(-1, 1)
                position_z = np.repeat(point[k][2], 400).reshape(-1, 1)
                position = np.concatenate((position_x, position_y, position_z),axis=1)
                dis2 = find_dis2(position)
                dis2 = dis2.reshape(-1)
                if (dis2>=0).all():
                    judge2 = 1

                if (judge1 > 0 or judge2 > 0) and (dis_origin > point_dis_coordinate[k]):
                    dis_origin = point_dis_coordinate[k]
                    know = point[k]
                        
            if (dis_origin != 100) and (dis_origin > location[0]):
                location[0] = dis_origin; location[1] = know; location[2] = [i, j]
        location[2][0] = theta1 * location[2][0] + 10; location[2][1] = theta2 * location[2][1] - 10
        self.location = location

        R = self.cylinder10.get_rotation_matrix_from_xyz((location[2][0]*np.pi / 180, 0, 0))
        mesh_cylinderchange1 = copy.deepcopy(self.cylinder10)
        mesh_cylinderchange1.rotate(R, center=point_coordinate)
        R = self.cylinder10.get_rotation_matrix_from_xyz((0, location[2][1]*np.pi / 180, 0))
        mesh_cylinderchange = copy.deepcopy(mesh_cylinderchange1)
        mesh_cylinderchange.rotate(R, center=point_coordinate)
        self.cylinder = copy.deepcopy(mesh_cylinderchange)

        R = self.cylinder101.get_rotation_matrix_from_xyz((location[2][0]*np.pi / 180, 0, 0))
        mesh_cylinderchange1 = copy.deepcopy(self.cylinder101)
        mesh_cylinderchange1.rotate(R, center=point_coordinate)
        R = self.cylinder10.get_rotation_matrix_from_xyz((0, location[2][1]*np.pi / 180, 0))
        mesh_cylinderchange = copy.deepcopy(mesh_cylinderchange1)
        mesh_cylinderchange.rotate(R, center=point_coordinate)
        self.cylinder101 = copy.deepcopy(mesh_cylinderchange)

    def find_nail2(self):
        import torch
        from sko.GA import GA

        def dis(x, y):
            return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)

        def find_dis2(point):
            query_point = o3d.core.Tensor([point], dtype=o3d.core.Dtype.Float32)
            return scene.compute_signed_distance(query_point)

        mesh = self.mesh; point_coordinate = (0, 0, 0)
        mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh2)

        def obj_func(z):
            i, j = z

            theta_y = i; theta_z = j
            R = mesh.get_rotation_matrix_from_xyz((0, theta_y * np.pi / 180, 0))
            mesh_cylinderchange1 = copy.deepcopy(self.cylinder10)
            mesh_cylinderchange1.rotate(R, center=point_coordinate)
            R = mesh.get_rotation_matrix_from_xyz((0, 0, theta_z * np.pi / 180))
            mesh_cylinderchange = copy.deepcopy(mesh_cylinderchange1)
            mesh_cylinderchange.rotate(R, center=point_coordinate)
            dis_origin = 100
            pcd2 = mesh_cylinderchange.sample_points_uniformly(number_of_points=150)
            point = np.asarray(pcd2.points)
            point_dis_coordinate = np.array([dis(point[k], point_coordinate) for k in range(150)])
            for k in range(150):
                if (point_dis_coordinate[k] >= dis_origin) or (point_dis_coordinate[k] <= 5):
                    continue

                judge1 = -1; judge2 = -1
                position_x = np.arange(0, 40, 0.1) + point[k][0]
                position_x = position_x.reshape(-1, 1)
                position_y = np.repeat(point[k][1], 400).reshape(-1, 1)
                position_z = np.repeat(point[k][2], 400).reshape(-1, 1)
                position = np.concatenate((position_x, position_y, position_z),axis=1)
                dis2 = find_dis2(position)
                dis2 = dis2.reshape(-1)
                if (dis2>=0).all():
                    judge1 = 1
                    
                position_x = np.arange(-40, 0, 0.1) + point[k][0]
                position_x = position_x.reshape(-1, 1)
                position_y = np.repeat(point[k][1], 400).reshape(-1, 1)
                position_z = np.repeat(point[k][2], 400).reshape(-1, 1)
                position = np.concatenate((position_x, position_y, position_z),axis=1)
                dis2 = find_dis2(position)
                dis2 = dis2.reshape(-1)
                if (dis2>=0).all():
                    judge2 = 1

                if (judge1 > 0 or judge2 > 0) and (dis_origin > point_dis_coordinate[k]):
                    dis_origin = point_dis_coordinate[k]
                    know = point[k]
            return -dis_origin

        # 基于GPU加速的遗传算法
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ga = GA(func=obj_func, n_dim=2, size_pop=10, max_iter=50, prob_mut=0.001, lb=[10, 0], ub=[15, 360], precision=1e-7)
        ga.to(device=device)
        best_x, best_y = ga.run()

        location = [-best_y, [], best_x]
        self.location = location

        R = mesh.get_rotation_matrix_from_xyz((0, location[2][0]*np.pi / 180, 0))
        mesh_cylinderchange1 = copy.deepcopy(self.cylinder10)
        mesh_cylinderchange1.rotate(R, center=point_coordinate)
        R = mesh.get_rotation_matrix_from_xyz((0, 0, location[2][1]*np.pi / 180))
        mesh_cylinderchange = copy.deepcopy(mesh_cylinderchange1)
        mesh_cylinderchange.rotate(R, center=point_coordinate)
        self.cylinder = copy.deepcopy(mesh_cylinderchange)

    def find_handle(self, file_name):
        self.cylinder2 = o3d.io.read_triangle_mesh(file_name)
        self.cylinder2.compute_vertex_normals()
        point_coordinate = (0, 0, 0)
        R = self.cylinder10.get_rotation_matrix_from_xyz((0, self.location[2][1]*np.pi / 180, 0))
        mesh_cylinderchange1 = copy.deepcopy(self.cylinder2)
        mesh_cylinderchange1.rotate(R, center=point_coordinate)
        R = self.cylinder10.get_rotation_matrix_from_xyz((self.location[2][0]*np.pi / 180, 0, 0))
        mesh_cylinderchange = copy.deepcopy(mesh_cylinderchange1)
        mesh_cylinderchange.rotate(R, center=point_coordinate)
        self.cylinder2 = copy.deepcopy(mesh_cylinderchange)

    def find_guide(self):
        mesh1 = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh1)
        a=np.array([])
        r_circle = self.r
        r_circle /= 2 / 3
        p1 = np.array(self.mesh.vertices)
        p1 = p1[self.id[0]]

        p = []
        for i in range(180):
            for j in range(180):
                for k in range(5):
                    p.append([i, j, k])
        for z1 in tqdm(p):
            i = z1[0]; j = z1[1]; k = z1[2]
            x=(-r_circle / 2) + r_circle / 180 * i; y=(p1[1]) - r_circle / 180 * j; z = (-5) + 0.8 * k
            # x=(-r_circle / 2) + r_circle / 180 * i; y= - r_circle / 180 * j; z = 0.8 * k
            query_point = o3d.core.Tensor([[x,y,z]],dtype=o3d.core.Dtype.Float32)
            ans = scene.compute_closest_points(query_point)
            points=ans['points'].numpy()
            triangle=ans['primitive_ids'][0].item()
            a=np.append(a,triangle)
            a=a.astype(int)

        mesh2 = copy.deepcopy(self.mesh)
        mesh2.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh2.triangles)[a])
        mesh2.triangle_normals = o3d.utility.Vector3dVector(
        np.asarray(mesh2.triangle_normals)[a])
        mesh2.paint_uniform_color([0.1, 0.1, 0.7])

        # o3d.visualization.draw_geometries([mesh2, self.cylinder2])

        mesh2.compute_vertex_normals()
        pcd1 = mesh2.sample_points_uniformly(number_of_points=10000)

        xyz = np.asarray(pcd1.points)
        xyz2 = []
        for i in range(10000):
            if (xyz[i][0])**2 + (xyz[i][1])**2 > 2.4**2:
                xyz2.append(xyz[i])
        xyz2 = np.array(xyz2)
        xyz = copy.deepcopy(xyz2)
        p = []
        z1 = []
        for i in range(xyz.shape[0]):
            for j in range(10):
                z1.append([i, j])
        for z in tqdm(z1):
            i = z[0]; j = z[1]
            q = [xyz[i, 0], xyz[i, 1], xyz[i, 2] - j * 0.5]
            p.append(q)
        p = np.array(p)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(p)
        self.guide_pcd = pcd2

        mesh4 = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd2, alpha=2)
        mesh4.compute_vertex_normals()
        mesh4.paint_uniform_color([0, 0.8,0.8])
        self.guide_mesh = mesh4
        self.guide_mesh.paint_uniform_color([0.1, 0.1, 0.7])
    
    def show(self, l):
        pl = pv.Plotter()
        for i in range(len(l)):
            o3d.io.write_triangle_mesh('%d.stl'%i, l[i])
            p = pv.read('%d.stl'%i)
            _ = pl.add_mesh(p)

        pl.camera_position = 'xz'
        pl.show()

    def save(self):
        o3d.io.write_triangle_mesh('nail1.stl', self.cylinder)
        o3d.io.write_triangle_mesh('nail.stl', self.cylinder101)
        o3d.io.write_triangle_mesh('guide.stl', self.guide_mesh)
        o3d.io.write_triangle_mesh('handle.stl', self.cylinder2)
        o3d.io.write_triangle_mesh('mesh.stl', self.mesh)
        o3d.io.write_triangle_mesh('jizuo.stl', self.jizuo)

    def go_back(self):
        n = len(self.change)
        for i in range(n):
            p = self.change[n-1-i]
            if p[0]=='translate':
                self.mesh = copy.deepcopy(self.mesh).translate(p[1])
                self.cylinder = copy.deepcopy(self.cylinder).translate(p[1])
                self.cylinder2 = copy.deepcopy(self.cylinder2).translate(p[1])
                self.guide_mesh = copy.deepcopy(self.guide_mesh).translate(p[1])
                self.jizuo = copy.deepcopy(self.jizuo).translate(p[1])
                self.cylinder101 = copy.deepcopy(self.cylinder101).translate(p[1])
            else:
                self.mesh = self.mesh.rotate(p[1], center=(0, 0, 0))
                self.cylinder = self.cylinder.rotate(p[1], center=(0, 0, 0))
                self.cylinder2 = self.cylinder2.rotate(p[1], center=(0, 0, 0))
                self.guide_mesh = self.guide_mesh.rotate(p[1], center=(0, 0, 0))
                self.jizuo = self.jizuo.rotate(p[1], center=(0, 0, 0))
                self.cylinder101 = self.cylinder101.rotate(p[1], center=(0, 0, 0))
    
    def find_jizuo(self, filename):
        '''def dis(x, y):
            return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)

        def find_angle(p1, p2, p3):
            l1 = dis(p1, p2); l2 = dis(p2, p3); l3 = dis(p1, p3)
            cos = (l1 ** 2 + l2 ** 2 - l3 ** 2) / (2 * l1 * l2)
            return math.acos(cos)/np.pi

        # 改变基座位置
        def change_jizuo(mesh_jizuo1):
            a = - mesh_jizuo1.get_center() + [0, 0, 0] + [0, 0, 1.5]
            mesh_jizuo = copy.deepcopy(mesh_jizuo1).translate(tuple(a))
            mesh_jizuo.compute_vertex_normals()
            return mesh_jizuo

        # 旋转基座
        def rotate_jizuo(mesh_jizuo, direction_vector):
            point_coordinate = [0, 0, 0]
            # 向量OC为基座方向向量
            vector_oc = mesh_jizuo.get_center() - bottom_center
            vector_oc = [vector_oc[0] / vector_oc[2], vector_oc[1] / vector_oc[2], 1]
            # print(vector_oc)

            # 向量OD为钉子方向向量
            vector_od = [direction_vector[0] / direction_vector[2], direction_vector[1] / direction_vector[2], 1]
            # print(vector_od)

            # 基座方向向量与z轴夹角
            theta = find_angle(vector_oc, [0, 0, 0], [0, 0, 1])

            # 钉子方向向量与z轴夹角
            beta = find_angle(vector_od, [0, 0, 0], [0, 0, 1])

            # 第一次旋转
            vector_oc2 = [0, np.sin(np.pi * theta) / np.cos(np.pi * theta), 1]
            alpha = find_angle(vector_oc, [0, 0, 1], vector_oc2)
            
            R = self.mesh.get_rotation_matrix_from_xyz((0, 0, - np.pi * alpha))
            mesh_jizuofirst = copy.deepcopy(mesh_jizuo)
            mesh_jizuofirst.rotate(R, center=point_coordinate)

            # 第二次旋转
            R = self.mesh.get_rotation_matrix_from_xyz((np.pi * theta - np.pi * beta, 0, 0))
            mesh_jizuosecond = copy.deepcopy(mesh_jizuofirst)
            mesh_jizuosecond.rotate(R, center=point_coordinate)

            # 第三次旋转
            vector_oc3 = [0, np.sin(np.pi * beta) / np.cos(np.pi * beta), 1]
            delte = find_angle(vector_oc3, [0, 0, 1], vector_od)
            if vector_od[0] < 0:
                delte = - delte

            R = self.mesh.get_rotation_matrix_from_xyz((0, 0, - np.pi * delte))
            mesh_jizuothird = copy.deepcopy(mesh_jizuosecond)
            mesh_jizuothird.rotate(R, center=point_coordinate)
            return mesh_jizuothird
        
        def find_center(p1, p2, p3):
            x1 = p1[0];y1 = p1[1];z1 = p1[2]
            x2 = p2[0];y2 = p2[1];z2 = p2[2]
            x3 = p3[0];y3 = p3[1];z3 = p3[2]
            a1 = (y1*z2 - y2*z1 - y1*z3 + y3*z1 + y2*z3 - y3*z2)
            b1 = -(x1*z2 - x2*z1 - x1*z3 + x3*z1 + x2*z3 - x3*z2)
            c1 = (x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2)
            d1 = -(x1*y2*z3 - x1*y3*z2 - x2*y1*z3 + x2*y3*z1 + x3*y1*z2 - x3*y2*z1)
            a2 = 2 * (x2 - x1)
            b2 = 2 * (y2 - y1)
            c2 = 2 * (z2 - z1)
            d2 = x1*x1 + y1*y1 + z1*z1 - x2*x2 - y2*y2 - z2*z2
            a3 = 2 * (x3 - x1)
            b3 = 2 * (y3 - y1)
            c3 = 2 * (z3 - z1)
            d3 = x1*x1 + y1*y1 + z1*z1 - x3*x3 - y3*y3 - z3*z3
            x = -(b1*c2*d3 - b1*c3*d2 - b2*c1*d3 + b2*c3*d1 + b3*c1*d2 - b3*c2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
            y = (a1*c2*d3 - a1*c3*d2 - a2*c1*d3 + a2*c3*d1 + a3*c1*d2 - a3*c2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
            z = -(a1*b2*d3 - a1*b3*d2 - a2*b1*d3 + a2*b3*d1 + a3*b1*d2 - a3*b2*d1) / (a1*b2*c3 - a1*b3*c2 - a2*b1*c3 + a2*b3*c1 + a3*b1*c2 - a3*b2*c1)
            return x, y, z

        #读入基座文件
        mesh_jizuo1 = o3d.io.read_triangle_mesh(filename)
        mesh_jizuo1.compute_vertex_normals()

        #移动基座到球心
        mesh_jizuo1 = change_jizuo(mesh_jizuo1)

        # o3d.visualization.draw_geometries([self.mesh, mesh_jizuo, self.cylinder])

        #计算基座定向边框的顶点
        obb = mesh_jizuo1.get_oriented_bounding_box()
        m1 = np.asarray(obb.get_box_points())[0]
        m2 = np.asarray(obb.get_box_points())[1]
        m3 = np.asarray(obb.get_box_points())[2]

        #计算基座底面中心
        x1, y1, z1 = find_center(m1, m2, m3)
        bottom_center = [x1, y1, z1]

        #计算钉子方向向量
        direction_vector = np.asarray(self.cylinder.vertices)[0] - np.asarray(self.cylinder.vertices)[1]
        print(direction_vector)

        #旋转基座
        mesh_jizuonew = rotate_jizuo(mesh_jizuo1, direction_vector)
        mesh_jizuonew = change_jizuo(mesh_jizuonew)
        self.jizuo = copy.deepcopy(mesh_jizuonew)

        #展示一下
        # o3d.visualization.draw_geometries([mesh_cylindernew, mesh, mesh_sphere2, mesh_sphere, mesh_jizuonew])'''
    
        def dis(x, y):
            return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)
    
        def find_angle(p1, p2, p3):
            l1 = dis(p1, p2); l2 = dis(p2, p3); l3 = dis(p1, p3)
            cos = (l1 ** 2 + l2 ** 2 - l3 ** 2) / (2 * l1 * l2)
            return math.acos(cos)/np.pi
        
        '''mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

        self.jizuo = o3d.io.read_triangle_mesh(filename)
        self.jizuo.compute_vertex_normals()
        # self.show([self.mesh, self.jizuo, self.mesh_frame])
        print ('基座中心', self.jizuo.get_center())

        (x, y, z) = self.jizuo.get_center()
        theta = find_angle([x, y, 0], [0, 0, 0], [0, 1, 0])
        R = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi * theta))
        mesh_cylinder = copy.deepcopy(self.jizuo)
        mesh_cylinder.rotate(R, center=[0, 0, 0])
        print ('第一次旋转后基座中心', mesh_cylinder.get_center())

        self.show([self.mesh, mesh_cylinder, self.mesh_frame])

        (x, y, z) = mesh_cylinder.get_center()
        theta = find_angle([0, y, z], [0, 0, 0], [0, 0, 1])
        R = mesh.get_rotation_matrix_from_xyz((np.pi * theta - np.pi, 0, 0))
        mesh_cylinder = copy.deepcopy(mesh_cylinder)
        mesh_cylinder.rotate(R, center=[0, 0, 0])
        print ('第二次旋转后基座中心', mesh_cylinder.get_center())

        self.jizuo = copy.deepcopy(mesh_cylinder)

        self.show([self.mesh, mesh_cylinder, self.mesh_frame])
        print ('基座中心', self.jizuo.get_center())'''

        
        def change_jizuo(mesh_jizuo1):
            a = - mesh_jizuo1.get_center() + [0, 0, 0] + [0, 0, 2]
            mesh_jizuo = copy.deepcopy(mesh_jizuo1).translate(tuple(a))
            mesh_jizuo.compute_vertex_normals()
            return mesh_jizuo

        self.jizuo = o3d.io.read_triangle_mesh(filename)
        self.jizuo.compute_vertex_normals()
        # self.jizuo = copy.deepcopy(change_jizuo(self.jizuo)) # 注意 这里被注释后是正常的移动
        print ('基座中心', self.jizuo.get_center())
        print (self.location[2][0] / 180, self.location[2][1] / 180)

        R = self.cylinder10.get_rotation_matrix_from_xyz((0, self.location[2][1]*np.pi / 180, 0))
        mesh_cylinderchange1 = copy.deepcopy(self.jizuo)
        mesh_cylinderchange1.rotate(R, center=[0, 0, 0])
        R = self.cylinder10.get_rotation_matrix_from_xyz((self.location[2][0]*np.pi / 180, 0, 0))
        mesh_cylinderchange = copy.deepcopy(mesh_cylinderchange1)
        mesh_cylinderchange.rotate(R, center=[0, 0, 0])
        self.show([self.cylinder101, mesh_cylinderchange])

        theta1 = self.location[2][0] * np.pi / 180; theta2 = self.location[2][1] * np.pi / 180
        y = - np.sin(theta1) * 8; x = np.cos(theta1) * np.cos(theta2) * 8; z = np.cos(theta1) * np.sin(theta2) * 8

        print ('平移：', x, y, z)
        
        self.jizuo = copy.deepcopy(mesh_cylinderchange).translate(tuple([z, y, x]))

        print ('基座中心', self.jizuo.get_center())

s = scapula('mesh4.stl')
# s.select_points2([10787, 16263, 14676])
s.select_points1()
s.computer_circle()

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 50)
mesh_frame.compute_vertex_normals()

s.move_center_to_O()
s.find_vector('./6.5mm/nail.STL', 6.5)

'''y1 = o3d.geometry.TriangleMesh.create_cylinder(radius=3.25, height=100)
y1.compute_vertex_normals()
print ('圆柱顶点', np.asarray(y1.vertices)[0], np.asarray(y1.vertices)[1])

y2 = o3d.geometry.TriangleMesh.create_cylinder(radius=3.25, height=100)
y2.compute_vertex_normals()
R = y2.get_rotation_matrix_from_xyz((50 * np.pi / 180, 0, 0))
y2.rotate(R, center=(0, 0, 0))'''

b1 = o3d.geometry.TriangleMesh.create_sphere(radius=10)
print(b1)
b1 = b1.translate((50, 0, 0))
b1.compute_vertex_normals()
# s.show([s.mesh_frame, y2, y1, s.mesh, b1])

print ('\n基于步长穷举的钉位置寻找')
s.find_nail(5/10, 20/40, 800)
print ("钉长为:", s.location[0], "\n上/下倾角为:", s.location[2][0], "\n前/后倾角为:", s.location[2][1], '\n')
# s.show([s.cylinder, s.mesh])

s.find_handle('10.stl')
s.find_guide()
s.find_jizuo('./6.5mm/jizuo.stl')
# s.show([s.guide_mesh, s.cylinder2])
# s.show([s.cylinder, s.cylinder101, s.jizuo, mesh_frame, b1])
s.show([s.mesh, s.cylinder, s.cylinder101])
s.show([s.cylinder101, s.jizuo, s.mesh])
s.go_back()
s.save()
# s.show([s.mesh, s.guide_mesh, s.cylinder2, s.cylinder])
