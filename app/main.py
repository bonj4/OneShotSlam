import cv2 
import numpy as np
from utils import *
from data import Dataset
import time
import open3d as o3d

def BSlam(data,detector='orb', matching='BF',GoodP=10000, dist_threshold=0.5,):
    num_frame=len(data.imgs)
    
    points4d=np.array([[],[],[]]).T
    T_tot = np.eye(4)
    trajectory = np.zeros((num_frame, 4, 4))
    trajectory[0] = T_tot
    for idx in range(num_frame):
        print(idx)
        if idx==0: continue

        prev_img=data.imgs[idx-1]
        img=data.imgs[idx]
        # Extract features
        prev_kp, prev_des = extract_features(
            prev_img, detector=detector, GoodP=GoodP,)
        img_kp, img_des = extract_features(
            img, detector=detector, GoodP=GoodP,)
        # extract matches
        matches_unfilter = match_features(
            des1=prev_des, des2=img_des, matching=matching, detector=detector,)
        # filtering the matches
        if dist_threshold is not None:
            idx1,idx2,matches = filter_matches_distance(
                matches_unfilter, dist_threshold=dist_threshold)
        else:
            matches = matches_unfilter


        rmat, tvec, image1_points, image2_points = estimate_motion(
                matches=matches, kp1=prev_kp, kp2=img_kp, k=data.K,)
        

        Rt =poseRt(rmat,tvec)

                    
        T_tot = np.dot(T_tot,np.linalg.inv(Rt),)
        trajectory[idx] = T_tot
        print(T_tot)

        
        point1=np.dot(np.linalg.inv(data.K), np.concatenate([image1_points, np.ones((image1_points.shape[0], 1))], axis=1).T).T[:, 0:2]
        point2=np.dot(np.linalg.inv(data.K), np.concatenate([image2_points, np.ones((image2_points.shape[0], 1))], axis=1).T).T[:, 0:2]
        
        points3d =triangulate( trajectory[idx],  trajectory[idx-1], point2, point1)
        # points3d /= points3d[:,3:]

        points3d = cv2.convertPointsFromHomogeneous(points3d)
        points3d=points3d[points3d[:,0,2]>0]
        print(points3d[:,0].shape,points4d.shape)
        points4d = np.concatenate((points4d, points3d[:,0]))   

            

    return points4d,trajectory

if __name__=='__main__':
    s=time.perf_counter()
    data=Dataset()

    points3d,trajectory=BSlam(data=data)
    cv2.destroyAllWindows()
    print(time.perf_counter()-s)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    pcd = o3d.geometry.PointCloud()  
    pcd.points = o3d.utility.Vector3dVector(np.array(points3d))
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=np.array([0,0,0]))
    # Visualize the point cloud
    
    # o3d.visualization.draw_geometries([pcd,origin])
    view_ctl = vis.get_view_control()
    vis.add_geometry(pcd)
    vis.add_geometry(origin)

    vis.run()
    vis.destroy_window()

    plt.plot(trajectory[:,0,3],trajectory[:,2,3])
    plt.show()
    visualize_trajectory(trajectory[:,:3])