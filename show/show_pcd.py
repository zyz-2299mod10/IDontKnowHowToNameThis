import open3d as o3d
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str, default="./CFVS_data/pcd/pcd0.pcd", dest="path")
args = parser.parse_args()

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(args.path)
    o3d.visualization.draw_geometries([pcd])