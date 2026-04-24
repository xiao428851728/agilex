# 可視化點雲ply文件。
import open3d as o3d
import os
import sys

def visualize_ply(file_path):
    # 1. 檢查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 錯誤: 找不到文件 '{file_path}'")
        print("   請確認路徑是否正確，或者文件是否已生成。")
        return

    print(f"🔍 正在加載點雲: {file_path} ...")
    
    # 2. 讀取 PLY
    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        print(f"❌ 讀取失敗: {e}")
        return

    # 3. 檢查點雲是否爲空
    if pcd.is_empty():
        print("⚠️ 警告: 點雲是空的 (0 points)！")
        print("   原因可能是深度圖全爲0，或者高度過濾把所有點都濾掉了。")
        return

    num_points = len(pcd.points)
    print(f"✅ 成功加載! 共 {num_points} 個點")
    
    # 打印一些統計信息，幫助判斷坐標系範圍
    import numpy as np
    points = np.asarray(pcd.points)
    print("-" * 30)
    print(f"X 範圍: {points[:,0].min():.2f} ~ {points[:,0].max():.2f}")
    print(f"Y 範圍: {points[:,1].min():.2f} ~ {points[:,1].max():.2f}")
    print(f"Z 範圍: {points[:,2].min():.2f} ~ {points[:,2].max():.2f} (重點檢查這個!)")
    print("-" * 30)

    # 4. 創建可視化窗口
    # 添加一個坐標軸輔助看方向
    # 紅色(R) = X軸
    # 綠色(G) = Y軸
    # 藍色(B) = Z軸
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    # 為了讓點雲更明顯，如果點雲沒有顏色，我們可以給它統一塗成灰色
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

    print("🚀 正在打開 3D 窗口...")
    print("   [操作說明]")
    print("   - 左鍵拖拽: 旋轉")
    print("   - 滾輪: 縮放")
    print("   - Shift + 左鍵: 平移")
    print("   - +/- 鍵: 調整點的大小")
    
    o3d.visualization.draw_geometries(
        [pcd, axes], 
        window_name=f"PLY Viewer - {os.path.basename(file_path)}",
        width=1024, height=768,
        left=50, top=50
    )

if __name__ == "__main__":
    # --- 配置區域 ---
    # 在這裏修改你要看的文件名
    # 默認看 'vlfm_vis' 文件夾下的 debug_obstacle_cloud_0.ply
    
    target_file = "vlfm_vis/debug_obstacle_cloud (1).ply"
    
    # 如果你想看相機視角，可以改成這個：
    # target_file = "vlfm_vis/debug_camera_frame.ply"
    
    # 如果你想看世界坐標系，可以改成這個：
    # target_file = "vlfm_vis/debug_episodic_frame.ply"

    # 支持通過命令行傳參: python view_ply.py my_file.ply
    if len(sys.argv) > 1:
        target_file = sys.argv[1]

    visualize_ply(target_file)