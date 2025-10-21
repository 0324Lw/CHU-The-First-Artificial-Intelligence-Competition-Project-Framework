"""
管道点云缺陷检测系统
包含特征提取、模型训练和缺陷检测三个核心模块
"""

# ==================== 导入必要的库 ====================
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import os
from collections import defaultdict


# ==================== 特征提取模块 ====================
def extract_features_universal(points, pipe_type, pipe_radius=100.0, bend_radius=800.0, density_radius=5.0):
    """
    为点云中的每个点提取通用的、无偏见的几何特征
    对直管和弯管均适用

    参数:
        points: 输入的点云数据 (N, 3)
        pipe_type: 管道类型 ('straight' 或 'bent')
        pipe_radius: 管道的理想半径
        bend_radius: 弯管的弯曲半径
        density_radius: 计算局部点密度时使用的邻域半径

    返回:
        每个点的特征矩阵，形状为 (N, num_features)
    """
    num_points = points.shape[0]
    features = []

    # 计算每个点的局部半径和理想法线
    if pipe_type == 'straight':
        # 对于直管，中心线是Z轴
        centerline_points = np.zeros_like(points)
        centerline_points[:, 2] = points[:, 2]

        # 局部半径是到Z轴的距离
        local_radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

        # 理想法线是从Z轴指向该点的向量
        ideal_normals = points - centerline_points

    elif pipe_type == 'bent':
        # 对于弯管，中心线是XY平面上半径为bend_radius的圆弧
        points_shifted = points.copy()
        points_shifted[:, 0] += bend_radius

        dist_from_bend_center_xy = np.sqrt(points_shifted[:, 0] ** 2 + points_shifted[:, 1] ** 2)
        dist_from_bend_center_xy[dist_from_bend_center_xy == 0] = 1e-6  # 避免除零

        # 计算每个点在中心线上的投影点
        centerline_points_x = points_shifted[:, 0] * (bend_radius / dist_from_bend_center_xy)
        centerline_points_y = points_shifted[:, 1] * (bend_radius / dist_from_bend_center_xy)
        centerline_points_x -= bend_radius

        centerline_points = np.zeros_like(points)
        centerline_points[:, 0] = centerline_points_x
        centerline_points[:, 1] = centerline_points_y

        # 局部半径是点到其中心线投影点的3D距离
        local_radii = np.linalg.norm(points - centerline_points, axis=1)

        # 理想法线是从中心线投影点指向表面点的向量
        ideal_normals = points - centerline_points

    else:
        raise ValueError(f"未知的 pipe_type: {pipe_type}")

    # 归一化理想法线向量
    norm_ideal = np.linalg.norm(ideal_normals, axis=1, keepdims=True)
    norm_ideal[norm_ideal == 0] = 1
    ideal_normals /= norm_ideal

    # 特征1: 到理想表面的距离
    dist_to_surface = local_radii - pipe_radius
    features.append(dist_to_surface)

    # 特征2: 局部点密度
    kdtree = KDTree(points)
    density = kdtree.query_radius(points, r=density_radius, count_only=True)
    features.append(density)

    # 特征3: 法线向量偏差
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    point_normals = np.asarray(pcd.normals)

    # 确保实际法线方向与理想法线方向大体一致
    dot_products_orientation = np.sum(point_normals * ideal_normals, axis=1)
    point_normals[dot_products_orientation < 0] *= -1.0

    # 计算修正后的法线夹角
    dot_product = np.clip(np.sum(point_normals * ideal_normals, axis=1), -1.0, 1.0)
    angle_deviation = np.arccos(dot_product)
    features.append(angle_deviation)

    # 将所有特征堆叠成一个矩阵
    feature_matrix = np.vstack(features).T

    return feature_matrix


# ==================== 模型训练模块 ====================
def train_model(train_data_dir, model_save_path, pipe_radius=100.0, bend_radius=800.0):
    """
    训练两阶段式精炼分类器 (RF + XGBoost)

    参数:
        train_data_dir: 训练数据目录
        model_save_path: 模型保存路径
        pipe_radius: 管道半径
        bend_radius: 弯管弯曲半径
    """
    if not os.path.exists(train_data_dir):
        print(f"错误: 训练集目录 '{train_data_dir}' 不存在。")
        return

    train_files = sorted([f for f in os.listdir(train_data_dir) if f.endswith('.xyz')])
    if not train_files:
        print(f"错误: 在训练集目录 '{train_data_dir}' 中未找到 .xyz 文件。")
        return

    all_features = []
    all_labels = []

    print(f"将从 {len(train_files)} 个训练文件中加载数据并提取通用特征...")
    for i, filename in enumerate(train_files):
        print(f"  处理文件: {filename} ({i + 1}/{len(train_files)})")
        file_path = os.path.join(train_data_dir, filename)

        if 'bent' in filename:
            pipe_type = 'bent'
        elif 'straight' in filename:
            pipe_type = 'straight'
        else:
            print(f"  警告: 无法从文件名 '{filename}' 判断管道类型，跳过。")
            continue

        loaded_data = np.loadtxt(file_path, skiprows=1)
        points = loaded_data[:, :3]
        labels = loaded_data[:, 3].astype(int)

        features = extract_features_universal(
            points,
            pipe_type=pipe_type,
            pipe_radius=pipe_radius,
            bend_radius=bend_radius
        )

        all_features.append(features)
        all_labels.append(labels)

    X_train = np.vstack(all_features)
    y_train = np.hstack(all_labels)

    print(f"\n特征提取完成。总共 {X_train.shape[0]} 个点用于训练。")
    print(f"特征维度: {X_train.shape[1]}")
    print(f"缺陷点比例: {np.mean(y_train):.2%}")

    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    print(f"类别权重: {class_weight_dict}")

    # 第一阶段：训练随机森林
    print("\n--- 阶段1: 训练随机森林初筛模型 ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=class_weight_dict,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("随机森林模型训练完成。")

    # 使用RF的预测概率作为新特征
    print("\n--- 使用RF概率增强特征集 ---")
    rf_proba_features = rf_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
    X_train_augmented = np.hstack([X_train, rf_proba_features])
    print(f"增强后的特征维度: {X_train_augmented.shape[1]}")

    # 第二阶段：训练XGBoost精炼模型
    print("\n--- 阶段2: 训练XGBoost精炼判别模型 ---")
    scale_pos_weight = class_weight_dict[1] / class_weight_dict[0]

    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist'
    )

    xgb_model.fit(X_train_augmented, y_train, verbose=False)
    print("XGBoost模型训练完成!")

    # 将两个模型打包保存
    models_to_save = {
        'rf_model': rf_model,
        'xgb_model': xgb_model
    }
    joblib.dump(models_to_save, model_save_path)
    print(f"\n创新的两阶段模型已成功保存到: {model_save_path}")


# ==================== 缺陷检测模块 ====================
def find_optimal_eps(points, min_samples):
    """
    使用K-距离图和KneeLocator自动寻找最佳eps

    参数:
        points: 点云数据
        min_samples: 最小样本数

    返回:
        最优的eps值
    """
    k = min_samples
    neighbors = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    neighbors_fit = neighbors.fit(points)
    distances, indices = neighbors_fit.kneighbors(points)

    distances = np.sort(distances[:, k - 1], axis=0)
    x = np.arange(len(distances))

    # 使用kneed库找到拐点
    try:
        from kneed import KneeLocator
        kneedle = KneeLocator(x, distances, S=1.0, curve='convex', direction='increasing')
        if kneedle.elbow:
            return distances[kneedle.elbow]
    except ImportError:
        print("警告: 未安装kneed库，使用默认eps计算方式")

    # 在找不到明显拐点时，返回一个基于距离均值和标准差的估计值
    return np.mean(distances) + 2 * np.std(distances)


def cluster_defects_and_get_info_adaptive_eps(points, predicted_labels, min_samples=150):
    """
    对检测出的缺陷点进行聚类,并动态计算eps

    参数:
        points: 原始点云
        predicted_labels: 预测的标签
        min_samples: 最小样本数

    返回:
        缺陷区域信息列表
    """
    defect_points = points[predicted_labels == 1]
    if defect_points.shape[0] < min_samples:
        return []

    # 动态计算eps
    optimal_eps = find_optimal_eps(defect_points, min_samples)

    # 使用计算出的eps进行DBSCAN
    dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples, n_jobs=-1)
    cluster_labels = dbscan.fit_predict(defect_points)

    unique_labels = set(cluster_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    defect_regions = []
    for label in unique_labels:
        cluster_mask = (cluster_labels == label)
        cluster_points = defect_points[cluster_mask]

        if cluster_points.shape[0] > 0:
            center = np.mean(cluster_points, axis=0)
            min_coords = np.min(cluster_points, axis=0)
            max_coords = np.max(cluster_points, axis=0)
            defect_regions.append({
                'center': center,
                'bbox_min': min_coords,
                'bbox_max': max_coords,
                'point_count': cluster_points.shape[0],
                'cluster_id': label
            })

    return defect_regions


def detect_defects(test_data_dir, model_path, report_file, pipe_radius=100.0, bend_radius=800.0):
    """
    使用训练好的模型检测缺陷

    参数:
        test_data_dir: 测试数据目录
        model_path: 模型路径
        report_file: 报告文件路径
        pipe_radius: 管道半径
        bend_radius: 弯管弯曲半径
    """
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在。请先运行训练函数。")
        return

    if not os.path.exists(test_data_dir):
        print(f"错误: 测试集目录 '{test_data_dir}' 不存在。")
        return

    print(f"正在从 '{model_path}' 加载创新的两阶段模型...")
    models = joblib.load(model_path)
    rf_model = models['rf_model']
    xgb_model = models['xgb_model']
    print("模型加载成功。")

    test_files = sorted([f for f in os.listdir(test_data_dir) if f.endswith('.xyz')])
    if not test_files:
        print(f"错误: 在测试集目录 '{test_data_dir}' 中未找到 .xyz 文件。")
        return

    all_metrics = []
    total_detected_defects = 0

    with open(report_file, 'w', encoding='utf-8') as report_f:
        print(f"开始在 {len(test_files)} 个测试集上评估标准DBSCAN模型...")
        report_f.write(f"开始在 {len(test_files)} 个测试集上评估标准DBSCAN模型...\n")

        for i, filename in enumerate(test_files):
            header = f"\n{'=' * 25} 处理文件: {filename} ({i + 1}/{len(test_files)}) {'=' * 25}"
            print(header)
            report_f.write(header + '\n')

            file_path = os.path.join(test_data_dir, filename)

            pipe_type = 'bent' if 'bent' in filename else 'straight'
            loaded_data = np.loadtxt(file_path, skiprows=1)
            points = loaded_data[:, :3]
            truth_labels = loaded_data[:, 3].astype(int)

            info_msg = f"  已加载 {points.shape[0]} 个点，管道类型: {pipe_type}"
            print(info_msg)
            report_f.write(info_msg + '\n')

            print("  正在提取通用特征...")
            report_f.write("  正在提取通用特征...\n")

            features = extract_features_universal(points, pipe_type=pipe_type,
                                                  pipe_radius=pipe_radius, bend_radius=bend_radius)

            print("  正在使用两阶段模型预测...")
            report_f.write("  正在使用两阶段模型预测...\n")

            rf_proba_features = rf_model.predict_proba(features)[:, 1].reshape(-1, 1)
            features_augmented = np.hstack([features, rf_proba_features])
            predicted_labels = xgb_model.predict(features_augmented)

            print("  正在使用标准DBSCAN进行聚类...")
            report_f.write("  正在使用标准DBSCAN进行聚类...\n")

            defect_clusters = cluster_defects_and_get_info_adaptive_eps(
                points, predicted_labels, min_samples=150
            )
            total_detected_defects += len(defect_clusters)

            result_msg = f"\n  检测结果 (标准DBSCAN):\n    - 共检测到 {len(defect_clusters)} 个独立的缺陷区域。"
            print(result_msg)
            report_f.write(result_msg + '\n')

            if not defect_clusters:
                no_defect_msg = "    - 未找到缺陷。"
                print(no_defect_msg)
                report_f.write(no_defect_msg + '\n')
            else:
                for j, region in enumerate(defect_clusters):
                    center = region['center']
                    cluster_info = f"    - 缺陷 {j + 1} (ID:{region['cluster_id']}) 中心: (X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}), 点数: {region['point_count']}"
                    print(cluster_info)
                    report_f.write(cluster_info + '\n')

            precision = precision_score(truth_labels, predicted_labels, zero_division=0)
            recall = recall_score(truth_labels, predicted_labels, zero_division=0)
            f1 = f1_score(truth_labels, predicted_labels, zero_division=0)
            accuracy = accuracy_score(truth_labels, predicted_labels)
            all_metrics.append({'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy})

            eval_msg = f"\n  评估结果 (逐点比较):\n    - 精确率 (Precision): {precision:.4f}\n    - 召回率 (Recall):    {recall:.4f}\n    - F1分数 (F1-Score):  {f1:.4f}\n    - 整体准确率 (Accuracy): {accuracy:.4f}"
            print(eval_msg)
            report_f.write(eval_msg + '\n')

        if all_metrics:
            avg_precision = np.mean([m['precision'] for m in all_metrics])
            avg_recall = np.mean([m['recall'] for m in all_metrics])
            avg_f1 = np.mean([m['f1'] for m in all_metrics])
            avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
            avg_detected_count = total_detected_defects / len(test_files)

            summary_header = f"\n{'=' * 30} 最终评估总结 (标准DBSCAN模型) {'=' * 30}"
            print(summary_header)
            report_f.write(summary_header + '\n')

            summary_msg = f"在 {len(test_files)} 个测试集上的平均性能指标:\n  - 平均检测缺陷数 (DBSCAN): {avg_detected_count:.2f}\n  - 平均精确率:   {avg_precision:.4f}\n  - 平均召回率:   {avg_recall:.4f}\n  - 平均F1分数:   {avg_f1:.4f}\n  - 平均准确率:   {avg_accuracy:.4f}"
            print(summary_msg)
            report_f.write(summary_msg + '\n')

    print(f"\n--- 脚本执行完毕 ---")
    print(f"详细的标准DBSCAN评估报告已保存至: {report_file}")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 配置参数
    TRAIN_DATA_DIR = "defect_datasets_mixed/train_test"
    MODEL_SAVE_PATH = "two_stage_defect_model.joblib"
    TEST_DATA_DIR = "defect_datasets_mixed/test"
    REPORT_FILE = "dbscan_evaluation_report.txt"
    PIPE_RADIUS = 100.0
    BEND_RADIUS = 800.0

    # 训练模型
    # train_model(TRAIN_DATA_DIR, MODEL_SAVE_PATH, PIPE_RADIUS, BEND_RADIUS)

    # 检测缺陷
    # detect_defects(TEST_DATA_DIR, MODEL_SAVE_PATH, REPORT_FILE, PIPE_RADIUS, BEND_RADIUS)