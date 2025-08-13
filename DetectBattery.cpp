#include <iostream>
#include <vector>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>

struct BatteryParams {
    pcl::PointXYZ center;
    pcl::PointXYZ axis;
    float radius;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
};

int main(int argc, char** argv)
{
    // 1. 读取点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile<pcl::PointXYZ>("D:\\Soft\\VisualStudio2022\\Projects\\PCL_test\\数据\\电池补充\\data10.ply", *cloud) == -1)
    {
        PCL_ERROR("读取文件失败\n");
        return -1;
    }
    std::cout << "成功读取点云，点数: " << cloud->size() << std::endl;

    // 2. 下采样
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(0.5f, 0.5f, 0.5f);
    voxel_filter.filter(*cloud_filtered);

    // 3. 去除离群点
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clean(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_filtered);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud_clean);
    std::cout << "去除离群点，剩余点数: " << cloud_clean->size() << std::endl;

    // 4. RANSAC平面分割
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_plane(new pcl::PointCloud<pcl::PointXYZ>);
    {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(5000);
        seg.setDistanceThreshold(8.0);

        seg.setInputCloud(cloud_clean);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            std::cerr << "未检测到平面！" << std::endl;
        }
        else {
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(cloud_clean);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*cloud_no_plane);
            std::cout << "平面分割后剩余点数: " << cloud_no_plane->size() << std::endl;
        }
    }

    // 5. 欧几里得聚类
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_no_plane);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(2.0);  // 2cm距离阈值
    ec.setMinClusterSize(100);    // 最小点数
    ec.setMaxClusterSize(10000);  // 最大点数
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_no_plane);
    ec.extract(cluster_indices);

    std::cout << "\n发现聚类数量: " << cluster_indices.size() << std::endl;

    // 存储电池参数
    std::vector<BatteryParams> battery_params;

    // 6. 对每个聚类进行圆柱分割
    int cluster_count = 1;
    for (const auto& indices : cluster_indices) {
        // 提取单个聚类点云
        std::cout << "检查第" << cluster_count << "个聚类是否有圆柱体..." << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*cloud_no_plane, indices.indices, *cluster_cloud);

        // 法线估计
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        ne.setInputCloud(cluster_cloud);
        ne.setSearchMethod(tree);
        ne.setKSearch(50);
        ne.compute(*normals);

        // 圆柱分割
        pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
        pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_CYLINDER);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setNormalDistanceWeight(0.1);
        seg.setMaxIterations(5000);
        seg.setDistanceThreshold(0.5);
        seg.setRadiusLimits(2.0, 20.0);
        seg.setInputCloud(cluster_cloud);
        seg.setInputNormals(normals);
        seg.segment(*inliers_cylinder, *coefficients_cylinder);
        
        if (inliers_cylinder->indices.empty()) {
            std::cerr << "该聚类中未检测到圆柱体！" << std::endl;
            continue;
        }
        else {
            std::cout << "找到一个圆柱体！" << std::endl;
        }

        // 提取圆柱参数
        BatteryParams param;
        param.center.x = coefficients_cylinder->values[0];
        param.center.y = coefficients_cylinder->values[1];
        param.center.z = coefficients_cylinder->values[2];
        param.axis.x = coefficients_cylinder->values[3];
        param.axis.y = coefficients_cylinder->values[4];
        param.axis.z = coefficients_cylinder->values[5];
        param.radius = coefficients_cylinder->values[6];
        param.cloud = cluster_cloud;

        battery_params.push_back(param);
        
        cluster_count++;
    }

    // 7. 输出结果
    std::cout << "\n===== 电池参数报告 =====" << std::endl;
    for (size_t i = 0; i < battery_params.size(); ++i) {
        const auto& p = battery_params[i];
        std::cout << "电池 " << i + 1 << ":\n"
            << "  中心坐标: (" << p.center.x << ", " << p.center.y << ", " << p.center.z << ")\n"
            << "  主轴方向: (" << p.axis.x << ", " << p.axis.y << ", " << p.axis.z << ")\n"
            << "  半径: " << p.radius << "\n"
            << "  点云点数: " << p.cloud->size() << "\n" << std::endl;
    }

    // 8. 可视化
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);

    // 显示原始点云（灰色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> original_color(cloud, 150, 150, 150);
    viewer->addPointCloud(cloud, original_color, "original_cloud");

    // 定义颜色数组
    const std::vector<std::array<int, 3>> colors = {
        {255, 0, 0},   // 红
        {0, 255, 0},   // 绿
        {0, 0, 255},   // 蓝
        {255, 255, 0}, // 黄
        {255, 0, 255}  // 品红
    };

    // 显示每个电池点云及轴线
    for (size_t i = 0; i < battery_params.size(); ++i) {
        const auto& p = battery_params[i];
        std::string cloud_name = "battery_" + std::to_string(i);
        std::string line_name = "axis_" + std::to_string(i);

        // 显示点云
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(
            p.cloud,
            colors[i % 5][0],
            colors[i % 5][1],
            colors[i % 5][2]
        );
        viewer->addPointCloud(p.cloud, color, cloud_name);

        // 显示圆柱轴线
        pcl::PointXYZ end_point;
        end_point.x = p.center.x + p.axis.x * 0.5;
        end_point.y = p.center.y + p.axis.y * 0.5;
        end_point.z = p.center.z + p.axis.z * 0.5;
        viewer->addLine<pcl::PointXYZ>(p.center, end_point,
            colors[i % 5][0] / 255.0, colors[i % 5][1] / 255.0, colors[i % 5][2] / 255.0,
            line_name);
    }

    viewer->addCoordinateSystem(1.0);
    viewer->spin();

    return 0;
}