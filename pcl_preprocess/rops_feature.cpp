/*
 * @Author: Yong Lei yong.lei@momenta.ai
 * @Date: 2023-02-03 10:53:30
 * @LastEditors: Yong Lei yong.lei@momenta.ai
 * @LastEditTime: 2023-02-07 14:09:51
 * @FilePath: /3D-ADS/pcl_preprocess/rops_feature.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <pcl/features/rops_estimation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <iostream>
#include <fstream>
#include<pcl/visualization/pcl_plotter.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>


using namespace std;


pcl::PolygonMesh greedy_projection(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
  // Normal estimation*
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud);
  n.setInputCloud (cloud);
  n.setSearchMethod (tree);
  // n.setKSearch (30);
  n.setRadiusSearch(0.1);

  cout<<"compute normals" <<endl;
  n.compute (*normals);
  //* normals should not contain the point normals + surface curvatures

  // Concatenate the XYZ and normal fields*
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
  //* cloud_with_normals = cloud + normals

  // Create search tree*
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
  tree2->setInputCloud (cloud_with_normals);

  // Initialize objects
  pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
  pcl::PolygonMesh triangles;

  // Set the maximum distance between connected points (maximum edge length)
  gp3.setSearchRadius (0.025);

  // Set typical values for the parameters
  gp3.setMu (2.5);
  gp3.setMaximumNearestNeighbors (100);
  gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
  gp3.setMinimumAngle(M_PI/18); // 10 degrees
  gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
  gp3.setNormalConsistency(false);

  // Get result
  gp3.setInputCloud (cloud_with_normals);
  gp3.setSearchMethod (tree2);
  cout<<"compute triangles" <<endl;
  gp3.reconstruct (triangles);

  // pcl::io::savePLYFile("result.ply", triangles);

  return triangles;
}




int main (int argc, char** argv)
{
  /*-----------------读取点云文件-----------------*/
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  if (pcl::io::loadPCDFile (argv[1], *cloud) == -1)
    return (-1);
  cout<< "read" << argv[1]<<endl;



  /*-----------------三角化-----------------*/
  pcl::PolygonMesh poly = greedy_projection(cloud);
  



  /*-----------------计算RoPS-----------------*/
  float support_radius = 0.01f;
  unsigned int number_of_partition_bins = 5;
  unsigned int number_of_rotations = 3;

  pcl::search::KdTree<pcl::PointXYZ>::Ptr search_method (new pcl::search::KdTree<pcl::PointXYZ>);
  search_method->setInputCloud (cloud);

  pcl::ROPSEstimation <pcl::PointXYZ, pcl::Histogram <135> > feature_estimator;
  feature_estimator.setSearchMethod (search_method);
  feature_estimator.setSearchSurface (cloud);
  feature_estimator.setInputCloud (cloud);
  // feature_estimator.setIndices (indices);
  feature_estimator.setTriangles (poly.polygons);
  feature_estimator.setRadiusSearch (support_radius);
  feature_estimator.setNumberOfPartitionBins (number_of_partition_bins);
  feature_estimator.setNumberOfRotations (number_of_rotations);
  feature_estimator.setSupportRadius (support_radius);

  pcl::PointCloud<pcl::Histogram <135> >::Ptr histograms (new pcl::PointCloud <pcl::Histogram <135> > ());

  cout<<"feature_estimator"<<endl;
  feature_estimator.compute (*histograms);




  
  /*-----------------存储文件-----------------*/
  cout<< "write "<<argv[2]<<endl;
  // //写文件
  string out = argv[2];
  ofstream Outfile(out);
  for(int i = 0;i<histograms->points.size();i++){
    for(int j = 0;j<135;j++){
      if(j< 134){
        Outfile << histograms->points[i].histogram[j] << ' ';
      }else{
        Outfile << histograms->points[i].histogram[j] << endl;
      }
    }
  }
  Outfile.close();
  cout<< "**---done---**"<<endl;




  /*-----------------可视化-----------------*/
  // pcl::visualization::PCLPlotter::Ptr plotter (new pcl::visualization::PCLPlotter());
  // plotter->addFeatureHistogram<pcl::Histogram <135>>(*histograms,135);
  // plotter->spin();

  return (0);
}
