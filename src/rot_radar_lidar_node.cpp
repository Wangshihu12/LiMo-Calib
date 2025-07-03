#include <iostream>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <deque>
#include "motor_lidar_calib/CustomMsg.h"
#include "lidardata.h"
#include <std_msgs/String.h>

#include <Eigen/Dense>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/PointStamped.h>
#include <pcl_conversions/pcl_conversions.h>

struct ParaConfig
{
  std::string lidar_topic;
  std::string radar_topic;
  std::string rotor_topic;

  std::string bagFilePath;

  double bag_start;
  double bag_durr;

  std::string rotlidarFilePath;

  // calibration
  Eigen::Matrix3d R_rot_li;
  Eigen::Matrix3d R_rot_ra;
};

ros::Publisher pub_livox_points;
ros::Publisher pub_radar_points;

ParaConfig config_;
std::deque<motor_lidar_calib::CustomMsgPtr> livox_lidar_messages_;
std::deque<sensor_msgs::PointCloudPtr> radar_messages_;
std::deque<std::pair<double,double>> rotor_messages_;

/**
 * [功能描述]：从ROS参数服务器加载配置参数，包括话题名称、文件路径、时间参数和标定参数
 * @param nh：ROS节点句柄，用于获取参数服务器中的配置参数
 * @return 无返回值，配置参数存储在全局变量config_中
 */
void loadConfig(ros::NodeHandle& nh)
{
  // 获取配置文件名称参数（当前未使用）
  std::string config_name;
  nh.param<std::string>("config_name", config_name, "/config/calib.yaml");

  // 获取传感器话题名称参数
  nh.param<std::string>("lidar_topic", config_.lidar_topic, "");        // 激光雷达话题名称
  //nh.param<std::string>("radar_topic", config_.radar_topic, "");      // 雷达话题名称（已注释）
  nh.param<std::string>("rotor_topic", config_.rotor_topic, "");        // 旋转电机话题名称

  // 获取数据包处理时间参数
  nh.param<double>("bag_start", config_.bag_start, 0);                  // 数据包开始处理时间，默认为0秒
  nh.param<double>("bag_durr", config_.bag_durr, -1);                  // 数据包处理持续时间，-1表示处理到结束

  // 获取文件路径参数
  nh.param<std::string>("rotlidarFilePath", config_.rotlidarFilePath, "");  // 旋转激光雷达数据输出文件路径
  nh.param<std::string>("bagFilePath", config_.bagFilePath, "");            // 输入数据包文件路径
  ROS_INFO("config path: %s\n",config_.bagFilePath.c_str());               // 输出配置路径信息

  // 获取标定参数：旋转角度字符串
  std::string calibration_str;
  nh.param<std::string>("calibration_angles", calibration_str, "0,-1.047,0");  // 标定角度，格式为"ax,ay,az"
  double ax,ay,az;  // 分别表示绕X、Y、Z轴的旋转角度
  sscanf(calibration_str.c_str(), "%lf,%lf,%lf", &ax, &ay, &az);  // 解析角度字符串

  // 构建旋转矩阵：从旋转电机坐标系到激光雷达坐标系的变换矩阵
  // 按照ZYX欧拉角顺序构建旋转矩阵
  config_.R_rot_li = Eigen::AngleAxisd(az,Eigen::Vector3d::UnitZ()) * 
                     Eigen::AngleAxisd(ay,Eigen::Vector3d::UnitY()) * 
                     Eigen::AngleAxisd(ax,Eigen::Vector3d::UnitX());
  
  // 构建旋转矩阵：从旋转电机坐标系到雷达坐标系的变换矩阵（当前为单位矩阵）
  config_.R_rot_ra = Eigen::AngleAxisd(0,Eigen::Vector3d::UnitZ());
}

void getValidRotorMeasures(double begTime, double endTime, std::vector<std::pair<double,double>>& validMeasures)
{
  for (auto& measure: rotor_messages_)
  {
    // std::cout<<"tmp time: " <<measure.first <<"\n";
    if(measure.first < begTime)
    {
      // rotor_messages_.pop_front();
    }
    if(measure.first > begTime && measure.first < endTime)
    {
      validMeasures.push_back(measure);
      // rotor_messages_.pop_front();
    }
    if(measure.first > endTime)
    {
      validMeasures.push_back(measure);
      break;
    }
  }

}

/**
 * [功能描述]：处理激光雷达消息，将激光雷达点云数据转换为机体坐标系，并发布变换后的点云
 * @param lidar_msg：激光雷达自定义消息指针，包含点云数据和时间戳信息
 * @return 无返回值，处理后的点云通过ROS话题发布
 */
void LiDARHandler(motor_lidar_calib::CustomMsgPtr &lidar_msg)
{
  ROS_INFO("lidar_topic");  // 输出调试信息
  
  // 创建原始点云容器
  RTPointCloud::Ptr raw_cloud(new RTPointCloud);
  raw_cloud->header.stamp = lidar_msg->header.stamp.toNSec();  // 设置点云时间戳（纳秒）
  
  // 遍历激光雷达消息中的所有点，转换为RTPoint格式
  for(const auto& p : lidar_msg->points){
    RTPoint point;
    int line_num = (int)p.line;  // 获取激光线号

    if(p.reflectivity < 1) continue;  // 过滤反射率过低的点（噪声点）
    
    // 填充点云数据
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    point.intensity = p.reflectivity;  // 强度值使用反射率
    point.time = p.offset_time/1e9;    // 时间偏移量转换为秒
    point.ring = (line_num);           // 激光线环号

    raw_cloud->push_back(point);  // 添加到原始点云
  }

  // 计算激光雷达数据的时间范围
  double beg_lidar_time = raw_cloud->points[0].time + raw_cloud->header.stamp/1e9;      // 第一个点的绝对时间
  double end_lidar_time = raw_cloud->points.back().time + raw_cloud->header.stamp/1e9;  // 最后一个点的绝对时间

  // 获取对应时间范围内的旋转电机角度测量数据
  std::vector<std::pair<double,double>> validMeasures;  // 存储有效的旋转角度测量值
  getValidRotorMeasures(beg_lidar_time-0.1, end_lidar_time+0.1, validMeasures);  // 扩展0.1秒的时间窗口

  ROS_INFO("validMeasures.size():%d",validMeasures.size());  // 输出有效测量数量

  // 检查旋转角度测量数据是否充足
  if(validMeasures.size() < 20)
  {
    return;  // 如果测量数据不足20个，则退出处理
  }

  // 将点云从激光雷达坐标系变换到机体坐标系
  RTPointCloud::Ptr transformed_cloud(new RTPointCloud);  // 创建变换后的点云容器
  
  for (size_t i = 0; i < raw_cloud->size(); i++)
  {
    RTPoint point = raw_cloud->points[i];  // 获取原始点
    RTPoint pt_trans = point;              // 创建变换后的点

    // 将点坐标转换为Eigen向量格式
    Eigen::Vector3d pt_e;
    pt_e[0] = point.x;
    pt_e[1] = point.y;
    pt_e[2] = point.z;
    
    double tmpTime = raw_cloud->header.stamp/1e9 + point.time;  // 计算点的绝对时间
    double rotAngular = 0;  // 旋转角度
    bool valid = false;     // 标记是否找到有效的旋转角度
    
    // 在旋转角度测量数据中查找对应时间的角度值
    for (size_t j = 0; j < validMeasures.size()-1; j++)
    {
      // 如果点的时间在两个测量值之间，进行线性插值
      if(tmpTime > validMeasures[j].first && tmpTime < validMeasures[j+1].first)
      {
        // 计算角度差，处理角度跳变（-π到π的跳变）
        double anglediff = validMeasures[j+1].second-validMeasures[j].second;

        if(anglediff > M_PI)          // 处理正向跳变
        {
          anglediff-=2*M_PI;
        }
        if(anglediff < -M_PI)         // 处理负向跳变
        {
          anglediff+=2*M_PI;
        }
        
        // 线性插值计算当前时间的旋转角度
        rotAngular = (tmpTime-validMeasures[j].first)/(validMeasures[j+1].first-validMeasures[j].first)*anglediff + validMeasures[j].second;
        valid = 1;  // 标记找到有效角度
        break;
      }
    }
    
    if(!valid)  // 如果没有找到有效的旋转角度，跳过该点
    {
      continue;
    }
    
    // 应用坐标变换：先应用标定矩阵，再应用旋转角度变换
    Eigen::Vector3d pt_e_trans = Eigen::AngleAxisd(rotAngular,Eigen::Vector3d::UnitZ()) * config_.R_rot_li*pt_e;
    pt_trans.x = pt_e_trans[0];  // 更新变换后的x坐标
    pt_trans.y = pt_e_trans[1];  // 更新变换后的y坐标
    pt_trans.z = pt_e_trans[2];  // 更新变换后的z坐标

    transformed_cloud->push_back(pt_trans);  // 添加到变换后的点云
  }
  
  // 发布变换后的点云
  sensor_msgs::PointCloud2 pcd_msg;
  pcl::toROSMsg(*transformed_cloud,pcd_msg);    // 将PCL点云转换为ROS消息格式
  pcd_msg.header.stamp = lidar_msg->header.stamp;  // 设置时间戳
  pcd_msg.header.frame_id = "map";              // 设置坐标系为map
  pub_livox_points.publish(pcd_msg);            // 发布点云消息
}

/**
 * [功能描述]：处理雷达消息，将雷达点云数据转换为机体坐标系，并发布变换后的点云
 * @param radar_msg：雷达点云消息指针，包含点云数据和时间戳信息
 * @return 无返回值，处理后的点云通过ROS话题发布
 */
void RadarHandler(sensor_msgs::PointCloudPtr &radar_msg)
{
  // 创建原始点云容器
  RTPointCloud::Ptr raw_cloud(new RTPointCloud);
  raw_cloud->header.stamp = radar_msg->header.stamp.toNSec();  // 设置点云时间戳（纳秒）
  
  // 遍历雷达消息中的所有点，转换为RTPoint格式
  for (size_t i = 0; i < radar_msg->points.size(); i++)
  {
    // 提取点坐标
    double x = radar_msg->points[i].x;
    double y = radar_msg->points[i].y;
    double z = radar_msg->points[i].z;
    radar_msg->channels.size();  // 获取通道数量（未使用）

    // 创建RTPoint并填充数据
    RTPoint point;
    point.x = x;
    point.y = y;
    point.z = z;
    point.time = 0;  // 雷达数据时间偏移为0（瞬时测量）

    raw_cloud->push_back(point);  // 添加到原始点云
  }

  // 计算雷达数据的时间范围（雷达是瞬时测量，开始和结束时间相同）
  double beg_lidar_time = raw_cloud->header.stamp/1e9;  // 雷达数据开始时间
  double end_lidar_time = raw_cloud->header.stamp/1e9;  // 雷达数据结束时间

  // 获取对应时间范围内的旋转电机角度测量数据
  std::vector<std::pair<double,double>> validMeasures;  // 存储有效的旋转角度测量值
  getValidRotorMeasures(beg_lidar_time-1, end_lidar_time+1, validMeasures);  // 扩展1秒的时间窗口

  ROS_INFO("radar validMeasures.size():%d",validMeasures.size());  // 输出有效测量数量

  // 检查旋转角度测量数据是否充足
  if(validMeasures.size() < 10)
  {
    return;  // 如果测量数据不足10个，则退出处理
  }

  // 判断旋转电机是否在运动（静止检测）
  double averageValue = 0;  // 平均角度值
  for (size_t i = 0; i < validMeasures.size() ; i++)
  {
    averageValue += validMeasures[i].second;  // 累加所有角度值
  }
  averageValue /= validMeasures.size();  // 计算平均值

  // 计算角度变化的最大差值
  double diff = 0;  // 最大角度差
  for (size_t i = 0; i < validMeasures.size()-1 ; i++)
  {
    double tmpdiff = abs(validMeasures[i+1].second - validMeasures[i].second);  // 相邻测量值的差值
    if(tmpdiff > diff)
    {
      diff = tmpdiff;  // 更新最大差值
    }
  }

  // 如果角度变化超过阈值，说明电机在运动，退出处理
  if(diff > 3. / 8192.0 *2*M_PI)  // 阈值约为0.0023弧度（约0.13度）
  {
    return;  // 只在电机静止时处理雷达数据
  }

  // 将点云从雷达坐标系变换到机体坐标系
  RTPointCloud::Ptr transformed_cloud(new RTPointCloud);  // 创建变换后的点云容器
  
  for (size_t i = 0; i < raw_cloud->size(); i++)
  {
    RTPoint point = raw_cloud->points[i];  // 获取原始点
    RTPoint pt_trans = point;              // 创建变换后的点

    // 将点坐标转换为Eigen向量格式
    Eigen::Vector3d pt_e;
    pt_e[0] = point.x;
    pt_e[1] = point.y;
    pt_e[2] = point.z;
    
    double tmpTime = raw_cloud->header.stamp/1e9 + point.time;  // 计算点的绝对时间
    double rotAngular = 0;  // 旋转角度
    bool valid = false;     // 标记是否找到有效的旋转角度
    
    // 在旋转角度测量数据中查找对应时间的角度值
    for (size_t j = 0; j < validMeasures.size()-1; j++)
    {
      // 如果点的时间在两个测量值之间，进行线性插值
      if(tmpTime > validMeasures[j].first && tmpTime < validMeasures[j+1].first)
      {
        // 线性插值计算当前时间的旋转角度（雷达不需要处理角度跳变）
        rotAngular = (tmpTime-validMeasures[j].first)/(validMeasures[j+1].first-validMeasures[j].first)*(validMeasures[j+1].second-validMeasures[j].second) + validMeasures[j].second;
        valid = 1;  // 标记找到有效角度
        break;
      }
    }
    
    if(!valid)  // 如果没有找到有效的旋转角度，跳过该点
    {
      continue;
    }
    
    // 应用坐标变换：先应用标定矩阵，再应用旋转角度变换
    Eigen::Vector3d pt_e_trans = Eigen::AngleAxisd(rotAngular,Eigen::Vector3d::UnitZ()) * config_.R_rot_ra*pt_e;
    pt_trans.x = pt_e_trans[0];  // 更新变换后的x坐标
    pt_trans.y = pt_e_trans[1];  // 更新变换后的y坐标
    pt_trans.z = pt_e_trans[2];  // 更新变换后的z坐标

    transformed_cloud->push_back(pt_trans);  // 添加到变换后的点云
  }
  
  // 发布变换后的点云
  sensor_msgs::PointCloud2 pcd_msg;
  pcl::toROSMsg(*transformed_cloud,pcd_msg);    // 将PCL点云转换为ROS消息格式
  pcd_msg.header.stamp = radar_msg->header.stamp;  // 设置时间戳
  pcd_msg.header.frame_id = "map";              // 设置坐标系为map
  pub_radar_points.publish(pcd_msg);            // 发布点云消息
}

/**
 * [功能描述]：离线处理rosbag数据，从bag文件中读取激光雷达和旋转电机数据进行处理
 * @param 无参数
 * @return 无返回值，处理完成后自动结束
 */
void SpinOffline()
{
  // 创建并打开rosbag文件
  rosbag::Bag bag;
  bag.open(config_.bagFilePath, rosbag::bagmode::Read);  // 以只读模式打开bag文件

  // 设置要处理的话题列表
  std::vector<std::string> topics;
  topics.push_back(config_.lidar_topic);  // 添加激光雷达话题
  //topics.push_back(config_.radar_topic);  // 雷达话题（已注释）
  topics.push_back(config_.rotor_topic);   // 添加旋转电机话题

  // 创建rosbag视图对象用于数据查询
  rosbag::View view_;      // 用于指定时间范围的视图
  rosbag::View view_full;  // 用于获取完整bag文件信息的视图
  view_full.addQuery(bag); // 添加完整bag文件查询
  
  // 计算数据处理的时间范围
  ros::Time time_init = view_full.getBeginTime();                    // 获取bag文件开始时间
  time_init += ros::Duration(config_.bag_start);                    // 加上配置的开始偏移时间
  ros::Time time_finish = (config_.bag_durr < 0)                    // 根据配置设置结束时间
                              ? view_full.getEndTime()               // 如果持续时间为负，处理到文件结束
                              : time_init + ros::Duration(config_.bag_durr);  // 否则处理指定持续时间
  
  // 创建指定话题和时间范围的查询视图
  view_.addQuery(bag, rosbag::TopicQuery(topics), time_init, time_finish);

  // 检查是否有数据可处理
  if (view_.size() == 0) {
    ROS_ERROR("No messages to play on specified topics.  Exiting.");  // 没有找到指定话题的消息
    ros::shutdown();  // 关闭ROS节点
    return;
  }

  // 记录最新的旋转电机时间戳，用于时间同步
  double latest_rotor_time = -1;
  
  // 遍历bag文件中的所有消息
  for (const rosbag::MessageInstance& m : view_) {
    ros::Time ros_bag_time = m.getTime();  // 获取消息时间戳
    
    // 处理激光雷达消息
    if (m.getTopic() == config_.lidar_topic)
    {
      ROS_INFO("lidar_topic");  // 输出调试信息
      auto lidar_msg = m.instantiate<motor_lidar_calib::CustomMsg>();  // 实例化激光雷达消息
      livox_lidar_messages_.push_back(lidar_msg);  // 添加到激光雷达消息队列

      // 处理激光雷达消息（需要等待足够的旋转电机数据）
      if(livox_lidar_messages_.size()>0)
      {
        auto lidar_msg = livox_lidar_messages_.front();  // 获取队列中最早的消息
        // 检查是否有足够的旋转电机数据（激光雷达时间+0.1秒 < 最新旋转电机时间）
        if((lidar_msg->header.stamp.toSec() + 0.1) < latest_rotor_time)
        {
          LiDARHandler(lidar_msg);          // 处理激光雷达数据
          livox_lidar_messages_.pop_front(); // 从队列中移除已处理的消息
        } 
      }
      // usleep(1e5);  // 休眠100毫秒（已注释）
    }
    
    // 处理雷达消息（整个部分已注释）
    // else if (m.getTopic() == config_.radar_topic)
    // {
    //   // ROS_INFO("radar_topic");
    //   auto radar_msg = m.instantiate<sensor_msgs::PointCloud>();
    //   radar_messages_.push_back(radar_msg);

    //   // handle lidar message
    //   if(radar_messages_.size()>0)
    //   {
    //     auto radar_msg = radar_messages_.front();
    //     if(radar_msg->header.stamp.toSec() < latest_rotor_time)
    //     {
    //       RadarHandler(radar_msg);
    //       radar_messages_.pop_front();
    //     } 
    //   }
    //   usleep(1e4);

    // }

    // 处理旋转电机消息
    else if (m.getTopic() == config_.rotor_topic)
    {
      ROS_INFO("rotor_topic");  // 输出调试信息
      
      std_msgs::String::Ptr point_msg = m.instantiate<std_msgs::String>();  // 实例化字符串消息
      double stamp = m.getTime().toSec();  // 获取消息时间戳（秒）
      //double stamp = point_msg->header.stamp.toSec();  // 替代方案（已注释）
      double position_value;  // 用于存储解析的位置值

      // 检查消息是否有效
      if (point_msg) {
        // 解析位置值：从"Position:xxx,"格式的字符串中提取位置值
        if (sscanf(point_msg->data.c_str(), "Position:%lf,", &position_value) == 1) {
            double angular = -position_value / 180 * M_PI;  // 将位置值转换为角度（弧度）
            rotor_messages_.push_back(std::make_pair(stamp, angular));  // 添加到旋转电机消息队列
            latest_rotor_time = stamp;  // 更新最新旋转电机时间戳
        } else {
            ROS_WARN("Failed to parse Position from rotor message: %s", point_msg->data.c_str());  // 解析失败警告
        }
      }
    }
  }
  // visualize  // 可视化（待实现）
}

void saveLiDARRotorFile(std::string filename)
{

  FILE* file = fopen(filename.c_str(),"w");
  for (motor_lidar_calib::CustomMsgPtr lidar_msg: livox_lidar_messages_)
  {
    // ROS_INFO("%lf",msg->header.stamp.toSec());
    RTPointCloud::Ptr raw_cloud(new RTPointCloud);
    raw_cloud->header.stamp = lidar_msg->header.stamp.toNSec();
    // raw_cloud_->clear();
    for(const auto& p : lidar_msg->points){
      RTPoint point;
      int line_num = (int)p.line;
      
      if(p.reflectivity < 1) continue;
      point.x = p.x;
      point.y = p.y;
      point.z = p.z;
      point.intensity = p.reflectivity;
      point.time = p.offset_time/1e9;
      point.ring = (line_num);

      raw_cloud->push_back(point);
    }

    // find the corresponding rotor angulars
    double beg_lidar_time = raw_cloud->points[0].time + raw_cloud->header.stamp/1e9;
    double end_lidar_time = raw_cloud->points.back().time + raw_cloud->header.stamp/1e9;

    std::vector<std::pair<double,double>> validMeasures;
    getValidRotorMeasures(beg_lidar_time-0.1, end_lidar_time+0.1, validMeasures);

    ROS_INFO("validMeasures.size():%d",validMeasures.size());

    if(validMeasures.size() < 20)
    {
      return;
    }

    // transform to body frame
    RTPointCloud::Ptr transformed_cloud(new RTPointCloud);
    for (size_t i = 0; i < raw_cloud->size(); i++)
    {
      RTPoint point = raw_cloud->points[i];
      RTPoint pt_trans = point;

      Eigen::Vector3d pt_e;
      pt_e[0] = point.x;
      pt_e[1] = point.y;
      pt_e[2] = point.z;
      double tmpTime = raw_cloud->header.stamp/1e9 + point.time;
      double rotAngular = 0;
      bool valid = false;
      for (size_t j = 0; j < validMeasures.size()-1; j++)
      {
        if(tmpTime > validMeasures[j].first && tmpTime < validMeasures[j+1].first)
        {
          double anglediff = validMeasures[j+1].second-validMeasures[j].second;

          if(anglediff > M_PI)
          {
            anglediff-=2*M_PI;
          }
          if(anglediff < -M_PI)
          {
            anglediff+=2*M_PI;
          }
          rotAngular = (tmpTime-validMeasures[j].first)/(validMeasures[j+1].first-validMeasures[j].first)*anglediff + validMeasures[j].second;
          valid = 1;
          break;
        }
      }
      if(!valid)
      {
        continue;
      }
      Eigen::Vector3d pt_e_trans = Eigen::AngleAxisd(rotAngular,Eigen::Vector3d::UnitZ()) * config_.R_rot_li*pt_e;
      pt_trans.x = pt_e_trans[0];
      pt_trans.y = pt_e_trans[1];
      pt_trans.z = pt_e_trans[2];

      transformed_cloud->push_back(pt_trans);

      // save to a data log
      // Initial (x y z) Raw (x y z) intensity angle time
      fprintf(file,"%lf %lf %lf %lf %lf %lf %d %lf %lf \n", pt_trans.x, pt_trans.y, pt_trans.z, point.x, point.y, point.z, static_cast<int>(point.intensity), rotAngular, tmpTime);
    }
  }

  fclose(file);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "rot_radar_lidar");
  ros::NodeHandle nh("~");
  loadConfig(nh);

  pub_livox_points = nh.advertise<sensor_msgs::PointCloud2>("/livox/points", 10);
 // pub_radar_points = nh.advertise<sensor_msgs::PointCloud2>("/radar/points", 10);

  SpinOffline();

  // save to LiDAR Rotor file
  saveLiDARRotorFile(config_.rotlidarFilePath);

  return 0;
}