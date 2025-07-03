#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ceres/ceres.h>
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

struct PlaneFactorData{
    std::vector<Eigen::Vector3d> rawPts;     
    std::vector<double> rotorAngles;  
    Eigen::Vector3d norm;                    
};

struct PolarNormal {
    double theta;  // angle from z-axis [0, pi]
    double phi;    // angle in x-y plane [-pi, pi]
};

PolarNormal cartesianToPolar(const Eigen::Vector3d& normal) {
    PolarNormal polar;
    polar.theta = std::acos(normal.z());
    polar.phi = std::atan2(normal.y(), normal.x());
    return polar;
}
struct CorrespondenceInfo{
    Eigen::Vector3d ref_pt;       
    Eigen::Vector3d src_pt;     
    double ref_angle;          
    double src_angle;            
    double ref_angle_speed;       
    double src_angle_speed;       
    Eigen::Vector3d norm;      
    double weight = 1.0;            
};

class LocalParameterizationSE3 : public ceres::LocalParameterization {
public:
  virtual ~LocalParameterizationSE3() {}

 
  virtual bool Plus(const double * T_raw, const double * delta_raw,
                    double * T_plus_delta_raw) const {
   
    const Eigen::Map<const Sophus::SE3d> T(T_raw);
    const Eigen::Map<const Eigen::Matrix<double,6,1> > delta(delta_raw);
    Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);
    
    T_plus_delta = T * Sophus::SE3d::exp(delta);
    return true;
  }


  virtual bool ComputeJacobian(const double * T_raw, double * jacobian_raw)
    const {
    const Eigen::Map<const Sophus::SE3d> T(T_raw);
    Eigen::Map<Eigen::Matrix<double,Sophus::SE3d::num_parameters, Sophus::SE3d::DoF,Eigen::RowMajor> > jacobian(jacobian_raw);
    
    jacobian = T.Dx_this_mul_exp_x_at_0();
    return true;
  }

  virtual int GlobalSize() const {
    return Sophus::SE3d::num_parameters;  
  }

  virtual int LocalSize() const {
    return Sophus::SE3d::DoF;             
  }
};


class LocalParameterizationXY : public ceres::LocalParameterization {
public:
  virtual ~LocalParameterizationXY() {}

  virtual bool Plus(const double * T_raw, const double * delta_raw,
                    double * T_plus_delta_raw) const {
    
    const Eigen::Map<const Eigen::Matrix<double,3,1> > T(T_raw);
    const Eigen::Map<const Eigen::Matrix<double,2,1> > delta(delta_raw);
    Eigen::Map<Eigen::Matrix<double,3,1>> T_plus_delta(T_plus_delta_raw);
    T_plus_delta.setZero();

    T_plus_delta = T + Eigen::Vector3d(delta(0), delta(1), 0);
    return true;
  }

  virtual bool ComputeJacobian(const double * T_raw, double * jacobian_raw) const {
    const Eigen::Map<const Eigen::Matrix<double,3,1>> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, 3, 2> > jacobian(jacobian_raw);
    jacobian.setZero();
    
    jacobian.block<2, 2>(0, 0).setIdentity();
    return true;
  }

  virtual int GlobalSize() const {
    return 3;
  }

  virtual int LocalSize() const {
    return 2;
  }
};

class RollPitchSO3Parameterization : public ceres::LocalParameterization {
public:
  virtual ~RollPitchSO3Parameterization() {}



  virtual bool Plus(const double * T_raw, const double * delta_raw,
                    double * T_plus_delta_raw) const {
    const Eigen::Map<const Sophus::SO3d> T(T_raw);
    Eigen::Map<Sophus::SO3d> T_plus_delta(T_plus_delta_raw);

   
    Eigen::Matrix<double, 3, 1> delta;
    delta.setZero();
    delta[0] = delta_raw[0]; 
    delta[1] = delta_raw[1]; 


    T_plus_delta = T * Sophus::SO3d::exp(delta);
    return true;
  }


  virtual bool ComputeJacobian(const double * T_raw, double * jacobian_raw) const {
    const Eigen::Map<const Sophus::SO3d> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, Sophus::SO3d::num_parameters, 2, Eigen::RowMajor>> jacobian(jacobian_raw);

    jacobian.setZero();

    
    Eigen::Matrix<double, 4, 3> J = T.Dx_this_mul_exp_x_at_0();
    
    jacobian.block<4, 1>(0, 0) = J.col(0); 
    jacobian.block<4, 1>(0, 1) = J.col(1); 

    return true;
  }

  virtual int GlobalSize() const {
    return Sophus::SO3d::num_parameters; 
  }

  virtual int LocalSize() const {
    return 2;
  }
};


template <typename T>
Eigen::Matrix<T,3,3> skewMat(Eigen::Matrix<T,3,1> v)
{
    Eigen::Matrix<T,3,3> result;

    result(0,1) = -v(2); 
    result(0,2) =  v(1); 
    result(1,0) =  v(2); 
    result(1,2) = -v(0);
    result(2,0) = -v(1); 
    result(2,1) =  v(0);


    result(0,0) = T(1); 
    result(1,1) = T(1); 
    result(2,2) = T(1);

    return result;
} 


struct PlaneFunctor {
    
    PlaneFunctor(CorrespondenceInfo& corres)
    :corres_(corres)
    {
    }
    
    
    template <typename T>
    bool operator()(const T* const ex_a_l_so3_, const T* const ex_a_l_trans_, const T* const dt,
        T* residual_) const {
        
        
        Eigen::Map<const Sophus::SO3<T>> ex_a_l_so3(ex_a_l_so3_);
        Eigen::Map<const Eigen::Matrix<T,3,1>> ex_a_l_trans(ex_a_l_trans_);
        
        
        Eigen::Matrix<T,3,3> refTimeMat =  skewMat(Eigen::Matrix<T,3,1>(T(0), T(0), T(corres_.ref_angle_speed) * (*dt)));
        Eigen::Matrix<T,3,3> srcTimeMat =  skewMat(Eigen::Matrix<T,3,1>(T(0), T(0), T(corres_.src_angle_speed) * (*dt)));
        
       
        Eigen::Matrix<T,3,1> refpt_w = Eigen::AngleAxisd(corres_.ref_angle, Eigen::Vector3d::UnitZ()).cast<T>() * 
            (ex_a_l_so3.matrix() * corres_.ref_pt.cast<T>() + ex_a_l_trans);
        Eigen::Matrix<T,3,1> srcpt_w = Eigen::AngleAxisd(corres_.src_angle, Eigen::Vector3d::UnitZ()).cast<T>() * 
            (ex_a_l_so3.matrix() * corres_.src_pt.cast<T>() + ex_a_l_trans);

    
        
        *residual_ = T(corres_.weight) * (corres_.norm.cast<T>().transpose() * (refpt_w - srcpt_w))(0);



        return true;
    }

    
    static ceres::CostFunction* Create(CorrespondenceInfo& corres) {
        return new ceres::AutoDiffCostFunction<PlaneFunctor, 1, 4, 3, 1>(
            new PlaneFunctor(corres));
    }

private:
    CorrespondenceInfo corres_; 
};


namespace py = pybind11;


void hello_from_cpp() {
    std::cout << "Hello from C++!" << std::endl;
}


void ProjectionUsingCalibParam(std::string pcdFilename, std::vector<double> mountingParam, py::array_t<double> raw_data)
{
    
    py::buffer_info buffer_info_raw = raw_data.request();
    int rows = buffer_info_raw.shape[0];
    int cols = buffer_info_raw.shape[1];

    std::cout << "rows: " << rows << "\n";
    std::cout << "cols: " << cols << "\n";

    
    double* input_ptr = static_cast<double*>(buffer_info_raw.ptr);

    
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);

    
    for (size_t i = 0; i < rows; i++)
    {
        Eigen::Vector3d pt_l;
  
        pt_l(0) = input_ptr[i*cols + 3];
        pt_l(1) = input_ptr[i*cols + 4];
        pt_l(2) = input_ptr[i*cols + 5];

        double intensity = input_ptr[i*cols + 6];
        double rotAngular = input_ptr[i*cols + 7];

     
        Eigen::Matrix3d R_rot_li;
        R_rot_li = Eigen::AngleAxisd(mountingParam[0], Eigen::Vector3d::UnitZ()) *
                   Eigen::AngleAxisd(mountingParam[1], Eigen::Vector3d::UnitY()) *
                   Eigen::AngleAxisd(mountingParam[2], Eigen::Vector3d::UnitX());

       
        Eigen::Vector3d t(mountingParam[3], mountingParam[4], mountingParam[5]);

   
        Eigen::Vector3d pt_w = Eigen::AngleAxisd(rotAngular, Eigen::Vector3d::UnitZ()) * (R_rot_li * pt_l + t);
        pcl::PointXYZI pt_pcl;
        pt_pcl.x = pt_w(0);
        pt_pcl.y = pt_w(1);
        pt_pcl.z = pt_w(2);
        pt_pcl.intensity = int(intensity);

        cloud->push_back(pt_pcl);
    }

  
    pcl::io::savePCDFileBinary(pcdFilename, *cloud);
}



Eigen::Vector3d calculateEulerAngles(const Eigen::Matrix3d& R) {
    double phi, theta, psi;


    theta = std::asin(-R(2, 0));

    if (std::abs(std::cos(theta)) > 1e-6) {
      
        phi = std::atan2(R(2, 1), R(2, 2));  
        psi = std::atan2(R(1, 0), R(0, 0));   
    } else {
     
        phi = 0;
        psi = std::atan2(-R(0, 1), R(1, 1));
    }


    return Eigen::Vector3d(psi, theta, phi);
}


/**
 * [功能描述]：激光雷达-旋转平台标定函数，通过优化求解激光雷达与旋转平台之间的外参
 * @param mountingParam：初始安装参数向量，包含旋转角度(3个)、平移向量(3个)和时间偏移(1个)
 * @param raw_data：原始数据数组，包含变换后的点云坐标、原始坐标、强度、旋转角度、时间戳等
 * @param optimizeTime：优化迭代次数
 * @param voxelSize：体素化降采样的体素大小
 * @return 返回优化后的最终安装参数向量
 */
std::vector<double> Calib(std::vector<double> mountingParam, py::array_t<double> raw_data, int optimizeTime, double voxelSize)
{
    // 初始化最终安装参数向量（7个参数：3个旋转角度 + 3个平移 + 1个时间偏移）
    std::vector<double> finalMountingParam;
    finalMountingParam.resize(7);
    finalMountingParam[0] = mountingParam[0];  // 绕Z轴旋转角度
    finalMountingParam[1] = mountingParam[1];  // 绕Y轴旋转角度
    finalMountingParam[2] = mountingParam[2];  // 绕X轴旋转角度
    finalMountingParam[3] = mountingParam[3];  // X方向平移
    finalMountingParam[4] = mountingParam[4];  // Y方向平移
    finalMountingParam[5] = mountingParam[5];  // Z方向平移
    finalMountingParam[6] = mountingParam[6];  // 时间偏移

    // 获取原始数据的维度信息
    py::buffer_info buffer_info_raw = raw_data.request();
    int rows = buffer_info_raw.shape[0];  // 数据行数（点的数量）
    int cols = buffer_info_raw.shape[1];  // 数据列数（每个点的属性数量）

    std::cout << "rows: " << rows << "\n";
    std::cout << "cols: " << cols << "\n";
    std::cout << "voxelSize: " << voxelSize << "\n";

    // 初始化外参：旋转矩阵（使用Sophus库）和平移向量
    Sophus::SO3d ex_a_l_so3;     // 旋转矩阵（SO3群）
    Eigen::Vector3d ex_a_l_trans; // 平移向量
   
    // 根据ZYX欧拉角构建初始旋转矩阵
    Eigen::Quaterniond q = Eigen::Quaterniond((Eigen::AngleAxisd(finalMountingParam[0], Eigen::Vector3d::UnitZ()) *
                                                 Eigen::AngleAxisd(finalMountingParam[1], Eigen::Vector3d::UnitY()) *
                                                 Eigen::AngleAxisd(finalMountingParam[2], Eigen::Vector3d::UnitX())).matrix());
    ex_a_l_so3.setQuaternion(q);  // 设置旋转矩阵
    ex_a_l_trans = Eigen::Vector3d(finalMountingParam[3], finalMountingParam[4], finalMountingParam[5]);  // 设置平移向量

    // 获取原始数据指针
    double* input_ptr = static_cast<double*>(buffer_info_raw.ptr);

    // 进行多次优化迭代
    for (size_t i = 0; i < optimizeTime; i++)
    {
        // 创建点云容器
        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>);      // 原始点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);          // 变换后的点云

        // 创建平面特征数据和对应关系容器
        std::vector<PlaneFactorData> factors;            // 平面特征数据
        std::vector<CorrespondenceInfo> allCorres;       // 所有对应关系

        // 存储旋转角度和角速度信息
        std::vector<double> rotorAngles;   // 旋转角度数组
        std::vector<double> rotorSpeeds;   // 旋转角速度数组
    
        // 遍历所有数据点，进行坐标变换
        for (size_t j = 1; j < rows; j++)
        {
            // 提取激光雷达坐标系下的点坐标
            Eigen::Vector3d pt_l;
            pt_l(0) = input_ptr[j*cols + 3];  // 原始点云X坐标
            pt_l(1) = input_ptr[j*cols + 4];  // 原始点云Y坐标
            pt_l(2) = input_ptr[j*cols + 5];  // 原始点云Z坐标

            // 过滤距离过近的点（可能是噪声）
            if (pt_l.norm() < 5)
            {
                continue;
            }
            
            // 提取点的属性信息
            double intensity = input_ptr[j*cols + 6];    // 强度值
            double rotAngular = input_ptr[j*cols + 7];   // 旋转角度

            // 计算旋转角速度（当前角度与前一个角度的差值除以时间差）
            double rotSpeed = (input_ptr[j*cols + 7] - input_ptr[(j-1)*cols + 7]) / ((input_ptr[j*cols + 8] - input_ptr[(j-1)*cols + 8]));

            // 存储角度和角速度信息
            rotorAngles.push_back(rotAngular);
            rotorSpeeds.push_back(rotSpeed);

            // 将点从激光雷达坐标系变换到世界坐标系
            // 变换顺序：1) 激光雷达坐标系到旋转平台坐标系  2) 旋转平台坐标系到世界坐标系
            Eigen::Vector3d pt_w = Eigen::AngleAxisd(rotAngular, Eigen::Vector3d::UnitZ()) *
                                   (ex_a_l_so3.matrix() * pt_l + ex_a_l_trans);
            
            // 创建PCL点云格式的点
            pcl::PointXYZ pt_pcl, pt_pcl_w;
            pt_pcl.x = pt_l(0);   // 原始点云
            pt_pcl.y = pt_l(1);
            pt_pcl.z = pt_l(2);
            raw_cloud->push_back(pt_pcl);

            pt_pcl_w.x = pt_w(0); // 变换后的点云
            pt_pcl_w.y = pt_w(1);
            pt_pcl_w.z = pt_w(2);
            cloud->push_back(pt_pcl_w);
        }

        // 构建KD树用于邻域搜索
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud);

        std::cout << "cloud size: " << cloud->size() << "\n";
        
        // 使用体素化降采样减少计算量
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_kernel(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(voxelSize, voxelSize, voxelSize);  // 设置体素大小
        sor.filter(*cloud_kernel);  // 执行降采样
        std::cout << "cloud_kernel size: " << cloud_kernel->size() << "\n";

        // 对每个降采样后的点进行平面特征提取
        for (size_t j = 0; j < cloud_kernel->size(); j++)
        {
            // 获取当前中心点
            pcl::PointXYZ centerPt = cloud_kernel->points[j];
            std::vector<int> pointIdxRadiusSearch;          // 半径搜索结果索引
            std::vector<float> pointRadiusSquaredDistance;  // 半径搜索结果距离

            // 在中心点周围搜索邻域点
            kdtree.radiusSearch(centerPt, 1.5 * voxelSize, pointIdxRadiusSearch, pointRadiusSquaredDistance);

            // 检查邻域点数量是否足够进行平面拟合
            if (pointIdxRadiusSearch.size() < 10)
            {
                continue;
            }

            // 收集邻域点的信息
            std::vector<Eigen::Vector3d> tmp_rawpts;      // 原始点坐标
            std::vector<double> tmp_rotors;               // 旋转角度
            std::vector<double> tmp_rotorspeeds;          // 旋转角速度
            Eigen::MatrixXd pointCloudMatrix(3, pointIdxRadiusSearch.size());  // 点云矩阵
            
            for (size_t k = 0; k < pointIdxRadiusSearch.size(); k++)
            {
                int idx = pointIdxRadiusSearch[k];
                pcl::PointXYZ pt_w = cloud->points[idx];
                pointCloudMatrix.col(k) = Eigen::Vector3d(pt_w.x, pt_w.y, pt_w.z);

                // 获取对应的原始点和旋转信息
                pcl::PointXYZ pt_l = raw_cloud->points[idx];
                double rotorAngle = rotorAngles[idx];

                tmp_rawpts.push_back(Eigen::Vector3d(pt_l.x, pt_l.y, pt_l.z));
                tmp_rotors.push_back(rotorAngle);
                tmp_rotorspeeds.push_back(rotorSpeeds[idx]);
            }

            // 使用SVD进行平面拟合
            Eigen::Matrix3Xd centeredPointCloud = pointCloudMatrix.colwise() - pointCloudMatrix.rowwise().mean();
            Eigen::JacobiSVD<Eigen::Matrix3Xd> svd(centeredPointCloud, Eigen::ComputeFullU);

            // 获取奇异值，用于判断平面特征的质量
            double sigma0 = svd.singularValues()(0);  // 最大奇异值
            double sigma1 = svd.singularValues()(1);  // 第二大奇异值
            double sigma2 = svd.singularValues()(2);  // 最小奇异值

            // 检查是否为良好的平面特征（最大奇异值与第二大奇异值的比值）
            if ((sigma0 - sigma1) / sigma0 > 0.5)
            {
                continue;  // 不是好的平面特征，跳过
            }
            else
            {
                // 提取平面法向量（最小奇异值对应的特征向量）
                Eigen::Vector3d normal = svd.matrixU().col(2);
                normal.normalize();

                // 创建平面特征数据
                PlaneFactorData tmpFactorData;
                tmpFactorData.norm = normal;
                tmpFactorData.rawPts = tmp_rawpts;
                tmpFactorData.rotorAngles = tmp_rotors;
                factors.push_back(tmpFactorData);

                // 为每个点对创建对应关系（用于优化）
                for (size_t k = 1; k < tmp_rawpts.size(); k++)
                {
                    CorrespondenceInfo tmpCorres;
                    tmpCorres.norm = normal;                    // 平面法向量
                    tmpCorres.ref_pt = tmp_rawpts[0];          // 参考点
                    tmpCorres.ref_angle = tmp_rotors[0];       // 参考点旋转角度
                    tmpCorres.src_pt = tmp_rawpts[k];          // 源点
                    tmpCorres.src_angle = tmp_rotors[k];       // 源点旋转角度
                    tmpCorres.ref_angle_speed = tmp_rotorspeeds[0];  // 参考点角速度
                    tmpCorres.src_angle_speed = tmp_rotorspeeds[k];  // 源点角速度
                    tmpCorres.weight = 1.0;                    // 权重

                    // 只保留角度差异较大的点对（确保几何约束有效）
                    if (fabs(tmpCorres.ref_angle - tmpCorres.src_angle) < M_PI / 3)
                    {
                        continue;
                    }
                    allCorres.push_back(tmpCorres);
                }
            }
        }
        
        std::cout << "allCorres size: " << allCorres.size() << "\n";
        
        // 构建Ceres优化问题
        ceres::Problem problem;
        
        // 添加旋转参数块（使用自定义的SO3参数化）
        problem.AddParameterBlock(ex_a_l_so3.data(), 4, new RollPitchSO3Parameterization());
        
        // 添加平移参数块
        problem.AddParameterBlock(ex_a_l_trans.data(), 3);

        // 添加时间偏移参数块
        problem.AddParameterBlock(&finalMountingParam[6], 1);

        // 为每个对应关系添加残差块
        for (size_t j = 0; j < allCorres.size(); j++)
        {
            ceres::CostFunction *cost = PlaneFunctor::Create(allCorres[j]);  // 创建平面约束的代价函数
            
            ceres::LossFunction* loss_function = new ceres::HuberLoss(0.3);  // 使用Huber损失函数增强鲁棒性
            problem.AddResidualBlock(cost,
                                     loss_function,
                                     ex_a_l_so3.data(), ex_a_l_trans.data(),
                                     &finalMountingParam[6]);
        }

        std::cout << "finish construct problem\n";

        // 设置Ceres求解器选项
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;                        // 线性求解器类型
        options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;       // 稀疏线性代数库
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;        // 信赖域策略
        options.minimizer_progress_to_stdout = true;                            // 输出优化进度
        options.num_threads = 20;                                               // 线程数

        // 执行优化
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        // 输出当前迭代的优化结果
        std::cout << "ex_a_l.rotationMatrix(): " << ex_a_l_so3.matrix() << "\n";
        std::cout << "ex_a_l.translation(): " << ex_a_l_trans << "\n";
        std::cout << "dt : " << finalMountingParam[6] << "\n";
    }
    
    // 将优化后的旋转矩阵转换为欧拉角
    Eigen::Vector3d angles = calculateEulerAngles(ex_a_l_so3.matrix());

    // 更新最终参数
    finalMountingParam[0] = angles[0];        // 绕Z轴旋转角度
    finalMountingParam[1] = angles[1];        // 绕Y轴旋转角度
    finalMountingParam[2] = angles[2];        // 绕X轴旋转角度
    finalMountingParam[3] = ex_a_l_trans[0];  // X方向平移
    finalMountingParam[4] = ex_a_l_trans[1];  // Y方向平移
    finalMountingParam[5] = ex_a_l_trans[2];  // Z方向平移

    // 返回最终的标定参数
    return finalMountingParam;
}

std::vector<double> Calib_reweight(std::vector<double> mountingParam, 
                                   py::array_t<double> raw_data, 
                                   int optimizeTime, double voxelSize)
{

    std::vector<double> finalMountingParam = mountingParam;
    py::buffer_info buffer_info_raw = raw_data.request();
    int rows = buffer_info_raw.shape[0];
    int cols = buffer_info_raw.shape[1];

    std::cout << "rows: " << rows << "\n";
    std::cout << "cols: " << cols << "\n";

    Sophus::SO3d ex_a_l_so3;
    Eigen::Vector3d ex_a_l_trans;
    Eigen::Quaterniond q = Eigen::Quaterniond((Eigen::AngleAxisd(finalMountingParam[0], Eigen::Vector3d::UnitZ()) *
                                              Eigen::AngleAxisd(finalMountingParam[1], Eigen::Vector3d::UnitY()) *
                                              Eigen::AngleAxisd(finalMountingParam[2], Eigen::Vector3d::UnitX())).matrix());
    ex_a_l_so3.setQuaternion(q);
    ex_a_l_trans = Eigen::Vector3d(finalMountingParam[3], finalMountingParam[4], finalMountingParam[5]);

    double* input_ptr = static_cast<double*>(buffer_info_raw.ptr);

    // Add variables needed for convergence checking
    double prev_error = std::numeric_limits<double>::max();
    const double convergence_threshold = 1e-6;


    std::vector<Eigen::Vector3d> final_normals;
    std::vector<double> final_weights;
    std::vector<Eigen::Vector3d> final_pre_normals;     
    std::vector<double> final_pre_weights; 

    for (size_t iter = 0; iter < optimizeTime; iter++)
    {
       
        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

   
        int k_neighbors = std::min(50, static_cast<int>(rows / 100));  
        std::vector<double> rotorAngles;
        std::vector<double> rotorSpeeds;
    

        for (size_t j = 1; j < rows; j++)
        {
            Eigen::Vector3d pt_l;
            pt_l(0) = input_ptr[j*cols + 3];
            pt_l(1) = input_ptr[j*cols + 4];
            pt_l(2) = input_ptr[j*cols + 5];

            // change the condition to filter points based on distance
            const double min_distance = 5.0;
            const double max_distance = 50.0;  // add the max distance limit
            double pt_norm = pt_l.norm();
            if (pt_norm < min_distance || pt_norm > max_distance)
                continue;
            
            double rotAngular = input_ptr[j*cols + 7];
            double rotSpeed = (input_ptr[j*cols + 7] - input_ptr[(j-1)*cols + 7]) /
                            (input_ptr[j*cols + 8] - input_ptr[(j-1)*cols + 8]);

     
            const double max_rot_speed = 10.0;  // rad/s
            if (std::abs(rotSpeed) > max_rot_speed)
                continue;

            rotorAngles.push_back(rotAngular);
            rotorSpeeds.push_back(rotSpeed);

            Eigen::Vector3d pt_w = Eigen::AngleAxisd(rotAngular, Eigen::Vector3d::UnitZ()) *
                                 (ex_a_l_so3.matrix() * pt_l + ex_a_l_trans);
            
            pcl::PointXYZ pt_pcl, pt_pcl_w;
            pt_pcl.x = pt_l(0); pt_pcl.y = pt_l(1); pt_pcl.z = pt_l(2);
            raw_cloud->push_back(pt_pcl);
            pt_pcl_w.x = pt_w(0); pt_pcl_w.y = pt_w(1); pt_pcl_w.z = pt_w(2);
            cloud->push_back(pt_pcl_w);
        }

        // establish the kdtree for nearest neighbor search
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud);
        
  
        double adjustedVoxelSize = voxelSize;
        if (cloud->size() > 100000) {
            adjustedVoxelSize *= std::sqrt(static_cast<double>(100000) / cloud->size());
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_kernel(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(adjustedVoxelSize, adjustedVoxelSize, adjustedVoxelSize);
        sor.filter(*cloud_kernel);
        std::cout << "cloud_kernel size: " << cloud_kernel->size() << "\n";

    
        std::vector<Eigen::Vector3d> bin_sampled_normals;
        std::vector<double> bin_sampled_weights;
        // define the static map to accumulate normals and weights in each bin
        static std::map<int, std::vector<std::pair<Eigen::Vector3d, double>>> binned_normals;

     
        std::vector<PlaneFactorData> factors;
        std::vector<CorrespondenceInfo> allCorres;
        
    
        for (size_t j = 0; j < cloud_kernel->size(); j++)
        {
            pcl::PointXYZ centerPt = cloud_kernel->points[j];
            std::vector<int> pointIdxKNN;
            std::vector<float> pointKNNDistance;

            if (kdtree.nearestKSearch(centerPt, k_neighbors, pointIdxKNN, pointKNNDistance) <= 0)
                continue;

           
            std::vector<double> distance_weights(pointIdxKNN.size());
            double max_dist = pointKNNDistance.back();
            for (size_t i = 0; i < pointIdxKNN.size(); i++) {
                distance_weights[i] = 1.0 - std::sqrt(pointKNNDistance[i] / max_dist);
            }

            std::vector<Eigen::Vector3d> tmp_rawpts;
            std::vector<double> tmp_rotors;
            std::vector<double> tmp_rotorspeeds;
            Eigen::MatrixXd pointCloudMatrix(3, pointIdxKNN.size());

            for (size_t k = 0; k < pointIdxKNN.size(); k++)
            {
                int idx = pointIdxKNN[k];
                pcl::PointXYZ pt_w = cloud->points[idx];
                pointCloudMatrix.col(k) = Eigen::Vector3d(pt_w.x, pt_w.y, pt_w.z);

                pcl::PointXYZ pt_l = raw_cloud->points[idx];
                tmp_rawpts.push_back(Eigen::Vector3d(pt_l.x, pt_l.y, pt_l.z));
                tmp_rotors.push_back(rotorAngles[idx]);
                tmp_rotorspeeds.push_back(rotorSpeeds[idx]);
            }

   
            Eigen::Vector3d mean = pointCloudMatrix.rowwise().mean();
            Eigen::MatrixXd centered = pointCloudMatrix.colwise() - mean;
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeFullU | Eigen::ComputeFullV);
            
            double sigma0 = svd.singularValues()(0);
            double sigma1 = svd.singularValues()(1);
            double sigma2 = svd.singularValues()(2);
            double trace = sigma0 + sigma1 + sigma2;
            
            // 
            double planarity = 2 * (sigma1 - sigma2) / trace;
            
          
            const double min_planarity = 0.7;
            const double max_condition_number = 100.0;
            if (planarity < min_planarity || sigma0 / sigma2 > max_condition_number)
                continue;

            
            {
        
            Eigen::Vector3d normal = svd.matrixU().col(2);
            normal.normalize();
        
            if (normal.z() < 0)
                normal = -normal;
       
            PolarNormal polar = cartesianToPolar(normal);
           
            const int theta_bins = 18;  // 180°/10°
            const int phi_bins = 36;    // 360°/10°
            int theta_bin = static_cast<int>(polar.theta * theta_bins / M_PI);
            int phi_bin = static_cast<int>((polar.phi + M_PI) * phi_bins / (2 * M_PI));
            theta_bin = std::min(std::max(theta_bin, 0), theta_bins - 1);
            phi_bin = std::min(std::max(phi_bin, 0), phi_bins - 1);
            int bin_index = theta_bin * phi_bins + phi_bin;
     
            binned_normals[bin_index].push_back({normal, planarity});
        }

        {
            PlaneFactorData tmpFactorData;
            Eigen::Vector3d normal = svd.matrixU().col(2);
            normal.normalize();
            tmpFactorData.norm = normal;
            tmpFactorData.rawPts = tmp_rawpts;
            tmpFactorData.rotorAngles = tmp_rotors;
            factors.push_back(tmpFactorData);
            for (size_t m = 1; m < tmp_rawpts.size(); m++)
            {
                CorrespondenceInfo tmpCorres;
                tmpCorres.norm = normal;
                tmpCorres.ref_pt = tmp_rawpts[0];
                tmpCorres.ref_angle = tmp_rotors[0];
                tmpCorres.src_pt = tmp_rawpts[m];
                tmpCorres.src_angle = tmp_rotors[m];
                tmpCorres.ref_angle_speed = tmp_rotorspeeds[0];
                tmpCorres.src_angle_speed = tmp_rotorspeeds[m];
                tmpCorres.weight = 1.0;
                if (fabs(tmpCorres.ref_angle - tmpCorres.src_angle) < M_PI / 3)
                    continue;
                allCorres.push_back(tmpCorres);
            }
        }
        
    
        if (j % 1000 == 0 || j == cloud_kernel->size() - 1)
        {
        
            for (auto& bin : binned_normals)
            {
                for (auto& nw : bin.second)
                {
                    final_pre_normals.push_back(nw.first);
                    final_pre_weights.push_back(nw.second);
                }
            }
           
            int total_count = 0;
            int non_empty_bins = 0;
            for (auto& bin : binned_normals)
            {
                if (!bin.second.empty())
                {
                    total_count += bin.second.size();
                    non_empty_bins++;
                }
            }
            int average_count = 0;
            if (non_empty_bins > 0)
                average_count = total_count / non_empty_bins;
            
      
            for (auto& bin : binned_normals)
            {
                if (!bin.second.empty())
                {
                    int count = bin.second.size();
                    std::vector<std::pair<Eigen::Vector3d, double>> sampled;
                    if (count > average_count && average_count > 0)
                    {
                        
                        std::vector<int> indices(count);
                        for (int i = 0; i < count; i++)
                            indices[i] = i;
                        std::random_shuffle(indices.begin(), indices.end());
                        for (int i = 0; i < average_count; i++)
                        {
                            sampled.push_back(bin.second[indices[i]]);
                        }
                    }
                    else
                    {
                        sampled = bin.second;
                    }
                
                    for (auto& nw : sampled)
                    {
                        final_normals.push_back(nw.first);
                        final_weights.push_back(nw.second);
                    }
                }
            }
        
            binned_normals.clear();
        }
    }  // end for (j in cloud_kernel)

    std::cout << "allCorres size: " << allCorres.size() << "\n";
    

    ceres::Problem problem;
    problem.AddParameterBlock(ex_a_l_so3.data(), 4, new RollPitchSO3Parameterization());
    problem.AddParameterBlock(ex_a_l_trans.data(), 3);
    problem.AddParameterBlock(&finalMountingParam[6], 1);
    for (size_t j = 0; j < allCorres.size(); j++)
    {
        ceres::CostFunction* cost = PlaneFunctor::Create(allCorres[j]);
        ceres::LossFunction* loss_function = new ceres::HuberLoss(0.3);
        problem.AddResidualBlock(cost, loss_function,
                                 ex_a_l_so3.data(), ex_a_l_trans.data(),
                                 &finalMountingParam[6]);
    }
    
    std::cout << "finish construct problem\n";
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 20;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "ex_a_l.rotationMatrix(): " << ex_a_l_so3.matrix() << "\n";
    std::cout << "ex_a_l.translation(): " << ex_a_l_trans << "\n";
    std::cout << "dt : " << finalMountingParam[6] << "\n";
    
    double current_error = summary.final_cost;
    double relative_error_change = std::abs(current_error - prev_error) / prev_error;
    if (relative_error_change < convergence_threshold && iter > 0) {
        std::cout << "Converged at iteration " << iter << std::endl;
        break;
    }
    prev_error = current_error;
}  // end for (iter)

    Eigen::Vector3d angles = calculateEulerAngles(ex_a_l_so3.matrix());
    finalMountingParam[0] = angles[0];
    finalMountingParam[1] = angles[1];
    finalMountingParam[2] = angles[2];
    finalMountingParam[3] = ex_a_l_trans[0];
    finalMountingParam[4] = ex_a_l_trans[1];
    finalMountingParam[5] = ex_a_l_trans[2];

// keep the original normals and weights
    if (!final_pre_normals.empty()) {
        std::ofstream pre_file("normal_vectors_before.txt");
        for (size_t i = 0; i < final_pre_normals.size(); ++i) {
            pre_file << final_pre_normals[i].x() << " " 
                    << final_pre_normals[i].y() << " " 
                    << final_pre_normals[i].z() << " " 
                    << final_pre_weights[i] << "\n";
        }
        pre_file.close();
        std::cout << "Saved pre-sampling normals to normal_vectors_before.txt, count: " << final_pre_normals.size() << "\n";
    }


    if (!final_normals.empty()) {
        std::ofstream post_file("normal_vectors_after.txt");
        for (size_t i = 0; i < final_normals.size(); ++i) {
            post_file << final_normals[i].x() << " " 
                    << final_normals[i].y() << " " 
                    << final_normals[i].z() << " " 
                    << final_weights[i] << "\n";
        }
        post_file.close();
        std::cout << "Saved post-sampling normals to normal_vectors_after.txt, count: " << final_normals.size() << "\n";
    }

return finalMountingParam;
}

PYBIND11_MODULE(pyCalib, m) {
    m.def("hello_from_cpp", &hello_from_cpp, "A function that says hello from C++.");
    m.def("ProjectionUsingCalibParam", &ProjectionUsingCalibParam, "ProjectionUsingCalibParam");
    m.def("Calib", &Calib, "Calib");
    m.def("Calib_reweight", &Calib_reweight, "Calib_reweight");
}