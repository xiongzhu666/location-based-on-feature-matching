#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <ctime>
#include <math.h>
#include <vector>
#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

using namespace std;
using namespace cv;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

float THRESH_FACTOR;
const int mRotationPatterns[8][9] = {
        1,2,3,
        4,5,6,
        7,8,9,

        4,1,2,
        7,5,3,
        8,9,6,

        7,4,1,
        8,5,2,
        9,6,3,

        8,7,4,
        9,5,1,
        6,3,2,

        9,8,7,
        6,5,4,
        3,2,1,

        6,9,8,
        3,5,7,
        2,1,4,

        3,6,9,
        2,5,8,
        1,4,7,

        2,3,6,
        1,5,9,
        4,7,8
};

// 5 level scales
const double mScaleRatios[5] = { 1.0, 1.0 / 2, 1.0 / sqrt(2.0), sqrt(2.0), 2.0 };

/*******************GMS-Match实现****************************/
class gms_matcher
{
public:
    // OpenCV Keypoints & Correspond Image Size & Nearest Neighbor Matches
    gms_matcher(const vector<KeyPoint> &vkp1, const Size size1, const vector<KeyPoint> &vkp2, const Size size2, const vector<DMatch> &vDMatches)
    {
        // Input initialize
        NormalizePoints(vkp1, size1, mvP1);
        NormalizePoints(vkp2, size2, mvP2);
        mNumberMatches = vDMatches.size();
        ConvertMatches(vDMatches, mvMatches);

        // Grid initialize
        mGridSizeLeft = Size(20, 20);
        mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;

        // Initialize the neihbor of left grid
        mGridNeighborLeft = Mat::zeros(mGridNumberLeft, 9, CV_32SC1);
        InitalizeNiehbors(mGridNeighborLeft, mGridSizeLeft);
    };
    ~gms_matcher() {};

private:

    // Normalized Points
    vector<Point2f> mvP1, mvP2;

    // Matches
    vector<pair<int, int> > mvMatches;

    // Number of Matches
    size_t mNumberMatches;

    // Grid Size
    Size mGridSizeLeft, mGridSizeRight;
    int mGridNumberLeft;
    int mGridNumberRight;

    // x	  : left grid idx
    // y      :  right grid idx
    // value  : how many matches from idx_left to idx_right
    Mat mMotionStatistics;

    //
    vector<int> mNumberPointsInPerCellLeft;

    // Inldex  : grid_idx_left
    // Value   : grid_idx_right
    vector<int> mCellPairs;

    // Every Matches has a cell-pair
    // first  : grid_idx_left
    // second : grid_idx_right
    vector<pair<int, int> > mvMatchPairs;

    // Inlier Mask for output
    vector<bool> mvbInlierMask;

    //
    Mat mGridNeighborLeft;
    Mat mGridNeighborRight;

public:

    // Get Inlier Mask
    // Return number of inliers
    int GetInlierMask(vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false);

private:

    // Normalize Key Points to Range(0 - 1)
    void NormalizePoints(const vector<KeyPoint> &kp, const Size &size, vector<Point2f> &npts) {
        const size_t numP = kp.size();
        const int width   = size.width;
        const int height  = size.height;
        npts.resize(numP);

        for (size_t i = 0; i < numP; i++)
        {
            npts[i].x = kp[i].pt.x / width;
            npts[i].y = kp[i].pt.y / height;
        }
    }

    // Convert OpenCV DMatch to Match (pair<int, int>)
    void ConvertMatches(const vector<DMatch> &vDMatches, vector<pair<int, int> > &vMatches) {
        vMatches.resize(mNumberMatches);
        for (size_t i = 0; i < mNumberMatches; i++)
        {
            vMatches[i] = pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
        }
    }

    int GetGridIndexLeft(const Point2f &pt, int type) {
        int x = 0, y = 0;

        if (type == 1) {
            x = floor(pt.x * mGridSizeLeft.width);
            y = floor(pt.y * mGridSizeLeft.height);

            if (y >= mGridSizeLeft.height || x >= mGridSizeLeft.width){
                return -1;
            }
        }

        if (type == 2) {
            x = floor(pt.x * mGridSizeLeft.width + 0.5);
            y = floor(pt.y * mGridSizeLeft.height);

            if (x >= mGridSizeLeft.width || x < 1) {
                return -1;
            }
        }

        if (type == 3) {
            x = floor(pt.x * mGridSizeLeft.width);
            y = floor(pt.y * mGridSizeLeft.height + 0.5);

            if (y >= mGridSizeLeft.height || y < 1) {
                return -1;
            }
        }

        if (type == 4) {
            x = floor(pt.x * mGridSizeLeft.width + 0.5);
            y = floor(pt.y * mGridSizeLeft.height + 0.5);

            if (y >= mGridSizeLeft.height || y < 1 || x >= mGridSizeLeft.width || x < 1) {
                return -1;
            }
        }

        return x + y * mGridSizeLeft.width;
    }

    int GetGridIndexRight(const Point2f &pt) {
        int x = floor(pt.x * mGridSizeRight.width);
        int y = floor(pt.y * mGridSizeRight.height);

        return x + y * mGridSizeRight.width;
    }

    // Assign Matches to Cell Pairs
    void AssignMatchPairs(int GridType);

    // Verify Cell Pairs
    void VerifyCellPairs(int RotationType);

    // Get Neighbor 9
    vector<int> GetNB9(const int idx, const Size& GridSize) {
        vector<int> NB9(9, -1);

        int idx_x = idx % GridSize.width;
        int idx_y = idx / GridSize.width;

        for (int yi = -1; yi <= 1; yi++)
        {
            for (int xi = -1; xi <= 1; xi++)
            {
                int idx_xx = idx_x + xi;
                int idx_yy = idx_y + yi;

                if (idx_xx < 0 || idx_xx >= GridSize.width || idx_yy < 0 || idx_yy >= GridSize.height)
                    continue;

                NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * GridSize.width;
            }
        }
        return NB9;
    }

    void InitalizeNiehbors(Mat &neighbor, const Size& GridSize) {
        for (int i = 0; i < neighbor.rows; i++)
        {
            vector<int> NB9 = GetNB9(i, GridSize);
            int *data = neighbor.ptr<int>(i);
            memcpy(data, &NB9[0], sizeof(int) * 9);
        }
    }

    void SetScale(int Scale) {
        // Set Scale
        mGridSizeRight.width = mGridSizeLeft.width  * mScaleRatios[Scale];
        mGridSizeRight.height = mGridSizeLeft.height * mScaleRatios[Scale];
        mGridNumberRight = mGridSizeRight.width * mGridSizeRight.height;

        // Initialize the neihbor of right grid
        mGridNeighborRight = Mat::zeros(mGridNumberRight, 9, CV_32SC1);
        InitalizeNiehbors(mGridNeighborRight, mGridSizeRight);
    }

    // Run
    int run(int RotationType);
};

int gms_matcher::GetInlierMask(vector<bool> &vbInliers, bool WithScale, bool WithRotation) {

    int max_inlier = 0;

    if (!WithScale && !WithRotation)
    {
        SetScale(0);
        max_inlier = run(1);
        vbInliers = mvbInlierMask;
        return max_inlier;
    }

    if (WithRotation && WithScale)
    {
        for (int Scale = 0; Scale < 5; Scale++)
        {
            SetScale(Scale);
            for (int RotationType = 1; RotationType <= 8; RotationType++)
            {
                int num_inlier = run(RotationType);

                if (num_inlier > max_inlier)
                {
                    vbInliers = mvbInlierMask;
                    max_inlier = num_inlier;
                }
            }
        }
        return max_inlier;
    }

    if (WithRotation && !WithScale)
    {
        SetScale(0);
        for (int RotationType = 1; RotationType <= 8; RotationType++)
        {
            int num_inlier = run(RotationType);

            if (num_inlier > max_inlier)
            {
                vbInliers = mvbInlierMask;
                max_inlier = num_inlier;
            }
        }
        return max_inlier;
    }

    if (!WithRotation && WithScale)
    {
        for (int Scale = 0; Scale < 5; Scale++)
        {
            SetScale(Scale);

            int num_inlier = run(1);

            if (num_inlier > max_inlier)
            {
                vbInliers = mvbInlierMask;
                max_inlier = num_inlier;
            }

        }
        return max_inlier;
    }

    return max_inlier;
}

void gms_matcher::AssignMatchPairs(int GridType) {

    for (size_t i = 0; i < mNumberMatches; i++)
    {
        Point2f &lp = mvP1[mvMatches[i].first];
        Point2f &rp = mvP2[mvMatches[i].second];

        int lgidx = mvMatchPairs[i].first = GetGridIndexLeft(lp, GridType);
        int rgidx = -1;

        if (GridType == 1)
        {
            rgidx = mvMatchPairs[i].second = GetGridIndexRight(rp);
        }
        else
        {
            rgidx = mvMatchPairs[i].second;
        }

        if (lgidx < 0 || rgidx < 0)	continue;

        mMotionStatistics.at<int>(lgidx, rgidx)++;
        mNumberPointsInPerCellLeft[lgidx]++;
    }

}

void gms_matcher::VerifyCellPairs(int RotationType) {

    const int *CurrentRP = mRotationPatterns[RotationType - 1];

    for (int i = 0; i < mGridNumberLeft; i++)
    {
        if (sum(mMotionStatistics.row(i))[0] == 0)
        {
            mCellPairs[i] = -1;
            continue;
        }

        int max_number = 0;
        for (int j = 0; j < mGridNumberRight; j++)
        {
            int *value = mMotionStatistics.ptr<int>(i);
            if (value[j] > max_number)
            {
                mCellPairs[i] = j;
                max_number = value[j];
            }
        }

        int idx_grid_rt = mCellPairs[i];

        const int *NB9_lt = mGridNeighborLeft.ptr<int>(i);
        const int *NB9_rt = mGridNeighborRight.ptr<int>(idx_grid_rt);

        int score = 0;
        double thresh = 0;
        int numpair = 0;

        for (size_t j = 0; j < 9; j++)
        {
            int ll = NB9_lt[j];
            int rr = NB9_rt[CurrentRP[j] - 1];
            if (ll == -1 || rr == -1)	continue;

            score += mMotionStatistics.at<int>(ll, rr);
            thresh += mNumberPointsInPerCellLeft[ll];
            numpair++;
        }

        thresh = THRESH_FACTOR * sqrt(thresh / numpair);

        if (score < thresh)
            mCellPairs[i] = -2;
    }
}

int gms_matcher::run(int RotationType) {

    mvbInlierMask.assign(mNumberMatches, false);

    // Initialize Motion Statisctics
    mMotionStatistics = Mat::zeros(mGridNumberLeft, mGridNumberRight, CV_32SC1);
    mvMatchPairs.assign(mNumberMatches, pair<int, int>(0, 0));

    for (int GridType = 1; GridType <= 4; GridType++)
    {
        // initialize
        mMotionStatistics.setTo(0);
        mCellPairs.assign(mGridNumberLeft, -1);
        mNumberPointsInPerCellLeft.assign(mGridNumberLeft, 0);

        AssignMatchPairs(GridType);
        VerifyCellPairs(RotationType);

        // Mark inliers
        for (size_t i = 0; i < mNumberMatches; i++)
        {
            if (mvMatchPairs[i].first >= 0) {
                if (mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)
                {
                    mvbInlierMask[i] = true;
                }
            }
        }
    }
    int num_inlier = sum(mvbInlierMask)[0];
    return num_inlier;
}

vector<float> robot_submap_location;              //= {200, 400}//假定机器人子图中的位置坐标
vector<vector<int> > kp1_location, kp2_location;          //定义图片特征点几何位置容器
vector<float> distance_submap;                            //定义存储子图距离的容器
int match_count = 0;                                      //定义匹配对数
vector<float> robot_final_location;                       //定义容器保存优化后的机器坐标

//定义ceres残差项结构体
//struct F{
//    template <typename T>
//    bool operator()(const T* const x1,
//                    const T* const x2,
//                    const T* const i,
//                    T* residual) const {
//        residual[0] = sqrt(pow(x1[0] - kp2_location[i][0], 2)
//                      + pow(x2[0] - kp2_location[i][1], 2)) - distance_submap[i];
//        return true;
//    }
//};

//#define USE_GPU
#ifdef USE_GPU
#include "/home/xz/Thirdparty/opencv-3.4.0/modules/cudafeatures2d/include/opencv2/cudafeatures2d.hpp"
#include "/home/xz/Thirdparty/opencv-3.4.0/modules/cudafilters/include/opencv2/cudafilters.hpp"
using cuda::GpuMat;
#endif

void GmsMatch(Mat &img1, Mat &img2);
Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);
double loss_fun(vector<float> &x);
vector<float> loss_dev(vector<float> &x);
vector<float> optimize(vector<float> &x, int num_iteration = 800, float learning_rate = 0.0001);
void parameter_load();
//void runImagePair() {
//    Mat img1 = imread("../data/submap.png");
//    Mat img2 = imread("../data/globalmap.png");
//
//    GmsMatch(img1, img2);
//}

int main(int argc, char* argv[])
{
    /****************外部传参***************/
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    robot_submap_location.push_back(stof(argv[3]));
    robot_submap_location.push_back(stof(argv[4]));

    THRESH_FACTOR = stof(argv[5]);                                  //设置筛选阈值

    vector<float> robot_global_location;                           //定义优化初始位置
    float img2_height = img2.rows;
    float img2_width = img2.cols;
    robot_global_location = {img2_width / 2, img2_height / 2};    //初始位置选在在图像中心
    //测试
    cout << img2_width << endl;

#ifdef USE_GPU
    int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0) { cuda::setDevice(0); }
#endif // USE_GPU

    //runImagePair();

    GmsMatch(img1, img2);

    /*******************ceres优化代码*************************/
//    ceres::Problem problem;
//    //定义机器优化初始位置, 默认为（0,0）
//    double x1 = robot_global_location[0];
//    double x2 = robot_global_location[1];
//    //将残差模块添加到优化目标中
//    for (int i = 0; i < match_count; i++) {
//        problem.AddResidualBlock(new AutoDiffCostFunction<F, 1, 1, 1, 1>(new F),
//                NULL,
//                &x1, &x2);
//    }
//    //参数选项设置
//    Solver::Options options;
//    options.max_num_iterations = 100;
//    options.linear_solver_type = ceres::DENSE_QR;
//    options.minimizer_progress_to_stdout = true;
//    //输出初始值
//    std::cout << "Initial x1 = " << x1
//              << ", x2 = " << x2
//              << "\n";
//    //开始优化
//    Solver::Summary summary;
//    Solve(options, &problem, &summary);
//    std::cout << summary.FullReport() << "\n";
//    std::cout << "Final x1 = " << x1
//              << ", x2 = " << x2
//              << "\n";

    cout << "hello-SLAM" << endl;                                 ///我只是萌萌的分割线
    cout << "hello-SLAM" << endl;                                 ///我只是萌萌的分割线
    cout << "hello-SLAM" << endl;                                 ///我只是萌萌的分割线

//    for (int i = 0; i < match_count; ++i) {
//        cout << distance_submap[i] << endl;
//    }
    /******************调用梯度下降函数模块*******************/
    robot_final_location = optimize(robot_global_location);      //调用优化函数

    circle(img1,Point(robot_submap_location[0], robot_submap_location[1]),8,CV_RGB(255,0,0),2);
    imshow("机器在子图中的位置", img1);
    circle(img2,Point(robot_final_location[0], robot_final_location[1]),8,CV_RGB(255,0,0),2);
    imshow("定位机器在全局的位置", img2);
    waitKey();
    return 0;

}

void GmsMatch(Mat &img1, Mat &img2) {
    vector<KeyPoint> kp1, kp2;
    Mat d1, d2;
    vector<DMatch> matches_all, matches_gms;

    Ptr<ORB> orb = ORB::create(10000);
    orb->setFastThreshold(0);

    orb->detectAndCompute(img1, Mat(), kp1, d1);
    orb->detectAndCompute(img2, Mat(), kp2, d2);

#ifdef USE_GPU
    GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(d1, d2, matches_all);
#endif

    // GMS filter
    std::vector<bool> vbInliers;
    gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
    int num_inliers = gms.GetInlierMask(vbInliers, true, true);
    cout << "Get total " << num_inliers << " matches." << endl;

    // collect matches
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            matches_gms.push_back(matches_all[i]);
        }
    }
    // draw matching
    Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 2);
    imshow("show", show);

    /*********************提取GMS匹配特征点的坐标*********************/
    int index_queryIdx, index_trainIdx;                                 //定义匹配索引
    for (int i = 0; i < matches_gms.size(); i++)                        //将匹配的特征点坐标赋给point
    {
        index_queryIdx = matches_gms.at(i).queryIdx;                    //取图1的匹配点的索引
        index_trainIdx = matches_gms.at(i).trainIdx;                    //取图2的匹配点的索引
        //对单个点的横纵坐标赋值
        vector<int> kp2_simple, kp1_simple;
        kp1_simple.push_back(int(kp1.at(index_queryIdx).pt.x));
        kp1_simple.push_back(int(kp1.at(index_queryIdx).pt.y));
        kp2_simple.push_back(int(kp2.at(index_trainIdx).pt.x));
        kp2_simple.push_back(int(kp2.at(index_trainIdx).pt.y));
        //测试,打印
        //cout << kp1_simple[0] << " " << kp1_simple[1] << " " << kp2_simple[0] << " " << kp2_simple[1] << endl;
        //将单个匹配点坐标压栈到匹配点坐标容器
        kp1_location.push_back(kp1_simple);
        kp2_location.push_back(kp2_simple);

        match_count ++;            //匹配点对数计数
    }
    /*********计算子图机器位置到特征点的距离：遍历匹配点，计算机器到子图特征点的距离，并保存在距离容器distance_submap*********/
    for(int i = 0; i < matches_gms.size(); i++)
    {
        float distance;
        distance = sqrt(pow(robot_submap_location[0] - kp1_location[i][0], 2)
                        + pow(robot_submap_location[1] - kp1_location[i][1], 2));
        distance_submap.push_back(distance);
        //测试，打印距离容器
        //cout << distance_submap[i] << endl;
    }
    cout << "匹配对数为：" << match_count << endl;
}

Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type) {
    const int height = max(src1.rows, src2.rows);
    const int width = src1.cols + src2.cols;
    Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
    src1.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
    src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

    if (type == 1)
    {
        for (size_t i = 0; i < inlier.size(); i++)
        {
            Point2f left = kpt1[inlier[i].queryIdx].pt;
            Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
            line(output, left, right, Scalar(0, 255, 255));
        }
    }
    else if (type == 2)
    {
        for (size_t i = 0; i < inlier.size(); i++)
        {
            Point2f left = kpt1[inlier[i].queryIdx].pt;
            Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
            line(output, left, right, Scalar(255, 0, 0));
        }

        for (size_t i = 0; i < inlier.size(); i++)
        {
            Point2f left = kpt1[inlier[i].queryIdx].pt;
            Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
            circle(output, left, 1, Scalar(0, 255, 255), 2);
            circle(output, right, 1, Scalar(0, 255, 0), 2);
        }
    }
    return output;
}

/****************************手写梯度下降定位函数********************************************/
/*********计算成本函数*************/
double loss_fun(vector<float> &x)
{
    double loss_distance = 0.0;
    for (int i = 0; i < match_count; i++) {
        double mid_1 = pow(x[0] - kp2_location[i][0], 2) + pow(x[1] - kp2_location[i][1], 2);
        double mid_2 = sqrt(mid_1);
        double mid_3 = mid_2 - distance_submap[i];
        double mid_4 = 0.5 * mid_3 * mid_3;
        loss_distance += mid_4;
    }
    return loss_distance;
}
/**********************计算成本函数梯度*************************/
vector<float> loss_dev(vector<float> &x)
{
    vector<float> dev_x_y;
    float dev_x = 0, dev_y = 0;
    for (int i = 0; i < match_count; i++){
        dev_x += (sqrt(pow(x[0] - kp2_location[i][0], 2) + pow(x[1] - kp2_location[i][1], 2)) - distance_submap[i]) *
                 (x[0] - kp2_location[i][0]) *
                 1 / sqrt(pow(x[0] - kp2_location[i][0], 2) + pow(x[1] - kp2_location[i][1], 2));
    }
    for (int i = 0; i < match_count; i++) {
        dev_y += (sqrt(pow(x[0] - kp2_location[i][0], 2) + pow(x[1] - kp2_location[i][1], 2)) - distance_submap[i]) *
                 (x[1] - kp2_location[i][1]) *
                 1 / sqrt(pow(x[0] - kp2_location[i][0], 2) + pow(x[1] - kp2_location[i][1], 2));
    }
    dev_x_y = {dev_x, dev_y};
    return dev_x_y;
}
/*******梯度下降实现*********/
vector<float> optimize(vector<float> &x, int num_iteration, float learning_rate)
{
    int num_optimize = 0;                                   //优化次数计数器
    vector<float> dev_cur;                                  //定义当前梯度
    float loss0, loss_cur;                                  //定义初始残差和当前残差
    cout << "初始位置为：" << x[0] << " " << x[1] << endl;
    loss0 = loss_fun(x);                                 //计算初始残差
    cout << "初始成本为：" << loss0 << endl;
    /*********梯度下降 + 坐标更新************/
    for (int i = 0; i < num_iteration; i++) {
        dev_cur = loss_dev(x);
        x[0] = x[0] - learning_rate * dev_cur[0];           //梯度下降更新坐标
        x[1] = x[1] - learning_rate * dev_cur[1];
        num_optimize += 1;
        loss_cur = loss_fun(x);                          //记录当前残差值
        cout << "优化次数为:" << num_optimize << endl;
        cout << "当前残差为：" << loss_cur << endl;
        cout << "当前坐标为：" << int(x[0]) << " " << int(x[1]) << endl;
    }
    return x;
}
void parameter_load()
{

}