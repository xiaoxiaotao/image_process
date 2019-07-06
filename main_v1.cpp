/*
* Author: ChengYantao
* Time:2019.07.04
* Function:对强光或者弱光下光照不均匀进行处理，使其分布一致
*/
#include <iostream>
#include <signal.h>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//RGB 对各个通道直方图均衡化
cv::Mat equalizehist(cv::Mat &inputframe){

    vector<Mat> BGR_plane;
    //分离BGR通道
    split(inputframe, BGR_plane);
    //分别对BGR通道进行直方图均衡化
    for (int i = 0; i < BGR_plane.size(); i++){
        equalizeHist(BGR_plane[i], BGR_plane[i]);
    }
    //合并通道
    merge(BGR_plane, inputframe);
    BGR_plane.clear();
    //return inputframe;
}
//图像锐化滤波
cv::Mat Sharp_Blur(cv::Mat frame){

    cv::Mat kernel(3,3,CV_32F,Scalar(0));
    kernel.at<float>(1,1) = 5.05;
    kernel.at<float>(0,1) = -1;
    kernel.at<float>(1,0) = -1;
    kernel.at<float>(2,1) = -1;
    kernel.at<float>(1,2) = -1;
    filter2D(frame,frame,frame.depth(),kernel);
    GaussianBlur(frame,frame,cv::Size(3,3),0,0);
    return frame;
}
//灰度图像抑制光照不均匀，处理后使其光照分布一致
void Adjust_Lighting(cv::Mat &image, int blockSize)
{
    if (image.channels() == 3) cvtColor(image, image, 7);
    double average = mean(image)[0];
    int rows_new = ceil(double(image.rows) / double(blockSize));
    int cols_new = ceil(double(image.cols) / double(blockSize));
    cv::Mat blockImage;
    blockImage = cv::Mat::zeros(rows_new, cols_new, CV_32FC1);
    for (int i = 0; i < rows_new; i++)
    {
        for (int j = 0; j < cols_new; j++)
        {
            int rowmin = i*blockSize;
            int rowmax = (i + 1)*blockSize;
            if (rowmax > image.rows) rowmax = image.rows;
            int colmin = j*blockSize;
            int colmax = (j + 1)*blockSize;
            if (colmax > image.cols) colmax = image.cols;
            cv::Mat imageROI = image(cv::Range(rowmin, rowmax), cv::Range(colmin, colmax));
            double temaver = mean(imageROI)[0];
            blockImage.at<float>(i, j) = temaver;
        }
    }
    blockImage = blockImage - average;
    cv::Mat blockImage2;
    cv::resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), INTER_CUBIC);
    cv::Mat image2;
    image.convertTo(image2, CV_32FC1);
    cv::Mat dst = image2 - blockImage2;
    dst.convertTo(image, CV_8UC1);
}
//RGB->YCrCb彩色视频直方图均衡化，防止光照不均匀,结合Adjust_Lighting使用
void Adjust_BGR_On_YUV(cv::Mat &frame)
{
    vector<cv::Mat> yuv_vec;
    cv::cvtColor(frame, frame, CV_BGR2YCrCb);
    split(frame,yuv_vec);//split Y U V


    //equalizeHist(yuv_vec[0],yuv_vec[0]);// Y channle
    Adjust_Lighting(yuv_vec[0],32);// Y channle


    merge(yuv_vec, frame);
    cv::cvtColor(frame, frame,CV_YCrCb2BGR);
    yuv_vec.clear();

}
//白平衡算法中的灰度世界法，改善图像发红发蓝发绿的现象
void Adjust_White_Banlance(cv::Mat &frame){

      //cv::Mat frame;
      vector<cv::Mat> g_vChannels;

      //分离通道
      split(frame,g_vChannels);
      cv::Mat imageBlueChannel = g_vChannels.at(0);
      cv::Mat imageGreenChannel = g_vChannels.at(1);
      cv::Mat imageRedChannel = g_vChannels.at(2);

      double imageBlueChannelAvg=0;
      double imageGreenChannelAvg=0;
      double imageRedChannelAvg=0;

      //求各通道的平均值
      imageBlueChannelAvg = mean(imageBlueChannel)[0];
      imageGreenChannelAvg = mean(imageGreenChannel)[0];
      imageRedChannelAvg = mean(imageRedChannel)[0];

      //求出个通道所占增益
      double K = (imageRedChannelAvg+imageGreenChannelAvg+imageRedChannelAvg)/3;
      double Kb = K/imageBlueChannelAvg;
      double Kg = K/imageGreenChannelAvg;
      double Kr = K/imageRedChannelAvg;

      //更新白平衡后的各通道BGR值
      addWeighted(imageBlueChannel,Kb,0,0,0,imageBlueChannel);
      addWeighted(imageGreenChannel,Kg,0,0,0,imageGreenChannel);
      addWeighted(imageRedChannel,Kr,0,0,0,imageRedChannel);

      merge(g_vChannels,frame);//图像各通道合并
}

int main()
{

    cv::Mat d_descriptors;
    vector<cv::KeyPoint> keyPoints;
    cv::VideoCapture capture;

    capture.open(0);
    cv::Mat frame;
    cv::Mat ShowKeypoints1;

    cv::Ptr<cv::ORB> d_orb = cv::ORB::create(1500, 1.2f, 10, 31, 0, 2, ORB::HARRIS_SCORE,31,20);
    int blockSize=64;
    cv::Mat frame_gray;

    while(true){

        capture.read(frame);
        if(frame.empty()){
            capture.open(0);
            continue;
        }
        cv::cvtColor(frame,frame_gray,CV_BGR2GRAY);
        imshow("original",frame_gray);

        Adjust_BGR_On_YUV(frame);
        //Adjust_White_Banlance(frame);

        //cv::cvtColor(frame, frame,CV_RGB2GRAY);
        //Equalizehist(frame,frame);

        frame=Sharp_Blur(frame);
        //Adjust_Lighting(frame,blockSize);

        //d_orb -> detectAndCompute(frame, Mat(), keyPoints, d_descriptors);
        //cout<<keyPoints.size()<<endl;
        cv::cvtColor(frame,frame,CV_BGR2GRAY);
        imshow("after deal",frame);
        waitKey(1);

    }
 

 
	return 0;
}

