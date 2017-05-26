#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "../practice_digtal_photogrammery/pdp.h"



void fixDisparity(cv::Mat& disp, int numberOfDisparities ) 
{
	cv::Mat disp_temp;
	// float minDisparity =13;// algorithm parameters that can be modified
	// for (int i = 0; i < disp.rows; i++)
	// {
	// 	for (int j = numberOfDisparities; j < disp.cols; j++)
	// 	{
	// 		if (disp.at(i,j) <= minDisparity) disp.at(i,j) = lastPixel;
	// 		else lastPixel = disp.at(i,j);
	// 	}
	// }
	// float lastPixel = 0;
 //    auto cn = disp.channels();
 //    auto width = disp.cols;
 //    auto height = disp.rows; 
    // for(int j = 0; j < height; ++j)
    // {
    //     float *p =disp.ptr<float>(j);   
    //     for(int i = numberOfDisparities; i < width; ++i)
    //     {
    //         if (*(p + i*cn) <= minDisparity) *(p + i*cn) = lastPixel;
    //         // if (abs(*(p + i*cn)-lastPixel) < 3) *(p + i*cn) = lastPixel;
    //         else lastPixel = *(p + i*cn);
    //     }
    // }
	int an = 4;	// algorithm parameters that can be modified
	cv::copyMakeBorder(disp, disp_temp, an,an,an,an, cv::BORDER_REPLICATE);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(an*2+1, an*2+1));
	cv::morphologyEx(disp_temp, disp_temp, CV_MOP_OPEN, element);
	cv::morphologyEx(disp_temp, disp_temp, CV_MOP_CLOSE, element);
	disp = disp_temp(cv::Range(an, disp_temp.rows-an), cv::Range(an, disp_temp.cols-an)).clone();

    cv::GaussianBlur(disp,disp,cv::Size(5,5),0.2);
    disp_temp = disp(cv::Range(0,disp.rows),cv::Range(numberOfDisparities,disp.cols)).clone();
    disp = disp_temp;
}

int main(int argc, char const *argv[])
{
    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2 };
    int alg = STEREO_SGBM;
	//img read
	cv::Mat left = cv::imread("../data/leftview.bmp");
	cv::Mat right = cv::imread("../data/rightview.bmp");
	auto width = left.cols;
	auto height = right.rows;
	int cn = left.channels();
    //=======================sgbm parameters init=========
	//set sgbm parameters 
	SGBM sgbm;
    double number_of_disparities = ((width/8) + 15) & -16;
	// double number_of_disparities = 256;
	double sad_window_size = 5;
	sgbm.preFilterCap(63)
		.SADWindowSize(sad_window_size)
		.P1(4*cn*sad_window_size*sad_window_size)
		.P2(32*cn*sad_window_size*sad_window_size)
		.numDisparities(number_of_disparities)
		.minDisparity(0)
		.uniquenessRatio(10)
		.speckleWindowSize(100) 
		.speckleRange(32)
		.disp12MaxDiff(1);
    // sgbm.fullDP = alg == STEREO_HH;

	cv::Mat leftb;
	cv::Mat rightb; 
    cv::copyMakeBorder(left, leftb, 0, 0, number_of_disparities, 0, IPL_BORDER_REPLICATE);
    cv::copyMakeBorder(right, rightb, 0, 0, number_of_disparities, 0, IPL_BORDER_REPLICATE); 
	

	cv::Mat disp;
	cv::Mat disp_temp;
	cv::Mat disp_show;
	sgbm(leftb,rightb,disp_temp);
	fixDisparity(disp_temp, number_of_disparities);
    disp_temp.convertTo(disp, CV_32FC1, 1.0/16);
    disp.convertTo(disp_show, CV_8U, 255.0/number_of_disparities);
    cv::namedWindow("left origin disparity",cv::WINDOW_NORMAL);
    cv::imshow("left origin disparity", disp_show);
	cv::waitKey();
	return 0;
}