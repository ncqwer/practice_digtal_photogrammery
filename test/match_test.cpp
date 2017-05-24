#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "../practice_digtal_photogrammery/dense_match.h"

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
    // double number_of_disparities = ((width/8) + 15) & -16;
	double number_of_disparities = 256;
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

	cv::Mat disp;
	cv::Mat disp_temp;
	cv::Mat disp_show;
	sgbm(left,right,disp_temp);
    disp_temp.convertTo(disp, CV_32FC1, 1.0/16);
    disp_temp.convertTo(disp_show, CV_8U, 255.0/number_of_disparities);
    cv::namedWindow("left origin disparity",cv::WINDOW_NORMAL);
    cv::imshow("left origin disparity", disp_show);
	cv::waitKey();
	return 0;
}