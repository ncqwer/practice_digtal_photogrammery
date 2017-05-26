#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "../practice_digtal_photogrammery/pdp.h"

cv::Mat forwardInterSection(
		const cv::Mat& left, 
		const cv::Mat& disp, 
		double B, double f, double x0, double y0,
		std::vector<cv::Point3f>& cloud_p,
		std::vector<cv::Point3i>& cloud_c)
{
	cloud_c.clear();
	cloud_p.clear();
	auto cn = left.channels();
	for (int y = 0; y < left.rows; y++)
	{
		auto p_left = left.ptr<uchar>(y);
		auto d_left = disp.ptr<float>(y);
		for (int x = 0; x < left.cols; x++)
		{
			float parallax = *(d_left + x);
			if (parallax == 0)
				break;
			cv::Point3f pPoint;
			cv::Point3i cPoint;
			float Z = f*B / parallax;
			float X = (x + 1 - x0)*Z / f;
			float Y = (y + 1 - y0)*Z / f;

			pPoint.x = X;
			pPoint.y = Y;
			pPoint.z = Z;

			cPoint.x = *(p_left + x*cn + 0);
			cPoint.y = *(p_left + x*cn + 1);
			cPoint.z = *(p_left + x*cn + 2);

			cloud_p.push_back(pPoint);
			cloud_c.push_back(cPoint);
		}
	}

	int hsj =1;
}




void fixDisparity(cv::Mat& disp, int numberOfDisparities ) 
{
	cv::Mat disp_temp;
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


void save(
		const std::string& filename,
		std::vector<cv::Point3f>& cloud_p,
		std::vector<cv::Point3i>& cloud_c)
{
	std::ofstream in(filename);

	auto pt_b = cloud_p.begin();
	auto color_b = cloud_c.begin();

	auto pt_e = cloud_p.end();
	for(;
		pt_b != pt_e;
		++pt_b,++color_b)
	{
		in
			<<pt_b->x<<" "
			<<pt_b->y<<" "
			<<pt_b->z<<" "
			<<color_b->x<<" "
			<<color_b->y<<" "
			<<color_b->z<<" "<<std::endl;
	}
}
int main(int argc, char const *argv[])
{
    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2 };
    int alg = STEREO_SGBM;
	//img read
	cv::Mat left = cv::imread("../data/1-2.lei.bmp");
	cv::Mat right = cv::imread("../data/1-2.rei.bmp");
	auto width = left.cols;
	auto height = right.rows;
	int cn = left.channels();
    //=======================sgbm parameters init=========
	//set sgbm parameters 
	SGBM sgbm;
    // double number_of_disparities = ((width/8) + 15) & -16;
	double number_of_disparities = 192;
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
	// sgbm(leftb,rightb,disp_temp);
	moduleCall_kernal(
		"SGBM",
		std::cout,
		sgbm,
		leftb,rightb,
		disp_temp);
	// fixDisparity(disp_temp, number_of_disparities);
	moduleCall_kernal(
		"fix disparity",
		std::cout,
		fixDisparity,
		disp_temp,number_of_disparities);
    disp_temp.convertTo(disp, CV_32FC1, 1.0/16);
    disp.convertTo(disp_show, CV_8U, 255.0/number_of_disparities);
    cv::namedWindow("left origin disparity",cv::WINDOW_NORMAL);
    cv::imshow("left origin disparity", disp_show);

	std::vector<cv::Point3f> cloud_p;
	std::vector<cv::Point3i> cloud_c; 
	forwardInterSection(
			left, 
			disp, 
			// double B, double f, double x0, double y0,
			1,1,left.cols/2,left.rows/2,
			cloud_p,
			cloud_c
			);
	// moduleCall_kernal(
	// 		"forwardInterSection",
	// 		std::cout,
	// 		forwardInterSection,
	// 		left,disp,1,1,0,0,
	// 		cloud_p,cloud_c);  // there is bug,cv::Mat release error
	std::cout
		<<"pt size"<<cloud_c.size()<<std::endl
		<<"pt size"<<cloud_p.size()<<std::endl;
	// save("pts.txt",cloud_p,cloud_c);
	moduleCall_kernal(
			"save pts",
			std::cout,
			save,
			"pts.txt",
			cloud_p,
			cloud_c);
	cv::waitKey();
	return 0;
}