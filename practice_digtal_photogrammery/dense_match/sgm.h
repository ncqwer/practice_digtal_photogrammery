#ifndef _DENSE_MATCH_SGM_H_
#define _DENSE_MATCH_SGM_H_ 
#include <iostream>

#include <opencv2/opencv.hpp>

class SGM
{
public:
	SGM()
	{
		for(size_t nr = 0; nr < 8; ++nr)
		{
			_Lrs[nr] = NULL;
			_min_Lrs[nr] = NULL;
		}
	}
	~SGM()
	{
		clear();
	}

	SGM(const SGM& rhs)=delete;
	SGM(SGM&& rhs) noexcept=delete;

	SGM& operator= (SGM rhs_copy)=delete;

	void operator() (
			const cv::Mat& left,
			const cv::Mat& right,
			cv::Mat& disp);

	void intialCostAndLr(
			const cv::Mat& left,
			const cv::Mat& right);

	void dynamicProgramming(
			cv::Mat& disp);

	SGM& numberDisparity(
			int numberDisparty)
	{
		_number_disparty = numberDisparty;
		return *this;
	}

	SGM& P1(
			int P1)
	{
		_P1 = P1;
		return *this;
	}

	SGM& P2(
			int P2)
	{
		_P2 = P2;
		return *this;
	}

	SGM& NPass(
			int NPass)
	{
		_npass = NPass;
		return *this;
	}
private:
	int getPos(
			int x,
			int y);

	
	float getCloser(
			float a,
			float b,
			float c,
			float target);

	
	void getBefore(
			const int nr,
			const int pass,
			const int x,
			const int y,
			int& x_before,
			int& y_before);

	void clear();

	float* _Lrs[8];
	float* _min_Lrs[8];
	int _number_disparty;
	int _width;
	int _height;
	int _npass;
	int _P2;
	int _P1;
};


#endif