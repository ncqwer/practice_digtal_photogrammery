#include "sgm.h"

int SGM::getPos(
		int x,
		int y)
{
	return (y*_width + x)*_number_disparty;
}

float SGM::getCloser(
		float a,
		float b,
		float c,
		float target)
{
	using std::fabs;
	float d1 = 1000;
	float ans;
	if(fabs(a-target) < d1)
	{
		d1 = fabs(a-target);
		ans = a;
	}
	if(fabs(b-target) < d1)
	{
		d1 = fabs(b-target);
		ans = b;
	}
	if(fabs(c-target) < d1)
	{
		d1 = fabs(c-target);
		ans = c;
	}
	return ans;
}

void SGM::intialCostAndLr(
		const cv::Mat& left,
		const cv::Mat& right)
{

	int sz = _number_disparty * _width * _height;

	for(int i = 0; i < 8; ++i) _Lrs[i] = new float[sz];
	for(int i = 0; i < 8; ++i) _min_Lrs[i] = new float[_width*_height];
	for(size_t y = 0; y < _height; ++y)
	{
		const uchar* l = left.ptr<uchar>(y);
		const uchar* r = left.ptr<uchar>(y);
		l += _number_disparty;
		r += _number_disparty;
		for(size_t x = 0; x < _width; ++x)
		{
			float l_middle = *(l + x );
			int index_before = x-1<0 ? 0:x-1;
			float l_before = *(l + index_before);
			int index_after = x+1>_width-1? _width-1:x+1;
			float l_after = *(l + index_after);
			float l_diff = (l_middle + l_before)/2;
			float l_plus = (l_middle + l_after)/2;

			int index = getPos(x,y);
			// std::cout<<"process: "<<x<<" , "<<y<<std::endl;
			for(size_t nr = 0 ; nr < 8; ++nr)
			{
				float* Lr = _Lrs[nr] + index;
				float* min_Lr = _min_Lrs[nr] + y*_width + x;
				float min_v = 10000;
				for(size_t d = 0; d < _number_disparty; ++d)
				{
					float r_middle = *(r + x - d);
					float r_before = *(r + index_before - d);
					float r_after = *(r + index_after - d);
					float r_diff = (r_middle + r_before)/2;
					float r_plus = (r_middle + r_after)/2;

					float v1 = getCloser(l_middle,l_plus,l_diff,r_middle);
					float v2 = getCloser(r_middle,r_plus,r_diff,l_middle);
					float v = std::min(v1,v2);
					*(Lr + d) = v;
					if(min_v > v) min_v = v;
				}
				*min_Lr = min_v;
			}
		}
	}
}

void SGM::getBefore(
		const int nr,
		const int pass,
		const int x,
		const int y,
		int& x_before,
		int& y_before)
{
	if(pass == 0)
	{
		if(nr == 0)
		{
			x_before = x +1;
			y_before = y -1;
		} 
		if(nr == 1)
		{
			x_before = x;
			y_before = y -1;
		}
		if(nr == 2)
		{
			x_before = x -1;
			y_before = y -1;
		}
		if(nr == 3)
		{
			x_before = x -1;
			y_before = y;
		}
	}
	else
	{
		if(nr == 0)
		{
			x_before = x -1;
			y_before = y +1;
		} 
		if(nr == 1)
		{
			x_before = x;
			y_before = y +1;
		}
		if(nr == 2)
		{
			x_before = x +1;
			y_before = y +1;
		}
		if(nr == 3)
		{
			x_before = x +1;
			y_before = y;
		}
	}
}

void SGM::clear()
{
	for(size_t nr = 0; nr < 8; ++nr)
	{
		if(_Lrs[nr] != NULL) delete []_Lrs[nr];
		if(_min_Lrs[nr] != NULL) delete []_min_Lrs[nr];
	}
	_width = 0;
	_height = 0;
}

void SGM::dynamicProgramming(
		cv::Mat& disp)
{
	using std::min;
	for(size_t pass = 0 ; pass < _npass; ++pass)
	{
		int x1,x2,y1,y2,dx,dy;
		if(pass == 0)
		{
			x1 = 1; x2 = _width-1; dx = 1;
			y1 = 1; y2 = _height-1; dy = 1;
		}
		else
		{
			x1 = _width-1; x2 = 1; dx = -1;
			y1 = _height-1; y2 = 1; dy = -1;
		}
		for(int y = y1; y != y2; y+=dy)
		{
			for(int x = x1; x != x2; x+=dx)
			{
				for(int nr = 0; nr < 4; ++nr)
				{
					int x_before,y_before;
					getBefore(nr,pass,x,y,x_before,y_before);
					float* Lr_before = _Lrs[nr + pass*4] + getPos(x_before,y_before);
					float* Lr = _Lrs[nr + pass*4] + getPos(x,y);
					float* min_Lr_before = _min_Lrs[nr + pass*4] + (y_before*_width+x_before);
					float min_v_before = *min_Lr_before;
					float L4 = min_v_before+_P2;
					float min_v = 100000;

					for(int d = 0; d < _number_disparty; ++d)
					{
						float L1 = *(Lr_before + d);
						int d2 = d-1 < 0 ? 0 : d;
						float L2 = *(Lr_before + d2) + _P1;
						int d3 = d+1 > _number_disparty-1 ? _number_disparty-1 : d+1;
						float L3 = *(Lr_before + d3) + _P1;
						*(Lr + d) += (min(L1,min(L2,min(L3,L4))) - min_v_before);
						if(*(Lr + d) < min_v)
						{
							min_v = *(Lr + d);
						}
					} 
					float* min_Lr = _min_Lrs[nr + pass*4] + (y*_width+x);
					*(min_Lr) = min_v;
				}
			}
		}
	}
	//fill dispf
	for(int y = 0; y < _height; ++y)
	{
		uchar *pdisp = disp.ptr<uchar>(y);
		for(int x = 0; x < _width; ++x)
		{
			float max_sum = -10000;
			int disparty = 0;
			for(int d = 0; d < _number_disparty; ++d)
			{
				float sum = 0;
				for(int nr = 0; nr < 8; ++nr)
				{
					float* Lr = _Lrs[nr] + getPos(x,y);
					sum += *(Lr+d);
				}
				if(max_sum < sum)
				{
					max_sum = sum;
					disparty = d;
				}
			}
			std::cout<<"x: "<<x<<" y: "<<y<<"  "<<disparty<<std::endl;
			*(pdisp + x) = disparty;
		}
	}
}

void SGM::operator() (
		const cv::Mat& left,
		const cv::Mat& right,
		cv::Mat& disp)
{
	clear();
	_width = left.cols - _number_disparty;
	_height = left.rows;
	intialCostAndLr(left,right);

	disp.create(cv::Size(_width,_height),CV_8U);
	dynamicProgramming(disp);
}