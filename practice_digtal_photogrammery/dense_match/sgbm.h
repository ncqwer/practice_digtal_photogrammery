#include <iostream>

#include <opencv2/opencv.hpp>
#ifndef _DENSE_MATCH_SGBM_H_
#define _DENSE_MATCH_SGBM_H_ 


class SGBM
{
public:
    enum { DISP_SHIFT=4, DISP_SCALE = (1<<DISP_SHIFT) };

    //! the default constructor
     SGBM();

    //! the full constructor taking all the necessary algorithm parameters
    SGBM(
            int minDisparity, 
            int numDisparities, 
            int SADWindowSize,
            int P1, int P2, int disp12MaxDiff, int preFilterCap,
            int uniquenessRatio, int speckleWindowSize, int speckleRange,
            bool fullDP );
    //! the destructor
    ~SGBM();

    SGBM& minDisparity(
            const int m)
    {
        _minDisparity = m;
        return *this;
    }

    SGBM& numDisparities(
            const int n)
    {
        _numberOfDisparities = n;
        return *this;
    }

    SGBM& SADWindowSize(
            const int sad)
    {
        _SADWindowSize = sad;
        return *this;
    }

    SGBM& P1(
            const int p)
    {
        _P1 = p;
        return *this;
    }

    SGBM& P2(
            const int p)
    {
        _P2 = p;
        return *this;
    }

    SGBM& disp12MaxDiff(
            const int d)
    {
        _disp12MaxDiff = d;
        return *this;
    }

    SGBM& preFilterCap(
            const int p)
    {
        _preFilterCap = p;
        return *this;
    }

    SGBM& uniquenessRatio(
            const int u)
    {
        _uniquenessRatio = u;
        return *this;
    }

    SGBM& speckleWindowSize(
            const int sz)
    {
        _speckleRange = sz;
        return *this;
    }

    SGBM& speckleRange(
            const int r)
    {
        _speckleRange = r;
        return *this;
    }

    SGBM& fullDP(
            const bool f)
    {
        _fullDP = f;
        return *this;
    }
    //! the stereo correspondence operator that computes disparity map for the specified rectified stereo pair
    void operator()(cv::InputArray left, cv::InputArray right,
                                                cv::OutputArray disp);

    int _minDisparity;
    int _numberOfDisparities;
    int _SADWindowSize;
    int _preFilterCap;
    int _uniquenessRatio;
    int _P1;
    int _P2;
    int _speckleWindowSize;
    int _speckleRange;
    int _disp12MaxDiff;
    bool _fullDP;

protected:
    cv::Mat buffer;
};

#endif




// static void calcPixelCostBT( const Mat& img1, const Mat& img2, int y,
//                             int minD, int maxD, CostType* cost,
//                             PixType* buffer, const PixType* tab,
//                             int tabOfs, int );


