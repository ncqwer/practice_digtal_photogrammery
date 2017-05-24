#include "sgbm.h"


typedef uchar PixType;
typedef short CostType;
typedef short DispType;

// using cv::InputArray;
typedef cv::InputArray InputArray;
typedef cv::OutputArray OutputArray;
// using cv::InputOutputArray;
typedef cv::InputOutputArray InputOutputArray;


enum { NR = 16, NR2 = NR/2 };

SGBM::SGBM()
{
    _minDisparity = _numberOfDisparities = 0;
    _SADWindowSize = 0;
    _P1 = _P2 = 0;
    _disp12MaxDiff = 0;
    _disp12MaxDiff = 0;
    _uniquenessRatio = 0;
    _speckleWindowSize = 0;
    _speckleWindowSize = 0;
    _fullDP = false;
}


SGBM::SGBM(
        int minDisparity, 
        int numDisparities, 
        int SADWindowSize,
        int P1, int P2, int disp12MaxDiff, int preFilterCap,
        int uniquenessRatio, int speckleWindowSize, int speckleRange,
        bool fullDP )
{
    _minDisparity = minDisparity;
    _numberOfDisparities = numDisparities;
    _SADWindowSize = SADWindowSize;
    _P1 = P1;
    _P2 = P2;
    _disp12MaxDiff = disp12MaxDiff;
    _disp12MaxDiff = disp12MaxDiff;
    _uniquenessRatio = uniquenessRatio;
    _speckleWindowSize = speckleWindowSize;
    _speckleWindowSize = speckleWindowSize;
    _fullDP = fullDP;
}


SGBM::~SGBM()
{
}


static void calcPixelCostBT( const cv::Mat& img1, const cv::Mat& img2, int y,
                            int minD, int maxD, CostType* cost,
                            PixType* buffer, const PixType* tab,
                            int tabOfs, int )
{
    int x, c, width = img1.cols, cn = img1.channels();
    int minX1 = std::max(-maxD, 0), maxX1 = width + std::min(-minD, 0);
    int minX2 = std::max(minX1 + minD, 0), maxX2 = std::min(maxX1 + maxD, width);
    int D = maxD - minD, width1 = maxX1 - minX1, width2 = maxX2 - minX2;
    const PixType *row1 = img1.ptr<PixType>(y), *row2 = img2.ptr<PixType>(y);
    PixType *prow1 = buffer + width2*2, *prow2 = prow1 + width*cn*2;

    tab += tabOfs;

    for( c = 0; c < cn*2; c++ )
    {
        prow1[width*c] = prow1[width*c + width-1] =
        prow2[width*c] = prow2[width*c + width-1] = tab[0];
    }

    int n1 = y > 0 ? -(int)img1.step : 0, s1 = y < img1.rows-1 ? (int)img1.step : 0;
    int n2 = y > 0 ? -(int)img2.step : 0, s2 = y < img2.rows-1 ? (int)img2.step : 0;

    if( cn == 1 )
    {
        for( x = 1; x < width-1; x++ )
        {
            prow1[x] = tab[(row1[x+1] - row1[x-1])*2 + row1[x+n1+1] - row1[x+n1-1] + row1[x+s1+1] - row1[x+s1-1]];
            prow2[width-1-x] = tab[(row2[x+1] - row2[x-1])*2 + row2[x+n2+1] - row2[x+n2-1] + row2[x+s2+1] - row2[x+s2-1]];

            prow1[x+width] = row1[x];
            prow2[width-1-x+width] = row2[x];
        }
    }
    else
    {
        for( x = 1; x < width-1; x++ )
        {
            prow1[x] = tab[(row1[x*3+3] - row1[x*3-3])*2 + row1[x*3+n1+3] - row1[x*3+n1-3] + row1[x*3+s1+3] - row1[x*3+s1-3]];
            prow1[x+width] = tab[(row1[x*3+4] - row1[x*3-2])*2 + row1[x*3+n1+4] - row1[x*3+n1-2] + row1[x*3+s1+4] - row1[x*3+s1-2]];
            prow1[x+width*2] = tab[(row1[x*3+5] - row1[x*3-1])*2 + row1[x*3+n1+5] - row1[x*3+n1-1] + row1[x*3+s1+5] - row1[x*3+s1-1]];

            prow2[width-1-x] = tab[(row2[x*3+3] - row2[x*3-3])*2 + row2[x*3+n2+3] - row2[x*3+n2-3] + row2[x*3+s2+3] - row2[x*3+s2-3]];
            prow2[width-1-x+width] = tab[(row2[x*3+4] - row2[x*3-2])*2 + row2[x*3+n2+4] - row2[x*3+n2-2] + row2[x*3+s2+4] - row2[x*3+s2-2]];
            prow2[width-1-x+width*2] = tab[(row2[x*3+5] - row2[x*3-1])*2 + row2[x*3+n2+5] - row2[x*3+n2-1] + row2[x*3+s2+5] - row2[x*3+s2-1]];

            prow1[x+width*3] = row1[x*3];
            prow1[x+width*4] = row1[x*3+1];
            prow1[x+width*5] = row1[x*3+2];

            prow2[width-1-x+width*3] = row2[x*3];
            prow2[width-1-x+width*4] = row2[x*3+1];
            prow2[width-1-x+width*5] = row2[x*3+2];
        }
    }

    memset( cost, 0, width1*D*sizeof(cost[0]) );

    buffer -= minX2;
    cost -= minX1*D + minD; // simplify the cost indices inside the loop
    for( c = 0; c < cn*2; c++, prow1 += width, prow2 += width )
    {
        int diff_scale = c < cn ? 0 : 2;

        // precompute
        //   v0 = std::min(row2[x-1/2], row2[x], row2[x+1/2]) and
        //   v1 = std::max(row2[x-1/2], row2[x], row2[x+1/2]) and
        for( x = minX2; x < maxX2; x++ )
        {
            int v = prow2[x];
            int vl = x > 0 ? (v + prow2[x-1])/2 : v;
            int vr = x < width-1 ? (v + prow2[x+1])/2 : v;
            int v0 = std::min(vl, vr); v0 = std::min(v0, v);
            int v1 = std::max(vl, vr); v1 = std::max(v1, v);
            buffer[x] = (PixType)v0;
            buffer[x + width2] = (PixType)v1;
        }

        for( x = minX1; x < maxX1; x++ )
        {
            int u = prow1[x];
            int ul = x > 0 ? (u + prow1[x-1])/2 : u;
            int ur = x < width-1 ? (u + prow1[x+1])/2 : u;
            int u0 = std::min(ul, ur); u0 = std::min(u0, u);
            int u1 = std::max(ul, ur); u1 = std::max(u1, u);

            int minDlocal = std::max(minD, x-width+1);
            int maxDlocal = std::min(maxD, x);
            int d;
            for( d = minD; d < minDlocal; d++ )
            {
                int v = prow2[0];
                int v0 = buffer[0];
                int v1 = buffer[width2];
                int c0 = std::max(0, u - v1); c0 = std::max(c0, v0 - u);
                int c1 = std::max(0, v - u1); c1 = std::max(c1, u0 - v);

                cost[x*D + d] = (CostType)(cost[x*D+d] + (std::min(c0, c1) >> diff_scale));
            }
            for( ; d < maxDlocal; d++ )
            {
                int v = prow2[width-x-1 + d];
                int v0 = buffer[width-x-1 + d];
                int v1 = buffer[width-x-1 + d + width2];
                int c0 = std::max(0, u - v1); c0 = std::max(c0, v0 - u);
                int c1 = std::max(0, v - u1); c1 = std::max(c1, u0 - v);

                cost[x*D + d] = (CostType)(cost[x*D+d] + (std::min(c0, c1) >> diff_scale));
            }
            for( ; d < maxD; d++ )
            {
                int v = prow2[width-1];
                int v0 = buffer[width-1];
                int v1 = buffer[width-1 + width2];
                int c0 = std::max(0, u - v1); c0 = std::max(c0, v0 - u);
                int c1 = std::max(0, v - u1); c1 = std::max(c1, u0 - v);

                cost[x*D + d] = (CostType)(cost[x*D+d] + (std::min(c0, c1) >> diff_scale));
            }
        }
    }
}


static void computeDisparitySGBM( const cv::Mat& img1, const cv::Mat& img2,
                                 cv::Mat& disp1, const SGBM& params,
                                 cv::Mat& buffer )
{
    const int ALIGN = 16;
    const int DISP_SHIFT = SGBM::DISP_SHIFT;
    const int DISP_SCALE = SGBM::DISP_SCALE;
    const CostType MAX_COST = SHRT_MAX;

    int minD = params._minDisparity, maxD = minD + params._numberOfDisparities;
    cv::Size _SADWindowSize;
    _SADWindowSize.width = _SADWindowSize.height = params._SADWindowSize > 0 ? params._SADWindowSize : 5;
    int ftzero = std::max(params._disp12MaxDiff, 15) | 1;
    int _uniquenessRatio = params._uniquenessRatio >= 0 ? params._uniquenessRatio : 10;
    int _disp12MaxDiff = params._disp12MaxDiff > 0 ? params._disp12MaxDiff : 1;
    int _P1 = params._P1 > 0 ? params._P1 : 2, _P2 = std::max(params._P2 > 0 ? params._P2 : 5, _P1+1);
    int k, width = disp1.cols, height = disp1.rows;
    int minX1 = std::max(-maxD, 0), maxX1 = width + std::min(-minD, 0);
    int D = maxD - minD, width1 = maxX1 - minX1;
    int INVALID_DISP = minD - 1, INVALID_DISP_SCALED = INVALID_DISP*DISP_SCALE;
    int SW2 = _SADWindowSize.width/2, SH2 = _SADWindowSize.height/2;
    int npasses = params._fullDP ? 2 : 1;
    const int TAB_OFS = 256*4, TAB_SIZE = 256 + TAB_OFS*2;
    PixType clipTab[TAB_SIZE];

    for( k = 0; k < TAB_SIZE; k++ )
        clipTab[k] = (PixType)(std::min(std::max(k - TAB_OFS, -ftzero), ftzero) + ftzero);

    if( minX1 >= maxX1 )
    {
        disp1 = cv::Scalar::all(INVALID_DISP_SCALED);
        return;
    }

    CV_Assert( D % 16 == 0 );

    // NR - the number of directions. the loop on x below that computes Lr assumes that NR == 8.
    // if you change NR, please, modify the loop as well.
    int D2 = D+16, NRD2 = NR2*D2;

    // the number of L_r(.,.) and min_k L_r(.,.) lines in the buffer:
    // for 8-way dynamic programming we need the current row and
    // the previous row, i.e. 2 rows in total
    const int NLR = 2;
    const int LrBorder = NLR - 1;

    // for each possible stereo match (img1(x,y) <=> img2(x-d,y))
    // we keep pixel difference cost (C) and the summary cost over NR directions (S).
    // we also keep all the partial costs for the previous line L_r(x,d) and also min_k L_r(x, k)
    size_t costBufSize = width1*D;
    size_t CSBufSize = costBufSize*(params._fullDP ? height : 1);
    size_t minLrSize = (width1 + LrBorder*2)*NR2, LrSize = minLrSize*D2;
    int hsumBufNRows = SH2*2 + 2;
    size_t totalBufSize = (LrSize + minLrSize)*NLR*sizeof(CostType) + // minLr[] and Lr[]
    costBufSize*(hsumBufNRows + 1)*sizeof(CostType) + // hsumBuf, pixdiff
    CSBufSize*2*sizeof(CostType) + // C, S
    width*16*img1.channels()*sizeof(PixType) + // temp buffer for computing per-pixel cost
    width*(sizeof(CostType) + sizeof(DispType)) + 1024; // disp2cost + disp2

    if( !buffer.data || !buffer.isContinuous() ||
        buffer.cols*buffer.rows*buffer.elemSize() < totalBufSize )
        buffer.create(1, (int)totalBufSize, CV_8U);

    // summary cost over different (nDirs) directions
    CostType* Cbuf = (CostType*)cv::alignPtr(buffer.data, ALIGN);
    memset(Cbuf, 0, CSBufSize*sizeof(CostType));
    CostType* Sbuf = Cbuf + CSBufSize;
    CostType* hsumBuf = Sbuf + CSBufSize;
    CostType* pixDiff = hsumBuf + costBufSize*hsumBufNRows;

    CostType* disp2cost = pixDiff + costBufSize + (LrSize + minLrSize)*NLR;
    DispType* disp2ptr = (DispType*)(disp2cost + width);
    PixType* tempBuf = (PixType*)(disp2ptr + width);

    // add _P2 to every C(x,y). it saves a few operations in the inner loops
    for( k = 0; k < width1*D; k++ )
        Cbuf[k] = (CostType)_P2;

    for( int pass = 1; pass <= npasses; pass++ )
    {
        int x1, y1, x2, y2, dx, dy;

        if( pass == 1 )
        {
            y1 = 0; y2 = height; dy = 1;
            x1 = 0; x2 = width1; dx = 1;
        }
        else
        {
            y1 = height-1; y2 = -1; dy = -1;
            x1 = width1-1; x2 = -1; dx = -1;
        }

        CostType *Lr[NLR]={0}, *minLr[NLR]={0};

        for( k = 0; k < NLR; k++ )
        {
            // shift Lr[k] and minLr[k] pointers, because we allocated them with the borders,
            // and will occasionally use negative indices with the arrays
            // we need to shift Lr[k] pointers by 1, to give the space for d=-1.
            // however, then the alignment will be imperfect, i.e. bad for SSE,
            // thus we shift the pointers by 8 (8*sizeof(short) == 16 - ideal alignment)
            Lr[k] = pixDiff + costBufSize + LrSize*k + NRD2*LrBorder + 8;
            memset( Lr[k] - LrBorder*NRD2 - 8, 0, LrSize*sizeof(CostType) );
            minLr[k] = pixDiff + costBufSize + LrSize*NLR + minLrSize*k + NR2*LrBorder;
            memset( minLr[k] - LrBorder*NR2, 0, minLrSize*sizeof(CostType) );
        }

        for( int y = y1; y != y2; y += dy )
        {
            int x, d;
            DispType* disp1ptr = disp1.ptr<DispType>(y);
            CostType* C = Cbuf + (!params._fullDP ? 0 : y*costBufSize);
            CostType* S = Sbuf + (!params._fullDP ? 0 : y*costBufSize);

            if( pass == 1 ) // compute C on the first pass, and reuse it on the second pass, if any.
            {
                int dy1 = y == 0 ? 0 : y + SH2, dy2 = y == 0 ? SH2 : dy1;

                for( k = dy1; k <= dy2; k++ )
                {
                    CostType* hsumAdd = hsumBuf + (std::min(k, height-1) % hsumBufNRows)*costBufSize;

                    if( k < height )
                    {
                        calcPixelCostBT( img1, img2, k, minD, maxD, pixDiff, tempBuf, clipTab, TAB_OFS, ftzero );

                        memset(hsumAdd, 0, D*sizeof(CostType));
                        for( x = 0; x <= SW2*D; x += D )
                        {
                            int scale = x == 0 ? SW2 + 1 : 1;
                            for( d = 0; d < D; d++ )
                                hsumAdd[d] = (CostType)(hsumAdd[d] + pixDiff[x + d]*scale);
                        }

                        if( y > 0 )
                        {
                            const CostType* hsumSub = hsumBuf + (std::max(y - SH2 - 1, 0) % hsumBufNRows)*costBufSize;
                            const CostType* Cprev = !params._fullDP || y == 0 ? C : C - costBufSize;

                            for( x = D; x < width1*D; x += D )
                            {
                                const CostType* pixAdd = pixDiff + std::min(x + SW2*D, (width1-1)*D);
                                const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*D, 0);
                                for( d = 0; d < D; d++ )
                                {
                                    int hv = hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
                                    C[x + d] = (CostType)(Cprev[x + d] + hv - hsumSub[x + d]);
                                }
                            }
                        }
                        else
                        {
                            for( x = D; x < width1*D; x += D )
                            {
                                const CostType* pixAdd = pixDiff + std::min(x + SW2*D, (width1-1)*D);
                                const CostType* pixSub = pixDiff + std::max(x - (SW2+1)*D, 0);

                                for( d = 0; d < D; d++ )
                                    hsumAdd[x + d] = (CostType)(hsumAdd[x - D + d] + pixAdd[d] - pixSub[d]);
                            }
                        }
                    }

                    if( y == 0 )
                    {
                        int scale = k == 0 ? SH2 + 1 : 1;
                        for( x = 0; x < width1*D; x++ )
                            C[x] = (CostType)(C[x] + hsumAdd[x]*scale);
                    }
                }

                // also, clear the S buffer
                for( k = 0; k < width1*D; k++ )
                    S[k] = 0;
            }

            // clear the left and the right borders
            memset( Lr[0] - NRD2*LrBorder - 8, 0, NRD2*LrBorder*sizeof(CostType) );
            memset( Lr[0] + width1*NRD2 - 8, 0, NRD2*LrBorder*sizeof(CostType) );
            memset( minLr[0] - NR2*LrBorder, 0, NR2*LrBorder*sizeof(CostType) );
            memset( minLr[0] + width1*NR2, 0, NR2*LrBorder*sizeof(CostType) );

            /*
             [formula 13 in the paper]
             compute L_r(p, d) = C(p, d) +
             std::min(L_r(p-r, d),
             L_r(p-r, d-1) + _P1,
             L_r(p-r, d+1) + _P1,
             min_k L_r(p-r, k) + _P2) - min_k L_r(p-r, k)
             where p = (x,y), r is one of the directions.
             we process all the directions at once:
             0: r=(-dx, 0)
             1: r=(-1, -dy)
             2: r=(0, -dy)
             3: r=(1, -dy)
             4: r=(-2, -dy)
             5: r=(-1, -dy*2)
             6: r=(1, -dy*2)
             7: r=(2, -dy)
             */
            for( x = x1; x != x2; x += dx )
            {
                int xm = x*NR2, xd = xm*D2;

                int delta0 = minLr[0][xm - dx*NR2] + _P2, delta1 = minLr[1][xm - NR2 + 1] + _P2;
                int delta2 = minLr[1][xm + 2] + _P2, delta3 = minLr[1][xm + NR2 + 3] + _P2;

                CostType* Lr_p0 = Lr[0] + xd - dx*NRD2;
                CostType* Lr_p1 = Lr[1] + xd - NRD2 + D2;
                CostType* Lr_p2 = Lr[1] + xd + D2*2;
                CostType* Lr_p3 = Lr[1] + xd + NRD2 + D2*3;

                Lr_p0[-1] = Lr_p0[D] = Lr_p1[-1] = Lr_p1[D] =
                Lr_p2[-1] = Lr_p2[D] = Lr_p3[-1] = Lr_p3[D] = MAX_COST;

                CostType* Lr_p = Lr[0] + xd;
                const CostType* Cp = C + x*D;
                CostType* Sp = S + x*D;

                int minL0 = MAX_COST, minL1 = MAX_COST, minL2 = MAX_COST, minL3 = MAX_COST;

                for( d = 0; d < D; d++ )
                {
                    int Cpd = Cp[d], L0, L1, L2, L3;

                    L0 = Cpd + std::min((int)Lr_p0[d], std::min(Lr_p0[d-1] + _P1, std::min(Lr_p0[d+1] + _P1, delta0))) - delta0;
                    L1 = Cpd + std::min((int)Lr_p1[d], std::min(Lr_p1[d-1] + _P1, std::min(Lr_p1[d+1] + _P1, delta1))) - delta1;
                    L2 = Cpd + std::min((int)Lr_p2[d], std::min(Lr_p2[d-1] + _P1, std::min(Lr_p2[d+1] + _P1, delta2))) - delta2;
                    L3 = Cpd + std::min((int)Lr_p3[d], std::min(Lr_p3[d-1] + _P1, std::min(Lr_p3[d+1] + _P1, delta3))) - delta3;

                    Lr_p[d] = (CostType)L0;
                    minL0 = std::min(minL0, L0);

                    Lr_p[d + D2] = (CostType)L1;
                    minL1 = std::min(minL1, L1);

                    Lr_p[d + D2*2] = (CostType)L2;
                    minL2 = std::min(minL2, L2);

                    Lr_p[d + D2*3] = (CostType)L3;
                    minL3 = std::min(minL3, L3);

                    Sp[d] = cv::saturate_cast<CostType>(Sp[d] + L0 + L1 + L2 + L3);
                }
                minLr[0][xm] = (CostType)minL0;
                minLr[0][xm+1] = (CostType)minL1;
                minLr[0][xm+2] = (CostType)minL2;
                minLr[0][xm+3] = (CostType)minL3;
            }

            if( pass == npasses )
            {
                for( x = 0; x < width; x++ )
                {
                    disp1ptr[x] = disp2ptr[x] = (DispType)INVALID_DISP_SCALED;
                    disp2cost[x] = MAX_COST;
                }

                for( x = width1 - 1; x >= 0; x-- )
                {
                    CostType* Sp = S + x*D;
                    int minS = MAX_COST, bestDisp = -1;

                    if( npasses == 1 )
                    {
                        int xm = x*NR2, xd = xm*D2;

                        int minL0 = MAX_COST;
                        int delta0 = minLr[0][xm + NR2] + _P2;
                        CostType* Lr_p0 = Lr[0] + xd + NRD2;
                        Lr_p0[-1] = Lr_p0[D] = MAX_COST;
                        CostType* Lr_p = Lr[0] + xd;

                        const CostType* Cp = C + x*D;

                        for( d = 0; d < D; d++ )
                        {
                            int L0 = Cp[d] + std::min((int)Lr_p0[d], std::min(Lr_p0[d-1] + _P1, std::min(Lr_p0[d+1] + _P1, delta0))) - delta0;

                            Lr_p[d] = (CostType)L0;
                            minL0 = std::min(minL0, L0);

                            int Sval = Sp[d] = cv::saturate_cast<CostType>(Sp[d] + L0);
                            if( Sval < minS )
                            {
                                minS = Sval;
                                bestDisp = d;
                            }
                        }
                        minLr[0][xm] = (CostType)minL0;
                    }
                    else
                    {
                        for( d = 0; d < D; d++ )
                        {
                            int Sval = Sp[d];
                            if( Sval < minS )
                            {
                                minS = Sval;
                                bestDisp = d;
                            }
                        }
                    }

                    for( d = 0; d < D; d++ )
                    {
                        if( Sp[d]*(100 - _uniquenessRatio) < minS*100 && std::abs(bestDisp - d) > 1 )
                            break;
                    }
                    if( d < D )
                        continue;
                    d = bestDisp;
                    int _x2 = x + minX1 - d - minD;
                    if( disp2cost[_x2] > minS )
                    {
                        disp2cost[_x2] = (CostType)minS;
                        disp2ptr[_x2] = (DispType)(d + minD);
                    }

                    if( 0 < d && d < D-1 )
                    {
                        // do subpixel quadratic interpolation:
                        //   fit parabola into (x1=d-1, y1=Sp[d-1]), (x2=d, y2=Sp[d]), (x3=d+1, y3=Sp[d+1])
                        //   then find minimum of the parabola.
                        int denom2 = std::max(Sp[d-1] + Sp[d+1] - 2*Sp[d], 1);
                        d = d*DISP_SCALE + ((Sp[d-1] - Sp[d+1])*DISP_SCALE + denom2)/(denom2*2);
                    }
                    else
                        d *= DISP_SCALE;
                    disp1ptr[x + minX1] = (DispType)(d + minD*DISP_SCALE);
                }

                for( x = minX1; x < maxX1; x++ )
                {
                    // we round the computed disparity both towards -inf and +inf and check
                    // if either of the corresponding disparities in disp2 is consistent.
                    // This is to give the computed disparity a chance to look valid if it is.
                    int d1 = disp1ptr[x];
                    if( d1 == INVALID_DISP_SCALED )
                        continue;
                    int _d = d1 >> DISP_SHIFT;
                    int d_ = (d1 + DISP_SCALE-1) >> DISP_SHIFT;
                    int _x = x - _d, x_ = x - d_;
                    if( 0 <= _x && _x < width && disp2ptr[_x] >= minD && std::abs(disp2ptr[_x] - _d) > _disp12MaxDiff &&
                       0 <= x_ && x_ < width && disp2ptr[x_] >= minD && std::abs(disp2ptr[x_] - d_) > _disp12MaxDiff )
                        disp1ptr[x] = (DispType)INVALID_DISP_SCALED;
                }
            }

            // now shift the cyclic buffers
            std::swap( Lr[0], Lr[1] );
            std::swap( minLr[0], minLr[1] );
        }
    }
}

typedef cv::Point_<short> Point2s;

void SGBM::operator ()( InputArray _left, InputArray _right,
                             OutputArray _disp )
{
    cv::Mat left = _left.getMat(), right = _right.getMat();
    CV_Assert( left.size() == right.size() && left.type() == right.type() &&
              left.depth() == cv::DataType<PixType>::depth );

    _disp.create( left.size(), CV_16S );
    cv::Mat disp = _disp.getMat();

    computeDisparitySGBM( left, right, disp, *this, buffer );
    medianBlur(disp, disp, 3);

    if( _speckleWindowSize > 0 )
        filterSpeckles(disp, (_minDisparity - 1)*DISP_SCALE, _speckleWindowSize, DISP_SCALE*_speckleWindowSize, buffer);
}

template <typename T>
void filterSpecklesImpl(cv::Mat& img, int newVal, int maxSpeckleSize, int maxDiff, cv::Mat& _buf)
{

    int width = img.cols, height = img.rows, npixels = width*height;
    size_t bufSize = npixels*(int)(sizeof(Point2s) + sizeof(int) + sizeof(uchar));
    if( !_buf.isContinuous() || !_buf.data || _buf.cols*_buf.rows*_buf.elemSize() < bufSize )
        _buf.create(1, (int)bufSize, CV_8U);

    uchar* buf = _buf.data;
    int i, j, dstep = (int)(img.step/sizeof(T));
    int* labels = (int*)buf;
    buf += npixels*sizeof(labels[0]);
    Point2s* wbuf = (Point2s*)buf;
    buf += npixels*sizeof(wbuf[0]);
    uchar* rtype = (uchar*)buf;
    int curlabel = 0;

    // clear out label assignments
    memset(labels, 0, npixels*sizeof(labels[0]));

    for( i = 0; i < height; i++ )
    {
        T* ds = img.ptr<T>(i);
        int* ls = labels + width*i;

        for( j = 0; j < width; j++ )
        {
            if( ds[j] != newVal )   // not a bad disparity
            {
                if( ls[j] )     // has a label, check for bad label
                {
                    if( rtype[ls[j]] ) // small region, zero out disparity
                        ds[j] = (T)newVal;
                }
                // no label, assign and propagate
                else
                {
                    Point2s* ws = wbuf; // initialize wavefront
                    Point2s p((short)j, (short)i);  // current pixel
                    curlabel++; // next label
                    int count = 0;  // current region cv::Size
                    ls[j] = curlabel;

                    // wavefront propagation
                    while( ws >= wbuf ) // wavefront not empty
                    {
                        count++;
                        // put neighbors onto wavefront
                        T* dpp = &img.at<T>(p.y, p.x);
                        T dp = *dpp;
                        int* lpp = labels + width*p.y + p.x;

                        if( p.y < height-1 && !lpp[+width] && dpp[+dstep] != newVal && std::abs(dp - dpp[+dstep]) <= maxDiff )
                        {
                            lpp[+width] = curlabel;
                            *ws++ = Point2s(p.x, p.y+1);
                        }

                        if( p.y > 0 && !lpp[-width] && dpp[-dstep] != newVal && std::abs(dp - dpp[-dstep]) <= maxDiff )
                        {
                            lpp[-width] = curlabel;
                            *ws++ = Point2s(p.x, p.y-1);
                        }

                        if( p.x < width-1 && !lpp[+1] && dpp[+1] != newVal && std::abs(dp - dpp[+1]) <= maxDiff )
                        {
                            lpp[+1] = curlabel;
                            *ws++ = Point2s(p.x+1, p.y);
                        }

                        if( p.x > 0 && !lpp[-1] && dpp[-1] != newVal && std::abs(dp - dpp[-1]) <= maxDiff )
                        {
                            lpp[-1] = curlabel;
                            *ws++ = Point2s(p.x-1, p.y);
                        }

                        // pop most recent and propagate
                        // NB: could try least recent, maybe better convergence
                        p = *--ws;
                    }

                    // assign label type
                    if( count <= maxSpeckleSize )   // speckle region
                    {
                        rtype[ls[j]] = 1;   // small region label
                        ds[j] = (T)newVal;
                    }
                    else
                        rtype[ls[j]] = 0;   // large region label
                }
            }
        }
    }
}


void filterSpeckles( InputOutputArray _img, double _newval, int maxSpeckleSize,
                         double _maxDiff, InputOutputArray __buf )
{
    cv::Mat img = _img.getMat();
    cv::Mat temp, &_buf = __buf.needed() ? __buf.getMatRef() : temp;
    CV_Assert( img.type() == CV_8UC1 || img.type() == CV_16SC1 );

    int newVal = cvRound(_newval);
    int maxDiff = cvRound(_maxDiff);

    if (img.type() == CV_8UC1)
        filterSpecklesImpl<uchar>(img, newVal, maxSpeckleSize, maxDiff, _buf);
    else
        filterSpecklesImpl<short>(img, newVal, maxSpeckleSize, maxDiff, _buf);
}

