#ifndef HEADER_USER_DEFINED_MYOPENCV
#define HEADER_USER_DEFINED_MYOPENCV

#define _CRT_SECURE_NO_WARNINGS

#include <opencv2/opencv.hpp>
#include <iostream>


enum ImreadType {
	GRAY,
	COLOR
};
enum MophologyType {
	DILATE,
	ERODE,
	OPENING,
	CLOSING,
	OPTOCL,
	CLTOOP
};

enum FilterType {
	FILTER2D,
	BLUR,
	GAUSSIAN,
	MEDIAN,
	BILATERAL,
	LAPLACIAN
};


/* [scale]	
0: GRAY
1: COLOR
*/
void imRead(cv::Mat* src, const char filename[], int scale);

void imWrite(cv::Mat* src, const char filename[]);

void imNormalHist1D(cv::Mat* src, cv::Mat* dst, int histSize);

/* [threshold_type]
0: Binary
1: Binary Inverted
2: Threshold Truncated
3: Threshold to Zero
4: Threshold to Zero Inverted
8: THRESH_OTSU
16: THRESH_TRIANGLE
*/
void imThreshold(cv::Mat* src, cv::Mat* dst, double threshold_value, double ifBinaryMax, int threshold_type);

/* [threshold_type]
0: THRESH_BINARY
1: THRESH_BINARY_INV
*/
void imAdpthreshold(cv::Mat* src, cv::Mat* dst, double BinaryMax, int threshold_type, int size, double constant);


/* [ element shape ]
0: MORPH_RECT
1: MORPH_CROSS
2: MORPH_ELLIPSE
�ѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤѤ�
[ Mopology type ]
0: DILATE
1: ERODE
2: OPENING (erode -> dilate)
3: CLOSING (dilate -> erode)
4: OPTOCL (opening -> closing)
5: CLTOOP (clsoing -> opening)
*/
void imMorphology(cv::Mat* src, cv::Mat* dst, int elementshape, int elementsize, int type);


/* [ filter type ]
0: FILTER2D
1: BLUR
2: GAUSSIAN
3: MEDIAN
4: BILATERAL
5: LAPLACIAN (SHARPENING)
*/
void imFilter(cv::Mat* src, cv::Mat* dst, int size, int type);


/* [ filter type ]
0: SQDIFF
1: SQDIFF_NORMED
2: TM_CCORR
3: TM_CCORR_NORMED
4: TM_COEFF
5: TM_COEFF_NORMED
*/
void imTemplateMaching(cv::Mat img, cv::Mat templ, cv::Mat* result, cv::Mat* img_display, int match_method);

#endif
#pragma once
