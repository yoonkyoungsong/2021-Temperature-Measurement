#include "myOpenCV.h"

/* [scale]
0: GRAY
1: COLOR
*/
void imRead(cv::Mat* src, const char filename[], int scale) {
	
	cv::String imname = filename;

	*src = imread(imname, scale);

	if ((*src).empty())	
	{
		std::cout << "File Read Failed : src is empty" << std::endl;
		cv::waitKey(0);
	}

}


void imWrite(cv::Mat* src, const char filename[]) {

	cv::String imname = filename;

	imwrite(imname, *src);

	if ((*src).empty())
	{
		std::cout << "File Write Failed : src is empty" << std::endl;
		cv::waitKey(0);
	}


}



void imNormalHist1D(cv::Mat* src, cv::Mat* dst, int histSize) {

	// Set the ranges for B,G,R
	float range[] = { 0, histSize }; //the upper boundary is exclusive
	const float* histRange = { range };

	// Set histogram param 
	bool uniform = true, accumulate = false;

	// Compute the histograms
	cv::Mat hist1D;
	cv::calcHist(src, 1, 0, cv::Mat(), hist1D, 1, &histSize, &histRange, uniform, accumulate);


	// Draw the histograms for B, G and R
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);


	cv::Mat histImage1D(hist_h, hist_w, CV_8UC1, cv::Scalar(255));
	normalize(hist1D, hist1D, 0, histImage1D.rows, cv::NORM_MINMAX, -1, cv::Mat());

	*dst = histImage1D;

	// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage1D, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist1D.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(hist1D.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0);
	}

}

/* [threshold_type]
0: THRESH_BINARY
1: THRESH_BINARY_INV 
2: THRESH_TRUNC 
3: THRESH_TOZERO 
4: THRESH_TOZERO_INV
7: THRESH_MASK
8: THRESH_OTSU
16: THRESH_TRIANGLE
*/
void imThreshold(cv::Mat* src, cv::Mat* dst, double threshold_value, double ifBinaryMax, int threshold_type) {
	threshold(*src, *dst, threshold_value, ifBinaryMax, threshold_type);
}

/* [threshold_type]
0: THRESH_BINARY
1: THRESH_BINARY_INV
*/
void imAdpthreshold(cv::Mat* src, cv::Mat* dst, double BinaryMax, int threshold_value, int size, double constant) {
	adaptiveThreshold(*src, *dst, BinaryMax, cv::BORDER_REPLICATE, threshold_value, size, constant);
}

/* [ element shape ]
0: MORPH_RECT
1: MORPH_CROSS
2: MORPH_ELLIPSE
天天天天天天天天天天天天天天天天天天
[ Mopology type ]
0: DILATE
1: ERODE
2: OPENING (erode -> dilate)
3: CLOSING (dilate -> erode)
4: OPTOCL (opening -> closing)
5: CLTOOP (clsoing -> opening)
*/
void imMorphology(cv::Mat* src, cv::Mat* dst, int elementshape, int elementsize, int type){
	
	cv::Mat element = cv::getStructuringElement(elementshape, cv::Size(elementsize, elementsize));

	switch (type)
	{
	case DILATE:
		dilate(*src, *dst, element);
		break;

	case ERODE:
		erode(*src, *dst, element);
		break;

	case OPENING:
		erode(*src, *dst, element);
		dilate(*dst, *dst, element);
		break;

	case CLOSING:
		dilate(*src, *dst, element);
		erode(*dst, *dst, element);
		break;

	case OPTOCL:
		erode(*src, *dst, element);
		dilate(*dst, *dst, element);
		dilate(*dst, *dst, element);
		erode(*dst, *dst, element);
		break;

	case CLTOOP:
		dilate(*src, *dst, element);
		erode(*dst, *dst, element);
		erode(*dst, *dst, element);
		dilate(*dst, *dst, element);
		break;

	}

}



/* [ filter type ]
0: FILTER2D
1: BLUR
2: GAUSSIAN
3: MEDIAN
4: BILATERAL
5: LAPLACIAN (SHARPENING)
*/
void imFilter(cv::Mat* src, cv::Mat* dst, int size, int type) {

	cv::Size kernelSize = cv::Size(size, size);
	int delta = 0;
	int ddepth;

	switch (type) {
	
	case FILTER2D:
		ddepth = -1;
	
		(*src).convertTo(*src, CV_8UC1);

		filter2D(*src, *dst, ddepth, size);
		break;
	
	case BLUR:
		cv::blur(*src, *dst, cv::Size(size, size), cv::Point(-1, -1));
		break;
	
	case GAUSSIAN:
		cv::GaussianBlur(*src, *dst, cv::Size(size, size), 0, 0);
		break;
	
	case MEDIAN:
		cv::medianBlur(*src, *dst, size);
		break;

	case BILATERAL:
		cv::bilateralFilter(*src, *dst, size, size * 2, size / 2);
		break;	
	
	case LAPLACIAN:
		int scale = 1;
		ddepth = CV_16S;

		cv::Mat Laplacian = *dst;

		cv::Laplacian(*src, Laplacian, ddepth, size, scale, delta, cv::BORDER_DEFAULT);
		(*src).convertTo(*src, CV_16S);
		
		*dst = *src - Laplacian;
		(*dst).convertTo(*dst, CV_8U);

		break;	
	}




}


/* [ filter type ]
0: SQDIFF
1: SQDIFF NORMED
2: TM CCORR
3: TM CCORR NORMED
4: TM COEFF
5: TM COEFF NORMED
*/
void imTemplateMaching(cv::Mat img, cv::Mat templ , cv::Mat* result, cv::Mat *img_display, int match_method) {

	img.copyTo(*img_display);

	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	(*result).create(result_rows, result_cols, CV_32FC1);

	cv::matchTemplate(img, templ, *result, match_method);
	cv::normalize(*result, *result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

	double minVal, maxVal;
	cv::Point minLoc, maxLoc, matchLoc;
	cv::minMaxLoc(*result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED) {
		matchLoc = minLoc;
	}
	else {
		matchLoc = maxLoc;
	}

	cv::rectangle(*img_display, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);
	cv::rectangle(*result, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);


}