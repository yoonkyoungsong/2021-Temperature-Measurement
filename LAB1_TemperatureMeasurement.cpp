/*------------------------------------------------------
				2021. 04. 02. Fri
	   Image Proccessing with Deep Learning
		  LAP #1: Tempurature Measurement
			 21600372  Yoonkyoung Song
------------------------------------------------------*/

#include "myOpenCV.h"

using namespace std;
using namespace cv;

int hmin = 0, hmax = 50, smin = 180, smax = 255, vmin = 80, vmax = 215;

int main()
{
	Mat image, image_disp, hsv, hue, mask, dst;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;


	VideoCapture cap("IR_DEMO_cut.avi"); //비디오 읽어오기

	bool bSuccess = cap.read(image); // 카메라를 read하는 함수

	if (!bSuccess)
	{
		cout << "Cannot find a frame from video stream/n";
	}

	/// 영상 저장
	//double fps = 30; // 영상 프레임
	//int  fourcc = VideoWriter::fourcc('x', 'v', 'i', 'd'); // 코덱 설정

	//VideoWriter outputVideo;
	////save.open(입력 영상, 코덱, 프레임, 영상 크기, 컬러)
	//outputVideo.open("LAB1_DEMO.avi", fourcc, fps, image.size(), 1);

	for (;;)
	{
		cap.read(image);
		image.copyTo(image_disp);

		if (image.empty()) {
			break;
		}

		/******** You can use RGB instead of HSV ********/
		cvtColor(image, hsv, COLOR_BGR2HSV);
		
		vector<Mat> channels;
		split(hsv, channels);

		Mat value(image.size(), CV_8UC1);
		value = channels[2];

		/******** Add Pre-Processing such as filtering etc  ********/
		imFilter(&hsv, &hsv, 5, GAUSSIAN);

		/******** set dst as the output of InRange ********/
		inRange(hsv, Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)), Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)), dst);

		/******** Add Post-Processing such as morphology etc  ********/
		imMorphology(&dst, &dst, MORPH_RECT, 5, OPENING);

		/********  Find All Contour   ********/
		findContours(dst, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		if (contours.size() > 0) {

			/// Find the Contour with the largest area ///
			int idx = 0, largestComp = 0;
			double maxArea = 0;
			for (; idx >= 0; idx = hierarchy[idx][0])
			{
				const vector<Point>& c = contours[idx];
				double area = fabs(contourArea(Mat(c)));
				if (area > maxArea)
				{
					maxArea = area;
					largestComp = idx; //제일 큰 영역의 컨투어 인덱스
				}
			}

			Rect boxPoint = boundingRect(contours[largestComp]);

			if (boxPoint.x > 70 && boxPoint.x < 270) {
				if (maxArea > 3000) {

					mask = Mat::zeros(image.size(), CV_8UC1);
					drawContours(mask, contours, largestComp, Scalar(255), CV_FILLED);
					bitwise_and(value, mask, value);

					int size_1D = boxPoint.width * boxPoint.height;
					Mat array_1D(1, size_1D, CV_8UC1);

					for (int y = 0; y < boxPoint.height; y++) {
						for (int x = 0; x < boxPoint.width; x++) {
							array_1D.at<uchar>(0, y * boxPoint.width + x) = value.at<uchar>(y + boxPoint.y, x + boxPoint.x);
						}
					}

					cv::sort(array_1D, array_1D, SORT_DESCENDING);

					double avg_max = 0;
					double avg = 0;


					for (int y = size_1D / 100; y < 2 * size_1D / 100; y++) {
						if (y < size_1D / 100 + 5) avg_max += array_1D.at<uchar>(0, y);
						avg += array_1D.at<uchar>(0, y);
					}

					avg_max = avg_max / 5;
					avg = avg / (size_1D / 100);

					avg_max = 4.943047496238989e-05 * pow(avg_max,2) + 0.052213625617881 * avg_max + 25;
					avg = 4.943047496238989e-05 * pow(avg, 2) + 0.052213625617881 * avg + 25;

					///  Draw the max Contour on Black-background  Image ////
					drawContours(image_disp, contours, largestComp, Scalar(0, 0, 255), 2, 8, hierarchy); //BGR순서대로 색상

					/// Draw the Contour Box on Original Image ///
					rectangle(image_disp, boxPoint, Scalar(255, 0, 255), 3);


					char savg[20];
					char savg_max[20];
					char Massage_max[50] = "max = ";
					char Massage_avg[50] = "  avg = ";
					sprintf(savg, "%.2lf", avg);
					strcat(Massage_avg, savg);
					sprintf(savg_max, "%.2lf", avg_max);
					strcat(Massage_max, savg_max);
					strcat(Massage_max, Massage_avg);

					if (avg_max > 38) {
						putText(image_disp, Massage_max, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
						putText(image_disp, "WARNING", Point(130, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
					}
					else {
						putText(image_disp, Massage_max, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
					}

				}
			}
		}

		/// 영상 저장
		//outputVideo << image_disp;
		imshow("tmeprature Measurement", image_disp);

		char c = (char)waitKey(10);
		if (c == 27)
			break;

	}

	return 0;
}