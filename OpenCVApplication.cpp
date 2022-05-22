// OpenCVApplication.cpp : Defines the entry point for the console application.
//

// https://drive.google.com/drive/folders/1fR3JeLpuCHtY7MtybVr9QGzB8bzxrZ3V <- python project

//HOG sau border tracing dupa mser si canny

#include "stdafx.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>

#include "canny.h"
#include "histogram.h"
#include "hsv.h"
#include "mser.h"
#include "opening.h"
#include "hog.h"

using namespace cv;
using namespace std;


Scalar RED = Scalar(0, 0, 255);
Scalar BLUE = Scalar(255, 0, 0);


pair<Mat, Mat> augment(pair<Mat, Mat> src) {
	Mat Bb, Bg, Br, Rb, Rg, Rr, B, R;

	extractChannel(src.second, Bb, 0);
	extractChannel(src.second, Bg, 1);
	extractChannel(src.second, Br, 2);

	extractChannel(src.first, Rb, 0);
	extractChannel(src.first, Rg, 1);
	extractChannel(src.first, Rr, 2);

	Bb = -0.5 * Br + 3 * Bb - 2 * Bg;
	Rr = 2 * Rr - 0.5 * Rb - 2 * Rg;

	merge(vector<Mat>({ Bb, Bg, Br }), B);
	merge(vector<Mat>({ Rb, Rg, Rr }), R);

	return make_pair(R, B);
}

void detectSigns(Mat src, vector<Rect> blue, vector<Rect> red) {
	for (Rect box : blue) {
		rectangle(src, box, CV_RGB(255, 255, 255), 2);
	}
	for (Rect box : red) {
		rectangle(src, box, CV_RGB(255, 255, 255), 2);
	}
	imshow("final", src);
}

int main(){
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src;
		src = imread(fname);
		imshow("original", src);
		int h = src.rows, w = src.cols;
		
		pair<Mat, Mat> dst = clearOpening(augment(convertToHSV(histogramEqualization(src))));
		Mat dstR, dstB;
		extractChannel(dst.first, dstR, R);
		extractChannel(dst.second, dstB, B);

		Mat cannyR = canny(dstR);
		Mat cannyB = canny(dstB);
		detectSigns(src, 
			checkHogs(dilation(cannyR), mser(dstR)),
			checkHogs(dilation(cannyB), mser(dstB)));
		waitKey();
	}
	return 0;
}