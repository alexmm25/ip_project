#include <vector>
#include <numeric>

#include "stdafx.h"
#include "common.h"

#include <fstream>
#include <sstream>

using namespace std;

#include "histogram.h"

vector<int> h(Mat src, COLOR c) {
	vector<int> h(256, 0);

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			h[src.at<Vec3b>(i, j)[c]]++;
		}

	return h;
}

vector<float> p(Mat src, COLOR c) {
	vector<float> p(256);
	vector<int> h1 = h(src, c);
	float M = src.rows * src.cols;

	for (int i = 0; i < 256; i++)
		p[i] = h1[i] / M;

	return p;
}

Mat histogramEqualization_gray(Mat src, COLOR c) {
	Mat dest(src.rows, src.cols, CV_8UC1, Scalar(255));

	vector<float> pd = p(src, c);
	vector<float> pc;
	pc.push_back(0.);

	for (int i = 1; i < pd.size(); i++) {
		pc.push_back(pc[i - 1] + pd[i]);
	}

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			int nw = 255 * pc[src.at<Vec3b>(i, j)[c]];
			if (nw > 255)
				dest.at<uchar>(i, j) = 255;
			else if (nw < 0)
				dest.at<uchar>(i, j) = 0;
			else dest.at<uchar>(i, j) = nw;
		}
	return dest;
}

Mat histogramEqualization(Mat src) {
	Mat destB = histogramEqualization_gray(src, B);
	Mat destG = histogramEqualization_gray(src, G);
	Mat destR = histogramEqualization_gray(src, R);

	Mat dest(src.rows, src.cols, CV_8UC3);

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			dest.at<Vec3b>(i, j)[B] = destB.at<uchar>(i, j);
			dest.at<Vec3b>(i, j)[G] = destG.at<uchar>(i, j);
			dest.at<Vec3b>(i, j)[R] = destR.at<uchar>(i, j);
		}
	return dest;
}