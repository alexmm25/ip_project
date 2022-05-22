#include <vector>
#include <numeric>

#include "stdafx.h"
#include "common.h"

#include <fstream>
#include <sstream>

using namespace std;

#include "hsv.h"

Mat convertToHue(Mat src) {
	int height = src.rows;
	int width = src.cols;
	Mat dstH = Mat(height, width, CV_8UC1);
	Mat dstS = Mat(height, width, CV_8UC1);
	Mat dstV = Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b vals = src.at<Vec3b>(i, j);
			float H, S, V, H_norm, S_norm, V_norm;
			float M, m, C;
			float b = (float)vals[0] / 255;
			float g = (float)vals[1] / 255;
			float r = (float)vals[2] / 255;
			M = max(r, max(g, b));
			m = min(r, min(g, b));
			C = (M - m);

			V = M;
			if (V)
				S = C / V;
			else S = 0;

			if (C) {
				if (M == r) H = 60 * (g - b) / C;
				if (M == g) H = 120 + 60 * (b - r) / C;
				if (M == b) H = 240 + 60 * (r - g) / C;
			}
			else H = 0;
			if (H < 0) H += 360;
			S_norm = S * 255;
			V_norm = V * 255;
			H_norm = H * 255 / 360;
			dstH.at<uchar>(i, j) = H_norm;
			dstS.at<uchar>(i, j) = S_norm;
			dstV.at<uchar>(i, j) = V_norm;
		}
	}

	Mat dst = src.clone();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
				// below: values from medium.com
			if (((dstH.at<uchar>(i, j) >= 0 && dstS.at<uchar>(i, j) >= 70 && dstV.at<uchar>(i, j) >= 60) &&
				(dstH.at<uchar>(i, j) <= 25 && dstS.at<uchar>(i, j) <= 255 && dstV.at<uchar>(i, j) <= 255))
				||
				((dstH.at<uchar>(i, j) >= 340 && dstS.at<uchar>(i, j) >= 70 && dstV.at<uchar>(i, j) >= 60) &&
					(dstH.at<uchar>(i, j) <= 360 && dstS.at<uchar>(i, j) <= 255 && dstV.at<uchar>(i, j) <= 255))
				||
				((dstH.at<uchar>(i, j) >= 200 && dstS.at<uchar>(i, j) >= 127 && dstV.at<uchar>(i, j) >= 0) &&
					(dstH.at<uchar>(i, j) <= 280 && dstS.at<uchar>(i, j) <= 255 && dstV.at<uchar>(i, j) <= 255)))
				continue;

			else dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
		}
	}

	return dst;
}

pair<Mat, Mat> convertToHSV(Mat src) {
	int height = src.rows;
	int width = src.cols;

	Mat hsv = Mat(height, width, CV_8UC3);
	cvtColor(src, hsv, COLOR_BGR2HSV);

	Mat maskRL(height, width, CV_8UC1);
	Mat maskRU(height, width, CV_8UC1);
	Mat maskR(height, width, CV_8UC1);
	Mat maskB(height, width, CV_8UC1);
	Mat mask(height, width, CV_8UC1);

	Mat dstR(height, width, CV_8UC3);
	Mat dstB(height, width, CV_8UC3);

	inRange(hsv, Scalar(0, 70, 60), Scalar(15, 255, 255), maskRL);
	inRange(hsv, Scalar(160, 70, 60), Scalar(180, 255, 255), maskRU);
	bitwise_or(maskRL, maskRU, maskR);

	bitwise_and(src, src, dstR, maskR);

	inRange(hsv, Scalar(100, 127, 0), Scalar(140, 255, 255), maskB);
	//bitwise_or(maskR, maskB, mask);

	bitwise_and(src, src, dstB, maskB);

	return make_pair(dstR, dstB);
}
