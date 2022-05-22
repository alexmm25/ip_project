#include "stdafx.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

#include "hog.h"
#include "canny.h"

float average(vector<int> v, int s, int e, int WH) {
	float sum = 0;

	for (int i = s; i <= e; i++) {
		sum += v[i];
	}
	return sum / (2 * WH + 1);
}

vector<int> hMax(vector<int> hist) {
	int WH = 7;
	float TH = 0.0003;

	vector<int> hmax;
	for (int k = WH; k <= 180 - WH; k++) {
		float v = average(hist, k - WH, k + WH, WH);
		bool greater = true;

		for (int i = k - WH; i <= k + WH; i++) {
			if (hist[k] < hist[i]) {
				greater = false;
				break;
			}
		}
		if (greater && hist[k] > (v + TH))
			hmax.push_back(k);
	}
	return hmax;
}

int showHistogram(const string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;
	int edges = 0;
	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		if (cvRound(hist[x] * scale) > 15)
			edges++;
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
	return edges;
}

Mat contours(Mat src) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	Mat img(src.rows, src.cols, CV_8UC1, Scalar(0));
	drawContours(img, contours, -1, Scalar(255), 1);
	return img;
}

bool hog(Mat src) {
	vector<int> hist(180, 0);
	pair<Mat, Mat> G = gradient(src);

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 255) {
				int val = (int)(G.second.at<float>(i, j) * (180.0 / 3.141592653589793238463));
				if (val < 0) val += 360;
				if (val == 0) continue;
				hist[val / 2] += G.first.at<float>(i, j);
			}
		}

	//showHistogram("hist", hist.data(), 180, 100);
	vector<int> histmax = hMax(hist); 
	if (histmax.size() >= 3 && histmax.size() <= 11 || histmax.size() > 160)
		return true;
	else return false;
}

vector<Rect> checkHogs(Mat src, vector<Rect> boxes) {
	vector<Rect> finalBoxes;
	Mat img = contours(src);

	for (Rect box : boxes)
		if (hog(Mat(img, box)))
			finalBoxes.push_back(box);
	return finalBoxes;
}