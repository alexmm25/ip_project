#include "stdafx.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

#include "hog.h"
#include "canny.h"

Point advance(Point p, int dir, int rows, int cols) {
	if (dir != -1) {
		int diry[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
		int dirx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

		p.x += dirx[dir];
		p.y += diry[dir];
	}

	if (p.x < 0)
		p.x = 0;
	if (p.y < 0)
		p.y = 0;
	if (p.y >= rows)
		p.y = rows - 1;
	if (p.x >= cols)
		p.x = cols - 1;

	return p;
}

Mat borderTracing(Mat src) {
	Mat contour(src.rows, src.cols, CV_8UC1, Scalar(0));
	Point p0;

	imshow("before border", src);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 255) {
				p0.x = j;
				p0.y = i;
				i = src.rows + 1;
				break;
			}
		}

	int dir = 7, n = 0;
	Point pn(p0), p1, pn_1;
	bool stored = false;

	do {
		n++;
		if (dir % 2 == 0)
			dir = (dir + 7) % 8;
		else dir = (dir + 6) % 8;

		pn_1 = pn;
		Point adv = advance(pn, dir, src.rows, src.cols);
		int i = 0;
		while (src.at<uchar>(adv) == 0 && i < 8) {
			dir = (dir + 1) % 8;
			adv = advance(pn, dir, src.rows, src.cols);
			i++;
		}
		pn = adv;
		if (!stored) {
			p1 = pn;
			stored = true;
		}
		contour.at<uchar>(pn) = 255;
	} while (!((pn == p1) && (pn_1 == p0) && (n >= 2)) && n < 1000);

	return contour;
}

void showHistogram(const string& name, int* hist, const int  hist_cols, const int hist_height)
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

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

bool hog(Mat src) {
	vector<int> hist(180);
	imshow("border", src);
	pair<Mat, Mat> G = gradient(src);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 255) {
				int val = (int)(G.second.at<float>(i, j) * (180.0 / 3.141592653589793238463));
				if (val < 0) val += 360;
				if (val == 0) continue;
				hist[val / 2]++;
			}
		}
	showHistogram("hist", hist.data(), 180, 100);
	return true;
}

vector<Rect> checkHogs(Mat src, vector<Rect> boxes) {
	vector<Rect> finalBoxes;

	for (Rect box : boxes) {
		Mat cropped(src, box);
		if (hog(borderTracing(cropped)))
			finalBoxes.push_back(box);
	}

	return finalBoxes;
}