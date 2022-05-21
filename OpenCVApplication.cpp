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

using namespace cv;
using namespace std;

enum COLOR {B, G, R};
Scalar RED = Scalar(0, 0, 255);
Scalar BLUE = Scalar(255, 0, 0);



void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

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

bool isInside(Mat img, int i, int j) {
	if (i >= 0 && i < img.rows && j >= 0 && j < img.cols)
		return true;
	else return false;
}

Mat getStructuralElem() {
	Mat_<uchar> struct_elem(3, 3, 255);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			if (i == 0 && j == 0 || i == 2 && j == 0 || i == 2 && j == 2 || i == 0 && j == 2)
				struct_elem(i, j) = 255;
			else struct_elem(i, j) = 0;
	return struct_elem;
}

Mat erosion(Mat src, Mat structuralElem) {
	Mat dst = src.clone();
	int SEi = structuralElem.rows / 2;
	int SEj = structuralElem.cols / 2;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0) {
				dst.at<uchar>(i, j) = 0;
				for (int ii = 0; ii < structuralElem.rows; ii++) {
					for (int jj = 0; jj < structuralElem.cols; jj++) {
						if (structuralElem.at<uchar>(ii, jj) == 0) {
							if (isInside(src, i + ii - SEi, j + jj - SEj)) {
								if (src.at<uchar>(i + ii - SEi, j + jj - SEj) == 255)
									dst.at<uchar>(i, j) = 255;
							}
						}
					}
				}
			}
		}
	}
	return dst;
}

Mat dilation(Mat src, Mat structuralElem) {
	Mat dst = src.clone();
	int SEi = structuralElem.rows / 2;
	int SEj = structuralElem.cols / 2;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0) {
				dst.at<uchar>(i, j) = 0;
				for (int ii = 0; ii < structuralElem.rows; ii++) {
					for (int jj = 0; jj < structuralElem.cols; jj++) {
						if (structuralElem.at<uchar>(ii, jj) == 0) {
							if (isInside(src, i + ii - SEi, j + jj - SEj)) {
								dst.at<uchar>(i + ii - SEi, j + jj - SEj) = 0;
							}
						}
					}
				}
			}
		}
	}
	return dst;
}

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
			/*if ((dstH.at<uchar>(i, j) >= 244 && dstH.at<uchar>(i, j) <= 250) || 
				(dstH.at<uchar>(i, j) >= 0 && dstH.at<uchar>(i, j) <= 11) ||
				(dstH.at<uchar>(i, j) >= 149 && dstH.at<uchar>(i, j) <= 181))*/

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

pair<vector<Rect>, vector<vector<Point>>> nonMaximumSuppression(vector<Rect> boxes, vector<vector<Point>> regions, float overlap_threshold)
{
	vector<float> areas;
	pair<vector<Rect>, vector<vector<Point>>> pick;          //indices of final detection boxes

	for (Rect box: boxes)
		areas.push_back(box.area());

	vector<size_t> idxs(areas.size());
	iota(idxs.begin(), idxs.end(), 0);
	stable_sort(idxs.begin(), idxs.end(), [&areas](size_t i1, size_t i2) {return areas[i1] < areas[i2]; });

	while (idxs.size() > 0)         
	{
		int last = idxs.size() - 1;
		int i = idxs[last];
		pick.first.push_back(boxes[i]);
		pick.second.push_back(regions[i]);

		vector<int> suppress;
		suppress.push_back(last);

		for (int pos = 0; pos < last; pos++)     
		{
			int j = idxs[pos];

			int xx1 = max(boxes[i].x, boxes[j].x);         
			int yy1 = max(boxes[i].y, boxes[j].y);         
			int xx2 = min(boxes[i].br().x, boxes[j].br().x);    
			int yy2 = min(boxes[i].br().y, boxes[j].br().y);    

			int w = max(0, xx2 - xx1 + 1);     
			int h = max(0, yy2 - yy1 + 1);   

			float overlap = float(w * h) / areas[j];

			if (overlap > overlap_threshold)       
				suppress.push_back(pos);
		}

		for (int p: suppress) {
			idxs[p] = -1;
		}

		for (int p = 0; p < idxs.size();)
		{
			if (idxs[p] == -1)
				idxs.erase(idxs.begin() + p);
			else
				p++;
		}
	}
	return pick;
}

vector<vector<Point>> mser(Mat img, COLOR c) {
	Ptr<MSER> ms = MSER::create(5, 200, 999000, 0.4, 0.4, 50, 1.0091);
	vector<vector<Point>> regions;
	vector<Rect> mser_bbox;
	ms->detectRegions(img, regions, mser_bbox);

	pair<vector<Rect>, vector<vector<Point>>> supressed = nonMaximumSuppression(mser_bbox, regions, 0.1);

	for (Rect box: supressed.first)
		rectangle(img, box, CV_RGB(255, 255, 255));

	imshow("mser"+c, img);

	return supressed.second;
}

vector<Rect> nonMaximumSuppression(vector<Rect> boxes, float overlap_threshold)
{
	vector<float> areas;
	vector<Rect> pick;          //indices of final detection boxes

	if (boxes.size() == 0)
		return pick;

	for (Rect box : boxes)
		areas.push_back(box.area());

	vector<size_t> idxs(boxes.size());
	iota(idxs.begin(), idxs.end(), 0);
	stable_sort(idxs.begin(), idxs.end(), [&boxes](size_t i1, size_t i2) {return boxes[i1].area() < boxes[i2].area(); });

	while (idxs.size() > 0)
	{
		int last = idxs.size() - 1;
		int i = idxs[last];
		pick.push_back(boxes[i]);

		vector<int> suppress;
		suppress.push_back(last);

		for (int pos = 0; pos < last; pos++)
		{
			int j = idxs[pos];

			int xx1 = max(boxes[i].x, boxes[j].x);
			int yy1 = max(boxes[i].y, boxes[j].y);
			int xx2 = min(boxes[i].br().x, boxes[j].br().x);
			int yy2 = min(boxes[i].br().y, boxes[j].br().y);

			int w = max(0, xx2 - xx1 + 1);
			int h = max(0, yy2 - yy1 + 1);

			float overlap = float(w * h) / areas[j];

			if (overlap > overlap_threshold)
				suppress.push_back(pos);
		}

		for (int p : suppress) {
			idxs[p] = -1;
		}

		for (int p = 0; p < idxs.size();)
		{
			if (idxs[p] == -1)
				idxs.erase(idxs.begin() + p);
			else
				p++;
		}
	}
	return pick;
}

bool sortBySecondElement(const pair<int, int>& a, const pair<int, int>& b) {
	return (a.second > b.second);
}

Mat mser(Mat src, Mat canny) {
	Mat img = src.clone();

	Ptr<MSER> ms = MSER::create(8, 200, 999000, 0.4, 0.4, 50, 1.0091);
	vector<vector<Point>> regions;
	vector<Rect> mser_bbox;
	ms->detectRegions(img, regions, mser_bbox);

	vector<Rect> boxes = nonMaximumSuppression(mser_bbox, 0.1);
	vector<Rect> boxesChecked;
	vector<Rect> boxesCanny;
	vector<pair<int, int>> indexedSizes;

	for (int i = 0; i < boxes.size(); i++)
		indexedSizes.push_back(make_pair(i, boxes[i].height * boxes[i].width));

	sort(indexedSizes.begin(), indexedSizes.end(), sortBySecondElement);

	for (int i = 0; i < (int) boxes.size(); i++) {
		float ratio2 = boxes[indexedSizes[i].first].height / (float)boxes[indexedSizes[i].first].width;
		float ratio1 = boxes[indexedSizes[i].first].width / (float)boxes[indexedSizes[i].first].height;

		if (ratio1 <= 0.5 || ratio1 > 1.2)
			continue;
		if (ratio2 <= 0.5)
			continue;

		//rectangle(img, boxes[indexedSizes[i].first], CV_RGB(255, 255, 255));
		boxesChecked.push_back(boxes[indexedSizes[i].first]);
	}

	for(int i = 0; i < canny.rows; i++) 
		for (int j = 0; j < canny.cols; j++) {
			if (canny.at<uchar>(i, j) == 255)
				for (auto box = boxesChecked.begin(); box != boxesChecked.end(); box++) {
				if ((*box).contains(Point2i(i, j))) {
					boxesCanny.push_back(*box);
					boxesChecked.erase(box--);
				}
			}
		}

	/*int stop = 0;
	for (int i = 0; i < (int)boxesChecked.size(); i++) {
		if (stop == 4) break;
		rectangle(img, boxesChecked[indexedSizes[i].first], CV_RGB(255, 255, 255));
		stop++;
	}*/

	for (Rect box : boxesCanny) {
		rectangle(img, box, CV_RGB(255, 255, 255));
	}
	return img;
}

Mat convexHulls(vector<vector<Point>> regions, int rows, int cols, Scalar color) {
	vector<vector<Point>> hulls;
	for (vector<Point> r : regions) {
		vector<Point> hull;
		convexHull(r, hull);
		hulls.push_back(hull);
	}
	Mat img(rows, cols, CV_8UC3, Scalar(0));
	fillPoly(img, hulls, color);
	return img;
}

#define P 5
int SE[2][P] = { {0, 0, 0, -1, 1},
							  {0, -1, 1, 0, 0} };

void apply(Mat& dest, int i, int j, int val, Mat src) {
	int cnt = 0;

	for (int k = 0; k < P; k++) {
		int x = i + SE[0][k], y = j + SE[1][k];

		if (x < 0)
			x = 0;
		if (y < 0)
			y = 0;
		if (y >= src.rows)
			y = src.rows - 1;
		if (x >= src.cols)
			x = src.cols - 1;

		if (src.at<uchar>(x, y) == 0)
			cnt++;

		if (val == 0)
			dest.at<uchar>(x, y) = val;
	}

	if (cnt == P && val == 255)
		dest.at<uchar>(i, j) = 0;
	else if (val == 255) {
		dest.at<uchar>(i, j) = 255;
	}
}

Mat dilation(Mat src) {
	Mat dest = src.clone();

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0) {
				apply(dest, i, j, 0, src);
			}
		}

	return dest;
}

Mat erosion(Mat src) {
	Mat dest = src.clone();

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0) {
				apply(dest, i, j, 255, src);
			}
		}

	return dest;
}

Mat clearRegions(Mat src) {
	Mat Rc, Gc, Bc, dst;

	extractChannel(src, Rc, R);
	extractChannel(src, Gc, G);
	extractChannel(src, Bc, B);

	merge(vector<Mat>{ dilation(erosion(Bc, Mat(3, 3, CV_8UC1, Scalar(1)))), Gc, dilation(erosion(Rc, Mat(3, 3, CV_8UC1, Scalar(1)))) }, dst);
	return dst;
}

Mat clearOpening(Mat dst) {
	return dilation(erosion(dst, getStructuralElem()), getStructuralElem());
}

void detectSigns(Mat src, Mat blue, Mat red) {
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if (blue.at<Vec3b>(i, j) == Vec3b(255, 255, 255) || red.at<Vec3b>(i, j) == Vec3b(255, 255, 255))
				src.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
	imshow("final", src);
}

int main(){
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src;
		src = imread(fname);
		imshow("original", src);

		int h = src.rows, w = src.cols;

		//pair<Mat, Mat> dst = augment(convertToHSV(histogramEqualization(src)));
		
		
		pair<Mat, Mat> dst = augment(convertToHSV(histogramEqualization(src)));

		imshow("augmetn", dst.first);

		//imshow("huls R", clearRegions(convexHulls(mser(dst.first, R), h, w, RED)));
		//imshow("huls B", clearRegions(convexHulls(mser(dst.second, B), h , w, BLUE)));

		Mat dstR = clearOpening(mser(dst.first, canny(dst.first, R)));
		Mat dstB = clearOpening(mser(dst.second, canny(dst.second, B)));

		mser(dstR, R);
		detectSigns(src, dstR, dstB);

		imshow("cannyB", canny(dst.second, B));
		imshow("cannyR", canny(dst.first, R));
		waitKey();
	}
	return 0;
}