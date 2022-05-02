// OpenCVApplication.cpp : Defines the entry point for the console application.
//

// https://drive.google.com/drive/folders/1fR3JeLpuCHtY7MtybVr9QGzB8bzxrZ3V <- python project

#include "stdafx.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

enum COLOR {B, G, R};

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

	Mat hsv = Mat(height, width, CV_8UC3);
	
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
			//hsv.at<Vec3b>(i, j)[0] = H_norm;
			//hsv.at<Vec3b>(i, j)[1] = S_norm;
			//hsv.at<Vec3b>(i, j)[2] = V_norm;

		}
	}

	cvtColor(src, hsv, COLOR_BGR2HSV);
	Mat maskRL(height, width, CV_8UC1);// = src.clone();
	Mat maskRU(height, width, CV_8UC1);
	Mat maskR(height, width, CV_8UC1);
	Mat maskB(height, width, CV_8UC1);
	Mat mask(height, width, CV_8UC1);
	Mat dst0(height, width, CV_8UC3);
	inRange(hsv, Scalar(0, 70, 60), Scalar(15, 255, 255), maskRL);
	inRange(hsv, Scalar(160, 70, 60), Scalar(180, 255, 255), maskRU);
	bitwise_or(maskRL, maskRU, maskR);

	inRange(hsv, Scalar(100, 127, 0), Scalar(140, 255, 255), maskB);
	bitwise_or(maskR, maskB, mask);

	bitwise_and(src, src, dst0, mask);

	Mat dst = src.clone();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if ((dstH.at<uchar>(i, j) >= 244 && dstH.at<uchar>(i, j) <= 250) || 
				(dstH.at<uchar>(i, j) >= 0 && dstH.at<uchar>(i, j) <= 11) ||
				(dstH.at<uchar>(i, j) >= 149 && dstH.at<uchar>(i, j) <= 181))

				// below: values from medium.com
			/*if (((dstH.at<uchar>(i, j) >= 0 && dstS.at<uchar>(i, j) >= 70 && dstV.at<uchar>(i, j) >= 60) &&
				(dstH.at<uchar>(i, j) <= 10 && dstS.at<uchar>(i, j) <= 255 && dstV.at<uchar>(i, j) <= 255))
				||
				((dstH.at<uchar>(i, j) >= 170 && dstS.at<uchar>(i, j) >= 70 && dstV.at<uchar>(i, j) >= 60) &&
					(dstH.at<uchar>(i, j) <= 180 && dstS.at<uchar>(i, j) <= 255 && dstV.at<uchar>(i, j) <= 255))
				||
				((dstH.at<uchar>(i, j) >= 100 && dstS.at<uchar>(i, j) >= 127 && dstV.at<uchar>(i, j) >= 0) &&
					(dstH.at<uchar>(i, j) <= 140 && dstS.at<uchar>(i, j) <= 255 && dstV.at<uchar>(i, j) <= 255)))*/
				continue;

			else dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
		}
	}

	return dst0;
}


Mat mser(Mat img) {
	Ptr<MSER> ms = MSER::create(8, 400, 4000);
	vector<vector<Point>> regions;
	vector<cv::Rect> mser_bbox;
	ms->detectRegions(img, regions, mser_bbox);

	for (int i = 0; i < regions.size(); i++)
		rectangle(img, mser_bbox[i], CV_RGB(0, 255, 0));
	return img;
}

int main(){
	// filtered_b = -0.5 * filtered_r + 3 * filtered_b - 2 * filtered_g
	// filtered_r = 4 * filtered_r - 0.5 * filtered_b - 2 * filtered_g
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src;
		src = imread(fname);
		imshow("image", convertToHue(histogramEqualization(src)));
		waitKey();
	}
	return 0;
}