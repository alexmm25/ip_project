// OpenCVApplication.cpp : Defines the entry point for the console application.
//

// https://drive.google.com/drive/folders/1fR3JeLpuCHtY7MtybVr9QGzB8bzxrZ3V <- python project

#include "stdafx.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

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

void histogramEqualization() {

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
	Mat dst = src.clone();
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
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (!((dstH.at<uchar>(i, j) >= 244 && dstH.at<uchar>(i, j) <= 255) || (dstH.at<uchar>(i, j) >= 0 && dstH.at<uchar>(i, j) <= 11) ||
				(dstH.at<uchar>(i, j) >= 149 && dstH.at<uchar>(i, j) <= 181)))
				// below: values from medium.com
			/*if((dstH.at<uchar>(i,j) >= 0 && dstS.at<uchar>(i, j) >= 70 && dstV.at<uchar>(i, j) >= 60) && 
				(dstH.at<uchar>(i, j) <= 10 && dstS.at<uchar>(i, j) <= 255 && dstV.at<uchar>(i, j) <= 255)
				||
				(dstH.at<uchar>(i, j) >= 170 && dstS.at<uchar>(i, j) >= 70 && dstV.at<uchar>(i, j) >= 60) &&
				(dstH.at<uchar>(i, j) <= 180 && dstS.at<uchar>(i, j) <= 255 && dstV.at<uchar>(i, j) <= 255)
				||
				(dstH.at<uchar>(i, j) >= 94 && dstS.at<uchar>(i, j) >= 127 && dstV.at<uchar>(i, j) >= 20) &&
				(dstH.at<uchar>(i, j) <= 126 && dstS.at<uchar>(i, j) <= 255 && dstV.at<uchar>(i, j) <= 200)
				)*/
			dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
		}
	}

	return dst;
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
		imshow("image", convertToHue(src));
		waitKey();
	}
	return 0;
}