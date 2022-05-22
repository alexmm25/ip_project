#include <vector>
#include <numeric>

#include "stdafx.h"
#include "common.h"

#include <fstream>
#include <sstream>

using namespace std;

#include "opening.h"
#include "histogram.h"


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
				struct_elem(i, j) = 0;
			else struct_elem(i, j) = 255;
	return struct_elem;
}

Mat erosion(Mat src, Mat structuralElem) {
	Mat dst = src.clone();
	int SEi = structuralElem.rows / 2;
	int SEj = structuralElem.cols / 2;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 255) {
				dst.at<uchar>(i, j) = 255;
				for (int ii = 0; ii < structuralElem.rows; ii++) {
					for (int jj = 0; jj < structuralElem.cols; jj++) {
						if (structuralElem.at<uchar>(ii, jj) == 255) {
							if (isInside(src, i + ii - SEi, j + jj - SEj)) {
								if (src.at<uchar>(i + ii - SEi, j + jj - SEj) == 0)
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
			if (src.at<uchar>(i, j) == 255) {
				dst.at<uchar>(i, j) = 255;
				for (int ii = 0; ii < structuralElem.rows; ii++) {
					for (int jj = 0; jj < structuralElem.cols; jj++) {
						if (structuralElem.at<uchar>(ii, jj) == 255) {
							if (isInside(src, i + ii - SEi, j + jj - SEj)) {
								dst.at<uchar>(i + ii - SEi, j + jj - SEj) = 255;
							}
						}
					}
				}
			}
		}
	}
	return dst;
}

Mat dilation(Mat src) {
	Mat structuralElem = getStructuralElem();
	return dilation(src, structuralElem);
}

pair<Mat, Mat> clearOpening(pair<Mat, Mat> dst) {
	Mat structuralElem = getStructuralElem();
	return make_pair(dilation(erosion(dst.first, structuralElem), structuralElem), 
		dilation(erosion(dst.second, structuralElem), structuralElem));
}