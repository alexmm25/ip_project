#include <vector>
#include <numeric>

#include "stdafx.h"
#include "common.h"

#include <fstream>
#include <sstream>

using namespace std;

#include "canny.h"


#define P 0.1

vector<int> h(Mat src) {
	vector<int> h(256, 0);

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			h[src.at<uchar>(i, j)]++;
		}

	return h;
}

Mat adaptive_th(Mat src) {
	Mat dst = src.clone();
	int height = src.rows;
	int width = src.cols;

	for (int j = 0; j < width; j++) {
		dst.at<uchar>(0, j) = 0;
		dst.at<uchar>(height - 1, j) = 0;
	}
	for (int i = 0; i < height; i++) {
		dst.at<uchar>(i, 0) = 0;
		dst.at<uchar>(i, width - 1) = 0;
	}

	vector<int> hist = h(dst);
	int noEdgePixels = P * ((height - 2) * (width - 2) - hist[0]);
	int tH;

	for (int i = hist.size() - 1; i > 0; i--) {
		tH = i;
		if (noEdgePixels >= 0)
			noEdgePixels -= hist[i];
		else break;
	}

	int tL = 0.4 * tH;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			if (dst.at<uchar>(i, j) < tL)
				dst.at<uchar>(i, j) = 0;
			else if (dst.at<uchar>(i, j) < tH)
				dst.at<uchar>(i, j) = 127;
			else dst.at<uchar>(i, j) = 255;
		}

	return dst;
}

Mat edge_linking(Mat src) {

	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 255) {
				queue<Point> Q;
				Q.push(Point(j, i));

				while (!Q.empty())
				{
					Point q = Q.front();
					Q.pop();
					for (int k = 0; k < 8; k++) {
						Point n(q.x + di[k], q.y + dj[k]);
						if (n.x < 0 || n.y < 0 || n.y >= src.rows || n.x >= src.cols)
							continue;
						if (src.at<uchar>(n.y, n.x) == 127) {
							src.at<uchar>(n.y, n.x) = 255;
							Q.push(n);
						}
					}
				}
			}
		}
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if (src.at<uchar>(i, j) == 127)
				src.at<uchar>(i, j) = 0;

	return src;
}

int apply(Mat src, int i, int j, vector<vector<int>> filter, int div) {
	int conv = 0;
	int w = filter.size();

	for (int k = 0; k < w; k++)
		for (int l = 0; l < w; l++) {
			int newI = i - w / 2 + k;
			int newJ = j - w / 2 + l;
			if (newI >= 0 && newJ >= 0 && newI < src.rows && newJ < src.cols)
				conv += filter[k][l] * src.at<uchar>(newI, newJ);
		}
	return conv / div;
}

vector<vector<float>> G2D(float sigma) {
	vector<vector<float>> g;

	int w = 6 * sigma + 0.5;
	w = w % 2 == 0 ? w + 1 : w;

	for (int i = 0; i < w; i++) {
		vector<float> v;
		for (int j = 0; j < w; j++) {
			v.push_back((1. / (2. * PI * sigma * sigma)) * exp(((i - w / 2) * (i - w / 2) + (j - w / 2) * (j - w / 2)) / (-2. * sigma * sigma)));
		}
		g.push_back(v);
		v.clear();
	}

	return g;
}

float sumG(vector<vector<float>> g) {
	float sum = 0;
	for (vector<float> g1 : g)
		for (float e : g1) {
			sum += e;
		}
	return sum;
}

Mat gaussian_filter2D(Mat src, float sigma) {
	Mat dest(src.rows, src.cols, CV_8UC1);

	vector<vector<float>> g = G2D(sigma);
	float sum = sumG(g);

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			float conv = 0;
			int h = g.size();
			int w = g[0].size();

			for (int k = 0; k < h; k++)
				for (int l = 0; l < w; l++) {
					int newI = i - h / 2 + k;
					int newJ = j - w / 2 + l;
					if (newI >= 0 && newJ >= 0 && newI < src.rows && newJ < src.cols)
						conv += g[k][l] * src.at<uchar>(newI, newJ);
				}
			dest.at<uchar>(i, j) = conv / sum;

			//dest.at<uchar>(i, j) = apply(src, i, j, g, sum);
		}

	return dest;
}


pair<Mat, Mat> gradient(Mat src) {

	int rows = src.rows;
	int cols = src.cols;

	Mat srcG = gaussian_filter2D(src, 0.5);

	Mat G(rows, cols, CV_32FC1);
	Mat Phi(rows, cols, CV_32FC1);
	vector<vector<int>> Sx({ { -1, 0, 1 }, {-2, 0, 2}, {-1, 0, 1} });
	vector<vector<int>> Sy({ { 1, 2, 1 }, {0, 0, 0}, {-1, -2, -1} });

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) {
			float gx = apply(srcG, i, j, Sx, 1);
			float gy = apply(srcG, i, j, Sy, 1);

			G.at<float>(i, j) = sqrt(gy * gy + gx * gx) / (4 * sqrt(2));
			Phi.at<float>(i, j) = atan2(gy, gx);
		}
	return make_pair(G, Phi);
}


Mat nonMaximaSupression(pair<Mat, Mat> src) {
	int rows = src.first.rows;
	int cols = src.first.cols;
	Mat G = src.first.clone();

	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) {
			float phi = src.second.at<float>(i, j);
			int i1, j1, i2, j2;

			if (phi >= -CV_PI / 8 && phi <= CV_PI / 8 || phi <= -7 * CV_PI / 8 || phi >= 7 * CV_PI / 8) // 0
				i1 = i, j1 = j - 1, i2 = i, j2 = j + 1;
			else if (phi >= CV_PI / 8 && phi <= 3 * CV_PI / 8 || phi >= -7 * CV_PI / 8 && phi <= -5 * CV_PI / 8) // 1
				i1 = i + 1, j1 = j - 1, i2 = i - 1, j2 = j + 1;
			else if (phi >= -5 * CV_PI / 8 && phi <= -3 * CV_PI / 8 || phi >= 3 * CV_PI / 8 && phi <= 5 * CV_PI / 8) // 2
				i1 = i - 1, j1 = j, i2 = i + 1, j2 = j;
			else if (phi >= -3 * CV_PI / 8 && phi <= -CV_PI / 8 || phi >= 5 * CV_PI / 8 && phi <= 7 * CV_PI / 8) // 3
				i1 = i - 1, j1 = j - 1, i2 = i + 1, j2 = j + 1;

			if (i1 >= 0 && i2 >= 0 && j1 >= 0 && j2 >= 0 && i1 < rows && i2 < rows && j1 < cols && j2 < cols)
			if (src.first.at<float>(i, j) >= src.first.at<float>(i1, j1) && src.first.at<float>(i, j) >= src.first.at<float>(i2, j2))
				continue;
			else G.at<float>(i, j) = 0;
		}
	return G;
}


Mat nonMax_gradient(Mat src) {
	Mat Gnms(src.rows, src.cols, CV_8UC1);

	Mat Gf = nonMaximaSupression(gradient(src));

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			Gnms.at<uchar>(i, j) = Gf.at<float>(i, j);

	return Gnms;
}

Mat canny(Mat src) {
	return edge_linking(adaptive_th(nonMax_gradient(src)));
}