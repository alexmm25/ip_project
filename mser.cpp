#include "stdafx.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

#include "mser.h"
#include "histogram.h"

vector<Rect> nonMaximumSuppression(vector<Rect> boxes, float overlap_threshold)
{
	vector<float> areas;
	vector<Rect> pick;          //indices of final detection boxes

	for (Rect box : boxes)
		areas.push_back(box.area());

	vector<size_t> idxs(areas.size());
	iota(idxs.begin(), idxs.end(), 0);
	stable_sort(idxs.begin(), idxs.end(), [&areas](size_t i1, size_t i2) {return areas[i1] < areas[i2]; });

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

vector<Rect> ratioCheck(int rows, int cols, vector<Rect> boxes) {
	vector<Rect> boxesChecked;
	vector<pair<int, int>> indexedSizes;

	for (int i = 0; i < boxes.size(); i++) 
		indexedSizes.push_back(make_pair(i, boxes[i].height * boxes[i].width));

	stable_sort(indexedSizes.begin(), indexedSizes.end(), [](pair<int, int> a, pair<int, int> b) {return a.second > b.second;});

	for (int i = 0; i < (int)boxes.size(); i++) {
		float ratio2 = boxes[indexedSizes[i].first].height / (float)boxes[indexedSizes[i].first].width;
		float ratio1 = boxes[indexedSizes[i].first].width / (float)boxes[indexedSizes[i].first].height;

		if (boxes[indexedSizes[i].first].height >= rows-1 || boxes[indexedSizes[i].first].width >= cols-1)
			continue;
		if (ratio1 <= 0.5 || ratio1 > 1.2)
			continue;
		if (ratio2 <= 0.5)
			continue;

		boxesChecked.push_back(boxes[indexedSizes[i].first]);
	}

	return boxesChecked;
}

vector<Rect> mser(Mat src, Mat canny) {
	Mat img = src.clone();

	Ptr<MSER> ms = MSER::create(19, 400, 99990, 0.1);
	vector<vector<Point>> regions;
	vector<Rect> mser_bbox;
	ms->detectRegions(img, regions, mser_bbox);

	vector<Rect> boxes = nonMaximumSuppression(ratioCheck(src.rows, src.cols, mser_bbox), 1.025);
	vector<Rect> boxesCanny;

	/*
	for (int i = 0; i < canny.rows; i++)
		for (int j = 0; j < canny.cols; j++) {
			if (canny.at<uchar>(i, j) == 255)
				for (auto box = boxesChecked.begin(); box != boxesChecked.end(); box++) {
					if ((*box).contains(Point2i(i, j))) {
						boxesCanny.push_back(*box);
						boxesChecked.erase(box--);
					}
				}
		}*/

	//for (Rect box : boxesCanny) {
		//rectangle(img, box, CV_RGB(255, 255, 255));
	//}
	return boxes;
}