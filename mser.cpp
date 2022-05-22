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
		indexedSizes.push_back(make_pair(i, boxes[i].area()));

	stable_sort(indexedSizes.begin(), indexedSizes.end(), [](pair<int, int> a, pair<int, int> b) {return a.second > b.second;});

	int i = 0;
	for (pair<int,int> index : indexedSizes) {
		if (i == 3) break;

		float ratio2 = boxes[index.first].height / (float)boxes[index.first].width;
		float ratio1 = boxes[index.first].width / (float)boxes[index.first].height;

		if (boxes[index.first].height >= rows - 1 || boxes[index.first].width >= cols - 1)
			continue;
		if (ratio1 <= 0.5 || ratio1 > 1.2)
			continue;
		if (ratio2 <= 0.5)
			continue;

		boxesChecked.push_back(boxes[index.first]);
		i++;
	}

	return boxesChecked;
}

vector<Rect> mser(Mat src) {
	Mat img = src.clone();

	Ptr<MSER> ms = MSER::create(19, 400, 99990, 0.1);
	vector<vector<Point>> regions;
	vector<Rect> mser_bbox;
	ms->detectRegions(img, regions, mser_bbox);

	return nonMaximumSuppression(ratioCheck(src.rows, src.cols, mser_bbox), 0.01);
}