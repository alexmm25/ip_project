#pragma once

std::pair<cv::Mat, cv::Mat> clearOpening(std::pair<cv::Mat, cv::Mat> dst);
cv::Mat dilation(cv::Mat src);