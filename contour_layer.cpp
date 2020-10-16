// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "contour_layer.h"
ContourSelectInstance::ContourSelectInstance() {
	roi_x_min = 0.01;
	roi_x_max = 0.99;
	roi_y_min = 0.01;
	roi_y_max = 0.99;
	bin_threshold = 120;
	minLength = 32;
	min_gap_ceil = 32;
	max_object_ssim = 0.0;
	cons_len = 0;
}

ContourSelectInstance::~ContourSelectInstance() {

}

int ContourSelectInstance::load_model(const char * path)
{
	//load file
	int ret = 0;
	FILE* fhandle = fopen(path, "rb+");
	if (fhandle != nullptr) { 
		//fread(&red_coef, sizeof(float), 1, fhandle);
		//fread(&green_coef, sizeof(float), 1, fhandle);
		//fread(&blue_coef, sizeof(float), 1, fhandle); 
		fread(&roi_x_min, sizeof(float), 1, fhandle);
		fread(&roi_x_max, sizeof(float), 1, fhandle);
		fread(&roi_y_min, sizeof(float), 1, fhandle);
		fread(&roi_y_max, sizeof(float), 1, fhandle);
	}
	fclose(fhandle);
	return ret;
}

int ContourSelectInstance::save_model(const char * path) {
	//load file
	int ret = 0;
	FILE* fhandle = fopen(path, "wb+");
	if (fhandle != nullptr) {
		//fwrite(&red_coef, sizeof(float), 1, fhandle);
		//fwrite(&green_coef, sizeof(float), 1, fhandle);
		//fwrite(&blue_coef, sizeof(float), 1, fhandle);

		fwrite(&roi_x_min, sizeof(float), 1, fhandle);
		fwrite(&roi_x_max, sizeof(float), 1, fhandle);
		fwrite(&roi_y_min, sizeof(float), 1, fhandle);
		fwrite(&roi_y_max, sizeof(float), 1, fhandle);
	}

	return ret;
}

void ContourSelectInstance::maximum_length_contours(const cv::Mat& amps) {
	const int M = 15, N = 15;
	const int KM = 7, KN = 7;
	const int width = amps.cols, height = amps.rows;
	//std::cout <<"Contour Detector paramter -> minLength:" << minLength << ", width:" << width << ", height:" << height << ", threshold:"  << bin_th << std::endl;
	//figure out the range that we should apply the filter to
	const long first_row = std::max(M / 2, (int)ceil(roi_y_min * height));
	const long first_col = std::max(N / 2, (int)ceil(roi_x_min * width));
	const long last_row = std::min(height - M / 2, (int)ceil(roi_y_max * height));
	const long last_col = std::min(width - N / 2, (int)ceil(roi_x_max * width));
	contour_loop_set.clear();
	cv::Size processSize = cv::Size(amps.cols, amps.rows);
	cv::Mat pix_used = cv::Mat::zeros(processSize, CV_32S);
	//const int incx1[8] = { 1, 1, 0, -1, -1, -1,  0,  1 };
	//const int incy1[8] = { 0, 1, 1,  1,  0, -1, -1, -1 };
	const int maxGAP = KM / 2;
	for (long r = first_row; r < last_row; ++r)
	{
		for (long c = first_col; c < last_col; ++c) {
			float norm = amps.at<float>(r, c);
			if (pix_used.at<int>(r, c) > 0) continue;
			int max_first_incr = 0, max_second_incr = 0;
			float first_norm = 0, second_norm = 0;
			int max_sum_norm = bin_threshold;
			long contour_length = 0;
			std::vector<cv::Point> line_cw; line_cw.clear();
			std::vector<cv::Point> line_ccw; line_ccw.clear();
			int max_incr1 = 0;
			if (norm > bin_threshold) {
				int max_sz1 = 1, c_start = c, r_start = r, c_end = c, r_end = r;
				for (int first_incr = 0; first_incr < 4; first_incr++) {
					int pos_incr_sz = 0, pos_gap = 0;
					int pos_c = c + incx1[first_incr], pos_r = r + incy1[first_incr];
					if ((pos_c < first_col) || (pos_c > last_col) || (pos_r < first_row) || (pos_r > last_row)) continue;
					first_norm = amps.at<float>(pos_r, pos_c);
					if (first_norm <= bin_threshold) pos_gap++;
					else pos_incr_sz++;
					while (pos_gap <= maxGAP) {
						pos_c = pos_c + incx1[first_incr]; pos_r = pos_r + incy1[first_incr];
						if ((pos_c < first_col) || (pos_c > last_col) || (pos_r < first_row) || (pos_r > last_row)) break;
						first_norm = amps.at<float>(pos_r, pos_c);
						if (first_norm <= bin_threshold) pos_gap++;
						else pos_incr_sz++;
					}

					int first_incr_neg = first_incr + 4;
					int neg_incr_sz = 0, neg_gap = 0;
					int neg_c = c + incx1[first_incr_neg], neg_r = r + incy1[first_incr_neg];
					if ((neg_c < first_col) || (neg_c > last_col) || (neg_r < first_row) || (neg_r > last_row)) continue;
					first_norm = amps.at<float>(neg_r, neg_c);
					if (first_norm <= bin_threshold) neg_gap++;
					else neg_incr_sz++;
					while (neg_gap <= maxGAP) {
						neg_c = neg_c + incx1[first_incr_neg]; neg_r = neg_r + incy1[first_incr_neg];
						if ((neg_c < first_col) || (neg_c > last_col) || (neg_r < first_row) || (neg_r > last_row)) break;
						first_norm = amps.at<float>(neg_r, neg_c);
						if (first_norm <= bin_threshold) neg_gap++;
						else neg_incr_sz++;
					}

					if (max_sz1 < 1 + pos_incr_sz + neg_incr_sz) {
						max_sz1 = 1 + pos_incr_sz + neg_incr_sz;
						max_incr1 = first_incr;
						c_start = neg_c; r_start = neg_r;
						c_end = pos_c; r_end = pos_r;
					}
				}
				if (max_sz1 < KM) continue;
				else contour_length = max_sz1;
				int max_recall = std::min(KM, max_sz1);
				int first_r, first_c;
				int last_start_r = r_start, last_start_c = c_start, last_end_r = r_end, last_end_c = c_end, last_incr1 = max_incr1;
				int prev_sz2 = max_sz1;
				//顺时针延伸
				while (true) {
					first_r = last_end_r; first_c = last_end_c;
					int max_r2s, max_c2s, max_sz_sum = prev_sz2, max_incr2 = 0, max_incr2_sz = 0, max_r2, max_c2;
					int second_incr = last_incr1 - 1; if (second_incr < 0) second_incr = 7;
					int recall = 0;
					while (recall < max_recall) {
						int incr_sz2 = 0;
						int second_c = first_c + incx1[second_incr], second_r = first_r + incy1[second_incr];
						if ((second_c > first_col) && (second_c < last_col) && (second_r > first_row) && (second_r < last_row)) {
							second_norm = amps.at<float>(second_r, second_c);
							int second_gap = 0;
							if (second_norm <= bin_threshold) second_gap++;
							else incr_sz2++;
							while (second_gap <= maxGAP) {
								second_c = second_c + incx1[second_incr]; second_r = second_r + incy1[second_incr];
								if ((second_c < first_col) || (second_c > last_col) || (second_r < first_row) || (second_r > last_row)) break;
								second_norm = amps.at<float>(second_r, second_c);
								if (second_norm <= bin_threshold) second_gap++;
								else incr_sz2++;
							}
						}
						if (max_sz_sum < incr_sz2 + prev_sz2 - recall) {
							max_sz_sum = incr_sz2 + prev_sz2 - recall;
							max_incr2 = second_incr;	max_incr2_sz = incr_sz2;
							max_r2s = first_r; max_c2s = first_c; max_r2 = second_r; max_c2 = second_c;
						}
						first_norm = amps.at<float>(first_r, first_c);
						if (first_norm > bin_threshold) recall++;
						first_c = first_c - incx1[last_incr1]; first_r = first_r - incy1[last_incr1];
					}
					first_r = last_end_r; first_c = last_end_c;
					second_incr = last_incr1 + 1; if (second_incr > 7) second_incr = 0;
					recall = 0;
					while (recall < max_recall) {
						int incr_sz2 = 0;
						int second_c = first_c + incx1[second_incr], second_r = first_r + incy1[second_incr];
						if ((second_c > first_col) && (second_c < last_col) && (second_r > first_row) && (second_r < last_row)) {
							second_norm = amps.at<float>(second_r, second_c);
							int second_gap = 0;
							if (second_norm <= bin_threshold) second_gap++;
							else incr_sz2++;
							while (second_gap <= maxGAP) {
								second_c = second_c + incx1[second_incr]; second_r = second_r + incy1[second_incr];
								if ((second_c < first_col) || (second_c > last_col) || (second_r < first_row) || (second_r > last_row)) break;
								second_norm = amps.at<float>(second_r, second_c);
								if (second_norm <= bin_threshold) second_gap++;
								else incr_sz2++;
							}
						}
						if (max_sz_sum < incr_sz2 + prev_sz2 - recall) {
							max_sz_sum = incr_sz2 + prev_sz2 - recall;
							max_incr2 = second_incr; max_incr2_sz = incr_sz2;
							max_r2s = first_r; max_c2s = first_c; max_r2 = second_r; max_c2 = second_c;
						}
						first_norm = amps.at<float>(first_r, first_c);
						if (first_norm > bin_threshold) recall++;
						first_c = first_c - incx1[last_incr1]; first_r = first_r - incy1[last_incr1];
					}

					if (max_incr2_sz < 1) {
						line_cw.push_back(cv::Point(last_start_c, last_start_r));
						line_cw.push_back(cv::Point(last_end_c, last_end_r));
						break;    //碰到结束端点，前进停止
					}	else {        //否则， 更新下一段
						line_cw.push_back(cv::Point(last_start_c, last_start_r));
						line_cw.push_back(cv::Point(max_c2s, max_r2s));
						last_start_r = max_r2s; last_start_c = max_c2s;
						last_end_r = max_r2; last_end_c = max_c2;
						last_incr1 = max_incr2;
						contour_length += max_incr2_sz;
						prev_sz2 = max_incr2_sz;
					}
					int second_r = max_r2s, second_c = max_c2s;
					bool circle_loop = false;
					while ((second_r != max_r2) || (second_c != max_c2)) {   //检查是否遇到回环， 防止循环
						second_c = second_c + incx1[max_incr2];	second_r = second_r + incy1[max_incr2];
						if (pix_used.at<float>(second_r, second_c) == 1) {
							circle_loop = true;
							break;
						}
						int init_r = r_start, init_c = c_start;
						while ((init_r != r_end) || (init_c != c_end)) {
							if ((second_r == init_r) && (second_c == init_c)) {
								circle_loop = true;
							}
							init_c = init_c + incx1[max_incr1]; init_r = init_r + incy1[max_incr1];
						}
						pix_used.at<float>(second_r, second_c) = 1;  //1: prev points in cur line. 2: already .
					}

					if (circle_loop) break; //发现回路循环 
					max_recall = std::min(KM, max_incr2_sz);
				}
				//逆时针延伸
				int primary_end_r = r_start, primary_end_c = c_start, primary_start_r = r_end, primary_start_c = c_end, primary_incr1 = max_incr1 + 4;
				//std::cout << "Initial search direction - Primary incr1:" << primary_incr1 << ", new line -> startr:" << r_end << ", startc:" << c_end <<
				//	", endr:" << r_start << ", endc" << c_start << std::endl;
				prev_sz2 = max_sz1;
				while (true) {
					first_r = primary_end_r; first_c = primary_end_c;
					int max_r2s, max_c2s, max_sz_sum = prev_sz2, max_incr2 = 0, max_incr2_sz = 0, max_r2, max_c2;
					int second_incr = primary_incr1 - 1;  if (second_incr < 0) second_incr = 7;
					int recall = 0;
					while (recall < max_recall) {
						int incr_sz2 = 0;
						int second_c = first_c + incx1[second_incr], second_r = first_r + incy1[second_incr];
						if ((second_c > first_col) && (second_c < last_col) && (second_r > first_row) && (second_r < last_row)) {
							second_norm = amps.at<float>(second_r, second_c);
							int second_gap = 0;
							if (second_norm <= bin_threshold) second_gap++;
							else incr_sz2++;
							while (second_gap <= maxGAP) {
								second_c = second_c + incx1[second_incr]; second_r = second_r + incy1[second_incr];
								if ((second_c < first_col) || (second_c > last_col) || (second_r < first_row) || (second_r > last_row)) break;
								second_norm = amps.at<float>(second_r, second_c);
								if (second_norm <= bin_threshold) second_gap++;
								else incr_sz2++;
							}
						}
						if (max_sz_sum < incr_sz2 + prev_sz2 - recall) {
							max_sz_sum = incr_sz2 + prev_sz2 - recall;
							max_incr2 = second_incr; max_incr2_sz = incr_sz2;
							max_r2s = first_r; max_c2s = first_c; max_r2 = second_r; max_c2 = second_c;
						}
						//std::cout << "First rc:" << first_c << "," << first_r << std::endl;
						first_norm = amps.at<float>(first_r, first_c);
						if (first_norm > bin_threshold) recall++;
						first_c = first_c - incx1[primary_incr1]; first_r = first_r - incy1[primary_incr1];
					}
					first_r = primary_end_r; first_c = primary_end_c;
					second_incr = primary_incr1 + 1;	if (second_incr > 7) second_incr = 0;
					recall = 0;
					while (recall < max_recall) {
						int incr_sz2 = 0;
						int second_c = first_c + incx1[second_incr], second_r = first_r + incy1[second_incr];
						if ((second_c > first_col) && (second_c < last_col) && (second_r > first_row) && (second_r < last_row)) {
							second_norm = amps.at<float>(second_r, second_c);
							int second_gap = 0;
							if (second_norm <= bin_threshold) second_gap++;
							else incr_sz2++;
							while (second_gap <= maxGAP) {
								second_c = second_c + incx1[second_incr]; second_r = second_r + incy1[second_incr];
								if ((second_c < first_col) || (second_c > last_col) || (second_r < first_row) || (second_r > last_row)) break;
								second_norm = amps.at<float>(second_r, second_c);
								if (second_norm <= bin_threshold) second_gap++;
								else incr_sz2++;
							}
						}
						if (max_sz_sum < incr_sz2 + prev_sz2 - recall) {
							max_sz_sum = incr_sz2 + prev_sz2 - recall;
							max_incr2 = second_incr; max_incr2_sz = incr_sz2;
							max_r2s = first_r; max_c2s = first_c; max_r2 = second_r; max_c2 = second_c;
						}
						first_norm = amps.at<float>(first_r, first_c);
						if (first_norm > bin_threshold) recall++;
						first_c = first_c - incx1[primary_incr1]; first_r = first_r - incy1[primary_incr1];
					}

					if (max_incr2_sz < 1) {
						line_ccw.push_back(cv::Point(primary_start_c, primary_start_r));
						line_ccw.push_back(cv::Point(primary_end_c, primary_end_r));
						break;    //碰到结束端点，前进停止
					}
					else {                          //否则， 更新下一段
						line_ccw.push_back(cv::Point(primary_start_c, primary_start_r));
						line_ccw.push_back(cv::Point(max_c2s, max_r2s));
						primary_start_r = max_r2s; primary_start_c = max_c2s;
						primary_end_r = max_r2; primary_end_c = max_c2;
						primary_incr1 = max_incr2;
						contour_length += max_incr2_sz;
						prev_sz2 = max_incr2_sz;
						//std::cout << "Update next search direction - Primary incr1:" << primary_incr1 << ", new line -> startr:" << max_r2s << ", startc:" << max_c2s <<
						//	", endr:" << max_r2 << ", endc" << max_c2 << std::endl;
					}
					int second_r = max_r2s, second_c = max_c2s;
					bool circle_loop = false;
					while ((second_r != max_r2) || (second_c != max_c2)) {   //检查是否遇到回环， 防止循环
						second_c = second_c + incx1[max_incr2]; second_r = second_r + incy1[max_incr2];
						if (pix_used.at<float>(second_r, second_c) == 1) {
							circle_loop = true;
							break;
						}
						int init_r = r_start, init_c = c_start;
						while ((init_r != r_end) || (init_c != c_end)) {
							if ((second_r == init_r) && (second_c == init_c)) {
								circle_loop = true;
							}
							init_c = init_c + incx1[max_incr1]; init_r = init_r + incy1[max_incr1];
						}
						pix_used.at<float>(second_r, second_c) = 1;  //1: prev points in cur line. 2: already .
					}

					if (circle_loop) break; //发现回路循环 
					max_recall = std::min(KM, max_incr2_sz);
				}

				if (contour_length > minLength) {

					int size_cw = line_cw.size(), size_ccw = line_ccw.size();
					if ((size_cw > 0) && (size_ccw > 0)) {
						std::vector<cv::Point> line_sequence; line_sequence.clear();
						for (int i = size_cw - 1; i >= 0; i--) line_sequence.push_back(line_cw[i]);
						for (int i = 0; i < size_ccw; i++) line_sequence.push_back(line_ccw[i]);
						contour_loop_set.push_back(line_sequence);
					}
					else if (size_cw > 0) contour_loop_set.push_back(line_cw);
					else if (size_ccw > 0) contour_loop_set.push_back(line_ccw);

				}
			}
		}
	}

}

void ContourSelectInstance::maximum_value_contours(const cv::Mat& amps) {
	int height = amps.rows, width = amps.cols;
	const int M = 13, N = 13;
	const int KM = 13, KN = 13;
	const long first_row = std::max(M / 2, (int)ceil(roi_y_min * height));
	const long first_col = std::max(N / 2, (int)ceil(roi_x_min * width));
	const long last_row = std::min(height - M / 2, (int)ceil(roi_y_max * height));
	const long last_col = std::min(width - N / 2, (int)ceil(roi_x_max * width));
	
	contour_loop_set.clear();
	cv::Size processSize = cv::Size(amps.cols, amps.rows);
	cv::Mat pix_used = cv::Mat::zeros(processSize, CV_32S);
	int incx1[8] = { 1, 1, 0, -1, -1, -1,  0,  1 };
	int incy1[8] = { 0, 1, 1,  1,  0, -1, -1, -1 };
	const int maxGAP = 2;// KM / 2;
	for (long r = first_row; r < last_row; ++r)
	{
		for (long c = first_col; c < last_col; ++c) {
			float norm = amps.at<float>(r, c);
			if (pix_used.at<int>(r, c) > 0) continue;
			if (norm <= bin_threshold) continue;
			int max_first_incr = 0, max_second_incr = 0;
			float first_norm = 0, second_norm = 0;
			long contour_length = 0;
			std::vector<cv::Point> line_cw; line_cw.clear();
			std::vector<cv::Point> line_ccw; line_ccw.clear();
			int max_incr1 = 0;

			bool circle_loop = false, dead_end = false;

			int max_sz1 = 1, c_start = c, r_start = r, c_end = c, r_end = r;
			float start_norm_sum = 0.0;
			for (int first_incr = 0; first_incr < 4; first_incr++) {
				int pos_incr_sz = 0, pos_gap = 0; float max_norm_pos = 0.0;
				int pos_c = c + incx1[first_incr], pos_r = r + incy1[first_incr];
				if ((pos_c < first_col) || (pos_c > last_col) || (pos_r < first_row) || (pos_r > last_row)) continue;
				first_norm = amps.at<float>(pos_r, pos_c);
				//if (first_norm <= bin_th) pos_gap++;
				//else pos_incr_sz++;
				if (first_norm > bin_threshold) {
					pos_incr_sz++;
					max_norm_pos += first_norm;
				}
				else pos_gap++;
				while (pos_gap <= maxGAP) {
					pos_c = pos_c + incx1[first_incr]; pos_r = pos_r + incy1[first_incr];
					if ((pos_c < first_col) || (pos_c > last_col) || (pos_r < first_row) || (pos_r > last_row)) break;
					first_norm = amps.at<float>(pos_r, pos_c);
					//if (first_norm <= bin_th) pos_gap++;
					//else pos_incr_sz++;
					if (first_norm > bin_threshold) {
						pos_incr_sz++;
						max_norm_pos += first_norm;
					}
					else pos_gap++;
				}

				int first_incr_neg = first_incr + 4;
				int neg_incr_sz = 0, neg_gap = 0; float max_norm_neg = 0.0;
				int neg_c = c + incx1[first_incr_neg], neg_r = r + incy1[first_incr_neg];
				if ((neg_c < first_col) || (neg_c > last_col) || (neg_r < first_row) || (neg_r > last_row)) continue;
				first_norm = amps.at<float>(neg_r, neg_c);
				//if (first_norm <= bin_th) neg_gap++;
				//else neg_incr_sz++;
				if (first_norm > bin_threshold) {
					neg_incr_sz++;
					max_norm_neg += first_norm;
				}
				else neg_gap++;
				while (neg_gap <= maxGAP) {
					neg_c = neg_c + incx1[first_incr_neg]; neg_r = neg_r + incy1[first_incr_neg];
					if ((neg_c < first_col) || (neg_c > last_col) || (neg_r < first_row) || (neg_r > last_row)) break;
					first_norm = amps.at<float>(neg_r, neg_c);
					//if (first_norm <= bin_th) neg_gap++;
					//else neg_incr_sz++;
					if (first_norm > bin_threshold) {
						neg_incr_sz++;
						max_norm_neg += first_norm;
					}
					else neg_gap++;
				}

				//if (max_sz1 < 1 + pos_incr_sz + neg_incr_sz) 
				if (start_norm_sum < norm + max_norm_pos + max_norm_neg)
				{
					max_sz1 = 1 + pos_incr_sz + neg_incr_sz;
					start_norm_sum = norm + max_norm_pos + max_norm_neg;
					max_incr1 = first_incr;
					c_start = neg_c; r_start = neg_r;
					c_end = pos_c; r_end = pos_r;
				}
			}
			if (max_sz1 < KM) continue;
			else contour_length += max_sz1;
			int max_recall = std::min(KM / 2, max_sz1);
			int first_r, first_c;
			int last_start_r = r_start, last_start_c = c_start, last_end_r = r_end, last_end_c = c_end, last_incr1 = max_incr1;
			//std::cout << "First search direction - Last incr1:" << last_incr1 << ", first line -> startr:" << r_start << ", startc:" << c_start <<
			//	", endr:" << r_end << ", endc" << c_end << std::endl;
			int prev_sz2 = max_sz1; float prev_max_norm = start_norm_sum;
			float angle_sum_r = 0.0;
			//顺时针延长
			while (true) {
				first_r = last_end_r; first_c = last_end_c;
				int max_r2s, max_c2s, max_sz_sum = prev_sz2, max_incr2 = 0, max_incr2_sz = 0, max_r2, max_c2;
				int second_incr = last_incr1 - 1; if (second_incr < 0) second_incr = 7;
				int recall = 0; float max_norm2 = prev_max_norm, recall_norm = 0.0, norm2 = 0.0;
				while (recall < max_recall) {
					int incr_sz2 = 0; norm2 = 0.0;
					int second_c = first_c + incx1[second_incr], second_r = first_r + incy1[second_incr];
					if ((second_c > first_col) && (second_c < last_col) && (second_r > first_row) && (second_r < last_row)) {
						second_norm = amps.at<float>(second_r, second_c);
						int second_gap = 0;
						if (second_norm > bin_threshold) {
							incr_sz2++;
							norm2 += second_norm;
						}
						else second_gap++;
						while (second_gap <= maxGAP) {
							second_c = second_c + incx1[second_incr]; second_r = second_r + incy1[second_incr];
							if ((second_c < first_col) || (second_c > last_col) || (second_r < first_row) || (second_r > last_row)) break;
							second_norm = amps.at<float>(second_r, second_c);
							if (second_norm > bin_threshold) {
								incr_sz2++;
								norm2 += second_norm;
							}
							else second_gap++;
						}
					}
					//if (max_sz_sum < incr_sz2 + prev_sz2 - recall)
					if (max_norm2 < norm2 + prev_max_norm - recall_norm)
					{
						max_norm2 = norm2 + prev_max_norm - recall_norm;
						max_sz_sum = incr_sz2 + prev_sz2 - recall;
						max_incr2 = second_incr;	max_incr2_sz = incr_sz2;
						max_r2s = first_r; max_c2s = first_c; max_r2 = second_r; max_c2 = second_c;
					}
					first_norm = amps.at<float>(first_r, first_c);
					if (first_norm > bin_threshold) {
						recall++; recall_norm += first_norm;
					}
					first_c = first_c - incx1[last_incr1]; first_r = first_r - incy1[last_incr1];
				}
				first_r = last_end_r; first_c = last_end_c;
				second_incr = last_incr1 + 1; if (second_incr > 7) second_incr = 0;
				recall = 0; recall_norm = 0.0; norm2 = 0.0;
				while (recall < max_recall) {
					int incr_sz2 = 0; norm2 = 0.0;
					int second_c = first_c + incx1[second_incr], second_r = first_r + incy1[second_incr];
					if ((second_c > first_col) && (second_c < last_col) && (second_r > first_row) && (second_r < last_row)) {
						second_norm = amps.at<float>(second_r, second_c);
						int second_gap = 0;
						if (second_norm > bin_threshold) {
							incr_sz2++;
							norm2 += second_norm;
						}
						else second_gap++;
						while (second_gap <= maxGAP) {
							second_c = second_c + incx1[second_incr]; second_r = second_r + incy1[second_incr];
							if ((second_c < first_col) || (second_c > last_col) || (second_r < first_row) || (second_r > last_row)) break;
							second_norm = amps.at<float>(second_r, second_c);
							if (second_norm > bin_threshold) {
								incr_sz2++;
								norm2 += second_norm;
							}
							else second_gap++;
						}
					}
					//if (max_sz_sum < incr_sz2 + prev_sz2 - recall)
					if (max_norm2 < norm2 + prev_max_norm - recall_norm)
					{
						max_norm2 = norm2 + prev_max_norm - recall_norm;
						max_sz_sum = incr_sz2 + prev_sz2 - recall;
						max_incr2 = second_incr; max_incr2_sz = incr_sz2;
						max_r2s = first_r; max_c2s = first_c; max_r2 = second_r; max_c2 = second_c;
					}
					first_norm = amps.at<float>(first_r, first_c);
					if (first_norm > bin_threshold) {
						recall++; recall_norm += first_norm;
					}
					first_c = first_c - incx1[last_incr1]; first_r = first_r - incy1[last_incr1];
				}

				if (max_incr2_sz < 1) {
					line_cw.push_back(cv::Point(last_start_c, last_start_r));
					line_cw.push_back(cv::Point(last_end_c, last_end_r));
					dead_end = true;
					break;    //碰到结束端点，前进停止
				}
				else {                          //否则， 更新下一段
					line_cw.push_back(cv::Point(last_start_c, last_start_r));
					line_cw.push_back(cv::Point(max_c2s, max_r2s));
					last_start_r = max_r2s; last_start_c = max_c2s;
					last_end_r = max_r2; last_end_c = max_c2;
					last_incr1 = max_incr2;
					contour_length += max_incr2_sz;
					prev_sz2 = max_incr2_sz;
					prev_max_norm = max_norm2;
					//line_2d.push_back(cv::Point(last_start_c, last_start_r));
					//line_2d.push_back(cv::Point(last_end_c, last_end_r));
					//std::cout << "Update next search direction - Last incr1:" << last_incr1 << ", new line -> startr:" << max_r2s << ", startc:" << max_c2s <<
					//	", endr:"<< max_r2 << ", endc" << max_c2 << std::endl;
				}
				int second_r = max_r2s, second_c = max_c2s;
				while ((second_r != max_r2) || (second_c != max_c2)) {   //检查是否遇到回环， 防止循环
					second_c = second_c + incx1[max_incr2];	second_r = second_r + incy1[max_incr2];
					if (pix_used.at<float>(second_r, second_c) == 1) {
						circle_loop = true;
						line_cw.push_back(cv::Point(last_start_c, last_start_r));
						line_cw.push_back(cv::Point(second_c, second_r));
						//dead_end = true;
						break;
					}
					int init_r = r_start, init_c = c_start;
					while ((init_r != r_end) || (init_c != c_end)) {
						if ((second_r == init_r) && (second_c == init_c)) {
							circle_loop = true;
							line_cw.push_back(cv::Point(last_start_c, last_start_r));
							line_cw.push_back(cv::Point(second_c, second_r));
							break;
						}
						init_c = init_c + incx1[max_incr1]; init_r = init_r + incy1[max_incr1];
					}
					pix_used.at<float>(second_r, second_c) = 1;  //1: prev points in cur line. 2: already .
				}
				if (dead_end) break;
				if (circle_loop) break; //发现回路循环 
				max_recall = std::min(KM, max_incr2_sz);
			}

			if (!circle_loop) {
				max_recall = std::min(KM / 2, max_sz1);
				//逆时针延长
				int primary_end_r = r_start, primary_end_c = c_start, primary_start_r = r_end, primary_start_c = c_end, primary_incr1 = max_incr1 + 4;
				//std::cout << "Initial search direction - Primary incr1:" << primary_incr1 << ", new line -> startr:" << r_end << ", startc:" << c_end <<
				//	", endr:" << r_start << ", endc" << c_start << std::endl;
				prev_sz2 = max_sz1; prev_max_norm = start_norm_sum;
				while (true) {
					first_r = primary_end_r; first_c = primary_end_c;
					int max_r2s, max_c2s, max_sz_sum = prev_sz2, max_incr2 = 0, max_incr2_sz = 0, max_r2, max_c2;
					int second_incr = primary_incr1 - 1;  if (second_incr < 0) second_incr = 7;

					int recall = 0; float max_norm2 = prev_max_norm, recall_norm = 0.0, norm2 = 0.0;
					while (recall < max_recall) {
						int incr_sz2 = 0; norm2 = 0.0;
						int second_c = first_c + incx1[second_incr], second_r = first_r + incy1[second_incr];
						if ((second_c > first_col) && (second_c < last_col) && (second_r > first_row) && (second_r < last_row)) {
							second_norm = amps.at<float>(second_r, second_c);
							int second_gap = 0;
							if (second_norm > bin_threshold) {
								incr_sz2++;
								norm2 += second_norm;
							}
							else second_gap++;
							while (second_gap <= maxGAP) {
								second_c = second_c + incx1[second_incr]; second_r = second_r + incy1[second_incr];
								if ((second_c < first_col) || (second_c > last_col) || (second_r < first_row) || (second_r > last_row)) break;
								second_norm = amps.at<float>(second_r, second_c);
								if (second_norm > bin_threshold) {
									incr_sz2++;
									norm2 += second_norm;
								}
								else second_gap++;
							}
						}
						//if (max_sz_sum < incr_sz2 + prev_sz2 - recall) 
						if (max_norm2 < norm2 + prev_max_norm - recall_norm)
						{
							max_norm2 = norm2 + prev_max_norm - recall_norm;
							max_sz_sum = incr_sz2 + prev_sz2 - recall;
							max_incr2 = second_incr; max_incr2_sz = incr_sz2;
							max_r2s = first_r; max_c2s = first_c; max_r2 = second_r; max_c2 = second_c;
						}
						first_norm = amps.at<float>(first_r, first_c);
						if (first_norm > bin_threshold) {
							recall++; recall_norm += first_norm;
						}
						first_c = first_c - incx1[primary_incr1]; first_r = first_r - incy1[primary_incr1];
					}
					first_r = primary_end_r; first_c = primary_end_c;
					second_incr = primary_incr1 + 1;	if (second_incr > 7) second_incr = 0;
					recall = 0; recall_norm = 0.0;
					while (recall < max_recall) {
						int incr_sz2 = 0; norm2 = 0.0;
						int second_c = first_c + incx1[second_incr], second_r = first_r + incy1[second_incr];
						if ((second_c > first_col) && (second_c < last_col) && (second_r > first_row) && (second_r < last_row)) {
							second_norm = amps.at<float>(second_r, second_c);
							int second_gap = 0;
							if (second_norm > bin_threshold) {
								incr_sz2++;
norm2 += second_norm;
							}
							else second_gap++;
							while (second_gap <= maxGAP) {
								second_c = second_c + incx1[second_incr]; second_r = second_r + incy1[second_incr];
								if ((second_c < first_col) || (second_c > last_col) || (second_r < first_row) || (second_r > last_row)) break;
								second_norm = amps.at<float>(second_r, second_c);
								if (second_norm > bin_threshold) {
									incr_sz2++;
									norm2 += second_norm;
								}
								else second_gap++;
							}
						}
						//if (max_sz_sum < incr_sz2 + prev_sz2 - recall) 
						if (max_norm2 < norm2 + prev_max_norm - recall_norm)
						{
							max_norm2 = norm2 + prev_max_norm - recall_norm;
							max_sz_sum = incr_sz2 + prev_sz2 - recall;
							max_incr2 = second_incr; max_incr2_sz = incr_sz2;
							max_r2s = first_r; max_c2s = first_c; max_r2 = second_r; max_c2 = second_c;
						}
						first_norm = amps.at<float>(first_r, first_c);
						if (first_norm > bin_threshold) {
							recall++; recall_norm += first_norm;
						}
						first_c = first_c - incx1[primary_incr1]; first_r = first_r - incy1[primary_incr1];
					}

					if (max_incr2_sz < 1) {
						line_ccw.push_back(cv::Point(primary_start_c, primary_start_r));
						line_ccw.push_back(cv::Point(primary_end_c, primary_end_r));
						break;    //碰到结束端点，前进停止
					}
					else {        //否则， 更新下一段
						line_ccw.push_back(cv::Point(primary_start_c, primary_start_r));
						line_ccw.push_back(cv::Point(max_c2s, max_r2s));
						primary_start_r = max_r2s; primary_start_c = max_c2s;
						primary_end_r = max_r2; primary_end_c = max_c2;
						primary_incr1 = max_incr2;
						contour_length += max_incr2_sz;
						prev_sz2 = max_incr2_sz;
						prev_max_norm = max_norm2;
						//std::cout << "Update next search direction - Primary incr1:" << primary_incr1 << ", new line -> startr:" << max_r2s << ", startc:" << max_c2s <<
						//	", endr:" << max_r2 << ", endc" << max_c2 << std::endl;
					}
					int second_r = max_r2s, second_c = max_c2s;
					while ((second_r != max_r2) || (second_c != max_c2)) {   //检查是否遇到回环， 防止循环
						second_c = second_c + incx1[max_incr2]; second_r = second_r + incy1[max_incr2];
						if (pix_used.at<float>(second_r, second_c) == 1) {
							line_ccw.push_back(cv::Point(primary_start_c, primary_start_r));
							line_ccw.push_back(cv::Point(second_c, second_r));
							circle_loop = true;
							//dead_end = true;
							break;
						}
						int init_r = r_start, init_c = c_start;
						while ((init_r != r_end) || (init_c != c_end)) {
							if ((second_r == init_r) && (second_c == init_c)) {
								line_ccw.push_back(cv::Point(primary_start_c, primary_start_r));
								line_ccw.push_back(cv::Point(init_c, init_r));
								circle_loop = true;
								break;
							}
							init_c = init_c + incx1[max_incr1]; init_r = init_r + incy1[max_incr1];
						}
						pix_used.at<float>(second_r, second_c) = 1;  //1: prev points in cur line. 2: already .
					}
					if (dead_end) break;
					if (circle_loop) break; //发现回路循环 
					max_recall = std::min(KM, max_incr2_sz);
				}
			}

			if (contour_length > minLength) {
				int init_r = r_start, init_c = c_start;
				while ((init_r != r_end) || (init_c != c_end)) {
					pix_used.at<float>(init_r, init_c) = 1;
					init_c = init_c + incx1[max_incr1]; init_r = init_r + incy1[max_incr1];
				}
				pix_used.at<float>(r_end, c_end) = 1;
				//if (circle_loop || dead_end) {
				int size_cw = line_cw.size(), size_ccw = line_ccw.size();
				if ((size_cw > 0) && (size_ccw > 0)) {
					std::vector<cv::Point> line_sequence; line_sequence.clear();

					for (int i = size_cw - 1; i >= 0; i--) line_sequence.push_back(line_cw[i]);
					//line_sequence.push_back(line_ccw[1]);
					//if(size_ccw > 2)
					for (int i = 0; i < size_ccw; i++) line_sequence.push_back(line_ccw[i]);
					contour_loop_set.push_back(line_sequence);
				}
				else if (size_cw > 0) contour_loop_set.push_back(line_cw);
				else if (size_ccw > 0) contour_loop_set.push_back(line_ccw);
			}
		}
	}
}

float ContourSelectInstance::get_sum_gray_vert(const cv::Mat& amps, int r, int c, int m, int n) {// | direction
	float sum_gray = 0.0;
	sum_gray += amps.at<float>(r, c) + amps.at<float>(r, c - 1) + amps.at<float>(r, c + 1);
	sum_gray += amps.at<float>(r - 1, c) + amps.at<float>(r - 1, c - 1) + amps.at<float>(r - 1, c + 1);
	sum_gray += amps.at<float>(r + 1, c) + amps.at<float>(r + 1, c - 1) + amps.at<float>(r + 1, c + 1);
	if (m == 0) {
		sum_gray += amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r - 2, c - 0);
		sum_gray += amps.at<float>(r - 3, c - 2) + amps.at<float>(r - 3, c - 3) + amps.at<float>(r - 3, c - 1);
	}
	else if (m == 1) {
		sum_gray += amps.at<float>(r - 2, c - 0) + amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 2, c + 1);
		sum_gray += amps.at<float>(r - 3, c - 0) + amps.at<float>(r - 3, c - 1) + amps.at<float>(r - 3, c + 1);
	}
	else if (m == 2) {
		sum_gray += amps.at<float>(r - 2, c + 1) + amps.at<float>(r - 2, c - 0) + amps.at<float>(r - 2, c + 2);
		sum_gray += amps.at<float>(r - 3, c + 2) + amps.at<float>(r - 3, c + 1) + amps.at<float>(r - 3, c + 3);
	}
	if (n == 0) {
		sum_gray += amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 2, c - 2) + amps.at<float>(r + 2, c - 0);
		sum_gray += amps.at<float>(r + 3, c - 2) + amps.at<float>(r + 3, c - 3) + amps.at<float>(r + 3, c - 1);
	}
	else if (n == 1) {
		sum_gray += amps.at<float>(r + 2, c - 0) + amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 2, c + 1);
		sum_gray += amps.at<float>(r + 3, c - 0) + amps.at<float>(r + 3, c - 1) + amps.at<float>(r + 3, c + 1);
	}
	else if (n == 2) {
		sum_gray += amps.at<float>(r + 2, c + 1) + amps.at<float>(r + 2, c - 0) + amps.at<float>(r + 2, c + 2);
		sum_gray += amps.at<float>(r + 3, c + 2) + amps.at<float>(r + 3, c + 1) + amps.at<float>(r + 3, c + 3);
	}
	return sum_gray;
}

float ContourSelectInstance::get_sum_gray_horz(const cv::Mat& amps, int r, int c, int m, int n) {  // --- direction
	float sum_gray = amps.at<float>(r, c) + amps.at<float>(r - 1, c) + amps.at<float>(r + 1, c);
	sum_gray += amps.at<float>(r, c-1) + amps.at<float>(r - 1, c-1) + amps.at<float>(r + 1, c-1);
	sum_gray += amps.at<float>(r, c+1) + amps.at<float>(r - 1, c+1) + amps.at<float>(r + 1, c+1);
	if ( m == 0 ) {
		sum_gray += amps.at<float>(r - 1, c - 2) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r, c - 2);
		sum_gray += amps.at<float>(r - 2, c - 3) + amps.at<float>(r - 3, c - 3) + amps.at<float>(r-1, c - 3);
	}
	else if (m == 1) {
		sum_gray += amps.at<float>(r - 0, c - 2) + amps.at<float>(r - 1, c - 2) + amps.at<float>(r + 1, c - 2);
		sum_gray += amps.at<float>(r - 0, c - 3) + amps.at<float>(r - 1, c - 3) + amps.at<float>(r + 1, c - 3);
	}	else if (m == 2) {
		sum_gray += amps.at<float>(r + 1, c - 2) + amps.at<float>(r + 2, c - 2) + amps.at<float>(r, c - 2);
		sum_gray += amps.at<float>(r + 2, c - 3) + amps.at<float>(r + 1, c - 3) + amps.at<float>(r + 3, c - 3);
	}
	if (n == 0) {
		sum_gray += amps.at<float>(r - 1, c + 2) + amps.at<float>(r - 2, c + 2) + amps.at<float>(r, c + 2);
		sum_gray += amps.at<float>(r - 2, c + 3) + amps.at<float>(r - 3, c + 3) + amps.at<float>(r - 1, c + 3);		
	} else if (n == 1) {
		sum_gray += amps.at<float>(r - 0, c + 2) + amps.at<float>(r - 1, c + 2) + amps.at<float>(r + 1, c + 2);
		sum_gray += amps.at<float>(r - 0, c + 3) + amps.at<float>(r - 1, c + 3) + amps.at<float>(r + 1, c + 3);
	} else if (n == 2) {
		sum_gray += amps.at<float>(r + 1, c + 2) + amps.at<float>(r + 2, c + 2) + amps.at<float>(r, c + 2);
		sum_gray += amps.at<float>(r + 2, c + 3) + amps.at<float>(r + 1, c + 3) + amps.at<float>(r + 3, c + 3);
	}
	return sum_gray;
}

float ContourSelectInstance::get_sum_gray_band(const cv::Mat& amps, int r, int c, int direction0, int direction1, int direction2) {  // --- direction
	float sum_gray = 0.0;
	int direction_pre = direction0, direction_next = direction2;
	if ((direction1 == 0) || (direction1 == 4)) {
		sum_gray += amps.at<float>(r, c) + amps.at<float>(r - 1, c) + amps.at<float>(r + 1, c);
		sum_gray += amps.at<float>(r, c - 1) + amps.at<float>(r - 1, c - 1) + amps.at<float>(r + 1, c - 1);
		sum_gray += amps.at<float>(r, c + 1) + amps.at<float>(r - 1, c + 1) + amps.at<float>(r + 1, c + 1);		
		if (direction1 == 4) { direction_pre = (direction2 + 4) % 8; direction_next = (direction0 + 4) % 8; }
		if (direction_pre == 1) {
			sum_gray += amps.at<float>(r - 1, c - 2) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r, c - 2);
			sum_gray += amps.at<float>(r - 2, c - 3) + amps.at<float>(r - 3, c - 3) + amps.at<float>(r - 1, c - 3);
		}
		else if (direction_pre == 0) {
			sum_gray += amps.at<float>(r - 0, c - 2) + amps.at<float>(r - 1, c - 2) + amps.at<float>(r + 1, c - 2);
			sum_gray += amps.at<float>(r - 0, c - 3) + amps.at<float>(r - 1, c - 3) + amps.at<float>(r + 1, c - 3);
		}
		else if (direction_pre == 7) {
			sum_gray += amps.at<float>(r + 1, c - 2) + amps.at<float>(r + 2, c - 2) + amps.at<float>(r, c - 2);
			sum_gray += amps.at<float>(r + 2, c - 3) + amps.at<float>(r + 1, c - 3) + amps.at<float>(r + 3, c - 3);
		}
		if (direction_next == 7) {
			sum_gray += amps.at<float>(r - 1, c + 2) + amps.at<float>(r - 2, c + 2) + amps.at<float>(r, c + 2);
			sum_gray += amps.at<float>(r - 2, c + 3) + amps.at<float>(r - 3, c + 3) + amps.at<float>(r - 1, c + 3);
		}
		else if (direction_next == 0) {
			sum_gray += amps.at<float>(r - 0, c + 2) + amps.at<float>(r - 1, c + 2) + amps.at<float>(r + 1, c + 2);
			sum_gray += amps.at<float>(r - 0, c + 3) + amps.at<float>(r - 1, c + 3) + amps.at<float>(r + 1, c + 3);
		}
		else if (direction_next == 1) {
			sum_gray += amps.at<float>(r + 1, c + 2) + amps.at<float>(r + 2, c + 2) + amps.at<float>(r, c + 2);
			sum_gray += amps.at<float>(r + 2, c + 3) + amps.at<float>(r + 1, c + 3) + amps.at<float>(r + 3, c + 3);
		}
	}	else if((direction1 == 2) || (direction1 == 6)) {
		if (direction1 == 6) { direction_pre = (direction2 - 4) % 8; direction_next = (direction0 - 4) % 8; }
		sum_gray += amps.at<float>(r, c) + amps.at<float>(r, c - 1) + amps.at<float>(r, c + 1);
		sum_gray += amps.at<float>(r - 1, c) + amps.at<float>(r - 1, c - 1) + amps.at<float>(r - 1, c + 1);
		sum_gray += amps.at<float>(r + 1, c) + amps.at<float>(r + 1, c - 1) + amps.at<float>(r + 1, c + 1);
		if (direction_pre == 1) {
			sum_gray += amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r - 2, c - 0);
			sum_gray += amps.at<float>(r - 3, c - 2) + amps.at<float>(r - 3, c - 3) + amps.at<float>(r - 3, c - 1);
		}
		else if (direction_pre == 2) {
			sum_gray += amps.at<float>(r - 2, c - 0) + amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 2, c + 1);
			sum_gray += amps.at<float>(r - 3, c - 0) + amps.at<float>(r - 3, c - 1) + amps.at<float>(r - 3, c + 1);
		}
		else if (direction_pre == 3) {
			sum_gray += amps.at<float>(r - 2, c + 1) + amps.at<float>(r - 2, c - 0) + amps.at<float>(r - 2, c + 2);
			sum_gray += amps.at<float>(r - 3, c + 2) + amps.at<float>(r - 3, c + 1) + amps.at<float>(r - 3, c + 3);
		}
		if (direction_next == 3) {
			sum_gray += amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 2, c - 2) + amps.at<float>(r + 2, c - 0);
			sum_gray += amps.at<float>(r + 3, c - 2) + amps.at<float>(r + 3, c - 3) + amps.at<float>(r + 3, c - 1);
		}
		else if (direction_next == 2) {
			sum_gray += amps.at<float>(r + 2, c - 0) + amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 2, c + 1);
			sum_gray += amps.at<float>(r + 3, c - 0) + amps.at<float>(r + 3, c - 1) + amps.at<float>(r + 3, c + 1);
		}
		else if (direction_next == 1) {
			sum_gray += amps.at<float>(r + 2, c + 1) + amps.at<float>(r + 2, c - 0) + amps.at<float>(r + 2, c + 2);
			sum_gray += amps.at<float>(r + 3, c + 2) + amps.at<float>(r + 3, c + 1) + amps.at<float>(r + 3, c + 3);
		}
	}
	else if ( (direction1 == 1) || (direction1 == 5) ) {
		if(direction1 == 5) { direction_pre = (direction2 - 4) % 8; direction_next = (direction0 - 4) % 8; }
		if (direction_pre == 0) { 
			if (direction_next == 0) {
				sum_gray += amps.at<float>(r-1,c-3)+amps.at<float>(r-1,c-2)+amps.at<float>(r-1,c-1)+amps.at<float>(r, c)+amps.at<float>(r+1,c+1)+amps.at<float>(r+1,c+2)+amps.at<float>(r+1,c+3);
				sum_gray += amps.at<float>(r-2,c-3)+amps.at<float>(r-2,c-2)+amps.at<float>(r-2,c-1)+amps.at<float>(r-1, c)+amps.at<float>(r,c+1)+amps.at<float>(r,c+2)+amps.at<float>(r,c+3);
				sum_gray += amps.at<float>(r-0,c-3)+amps.at<float>(r,c-2)+amps.at<float>(r,c-1)+amps.at<float>(r+1, c)+amps.at<float>(r+2,c+1)+amps.at<float>(r+2,c+2)+amps.at<float>(r+2,c+3);
			}
			else if(direction_next == 1) {
				sum_gray += amps.at<float>(r-1,c-3)+amps.at<float>(r-1,c-2)+amps.at<float>(r-1,c-1)+amps.at<float>(r, c)+amps.at<float>(r+1,c+1)+amps.at<float>(r+2,c+2)+amps.at<float>(r+3,c+3);
				sum_gray += amps.at<float>(r-2,c-3)+amps.at<float>(r-2,c-2)+amps.at<float>(r-2,c-1)+amps.at<float>(r-1, c)+amps.at<float>(r,c+1)+amps.at<float>(r+1,c+2)+amps.at<float>(r+2,c+3);
				sum_gray += amps.at<float>(r-0,c-3)+amps.at<float>(r,c-2)+amps.at<float>(r,c-1)+amps.at<float>(r+1, c)+amps.at<float>(r+2,c+1)+amps.at<float>(r+3,c+2)+amps.at<float>(r+4,c+3);
			}
			else if (direction_next == 2) {
				sum_gray += amps.at<float>(r-1,c-3)+amps.at<float>(r-1,c-2)+amps.at<float>(r-1,c-1)+amps.at<float>(r, c)+amps.at<float>(r+1,c+1)+amps.at<float>(r+2,c+1)+amps.at<float>(r+3,c+1);
				sum_gray += amps.at<float>(r-2,c-3)+amps.at<float>(r-2,c-2)+amps.at<float>(r-2,c-1)+amps.at<float>(r-1, c)+amps.at<float>(r,c+1)+
							amps.at<float>(r+1,c+2)+amps.at<float>(r+2,c+2)+amps.at<float>(r+3, c+2);
				sum_gray += amps.at<float>(r,c-3)+amps.at<float>(r,c-2)+amps.at<float>(r,c-1)+amps.at<float>(r+1,c)+amps.at<float>(r+2,c)+amps.at<float>(r+3, c);
			}
		} else if(direction_pre == 1) {
			if (direction_next == 0) {
				sum_gray += amps.at<float>(r-3,c-3)+amps.at<float>(r-2,c-2)+amps.at<float>(r-1,c-1)+amps.at<float>(r,c)+amps.at<float>(r+1,c+1)+amps.at<float>(r+1,c+2)+amps.at<float>(r+1,c+3);
				sum_gray += amps.at<float>(r-4,c-3)+amps.at<float>(r-3,c-2)+amps.at<float>(r-2,c-1)+amps.at<float>(r-1,c)+amps.at<float>(r,c+1)+amps.at<float>(r,c+2)+amps.at<float>(r,c+3);
				sum_gray += amps.at<float>(r-2,c-3)+amps.at<float>(r-1,c-2)+amps.at<float>(r-0,c-1)+amps.at<float>(r+1,c)+amps.at<float>(r+2,c+1)+amps.at<float>(r+2,c+2)+amps.at<float>(r+2,c+3);
			}
			else if (direction_next == 1) {
				sum_gray += amps.at<float>(r-3, c-3)+amps.at<float>(r-2, c-2)+amps.at<float>(r-1, c-1)+amps.at<float>(r, c)+amps.at<float>(r+1, c+1)+amps.at<float>(r+2, c+2)+amps.at<float>(r+3, c+3);
				sum_gray += amps.at<float>(r-2, c-3)+amps.at<float>(r-1, c-2)+amps.at<float>(r-0, c-1)+amps.at<float>(r+1, c)+amps.at<float>(r+2, c+1)+amps.at<float>(r+3, c+2)+amps.at<float>(r+4, c+3);
				sum_gray += amps.at<float>(r-4, c-3)+amps.at<float>(r-3, c-2)+amps.at<float>(r-2, c-1)+amps.at<float>(r-1, c)+amps.at<float>(r, c+1)+amps.at<float>(r+1, c+2)+amps.at<float>(r+2, c+3);
			}
			else if (direction_next == 2) {
				sum_gray += amps.at<float>(r-3, c-3)+amps.at<float>(r-2, c-2)+amps.at<float>(r-1, c-1)+amps.at<float>(r, c)+amps.at<float>(r+1, c+1)+amps.at<float>(r+2,c+1) + amps.at<float>(r+3, c+1);
				sum_gray += amps.at<float>(r-3, c-4)+amps.at<float>(r-2, c-3)+amps.at<float>(r-1, c-2)+amps.at<float>(r, c-1)+amps.at<float>(r+1, c)+amps.at<float>(r+2,c) + amps.at<float>(r+3, c);
				sum_gray += amps.at<float>(r-3, c-2)+amps.at<float>(r-2, c-1)+amps.at<float>(r-1, c)+amps.at<float>(r, c+1)+amps.at<float>(r+1, c+2)+amps.at<float>(r+2,c+2) + amps.at<float>(r+3, c+2);
			}
		}
		else if(direction_pre == 2) {
			if (direction_next == 0) {
				sum_gray += amps.at<float>(r-3, c-1)+amps.at<float>(r-2,c-1)+amps.at<float>(r-1,c-1)+amps.at<float>(r,c)+amps.at<float>(r+1,c+1)+amps.at<float>(r+1,c+2)+amps.at<float>(r+1, c+3);
				sum_gray += amps.at<float>(r-3, c-2)+amps.at<float>(r-2,c-2)+amps.at<float>(r-1,c-2)+amps.at<float>(r,c-1)+amps.at<float>(r+1,c)+
							amps.at<float>(r + 2, c + 1) + amps.at<float>(r+2,c+2)+amps.at<float>(r+2, c+3);
				sum_gray += amps.at<float>(r-3, c)+amps.at<float>(r-2,c)+amps.at<float>(r-1,c)+
							amps.at<float>(r,c+1)+amps.at<float>(r,c+2)+amps.at<float>(r, c+3);
			}
			else if (direction_next == 1) {
				sum_gray += amps.at<float>(r-3, c-1)+amps.at<float>(r-2, c-1)+amps.at<float>(r-1, c-1)+amps.at<float>(r, c)+amps.at<float>(r+1, c+1)+amps.at<float>(r+2, c+2)+amps.at<float>(r+3, c+3);
				sum_gray += amps.at<float>(r-3, c-2)+amps.at<float>(r-2, c-2)+amps.at<float>(r-1, c-2)+amps.at<float>(r, c-1)+amps.at<float>(r+1, c)+amps.at<float>(r+2, c+1)+amps.at<float>(r+3, c+2);
				sum_gray += amps.at<float>(r-3, c)+amps.at<float>(r-2, c)+amps.at<float>(r-1, c)+amps.at<float>(r, c+1)+amps.at<float>(r+1, c+2)+amps.at<float>(r+2, c+3)+amps.at<float>(r+3, c+4);
			}
			else if (direction_next == 2) {
				sum_gray += amps.at<float>(r-3, c-1)+amps.at<float>(r-2, c-1)+amps.at<float>(r-1,c-1)+amps.at<float>(r,c)+amps.at<float>(r+1, c+1)+amps.at<float>(r+2, c+1)+amps.at<float>(r + 3, c + 1);
				sum_gray += amps.at<float>(r - 3, c - 2) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r - 1, c - 2) + amps.at<float>(r, c - 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c) + amps.at<float>(r + 3, c);
				sum_gray += amps.at<float>(r - 3, c) + amps.at<float>(r - 2, c) + amps.at<float>(r - 1, c) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c + 2) + amps.at<float>(r + 2, c + 2) + amps.at<float>(r + 3, c + 2);
			}
		}
	}
	else if ((direction1 == 3) || (direction1 == 7)) {
		if(direction1 == 7) { direction_pre = (direction2 - 4) % 8; direction_next = (direction0 - 4) % 8;}
		if(direction_pre == 2) {
			if (direction_next == 2) {
				sum_gray += amps.at<float>(r-3, c+1) + amps.at<float>(r-2, c+1)+amps.at<float>(r-1,c+1)+amps.at<float>(r,c)+amps.at<float>(r+1, c-1)+amps.at<float>(r+2, c-1)+amps.at<float>(r+3, c-1);
				sum_gray += amps.at<float>(r-3, c)+amps.at<float>(r-2, c)+amps.at<float>(r - 1, c)+amps.at<float>(r, c-1)+amps.at<float>(r+1,c-2)+amps.at<float>(r+2,c-2)+amps.at<float>(r+3, c-2);
				sum_gray += amps.at<float>(r-3, c+2) + amps.at<float>(r-2, c+2)+amps.at<float>(r-1,c+2)+amps.at<float>(r,c+1)+amps.at<float>(r+1, c)+amps.at<float>(r+2, c)+amps.at<float>(r+3, c);
			}
			else if (direction_next == 3) {
				sum_gray += amps.at<float>(r-3, c+1) + amps.at<float>(r-2, c+1)+amps.at<float>(r-1,c+1)+amps.at<float>(r,c)+amps.at<float>(r+1, c-1)+amps.at<float>(r+2, c-2)+amps.at<float>(r+3, c-3);
				sum_gray += amps.at<float>(r-3, c) + amps.at<float>(r-2, c)+amps.at<float>(r-1,c)+amps.at<float>(r,c-1)+amps.at<float>(r+1, c-2)+amps.at<float>(r+2, c-3)+amps.at<float>(r+3, c-4);
				sum_gray += amps.at<float>(r-3, c+2) + amps.at<float>(r-2, c+2)+amps.at<float>(r-1,c+2)+amps.at<float>(r,c+1)+amps.at<float>(r+1, c)+amps.at<float>(r+2, c-1)+amps.at<float>(r+3, c-2);
			}
			else if (direction_next == 4) {
				sum_gray += amps.at<float>(r-3, c+1) + amps.at<float>(r-2, c+1)+amps.at<float>(r-1,c+1)+amps.at<float>(r,c)+amps.at<float>(r+1, c-1)+amps.at<float>(r+1, c-2)+amps.at<float>(r+1, c-3);
				sum_gray += amps.at<float>(r-3, c+2) + amps.at<float>(r-2, c+2)+amps.at<float>(r-1,c+2) + amps.at<float>(r+2, c-1)+amps.at<float>(r+2, c-2)+amps.at<float>(r+2, c-3)+
							amps.at<float>(r+1,c) + amps.at<float>(r,c+1);
				sum_gray += amps.at<float>(r-3, c) + amps.at<float>(r-2, c)+amps.at<float>(r-1,c) + amps.at<float>(r, c-1)+amps.at<float>(r, c-2)+amps.at<float>(r, c-3);
			}
		}	else if(direction_pre == 3) {
			if (direction_next == 2) {
				sum_gray += amps.at<float>(r-3, c+3) + amps.at<float>(r-2, c+2)+amps.at<float>(r-1,c+1)+amps.at<float>(r,c)+amps.at<float>(r+1, c-1)+amps.at<float>(r+2, c-1)+amps.at<float>(r+3, c-1);
				sum_gray += amps.at<float>(r-3, c+2) + amps.at<float>(r-2, c+1)+amps.at<float>(r-1,c)+amps.at<float>(r,c-1)+amps.at<float>(r+1, c-2)+amps.at<float>(r+2, c-2)+amps.at<float>(r+3, c-2);
				sum_gray += amps.at<float>(r-3, c+4) + amps.at<float>(r-2, c+3)+amps.at<float>(r-1,c+2)+amps.at<float>(r,c+1)+amps.at<float>(r+1, c)+amps.at<float>(r+2, c)+amps.at<float>(r+3, c);
			}
			else if (direction_next == 3) {
				sum_gray += amps.at<float>(r-3, c+3) + amps.at<float>(r-2, c+2)+amps.at<float>(r-1,c+1)+amps.at<float>(r,c)+amps.at<float>(r+1, c-1)+amps.at<float>(r+2, c-2)+amps.at<float>(r+3, c-3);
				sum_gray += amps.at<float>(r-2, c+3) + amps.at<float>(r-1, c+2)+amps.at<float>(r,c+1)+amps.at<float>(r+1,c)+amps.at<float>(r+2, c-1)+amps.at<float>(r+3, c-2)+amps.at<float>(r+4, c-3);
				sum_gray += amps.at<float>(r-4, c+3) + amps.at<float>(r-3, c+2)+amps.at<float>(r-2,c+1)+amps.at<float>(r-1,c)+amps.at<float>(r, c-1)+amps.at<float>(r+1, c-2)+amps.at<float>(r+2, c-3);
			}
			else if (direction_next == 4) {				
				sum_gray += amps.at<float>(r-3, c+3) + amps.at<float>(r-2, c+2)+amps.at<float>(r-1,c+1)+amps.at<float>(r,c)+amps.at<float>(r+1, c-1)+amps.at<float>(r+1, c-2)+amps.at<float>(r+1, c-3);
				sum_gray += amps.at<float>(r-2, c+3) + amps.at<float>(r-1, c+2)+amps.at<float>(r,c+1)+amps.at<float>(r+1,c)+amps.at<float>(r+2, c-1)+amps.at<float>(r+2, c-2)+amps.at<float>(r+2, c-3);
				sum_gray += amps.at<float>(r-4, c+3) + amps.at<float>(r-3, c+2)+amps.at<float>(r-2,c+1)+amps.at<float>(r-1,c)+amps.at<float>(r, c-1)+amps.at<float>(r, c-2)+amps.at<float>(r, c-3);
			}
		}
		if (direction_pre == 4) {
			if (direction_next == 2) {
				sum_gray += amps.at<float>(r-1, c+3) + amps.at<float>(r-1, c+2)+amps.at<float>(r-1,c+1)+amps.at<float>(r,c)+amps.at<float>(r+1, c-1)+amps.at<float>(r+2, c-1)+amps.at<float>(r+3, c-1);
				sum_gray += amps.at<float>(r-2, c+3) + amps.at<float>(r-2, c+2)+amps.at<float>(r-2,c+1) + amps.at<float>(r+1, c-2)+amps.at<float>(r+2, c-2)+amps.at<float>(r+3, c-2) +
							amps.at<float>(r-1, c) + amps.at<float>(r, c-1);
				sum_gray += amps.at<float>(r, c+3) + amps.at<float>(r, c+2)+ amps.at<float>(r,c+1) + amps.at<float>(r+1, c)+amps.at<float>(r+2, c)+amps.at<float>(r+3, c);
			}
			else if (direction_next == 3) {
				sum_gray += amps.at<float>(r-1, c+3) + amps.at<float>(r-1, c+2)+amps.at<float>(r-1,c+1)+amps.at<float>(r,c)+amps.at<float>(r+1, c-1)+amps.at<float>(r+2, c-2)+amps.at<float>(r+3, c-3);
				sum_gray += amps.at<float>(r, c+3) + amps.at<float>(r, c+2)+amps.at<float>(r,c+1)+amps.at<float>(r+1,c)+amps.at<float>(r+2, c-1)+amps.at<float>(r+3, c-2)+amps.at<float>(r+4, c-3);
				sum_gray += amps.at<float>(r-2, c+3) + amps.at<float>(r-2, c+2)+amps.at<float>(r-2,c+1)+amps.at<float>(r-1,c)+amps.at<float>(r, c-1)+amps.at<float>(r+1, c-2)+amps.at<float>(r+2, c-3);
			}
			else if (direction_next == 4) {
				sum_gray += amps.at<float>(r-1, c+3) + amps.at<float>(r-1, c+2)+amps.at<float>(r-1,c+1)+amps.at<float>(r,c)+amps.at<float>(r+1, c-1)+amps.at<float>(r+1, c-2)+amps.at<float>(r+1, c-3);
				sum_gray += amps.at<float>(r, c+3) + amps.at<float>(r, c+2)+amps.at<float>(r,c+1)+amps.at<float>(r+1,c)+amps.at<float>(r+2, c-1)+amps.at<float>(r+2, c-2)+amps.at<float>(r+2, c-3);
				sum_gray += amps.at<float>(r-2, c+3) + amps.at<float>(r-2, c+2)+amps.at<float>(r-2,c+1)+amps.at<float>(r-1,c)+amps.at<float>(r, c-1)+amps.at<float>(r, c-2)+amps.at<float>(r, c-3);
			}
		}
	}

	return sum_gray;
}

float ContourSelectInstance::get_sum_gray_gauss(const cv::Mat& amps, int r, int c, int direction0, int direction1, int direction2) {  // --- direction
	float sum_gray = 0.0;
	int direction_pre = direction0, direction_next = direction2;
	if ((direction1 == 0) || (direction1 == 4)) {
		sum_gray += amps.at<float>(r, c) +amps.at<float>(r, c - 1)  +amps.at<float>(r, c + 1)   ;
		sum_gray += (amps.at<float>(r - 1, c)+ amps.at<float>(r - 1, c - 1) + amps.at<float>(r - 1, c + 1)) / 2;
		sum_gray += (amps.at<float>(r + 1, c)+ amps.at<float>(r + 1, c - 1) + amps.at<float>(r + 1, c + 1)) / 2;
		if (direction1 == 4) { direction_pre = (direction2 + 4) % 8; direction_next = (direction0 + 4) % 8; }
		if (direction_pre == 1) {
			sum_gray += amps.at<float>(r - 1, c - 2) + amps.at<float>(r - 2, c - 3);
			sum_gray += (amps.at<float>(r - 2, c - 2) + amps.at<float>(r, c - 2) + amps.at<float>(r - 3, c - 3) + amps.at<float>(r - 1, c - 3)) / 2;
		}
		else if (direction_pre == 0) {
			sum_gray += amps.at<float>(r - 0, c - 2) + amps.at<float>(r - 0, c - 3);
			sum_gray += (amps.at<float>(r - 1, c - 2) + amps.at<float>(r + 1, c - 2) + amps.at<float>(r - 1, c - 3) + amps.at<float>(r + 1, c - 3)) / 2;
		}
		else if (direction_pre == 7) {
			sum_gray += amps.at<float>(r + 1, c - 2) + amps.at<float>(r + 2, c - 3);
			sum_gray += (amps.at<float>(r + 2, c - 2) + amps.at<float>(r, c - 2) + amps.at<float>(r + 1, c - 3) + amps.at<float>(r + 3, c - 3)) / 2;
		}
		if (direction_next == 7) {
			sum_gray += amps.at<float>(r - 1, c + 2) + amps.at<float>(r - 2, c + 3);
			sum_gray += (amps.at<float>(r - 2, c + 2) + amps.at<float>(r, c + 2) + amps.at<float>(r - 3, c + 3) + amps.at<float>(r - 1, c + 3)) / 2;
		}
		else if (direction_next == 0) {
			sum_gray += amps.at<float>(r - 0, c + 2) + amps.at<float>(r - 0, c + 3);
			sum_gray += (amps.at<float>(r - 1, c + 2) + amps.at<float>(r + 1, c + 2) + amps.at<float>(r - 1, c + 3) + amps.at<float>(r + 1, c + 3)) / 2;
		}
		else if (direction_next == 1) {
			sum_gray += amps.at<float>(r + 1, c + 2) + amps.at<float>(r + 2, c + 3);
			sum_gray += (amps.at<float>(r + 2, c + 2) + amps.at<float>(r, c + 2) + amps.at<float>(r + 1, c + 3) + amps.at<float>(r + 3, c + 3)) / 2;
		}
	}
	else if ((direction1 == 2) || (direction1 == 6)) {
		if (direction1 == 6) { direction_pre = (direction2 - 4) % 8; direction_next = (direction0 - 4) % 8; }
		sum_gray += amps.at<float>(r, c) + amps.at<float>(r - 1, c) + amps.at<float>(r + 1, c);
		sum_gray += (amps.at<float>(r, c - 1) + amps.at<float>(r - 1, c - 1) + amps.at<float>(r + 1, c - 1)) / 2;
		sum_gray += (amps.at<float>(r, c + 1) + amps.at<float>(r - 1, c + 1) + amps.at<float>(r + 1, c + 1)) / 2;
		if (direction_pre == 1) {
			sum_gray += amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 3, c - 2);
			sum_gray += (amps.at<float>(r - 2, c - 2) + amps.at<float>(r - 2, c - 0) + amps.at<float>(r - 3, c - 3) + amps.at<float>(r - 3, c - 1)) / 2;
		}
		else if (direction_pre == 2) {
			sum_gray += amps.at<float>(r - 2, c - 0) + amps.at<float>(r - 3, c - 0) ;
			sum_gray += (amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 2, c + 1)+ amps.at<float>(r - 3, c - 1) + amps.at<float>(r - 3, c + 1)) / 2;
		}
		else if (direction_pre == 3) {
			sum_gray += amps.at<float>(r - 2, c + 1) + amps.at<float>(r - 2, c - 0) + amps.at<float>(r - 2, c + 2);
			sum_gray += amps.at<float>(r - 3, c + 2) + amps.at<float>(r - 3, c + 1) + amps.at<float>(r - 3, c + 3);
		}
		if (direction_next == 3) {
			sum_gray += amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 3, c - 2);
			sum_gray += (amps.at<float>(r + 2, c - 2) + amps.at<float>(r + 2, c - 0) + amps.at<float>(r + 3, c - 3) + amps.at<float>(r + 3, c - 1)) / 2;
		}
		else if (direction_next == 2) {
			sum_gray += amps.at<float>(r + 2, c - 0) + amps.at<float>(r + 3, c - 0);
			sum_gray += (amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 2, c + 1) + amps.at<float>(r + 3, c - 1) + amps.at<float>(r + 3, c + 1)) / 2;
		}
		else if (direction_next == 1) {
			sum_gray += amps.at<float>(r + 2, c + 1) + amps.at<float>(r + 3, c + 2);
			sum_gray += (amps.at<float>(r + 2, c - 0) + amps.at<float>(r + 2, c + 2) + amps.at<float>(r + 3, c + 1) + amps.at<float>(r + 3, c + 3)) / 2;
		}
	}
	else if ((direction1 == 1) || (direction1 == 5)) {
		if (direction1 == 5) { direction_pre = (direction2 - 4) % 8; direction_next = (direction0 - 4) % 8; }
		if (direction_pre == 0) {
			if (direction_next == 0) {
				sum_gray += amps.at<float>(r - 1, c - 3) + amps.at<float>(r - 1, c - 2) + amps.at<float>(r - 1, c - 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c + 1) + amps.at<float>(r + 1, c + 2) + amps.at<float>(r + 1, c + 3);
				sum_gray += (amps.at<float>(r - 2, c - 3) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 1, c) + amps.at<float>(r, c + 1) + amps.at<float>(r, c + 2) + amps.at<float>(r, c + 3)) / 2;
				sum_gray += (amps.at<float>(r - 0, c - 3) + amps.at<float>(r, c - 2) + amps.at<float>(r, c - 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c + 1) + amps.at<float>(r + 2, c + 2) + amps.at<float>(r + 2, c + 3)) / 2;
			}
			else if (direction_next == 1) {
				sum_gray += amps.at<float>(r - 1, c - 3) + amps.at<float>(r - 1, c - 2) + amps.at<float>(r - 1, c - 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c + 1) + amps.at<float>(r + 2, c + 2) + amps.at<float>(r + 3, c + 3);
				sum_gray += (amps.at<float>(r - 2, c - 3) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 1, c) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c + 2) + amps.at<float>(r + 2, c + 3)) / 2;
				sum_gray += (amps.at<float>(r - 0, c - 3) + amps.at<float>(r, c - 2) + amps.at<float>(r, c - 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c + 1) + amps.at<float>(r + 3, c + 2) + amps.at<float>(r + 4, c + 3)) / 2;
			}
			else if (direction_next == 2) {
				sum_gray += amps.at<float>(r - 1, c - 3) + amps.at<float>(r - 1, c - 2) + amps.at<float>(r - 1, c - 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c + 1) + amps.at<float>(r + 2, c + 1) + amps.at<float>(r + 3, c + 1);
				sum_gray += (amps.at<float>(r - 2, c - 3) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 1, c) + amps.at<float>(r, c + 1) +
					amps.at<float>(r + 1, c + 2) + amps.at<float>(r + 2, c + 2) + amps.at<float>(r + 3, c + 2)) / 2;
				sum_gray += (amps.at<float>(r, c - 3) + amps.at<float>(r, c - 2) + amps.at<float>(r, c - 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c) + amps.at<float>(r + 3, c)) / 2;
			}
		}
		else if (direction_pre == 1) {
			if (direction_next == 0) {
				sum_gray += amps.at<float>(r - 3, c - 3) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r - 1, c - 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c + 1) + amps.at<float>(r + 1, c + 2) + amps.at<float>(r + 1, c + 3);
				sum_gray += (amps.at<float>(r - 4, c - 3) + amps.at<float>(r - 3, c - 2) + amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 1, c) + amps.at<float>(r, c + 1) + amps.at<float>(r, c + 2) + amps.at<float>(r, c + 3)) / 2;
				sum_gray += (amps.at<float>(r - 2, c - 3) + amps.at<float>(r - 1, c - 2) + amps.at<float>(r - 0, c - 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c + 1) + amps.at<float>(r + 2, c + 2) + amps.at<float>(r + 2, c + 3)) / 2;
			}
			else if (direction_next == 1) {
				sum_gray += amps.at<float>(r - 3, c - 3) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r - 1, c - 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c + 1) + amps.at<float>(r + 2, c + 2) + amps.at<float>(r + 3, c + 3);
				sum_gray += (amps.at<float>(r - 2, c - 3) + amps.at<float>(r - 1, c - 2) + amps.at<float>(r - 0, c - 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c + 1) + amps.at<float>(r + 3, c + 2) + amps.at<float>(r + 4, c + 3)) / 2;
				sum_gray += (amps.at<float>(r - 4, c - 3) + amps.at<float>(r - 3, c - 2) + amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 1, c) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c + 2) + amps.at<float>(r + 2, c + 3)) / 2;
			}
			else if (direction_next == 2) {
				sum_gray += amps.at<float>(r - 3, c - 3) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r - 1, c - 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c + 1) + amps.at<float>(r + 2, c + 1) + amps.at<float>(r + 3, c + 1);
				sum_gray += (amps.at<float>(r - 3, c - 4) + amps.at<float>(r - 2, c - 3) + amps.at<float>(r - 1, c - 2) + amps.at<float>(r, c - 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c) + amps.at<float>(r + 3, c)) / 2;
				sum_gray += (amps.at<float>(r - 3, c - 2) + amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 1, c) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c + 2) + amps.at<float>(r + 2, c + 2) + amps.at<float>(r + 3, c + 2)) / 2;
			}
		}
		else if (direction_pre == 2) {
			if (direction_next == 0) {
				sum_gray += amps.at<float>(r - 3, c - 1) + amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 1, c - 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c + 1) + amps.at<float>(r + 1, c + 2) + amps.at<float>(r + 1, c + 3);
				sum_gray += (amps.at<float>(r - 3, c - 2) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r - 1, c - 2) + amps.at<float>(r, c - 1) + amps.at<float>(r + 1, c) +
					amps.at<float>(r + 2, c + 1) + amps.at<float>(r + 2, c + 2) + amps.at<float>(r + 2, c + 3)) / 2;
				sum_gray += (amps.at<float>(r - 3, c) + amps.at<float>(r - 2, c) + amps.at<float>(r - 1, c) +
					amps.at<float>(r, c + 1) + amps.at<float>(r, c + 2) + amps.at<float>(r, c + 3)) / 2;
			}
			else if (direction_next == 1) {
				sum_gray += amps.at<float>(r - 3, c - 1) + amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 1, c - 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c + 1) + amps.at<float>(r + 2, c + 2) + amps.at<float>(r + 3, c + 3);
				sum_gray += (amps.at<float>(r - 3, c - 2) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r - 1, c - 2) + amps.at<float>(r, c - 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c + 1) + amps.at<float>(r + 3, c + 2)) / 2;
				sum_gray += (amps.at<float>(r - 3, c) + amps.at<float>(r - 2, c) + amps.at<float>(r - 1, c) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c + 2) + amps.at<float>(r + 2, c + 3) + amps.at<float>(r + 3, c + 4)) / 2;
			}
			else if (direction_next == 2) {
				sum_gray += amps.at<float>(r - 3, c - 1) + amps.at<float>(r - 2, c - 1) + amps.at<float>(r - 1, c - 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c + 1) + amps.at<float>(r + 2, c + 1) + amps.at<float>(r + 3, c + 1);
				sum_gray += (amps.at<float>(r - 3, c - 2) + amps.at<float>(r - 2, c - 2) + amps.at<float>(r - 1, c - 2) + amps.at<float>(r, c - 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c) + amps.at<float>(r + 3, c)) / 2;
				sum_gray += (amps.at<float>(r - 3, c) + amps.at<float>(r - 2, c) + amps.at<float>(r - 1, c) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c + 2) + amps.at<float>(r + 2, c + 2) + amps.at<float>(r + 3, c + 2)) / 2;
			}
		}
	}
	else if ((direction1 == 3) || (direction1 == 7)) {
		if (direction1 == 7) { direction_pre = (direction2 - 4) % 8; direction_next = (direction0 - 4) % 8; }
		if (direction_pre == 2) {
			if (direction_next == 2) {
				sum_gray += amps.at<float>(r - 3, c + 1) + amps.at<float>(r - 2, c + 1) + amps.at<float>(r - 1, c + 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c - 1) + amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 3, c - 1);
				sum_gray += (amps.at<float>(r - 3, c) + amps.at<float>(r - 2, c) + amps.at<float>(r - 1, c) + amps.at<float>(r, c - 1) + amps.at<float>(r + 1, c - 2) + amps.at<float>(r + 2, c - 2) + amps.at<float>(r + 3, c - 2)) / 2;
				sum_gray += (amps.at<float>(r - 3, c + 2) + amps.at<float>(r - 2, c + 2) + amps.at<float>(r - 1, c + 2) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c) + amps.at<float>(r + 3, c)) / 2;
			}
			else if (direction_next == 3) {
				sum_gray += amps.at<float>(r - 3, c + 1) + amps.at<float>(r - 2, c + 1) + amps.at<float>(r - 1, c + 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c - 1) + amps.at<float>(r + 2, c - 2) + amps.at<float>(r + 3, c - 3);
				sum_gray += (amps.at<float>(r - 3, c) + amps.at<float>(r - 2, c) + amps.at<float>(r - 1, c) + amps.at<float>(r, c - 1) + amps.at<float>(r + 1, c - 2) + amps.at<float>(r + 2, c - 3) + amps.at<float>(r + 3, c - 4)) /2;
				sum_gray += (amps.at<float>(r - 3, c + 2) + amps.at<float>(r - 2, c + 2) + amps.at<float>(r - 1, c + 2) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 3, c - 2)) / 2;
			}
			else if (direction_next == 4) {
				sum_gray += amps.at<float>(r - 3, c + 1) + amps.at<float>(r - 2, c + 1) + amps.at<float>(r - 1, c + 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c - 1) + amps.at<float>(r + 1, c - 2) + amps.at<float>(r + 1, c - 3);
				sum_gray += (amps.at<float>(r - 3, c + 2) + amps.at<float>(r - 2, c + 2) + amps.at<float>(r - 1, c + 2) + amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 2, c - 2) + amps.at<float>(r + 2, c - 3) +
					amps.at<float>(r + 1, c) + amps.at<float>(r, c + 1)) / 2;
				sum_gray += (amps.at<float>(r - 3, c) + amps.at<float>(r - 2, c) + amps.at<float>(r - 1, c) + amps.at<float>(r, c - 1) + amps.at<float>(r, c - 2) + amps.at<float>(r, c - 3)) / 2;
			}
		}
		else if (direction_pre == 3) {
			if (direction_next == 2) {
				sum_gray += amps.at<float>(r - 3, c + 3) + amps.at<float>(r - 2, c + 2) + amps.at<float>(r - 1, c + 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c - 1) + amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 3, c - 1);
				sum_gray += (amps.at<float>(r - 3, c + 2) + amps.at<float>(r - 2, c + 1) + amps.at<float>(r - 1, c) + amps.at<float>(r, c - 1) + amps.at<float>(r + 1, c - 2) + amps.at<float>(r + 2, c - 2) + amps.at<float>(r + 3, c - 2)) / 2;
				sum_gray += (amps.at<float>(r - 3, c + 4) + amps.at<float>(r - 2, c + 3) + amps.at<float>(r - 1, c + 2) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c) + amps.at<float>(r + 3, c)) / 2;
			}
			else if (direction_next == 3) {
				sum_gray += amps.at<float>(r - 3, c + 3) + amps.at<float>(r - 2, c + 2) + amps.at<float>(r - 1, c + 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c - 1) + amps.at<float>(r + 2, c - 2) + amps.at<float>(r + 3, c - 3);
				sum_gray += (amps.at<float>(r - 2, c + 3) + amps.at<float>(r - 1, c + 2) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 3, c - 2) + amps.at<float>(r + 4, c - 3)) / 2;
				sum_gray += (amps.at<float>(r - 4, c + 3) + amps.at<float>(r - 3, c + 2) + amps.at<float>(r - 2, c + 1) + amps.at<float>(r - 1, c) + amps.at<float>(r, c - 1) + amps.at<float>(r + 1, c - 2) + amps.at<float>(r + 2, c - 3)) / 2;
			}
			else if (direction_next == 4) {
				sum_gray += amps.at<float>(r - 3, c + 3) + amps.at<float>(r - 2, c + 2) + amps.at<float>(r - 1, c + 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c - 1) + amps.at<float>(r + 1, c - 2) + amps.at<float>(r + 1, c - 3);
				sum_gray += (amps.at<float>(r - 2, c + 3) + amps.at<float>(r - 1, c + 2) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 2, c - 2) + amps.at<float>(r + 2, c - 3)) / 2;
				sum_gray += (amps.at<float>(r - 4, c + 3) + amps.at<float>(r - 3, c + 2) + amps.at<float>(r - 2, c + 1) + amps.at<float>(r - 1, c) + amps.at<float>(r, c - 1) + amps.at<float>(r, c - 2) + amps.at<float>(r, c - 3)) / 2;
			}
		}
		if (direction_pre == 4) {
			if (direction_next == 2) {
				sum_gray += amps.at<float>(r - 1, c + 3) + amps.at<float>(r - 1, c + 2) + amps.at<float>(r - 1, c + 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c - 1) + amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 3, c - 1);
				sum_gray += (amps.at<float>(r - 2, c + 3) + amps.at<float>(r - 2, c + 2) + amps.at<float>(r - 2, c + 1) + amps.at<float>(r + 1, c - 2) + amps.at<float>(r + 2, c - 2) + amps.at<float>(r + 3, c - 2) +
					amps.at<float>(r - 1, c) + amps.at<float>(r, c - 1)) / 2;
				sum_gray += (amps.at<float>(r, c + 3) + amps.at<float>(r, c + 2) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c) + amps.at<float>(r + 3, c)) / 2;
			}
			else if (direction_next == 3) {
				sum_gray += amps.at<float>(r - 1, c + 3) + amps.at<float>(r - 1, c + 2) + amps.at<float>(r - 1, c + 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c - 1) + amps.at<float>(r + 2, c - 2) + amps.at<float>(r + 3, c - 3);
				sum_gray += (amps.at<float>(r, c + 3) + amps.at<float>(r, c + 2) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 3, c - 2) + amps.at<float>(r + 4, c - 3)) / 2;
				sum_gray += (amps.at<float>(r - 2, c + 3) + amps.at<float>(r - 2, c + 2) + amps.at<float>(r - 2, c + 1) + amps.at<float>(r - 1, c) + amps.at<float>(r, c - 1) + amps.at<float>(r + 1, c - 2) + amps.at<float>(r + 2, c - 3)) / 2;
			}
			else if (direction_next == 4) {
				sum_gray += amps.at<float>(r - 1, c + 3) + amps.at<float>(r - 1, c + 2) + amps.at<float>(r - 1, c + 1) + amps.at<float>(r, c) + amps.at<float>(r + 1, c - 1) + amps.at<float>(r + 1, c - 2) + amps.at<float>(r + 1, c - 3);
				sum_gray += (amps.at<float>(r, c + 3) + amps.at<float>(r, c + 2) + amps.at<float>(r, c + 1) + amps.at<float>(r + 1, c) + amps.at<float>(r + 2, c - 1) + amps.at<float>(r + 2, c - 2) + amps.at<float>(r + 2, c - 3)) / 2;
				sum_gray += (amps.at<float>(r - 2, c + 3) + amps.at<float>(r - 2, c + 2) + amps.at<float>(r - 2, c + 1) + amps.at<float>(r - 1, c) + amps.at<float>(r, c - 1) + amps.at<float>(r, c - 2) + amps.at<float>(r, c - 3)) / 2;
			}
		}
	}

	return sum_gray;
}

bool ContourSelectInstance::extend_band_double(const cv::Mat& amps, struct pt_struct* init_pts, int zero_offset, bool rev_enable, cv::Mat& out_thva)//选择一个方向扩展线段
{
	int height = amps.rows, width = amps.cols;
	cv::Size processSize = cv::Size(amps.cols, amps.rows);
	cv::Mat pix_band = cv::Mat::zeros(processSize, CV_32S);

	memset(cur_pts, 0, sizeof(struct pt_struct) * MAX_PT_POOL);

	const long first_row = std::max(M / 2, (int)ceil(roi_y_min * height));
	const long first_col = std::max(N / 2, (int)ceil(roi_x_min * width));
	const long last_row = std::min(height - M / 2, (int)ceil(roi_y_max * height));
	const long last_col = std::min(width - N / 2, (int)ceil(roi_x_max * width));

	bool _ret = false;
	if(!rev_enable) {
		cur_pts[0].px = init_pts[zero_offset].px; cur_pts[0].py = init_pts[zero_offset].py; cur_pts[0].direction = init_pts[zero_offset].direction;
		cur_pts[0].size = 3; memset(cur_pts[0].mask, 0, 8 * sizeof(bool)); cur_pts[0].pixel_va = 0.0;
		if ((cur_pts[0].py < first_row) || (cur_pts[0].py > last_row) || (cur_pts[0].px < first_col) || (cur_pts[0].px > last_col)) return false;
		cur_pts[1].px = init_pts[zero_offset + 1].px; cur_pts[1].py = init_pts[zero_offset + 1].py; cur_pts[1].direction = init_pts[zero_offset + 1].direction;
		cur_pts[1].size = 3; memset(cur_pts[1].mask, 0, 8 * sizeof(bool)); cur_pts[1].pixel_va = 0.0;
		if ((cur_pts[1].py < first_row) || (cur_pts[1].py > last_row) || (cur_pts[1].px < first_col) || (cur_pts[1].px > last_col)) return false;
		cur_pts[2].px = init_pts[zero_offset + 2].px; cur_pts[2].py = init_pts[zero_offset + 2].py; cur_pts[2].direction = init_pts[zero_offset + 2].direction;
		cur_pts[2].size = 3; memset(cur_pts[2].mask, 0, 8 * sizeof(bool)); cur_pts[2].pixel_va = 0.0;
		if ((cur_pts[2].py < first_row) || (cur_pts[2].py > last_row) || (cur_pts[2].px < first_col) || (cur_pts[2].px > last_col)) return false;
		int cur_pts_sz = 3;
		int max_pts_sz = 0; float max_outv_sum = 0.0;
		int gap = 0;
		while (1) {
			int _incr0 = cur_pts[cur_pts_sz - 2].direction, _incr1 = cur_pts[cur_pts_sz - 1].direction;
			int _rs = cur_pts[cur_pts_sz - 1].py + incy1[_incr1], _cs = cur_pts[cur_pts_sz - 1].px + incx1[_incr1];
			int _rs_2 = _rs + incy1[_incr1], _cs_2 = _cs + incx1[_incr1];
			bool extend_or_recall = false;
			if ((_rs_2 < first_row) || (_rs_2 > last_row) || (_cs_2 < first_col) || (_cs_2 > last_col)) extend_or_recall = false;
			else {
				for (int incr_n1 = 0; incr_n1 <= 2; incr_n1++) {  //三个方向
					int _incr2 = (_incr1 + incr_n1 + 7) % 8; int _rs_3 = _rs_2 + incy1[_incr2], _cs_3 = _cs_2 + incx1[_incr2];
					if (out_thva.at<float>(_rs_3, _cs_3) > 1.0) {
						printf("loop detector detect loop x:%d, y:%d\n", _cs_3, _rs_3);
						cur_pts[cur_pts_sz - 1].mask[_incr2] = true;
					} else 
					if ( (!cur_pts[cur_pts_sz - 1].mask[_incr2]) ) {  //extend
						float cur_sum_deriv = get_sum_gray_gauss(amps, _rs, _cs, _incr0, _incr1, _incr2) / th_filter_sz;
						if (cur_sum_deriv > bin_threshold) {
							cur_pts_sz++;
							extend_or_recall = true;
							cur_pts[cur_pts_sz - 1].px = _cs_2;
							cur_pts[cur_pts_sz - 1].py = _rs_2;
							cur_pts[cur_pts_sz - 1].direction = _incr2;
							cur_pts[cur_pts_sz - 1].size = 3;
							cur_pts[cur_pts_sz - 1].pixel_va = cur_sum_deriv;
							memset(cur_pts[cur_pts_sz - 1].mask, 0, 8 * sizeof(bool));
							gap = 0;
							out_thva.at<float>(_rs_2, _cs_2) = cur_sum_deriv;
						}
						else {
							if(gap >= max_gap_3)	cur_pts[cur_pts_sz - 1].mask[_incr2] = true;
							else {
								cur_pts_sz++;
								extend_or_recall = true;
								cur_pts[cur_pts_sz - 1].px = _cs_2;
								cur_pts[cur_pts_sz - 1].py = _rs_2;
								cur_pts[cur_pts_sz - 1].direction = _incr2;
								cur_pts[cur_pts_sz - 1].size = 3;
								cur_pts[cur_pts_sz - 1].pixel_va = cur_sum_deriv;
								memset(cur_pts[cur_pts_sz - 1].mask, 0, 8 * sizeof(bool));
								gap++; out_thva.at<float>(_rs_2, _cs_2) = cur_sum_deriv;
							}
						}
					}
					if (extend_or_recall) break;
				}
			}
			if (!extend_or_recall) {    //recall回溯收缩
				if ( (gap > 0) && (cur_pts[cur_pts_sz - 1].pixel_va <= bin_threshold) ) gap--;
				cur_pts_sz--; out_thva.at<float>(_rs_2, _cs_2) = 0.0;
				int incr_nc = cur_pts[cur_pts_sz].direction;
				cur_pts[cur_pts_sz - 1].mask[incr_nc] = true;
			}
			if (cur_pts_sz >= slide_maximum_len) {//截止长度及比较长度
				//evaluate the founded points goal value计算目标值比较大小
				float cur_outv_sum = 0.0;
				for (int pts_i = 0; pts_i < cur_pts_sz; pts_i++) {
					int pts_c = cur_pts[pts_i].px, pts_r = cur_pts[pts_i].py;
					int pts_direction = cur_pts[pts_i].direction, pts_size = cur_pts[pts_i].size;
					for (int i = 0; i < pts_size; i++) {
						if (pix_band.at<float>(pts_r, pts_c) != 1) {
							pix_band.at<float>(pts_r, pts_c) = 1;
							cur_outv_sum += amps.at<float>(pts_r, pts_c);
						}
						if (pix_band.at<float>(pts_r - 1, pts_c) != 1) {
							pix_band.at<float>(pts_r - 1, pts_c) = 1;
							cur_outv_sum += amps.at<float>(pts_r - 1, pts_c);
						}
						if (pix_band.at<float>(pts_r + 1, pts_c) != 1) {
							pix_band.at<float>(pts_r + 1, pts_c) = 1;
							cur_outv_sum += amps.at<float>(pts_r + 1, pts_c);
						}
						if (pix_band.at<float>(pts_r, pts_c - 1) != 1) {
							pix_band.at<float>(pts_r, pts_c - 1) = 1;
							cur_outv_sum += amps.at<float>(pts_r, pts_c - 1);
						}
						if (pix_band.at<float>(pts_r, pts_c + 1) != 1) {
							pix_band.at<float>(pts_r, pts_c + 1) = 1;
							cur_outv_sum += amps.at<float>(pts_r, pts_c + 1);
						}

						pts_c += incx1[pts_direction]; pts_r += incy1[pts_direction];
					}
				}

				for (int pts_i = 0; pts_i < cur_pts_sz; pts_i++) {
					int pts_c = cur_pts[pts_i].px, pts_r = cur_pts[pts_i].py;
					int pts_direction = cur_pts[pts_i].direction, pts_size = cur_pts[pts_i].size;
					for (int i = 0; i < pts_size; i++) {
						pix_band.at<float>(pts_r, pts_c) = 0;
						pix_band.at<float>(pts_r - 1, pts_c) = 0;
						pix_band.at<float>(pts_r + 1, pts_c) = 0;
						pix_band.at<float>(pts_r, pts_c - 1) = 0;
						pix_band.at<float>(pts_r, pts_c + 1) = 0;
						pts_c += incx1[pts_direction]; pts_r += incy1[pts_direction];
						
					}
				}

				if (max_outv_sum < cur_outv_sum) { //found the new max value pts
					max_outv_sum = cur_outv_sum;
					memcpy(cur_pts_max, cur_pts, sizeof(struct pt_struct) * cur_pts_sz);
					max_pts_sz = cur_pts_sz;
				}
				if ((gap > 0) && (cur_pts[cur_pts_sz - 1].pixel_va <= bin_threshold)) gap--;
				cur_pts_sz--;
				int incr_nc = cur_pts[cur_pts_sz].direction;
				cur_pts[cur_pts_sz - 1].mask[incr_nc] = true;
			}
			else if (cur_pts_sz < 3) {
				break;
			}
		}

		if (max_pts_sz > 0) { //save the maximum current contour to cur_pts_max and record_pts
			memcpy(init_pts + zero_offset, cur_pts_max, max_pts_sz * sizeof(struct pt_struct));
			_ret = true;
		}
	} else if (rev_enable) {
		int direction_z = init_pts[zero_offset].direction, size_z = init_pts[zero_offset].size;
		cur_pts[0].px = init_pts[zero_offset].px + (size_z - 1) * incx1[direction_z]; cur_pts[0].py = init_pts[zero_offset].py + (size_z - 1) * incy1[direction_z];
		cur_pts[0].direction = (direction_z + 4) % 8;
		cur_pts[0].size = 3; memset(cur_pts[0].mask, 0, 8 * sizeof(bool));

		direction_z = init_pts[zero_offset - 1].direction; size_z = init_pts[zero_offset - 1].size;
		cur_pts[1].px = init_pts[zero_offset - 1].px + (size_z - 1) * incx1[direction_z]; cur_pts[1].py = init_pts[zero_offset - 1].py + (size_z - 1) * incy1[direction_z];
		cur_pts[1].direction = (direction_z + 4) % 8;
		cur_pts[1].size = 3; memset(cur_pts[1].mask, 0, 8 * sizeof(bool));

		direction_z = init_pts[zero_offset - 2].direction; size_z = init_pts[zero_offset - 2].size;
		cur_pts[2].px = init_pts[zero_offset - 2].px + (size_z - 1) * incx1[direction_z]; cur_pts[2].py = init_pts[zero_offset - 2].py + (size_z - 1) * incy1[direction_z];
		cur_pts[2].direction = (direction_z + 4) % 8;
		cur_pts[2].size = 3; memset(cur_pts[2].mask, 0, 8 * sizeof(bool));
		int cur_pts_sz = 3;
		int max_pts_sz = 0; float max_outv_sum = 0.0;
		int gap = 0;
		while (1) {
			int _incr0 = cur_pts[cur_pts_sz - 2].direction, _incr1 = cur_pts[cur_pts_sz - 1].direction;
			int _rs = cur_pts[cur_pts_sz - 1].py + incy1[_incr1], _cs = cur_pts[cur_pts_sz - 1].px + incx1[_incr1];
			int _rs_2 = _rs + incy1[_incr1], _cs_2 = _cs + incx1[_incr1];
			bool extend_or_recall = false;
			if ((_rs_2 < first_row) || (_rs_2 > last_row) || (_cs_2 < first_col) || (_cs_2 > last_col)) extend_or_recall = false;
			else {
				for (int incr_n1 = 0; incr_n1 <= 2; incr_n1++) {//扩展三个方向点
					int _incr2 = (_incr1 + incr_n1 + 7) % 8; int _rs_3 = _rs_2 + incy1[_incr2], _cs_3 = _cs_2 + incx1[_incr2];
					if (out_thva.at<float>(_rs_3, _cs_3) > 1.0) {
						//printf("loop detector detect loop x:%d, y:%d\n", _cs_3, _rs_3);
						cur_pts[cur_pts_sz - 1].mask[_incr2] = true;
					} else if ( (!cur_pts[cur_pts_sz - 1].mask[_incr2]) ) {  //extend
						float cur_sum_deriv = get_sum_gray_gauss(amps, _rs, _cs, _incr0, _incr1, _incr2) / th_filter_sz;
						if (cur_sum_deriv > bin_threshold) {
							cur_pts_sz++;
							extend_or_recall = true;
							cur_pts[cur_pts_sz - 1].px = _cs_2;
							cur_pts[cur_pts_sz - 1].py = _rs_2;
							cur_pts[cur_pts_sz - 1].direction = _incr2;
							cur_pts[cur_pts_sz - 1].size = 3;
							cur_pts[cur_pts_sz - 1].pixel_va = cur_sum_deriv;
							memset(cur_pts[cur_pts_sz - 1].mask, 0, 8 * sizeof(bool));
							gap = 0; out_thva.at<float>(_rs_2, _cs_2) = cur_sum_deriv;
						}
						else {
							if(gap >= max_gap_3) cur_pts[cur_pts_sz - 1].mask[_incr2] = true;
							else {
								cur_pts_sz++;
								extend_or_recall = true;
								cur_pts[cur_pts_sz - 1].px = _cs_2;
								cur_pts[cur_pts_sz - 1].py = _rs_2;
								cur_pts[cur_pts_sz - 1].direction = _incr2;
								cur_pts[cur_pts_sz - 1].size = 3;
								cur_pts[cur_pts_sz - 1].pixel_va = cur_sum_deriv;
								memset(cur_pts[cur_pts_sz - 1].mask, 0, 8 * sizeof(bool));
								gap++; out_thva.at<float>(_rs_2, _cs_2) = cur_sum_deriv;
							}
						}
					}
					if (extend_or_recall) break;
				}
			}
			if (!extend_or_recall) {    //recall回溯收缩
				if ((gap > 0) && (cur_pts[cur_pts_sz - 1].pixel_va <= bin_threshold)) gap--;
				cur_pts_sz--; out_thva.at<float>(_rs_2, _cs_2) = 0.0;
				int incr_nc = cur_pts[cur_pts_sz].direction;
				cur_pts[cur_pts_sz - 1].mask[incr_nc] = true;
			}
			if (cur_pts_sz >= slide_maximum_len) {//截止长度及比较长度
				//evaluate the founded points goal value计算目标值比较大小
				float cur_outv_sum = 0.0;
				for (int pts_i = 0; pts_i < cur_pts_sz; pts_i++) {
					int pts_c = cur_pts[pts_i].px, pts_r = cur_pts[pts_i].py;
					int pts_direction = cur_pts[pts_i].direction, pts_size = cur_pts[pts_i].size;
					for (int i = 0; i < pts_size; i++) {
						if (pix_band.at<float>(pts_r, pts_c) != 1) {
							pix_band.at<float>(pts_r, pts_c) = 1;
							cur_outv_sum += amps.at<float>(pts_r, pts_c);
						}
						if (pix_band.at<float>(pts_r - 1, pts_c) != 1) {
							pix_band.at<float>(pts_r - 1, pts_c) = 1;
							cur_outv_sum += amps.at<float>(pts_r - 1, pts_c);
						}
						if (pix_band.at<float>(pts_r + 1, pts_c) != 1) {
							pix_band.at<float>(pts_r + 1, pts_c) = 1;
							cur_outv_sum += amps.at<float>(pts_r + 1, pts_c);
						}
						if (pix_band.at<float>(pts_r, pts_c - 1) != 1) {
							pix_band.at<float>(pts_r, pts_c - 1) = 1;
							cur_outv_sum += amps.at<float>(pts_r, pts_c - 1);
						}
						if (pix_band.at<float>(pts_r, pts_c + 1) != 1) {
							pix_band.at<float>(pts_r, pts_c + 1) = 1;
							cur_outv_sum += amps.at<float>(pts_r, pts_c + 1);
						}

						pts_c += incx1[pts_direction]; pts_r += incy1[pts_direction];
					}
				}

				for (int pts_i = 0; pts_i < cur_pts_sz; pts_i++) {
					int pts_c = cur_pts[pts_i].px, pts_r = cur_pts[pts_i].py;
					int pts_direction = cur_pts[pts_i].direction, pts_size = cur_pts[pts_i].size;
					for (int i = 0; i < pts_size; i++) {
						pix_band.at<float>(pts_r, pts_c) = 0;
						pix_band.at<float>(pts_r - 1, pts_c) = 0;
						pix_band.at<float>(pts_r + 1, pts_c) = 0;
						pix_band.at<float>(pts_r, pts_c - 1) = 0;
						pix_band.at<float>(pts_r, pts_c + 1) = 0;

						pts_c += incx1[pts_direction]; pts_r += incy1[pts_direction];
					}
				}

				if (max_outv_sum < cur_outv_sum) { //found the new max value pts
					max_outv_sum = cur_outv_sum;
					memcpy(cur_pts_max, cur_pts, sizeof(struct pt_struct) * cur_pts_sz);
					max_pts_sz = cur_pts_sz;
				}
				if ((gap > 0) && (cur_pts[cur_pts_sz - 1].pixel_va <= bin_threshold)) gap--;
				cur_pts_sz--;
				int incr_nc = cur_pts[cur_pts_sz].direction;
				cur_pts[cur_pts_sz - 1].mask[incr_nc] = true;
			}
			else if (cur_pts_sz < 3) {
				break;
			}
		}

		if (max_pts_sz > 0) {
			for (int i = 0; i < max_pts_sz; i++) {
				memcpy(init_pts + zero_offset - i, cur_pts_max + i, 1 * sizeof(struct pt_struct));
				direction_z = init_pts[zero_offset - i].direction; size_z = init_pts[zero_offset - i].size;
				init_pts[zero_offset - i].px += (size_z - 1) * incx1[direction_z];
				init_pts[zero_offset - i].py += (size_z - 1) * incy1[direction_z];
				init_pts[zero_offset - i].direction = (direction_z + 4) % 8;
			}
			_ret = true;
		}
	}

	return _ret;
}

void ContourSelectInstance::maximum_density_double(const cv::Mat& amps) {
	int height = amps.rows, width = amps.cols;
	//const int M = 23, N = 23;
	//const int KM = 13, KN = 13;
	//const int BM = 5, BN = 5;
	const long first_row = std::max(M / 2, (int)ceil(roi_y_min * height));
	const long first_col = std::max(N / 2, (int)ceil(roi_x_min * width));
	const long last_row = std::min(height - M / 2, (int)ceil(roi_y_max * height));
	const long last_col = std::min(width - N / 2, (int)ceil(roi_x_max * width));
	//const int init_sz = 7;
	cv::Size processSize = cv::Size(amps.cols, amps.rows);
	cv::Mat pix_used = cv::Mat::zeros(processSize, CV_32S);
	cv::Mat pix_band = cv::Mat::zeros(processSize, CV_32S);
	contour_loop_set.clear();
	for (long r = first_row; r < last_row; ++r)
	{
		for (long c = first_col; c < last_col; ++c) {
			float max_norm_sum = 0.0;
			if (amps.at<float>(r, c) < bin_threshold) continue;
			bool used_flag = false;
			for (long m = 0; m < BM; ++m)
			{
				for (long n = 0; n < BN; ++n)
				{
					if ( pix_used.at<int>(r - BM / 2 + m, c - BN / 2 + n) > 0 ) {
						used_flag = true; break;
					}
				}
				if (used_flag) break;
			}
			if (used_flag) continue;
			int first_incr = 0; int first_incr_neg = 4;
			float max_deriv = -1.0; int max_m_n = 0;
			float sum_deriv = get_sum_gray_horz(amps, r, c, 0, 0);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 0;
				cur_pts[0].px = c - 3; cur_pts[0].py = r - 2; cur_pts[0].direction = 1;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 7;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 0, 1);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 1;
				cur_pts[0].px = c - 3; cur_pts[0].py = r - 2; cur_pts[0].direction = 1;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 0;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 0, 2);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 2;
				cur_pts[0].px = c - 3; cur_pts[0].py = r - 2; cur_pts[0].direction = 1;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 1;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 1, 0);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 3;
				cur_pts[0].px = c - 3; cur_pts[0].py = r - 0; cur_pts[0].direction = 0;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 7;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 1, 1);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 4;
				cur_pts[0].px = c - 3; cur_pts[0].py = r - 0; cur_pts[0].direction = 0;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 0;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 1, 2);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 5;
				cur_pts[0].px = c - 3; cur_pts[0].py = r - 0; cur_pts[0].direction = 0;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 1;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 2, 0);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 6;
				cur_pts[0].px = c - 3; cur_pts[0].py = r + 2; cur_pts[0].direction = 7;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 7;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 2, 1);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 7;
				cur_pts[0].px = c - 3; cur_pts[0].py = r + 2; cur_pts[0].direction = 7;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 0;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 2, 2);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 8;
				cur_pts[0].px = c - 3; cur_pts[0].py = r + 2; cur_pts[0].direction = 7;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 1;
			}

			sum_deriv = get_sum_gray_vert(amps, r, c, 0, 0);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 9;
				cur_pts[0].px = c - 2; cur_pts[0].py = r - 3; cur_pts[0].direction = 1;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 3;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 0, 1);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 10;
				cur_pts[0].px = c - 2; cur_pts[0].py = r - 3; cur_pts[0].direction = 1;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 2;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 0, 2);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 11;
				cur_pts[0].px = c - 2; cur_pts[0].py = r - 3; cur_pts[0].direction = 1;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 1;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 1, 0);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 12;
				cur_pts[0].px = c - 0; cur_pts[0].py = r - 3; cur_pts[0].direction = 2;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 3;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 1, 1);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 13;
				cur_pts[0].px = c - 0; cur_pts[0].py = r - 3; cur_pts[0].direction = 2;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 2;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 1, 2);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 14;
				cur_pts[0].px = c - 0; cur_pts[0].py = r - 3; cur_pts[0].direction = 2;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 1;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 2, 0);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 15;
				cur_pts[0].px = c + 2; cur_pts[0].py = r - 3; cur_pts[0].direction = 3;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 3;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 2, 1);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 16;
				cur_pts[0].px = c + 2; cur_pts[0].py = r - 3; cur_pts[0].direction = 3;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 2;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 2, 2);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 17;
				cur_pts[0].px = c + 2; cur_pts[0].py = r - 3; cur_pts[0].direction = 3;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 1;
			}
			cur_pts[0].size = 3; memset(cur_pts[0].mask, 0, 8 * sizeof(bool));
			cur_pts[1].size = 3; memset(cur_pts[1].mask, 0, 8 * sizeof(bool));
			cur_pts[2].size = 3; memset(cur_pts[2].mask, 0, 8 * sizeof(bool));

			int record_pts_sz = 0; int cur_pts_sz = 3;
			int zero_offset = MAX_PT_RECORD / 2;
			memset(record_pts, 0, MAX_PT_RECORD * sizeof(struct pt_struct));
			memcpy(record_pts + zero_offset, cur_pts, 3 * sizeof(struct pt_struct));
			if (max_deriv / th_filter_sz > bin_threshold) {
				int max_pts_sz = 0; float max_outv_sum = 0.0;
				bool extend_ret = extend_band_double(amps, record_pts, zero_offset, false, pix_used);
				if (extend_ret) {
					bool extend_ret_neg = extend_band_double(amps, record_pts, zero_offset + 1 + slide_maximum_len / 2, true, pix_used);
					bool extend_ret_pos = extend_band_double(amps, record_pts, zero_offset + slide_maximum_len / 2, false, pix_used);
					int _offset_begin = 0, _offset_end = 0;
					if (extend_ret_neg) _offset_begin = zero_offset - slide_maximum_len / 2 - 1;
					else  _offset_begin = zero_offset + slide_maximum_len / 2 - 1;
					if(extend_ret_pos) _offset_end = zero_offset + 3 * slide_maximum_len / 2 + 1;
					else _offset_end = zero_offset + 2 * slide_maximum_len / 2 + 1;
					std::vector<cv::Point> line_sequence; line_sequence.clear();
					for (int pts_i = _offset_begin; pts_i <= _offset_end; pts_i++) { //set used flag
						int pts_c = record_pts[pts_i].px, pts_r = record_pts[pts_i].py;
						int pts_direction = record_pts[pts_i].direction;
						int pts_len = record_pts[pts_i].size;
						if (pts_len > 0) {
							line_sequence.push_back(cv::Point(record_pts[pts_i].px, record_pts[pts_i].py));
							record_pts_sz++;
							for (int i = 0; i < pts_len; i++) {
								for (long m = 0; m < BM; ++m)
								{
									for (long n = 0; n < BN; ++n)
									{
										pix_used.at<int>(pts_r - BM / 2 + m, pts_c - BN / 2 + n) = 1;
									}
								}
								pts_c += incx1[pts_direction]; pts_r += incy1[pts_direction];
							}
						}
					}
					if (record_pts_sz > slide_maximum_len) contour_loop_set.push_back(line_sequence);
				}
			}

		}
	}
}

void ContourSelectInstance::maximum_density_depth(const cv::Mat& amps) {

	int height = amps.rows, width = amps.cols;
	//const int M = 23, N = 23;
	//const int KM = 13, KN = 13;
	//const int BM = 5, BN = 5;
	const long first_row = std::max(M / 2, (int)ceil(roi_y_min * height));
	const long first_col = std::max(N / 2, (int)ceil(roi_x_min * width));
	const long last_row = std::min(height - M / 2, (int)ceil(roi_y_max * height));
	const long last_col = std::min(width - N / 2, (int)ceil(roi_x_max * width));
	//const int init_sz = 7;
	cv::Size processSize = cv::Size(amps.cols, amps.rows);
	cv::Mat pix_used = cv::Mat::zeros(processSize, CV_32S);
	cv::Mat pix_cur_set = cv::Mat::zeros(processSize, CV_32F);
	contour_loop_set.clear();
	//int cons_std = contours_std.size();
	printf("Found contours length %d, must clear first.\n", cons_len);
	if (cons_len > 0) {
		for (int i = 0; i < cons_len; i++) {//struct pt_list contours_pool[MAX_POOL_SZ];
			if(contours_pool[i].pt_head) 
				free(contours_pool[i].pt_head);
		}
		memset(contours_pool, 0, MAX_POOL_SZ * sizeof(struct pt_list));
		cons_len = 0;
	}
	if (cross_len > 0) {//struct crossx_struct crossx_pool[MAX_POOL_SZ];
		memset(crossx_pool, 0, MAX_POOL_SZ * sizeof( struct crossx_struct ));
		cross_len = 0;
	}
	pts_distributed.clear();
	for (long r = first_row; r < last_row; ++r)
	{
		for (long c = first_col; c < last_col; ++c) {
			float max_norm_sum = 0.0;
			if (amps.at<float>(r, c) < bin_threshold) continue;
			bool used_flag = false;
			for (long m = 0; m < BM; ++m)
			{
				for (long n = 0; n < BN; ++n)
				{
					if (pix_used.at<int>(r - BM / 2 + m, c - BN / 2 + n) > 0) {
						used_flag = true; break;
					}
				}
				if (used_flag) break;
			}
			if (used_flag) continue;
			int first_incr = 0; int first_incr_neg = 4;
			float max_deriv = -1.0; int max_m_n = 0;
			float sum_deriv = get_sum_gray_horz(amps, r, c, 0, 0);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 0;
				cur_pts[0].px = c - 3; cur_pts[0].py = r - 2; cur_pts[0].direction = 1;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 7;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 0, 1);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 1;
				cur_pts[0].px = c - 3; cur_pts[0].py = r - 2; cur_pts[0].direction = 1;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 0;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 0, 2);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 2;
				cur_pts[0].px = c - 3; cur_pts[0].py = r - 2; cur_pts[0].direction = 1;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 1;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 1, 0);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 3;
				cur_pts[0].px = c - 3; cur_pts[0].py = r - 0; cur_pts[0].direction = 0;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 7;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 1, 1);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 4;
				cur_pts[0].px = c - 3; cur_pts[0].py = r - 0; cur_pts[0].direction = 0;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 0;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 1, 2);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 5;
				cur_pts[0].px = c - 3; cur_pts[0].py = r - 0; cur_pts[0].direction = 0;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 1;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 2, 0);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 6;
				cur_pts[0].px = c - 3; cur_pts[0].py = r + 2; cur_pts[0].direction = 7;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 7;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 2, 1);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 7;
				cur_pts[0].px = c - 3; cur_pts[0].py = r + 2; cur_pts[0].direction = 7;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 0;
			}
			sum_deriv = get_sum_gray_horz(amps, r, c, 2, 2);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 8;
				cur_pts[0].px = c - 3; cur_pts[0].py = r + 2; cur_pts[0].direction = 7;
				cur_pts[1].px = c - 1; cur_pts[1].py = r - 0; cur_pts[1].direction = 0;
				cur_pts[2].px = c + 1; cur_pts[2].py = r - 0; cur_pts[2].direction = 1;
			}

			sum_deriv = get_sum_gray_vert(amps, r, c, 0, 0);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 9;
				cur_pts[0].px = c - 2; cur_pts[0].py = r - 3; cur_pts[0].direction = 1;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 3;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 0, 1);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 10;
				cur_pts[0].px = c - 2; cur_pts[0].py = r - 3; cur_pts[0].direction = 1;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 2;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 0, 2);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 11;
				cur_pts[0].px = c - 2; cur_pts[0].py = r - 3; cur_pts[0].direction = 1;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 1;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 1, 0);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 12;
				cur_pts[0].px = c - 0; cur_pts[0].py = r - 3; cur_pts[0].direction = 2;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 3;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 1, 1);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 13;
				cur_pts[0].px = c - 0; cur_pts[0].py = r - 3; cur_pts[0].direction = 2;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 2;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 1, 2);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 14;
				cur_pts[0].px = c - 0; cur_pts[0].py = r - 3; cur_pts[0].direction = 2;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 1;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 2, 0);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 15;
				cur_pts[0].px = c + 2; cur_pts[0].py = r - 3; cur_pts[0].direction = 3;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 3;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 2, 1);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 16;
				cur_pts[0].px = c + 2; cur_pts[0].py = r - 3; cur_pts[0].direction = 3;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 2;
			}
			sum_deriv = get_sum_gray_vert(amps, r, c, 2, 2);
			if (sum_deriv > max_deriv) {
				max_deriv = sum_deriv; max_m_n = 17;
				cur_pts[0].px = c + 2; cur_pts[0].py = r - 3; cur_pts[0].direction = 3;
				cur_pts[1].px = c - 0; cur_pts[1].py = r - 1; cur_pts[1].direction = 2;
				cur_pts[2].px = c - 0; cur_pts[2].py = r + 1; cur_pts[2].direction = 1;
			}
			cur_pts[0].size = 3; memset(cur_pts[0].mask, 0, 8 * sizeof(bool));
			cur_pts[1].size = 3; memset(cur_pts[1].mask, 0, 8 * sizeof(bool));
			cur_pts[2].size = 3; memset(cur_pts[2].mask, 0, 8 * sizeof(bool));

			int record_pts_sz = 0; int cur_pts_sz = 3;
			int zero_offset = MAX_PT_RECORD / 2, positive_offset = zero_offset, negative_offset = zero_offset + slide_maximum_len;
			memset(record_pts, 0, MAX_PT_RECORD * sizeof(struct pt_struct));
			memcpy(record_pts + zero_offset, cur_pts, 3 * sizeof(struct pt_struct));
			if (max_deriv / th_filter_sz > bin_threshold) {
				int max_pts_sz = 0; float max_outv_sum = 0.0;
				bool extend_ret = extend_band_double(amps, record_pts, zero_offset, false, pix_cur_set);
				int _offset_begin = 0, _offset_end = 0;
				int loop_cnt = 0;
				bool extend_ret_neg = extend_ret, extend_ret_pos = extend_ret;
				while (extend_ret) {
					if (extend_ret_neg) {
						negative_offset = negative_offset - slide_maximum_len / 2 - 1;
						extend_ret_neg = extend_band_double(amps, record_pts, negative_offset, true, pix_cur_set);
						if (extend_ret_neg) _offset_begin = negative_offset - slide_maximum_len;
						else  _offset_begin = negative_offset;
					}
					if (extend_ret_pos) {
						positive_offset = positive_offset + slide_maximum_len / 2 + 1;
						extend_ret_pos = extend_band_double(amps, record_pts, positive_offset, false, pix_cur_set);
						if (extend_ret_pos) 	_offset_end = positive_offset + slide_maximum_len;
						else _offset_end = positive_offset;

					}
					extend_ret = extend_ret_neg || extend_ret_pos;
					if (loop_cnt++ >= 4)  break;
				}

				for (int pts_i = _offset_begin; pts_i <= _offset_end; pts_i++) { //set used flag
					int pts_c = record_pts[pts_i].px, pts_r = record_pts[pts_i].py;
					int pts_direction = record_pts[pts_i].direction;
					int pts_len = record_pts[pts_i].size;
					if (pts_len > 0) {						
						record_pts_sz++;
						//for (int i = 0; i < pts_len; i++) {				
						//	pts_c += incx1[pts_direction]; pts_r += incy1[pts_direction];							
						//}
					}
				}
				if ((record_pts_sz >= 1 * slide_maximum_len)) {
					std::vector<cv::Point> line_sequence; line_sequence.clear();
					cv::Mat distribute2d = cv::Mat::zeros(processSize, CV_32S);
					struct pt_list* list_head = contours_pool + cons_len;
					list_head->pt_head = (struct pt_struct*)malloc(record_pts_sz * sizeof(struct pt_struct));
					list_head->len = record_pts_sz;
					list_head->selection = false; list_head->sel_in = -1; list_head->sel_out = -1;
					list_head->branch_sz = 0; list_head->head_extend_en = true; list_head->tail_extend_en = true;
					int pts_ptr_idx = 0;
					for (int pts_i = _offset_begin; pts_i <= _offset_end; pts_i++) { //set used flag
						int pts_c = record_pts[pts_i].px, pts_r = record_pts[pts_i].py;
						int pts_direction = record_pts[pts_i].direction;
						int pts_len = record_pts[pts_i].size;
						if (pts_len > 0) {
							line_sequence.push_back(cv::Point(record_pts[pts_i].px, record_pts[pts_i].py));
							for (int i = 0; i < pts_len; i++) {
								for (long m = 0; m < BM; ++m)
								{
									for (long n = 0; n < BN; ++n)
									{
										pix_used.at<int>(pts_r - BM / 2 + m, pts_c - BN / 2 + n) = 1;
									}
								}
								pts_c += incx1[pts_direction]; pts_r += incy1[pts_direction];
								pix_cur_set.at<float>(pts_r, pts_c) = 0.0;
								distribute2d.at<int>(pts_r, pts_c) = 1;
							}
							memcpy(list_head->pt_head + pts_ptr_idx, record_pts + pts_i, sizeof(struct pt_struct));
							pts_ptr_idx++;
						}
					}
					contour_loop_set.push_back(line_sequence);
					cons_len++;
					if (cons_len > MAX_POOL_SZ) {
						printf("Error contour number bigger than maximum value\n"); while (1);
					}
					pts_distributed.push_back( distribute2d );
				}
			}

		}
	}

}

void ContourSelectInstance::maximum_band_contours(const cv::Mat& amps) {
	int height = amps.rows, width = amps.cols;
	//const int M = 13, N = 13;
	//const int KM = 11, KN = 11;
	//const int BM = 3, BN = 3;
	const long first_row = std::max(M / 2, (int)ceil(roi_y_min * height));
	const long first_col = std::max(N / 2, (int)ceil(roi_x_min * width));
	const long last_row = std::min(height - M / 2, (int)ceil(roi_y_max * height));
	const long last_col = std::min(width - N / 2, (int)ceil(roi_x_max * width));

	contour_loop_set.clear();
	cv::Size processSize = cv::Size(amps.cols, amps.rows);
	cv::Mat pix_used = cv::Mat::zeros(processSize, CV_32S);
	cv::Mat pix_status = cv::Mat::zeros(processSize, CV_32S);
	cv::Mat pix_band = cv::Mat::zeros(processSize, CV_32S);
	//const int incx1[8] = { 1, 1, 0, -1, -1, -1,  0,  1 };
	//const int incy1[8] = { 0, 1, 1,  1,  0, -1, -1, -1 };
	const int maxGAP = 3;
	int cur_status = 0;
	for (long r = first_row; r < last_row; ++r)
	{
		for (long c = first_col; c < last_col; ++c) {
			float norm = amps.at<float>(r, c);
			if (pix_used.at<int>(r, c) > 0) continue;
			if (norm <= bin_threshold) continue;
			int max_first_incr = 0, max_second_incr = 0;
			float first_norm = 0, second_norm = 0;
			
			int max_incr1 = 0;

			bool circle_flag = false, used_flag = false, outside_flag = false;
			float max_outv_sum = 0.0;
			int max_sz1 = 1, c_start = c, r_start = r, c_end = c, r_end = r;
			float start_norm_sum = 0.0;
			for (int first_incr = 0; first_incr < 4; first_incr++) {
				int pos_incr_sz = 0, pos_gap = 0; float max_norm_pos = 0.0;
				int pos_c = c + incx1[first_incr], pos_r = r + incy1[first_incr];
				if ((pos_c < first_col) || (pos_c > last_col) || (pos_r < first_row) || (pos_r > last_row)) continue;
				if (pix_used.at<int>(pos_r, pos_c) == 1) continue;
				first_norm = amps.at<float>(pos_r, pos_c);
				if (first_norm > bin_threshold) {
					pos_incr_sz++;
					max_norm_pos += first_norm;
				}
				else pos_gap++;
				while (pos_gap <= maxGAP) {
					pos_c = pos_c + incx1[first_incr]; pos_r = pos_r + incy1[first_incr];
					if ((pos_c < first_col) || (pos_c > last_col) || (pos_r < first_row) || (pos_r > last_row)) break;
					if (pix_used.at<int>(pos_r, pos_c) == 1) break;
					first_norm = amps.at<float>(pos_r, pos_c);
					//if (first_norm <= bin_th) pos_gap++;
					//else pos_incr_sz++;
					if (first_norm > bin_threshold) {
						pos_incr_sz++;
						max_norm_pos += first_norm;
					}
					else pos_gap++;
				}

				int first_incr_neg = first_incr + 4;
				int neg_incr_sz = 0, neg_gap = 0; float max_norm_neg = 0.0;
				int neg_c = c + incx1[first_incr_neg], neg_r = r + incy1[first_incr_neg];
				if ((neg_c < first_col) || (neg_c > last_col) || (neg_r < first_row) || (neg_r > last_row)) continue;
				if (pix_used.at<int>(neg_r, neg_c) == 1) continue;
				first_norm = amps.at<float>(neg_r, neg_c);
				if (first_norm > bin_threshold) {
					neg_incr_sz++;
					max_norm_neg += first_norm;
				}
				else neg_gap++;
				while (neg_gap <= maxGAP) {
					neg_c = neg_c + incx1[first_incr_neg]; neg_r = neg_r + incy1[first_incr_neg];
					if ((neg_c < first_col) || (neg_c > last_col) || (neg_r < first_row) || (neg_r > last_row)) break;
					if (pix_used.at<int>(neg_r, neg_c) == 1) break;
					first_norm = amps.at<float>(neg_r, neg_c);
					if (first_norm > bin_threshold) {
						neg_incr_sz++;
						max_norm_neg += first_norm;
					}
					else neg_gap++;
				}

				if (start_norm_sum < norm + max_norm_pos + max_norm_neg)
				{
					max_sz1 = 1 + pos_incr_sz + neg_incr_sz;
					start_norm_sum = norm + max_norm_pos + max_norm_neg;
					max_incr1 = first_incr;
					c_start = neg_c; r_start = neg_r;
					c_end = pos_c; r_end = pos_r;
				}
			}
			if (max_sz1 < KM) continue;
			cur_status++;
			bool search_end = false; 
			int init_direction = max_incr1; int init_r = r_end, init_c = c_end;
			int cur_pts_sz = 0;	long record_pts_sz = 0;
			cur_pts[cur_pts_sz].px = c_start;	cur_pts[cur_pts_sz].py = r_start;
			cur_pts[cur_pts_sz].direction = init_direction;		cur_pts[cur_pts_sz].direction_flag = 0;
			cur_pts[cur_pts_sz].size = max_sz1;
			cur_pts_sz++;
			for (int i = 0; i < max_sz1; i++) {
				int _c = c_start + i * incx1[init_direction];
				int _r = r_start + i * incy1[init_direction];
				pix_status.at<int>(_r, _c) = cur_status;
			}
			int recall_flag = 0;
			while (1) {
				while (1) {
					int neg_incr = (init_direction + 7) % 8;
					float neg_norm_sum = 0.0;
					int slide_c = init_c, slide_r = init_r;
					int sum_len = 0, cur_sz = pt_sz_min;
					outside_flag = false; circle_flag = false; used_flag = false;
					while (recall_flag == 0) {
						neg_norm_sum = 0.0;
						int neg_c = slide_c, neg_r = slide_r;
						for (int i = 0; i < pt_sz_min; i++) {    //一个线段最小长度pt_sz_min
							neg_c += incx1[neg_incr]; neg_r += incy1[neg_incr];
							if ((neg_c < first_col) || (neg_c > last_col) || (neg_r < first_row) || (neg_r > last_row)) { outside_flag = true; break; }
							if (pix_status.at<int>(neg_r, neg_c) == cur_status) { circle_flag = true; break; }
							if (pix_used.at<int>(neg_r, neg_c) == 1) { used_flag = true; break; }
							neg_norm_sum += amps.at<float>(neg_r, neg_c);
						}
						float avg_norm = neg_norm_sum / pt_sz_min;
						if (avg_norm > bin_threshold) {
							if (outside_flag | used_flag | circle_flag) break;
							sum_len += cur_sz;
							slide_c = slide_c + pt_recall_step * incx1[neg_incr];  //延伸线段最小单位pt_recall_step
							slide_r = slide_r + pt_recall_step * incy1[neg_incr];
							cur_sz = pt_recall_step;
							continue;
						}
						else break;
					}
					if (sum_len >= pt_sz_min) {//found new valid point
						cur_pts[cur_pts_sz].direction = neg_incr; cur_pts[cur_pts_sz].direction_flag = 0;
						cur_pts[cur_pts_sz].px = init_c;
						cur_pts[cur_pts_sz].py = init_r;
						cur_pts[cur_pts_sz].size = sum_len;
						init_direction = neg_incr;
						cur_pts_sz++;
						for (int i = 0; i <= sum_len; i++) {
							int neg_c = init_c + i * incx1[neg_incr];
							int neg_r = init_r + i * incy1[neg_incr];
							pix_status.at<int>(neg_r, neg_c) = cur_status;
						}
						init_c = init_c + (sum_len) * incx1[neg_incr];
						init_r = init_r + (sum_len) * incy1[neg_incr];
						continue;
					}

					int pos_incr = (init_direction + 1) % 8;
					float pos_norm_sum = 0.0;
					slide_c = init_c; slide_r = init_r;
					sum_len = 0; cur_sz = pt_sz_min;
					outside_flag = false; circle_flag = false; used_flag = false;
					while (1) {
						pos_norm_sum = 0.0;
						int pos_c = slide_c, pos_r = slide_r;
						for (int i = 0; i < pt_sz_min; i++) {
							pos_c += incx1[pos_incr]; pos_r += incy1[pos_incr];
							if ((pos_c < first_col) || (pos_c > last_col) || (pos_r < first_row) || (pos_r > last_row)) { outside_flag = true; break; }
							if (pix_status.at<int>(pos_r, pos_c) == cur_status) { circle_flag = true; break; }
							if (pix_used.at<int>(pos_r, pos_c) == 1) { used_flag = true; break; }
							pos_norm_sum += amps.at<float>(pos_r, pos_c);
						}

						float avg_norm = pos_norm_sum / pt_sz_min;
						if (avg_norm > bin_threshold) {
							if (outside_flag | used_flag | circle_flag) break;
							sum_len += cur_sz;
							slide_c = slide_c + pt_recall_step * incx1[pos_incr];  //延伸线段最小单位pt_recall_step
							slide_r = slide_r + pt_recall_step * incy1[pos_incr];
							cur_sz = pt_recall_step;
							continue;
						}
						else break;
					}

					if (sum_len >= pt_sz_min) {//found new valid point 
						cur_pts[cur_pts_sz].direction = pos_incr; cur_pts[cur_pts_sz].direction_flag = 1;
						cur_pts[cur_pts_sz].px = init_c;		cur_pts[cur_pts_sz].py = init_r;
						cur_pts[cur_pts_sz].size = sum_len;
						cur_pts_sz++;
						for (int i = 0; i <= sum_len; i++) {
							int pos_c = init_c + i * incx1[pos_incr];
							int pos_r = init_r + i * incy1[pos_incr];
							pix_status.at<int>(pos_r, pos_c) = cur_status;
						}
						init_direction = pos_incr;
						init_c = init_c + (sum_len) * incx1[pos_incr];
						init_r = init_r + (sum_len) * incy1[pos_incr];

						continue;
					}
					break;

				}
				
				//evaluate the founded points goal value
				float cur_outv_sum = 0.0;
				for (int pts_i = 0; pts_i < cur_pts_sz; pts_i++) {
					int pts_c = cur_pts[pts_i].px, pts_r = cur_pts[pts_i].py;
					int pts_direction = cur_pts[pts_i].direction, pts_size = cur_pts[pts_i].size;
					for (int i = 0; i < pts_size; i++) {
						for (long m = 0; m < BM; ++m)
						{
							for (long n = 0; n < BN; ++n)
							{
								if (pix_band.at<float>(pts_r - BM / 2 + m, pts_c - BN / 2 + n) != cur_status) {
									pix_band.at<float>(pts_r - BM / 2 + m, pts_c - BN / 2 + n) = cur_status;
									cur_outv_sum += amps.at<float>(pts_r - BM / 2 + m, pts_c - BN / 2 + n);
								}
								//cur_outv_sum += ( 0.5 * amps.at<float>(pts_r - BM / 2 + m, pts_c - BN / 2 + n) );
							}
						}
						pts_c += incx1[pts_direction]; pts_r += incy1[pts_direction];
					}
				}
				for (int pts_i = 0; pts_i < cur_pts_sz; pts_i++) {
					int pts_c = cur_pts[pts_i].px, pts_r = cur_pts[pts_i].py;
					int pts_direction = cur_pts[pts_i].direction, pts_size = cur_pts[pts_i].size;
					for (int i = 0; i < pts_size; i++) {
						for (long m = 0; m < BM; ++m)
						{
							for (long n = 0; n < BN; ++n)
							{
									pix_band.at<float>(pts_r - BM / 2 + m, pts_c - BN / 2 + n) = 0;
							}
						}
						pts_c += incx1[pts_direction]; pts_r += incy1[pts_direction];
					}
				}
				if (max_outv_sum < cur_outv_sum) { //found the new max value pts
					max_outv_sum = cur_outv_sum;
					memcpy(record_pts, cur_pts, sizeof(struct pt_struct) * cur_pts_sz);
					//for (int pts_i = 0; pts_i < cur_pts_sz; pts_i++) {
					//	record_pts[pts_i].px = cur_pts[pts_i].px; record_pts[pts_i].py = cur_pts[pts_i].py;
					//}
					record_pts_sz = cur_pts_sz;
				}
				//recall the point chain
				//int pts_i_recall = cur_pts_sz - 1;
				while (cur_pts_sz - 1 >= 0) { //shorten the length by pt_recall_step
					int pts_i_sz = cur_pts[cur_pts_sz - 1].size;
					int _d = cur_pts[cur_pts_sz - 1].direction ;
					int _r = cur_pts[cur_pts_sz - 1].py + pts_i_sz * incy1[_d];
					int _c = cur_pts[cur_pts_sz - 1].px + pts_i_sz * incx1[_d];
					if (pts_i_sz >= pt_sz_min + pt_recall_step) {
						for (int i = 0; i < pt_recall_step; i++) {
							pix_status.at<int>(_r, _c) = 0;
							_r -= incy1[_d];  _c -= incx1[_d];							
						}
						pts_i_sz -= pt_recall_step;
						cur_pts[cur_pts_sz - 1].size = pts_i_sz;
						init_direction = _d; //
						init_r = cur_pts[cur_pts_sz - 1].py + pts_i_sz * incy1[init_direction];
						init_c = cur_pts[cur_pts_sz - 1].px + pts_i_sz * incx1[init_direction];
						recall_flag = 0;
						break; 
					}
					else {						
						if (cur_pts_sz - 1 == 0) { search_end = true; break; }
						for(int i = 0; i < pts_i_sz; i++) {
							pix_status.at<int>(_r, _c) = 0;
							_r -= incy1[_d];  _c -= incx1[_d];
						}
						int direction_flag = cur_pts[cur_pts_sz - 1].direction_flag;
						cur_pts_sz--;
						if ((direction_flag == 0)) {
							recall_flag = 1; 
							init_direction = cur_pts[cur_pts_sz - 1].direction;
							init_r = cur_pts[cur_pts_sz].py;
							init_c = cur_pts[cur_pts_sz].px;
							break; 
						}
						else continue;
					}
				}
				//if (cur_status > 1024 * 1024) {
				//	printf("Find contour out of time\n");
				//	break;
				//}
				if (search_end) {
					printf("Find Contour with search end, cur_status:%d, max_outv_sum:%f, record points:%d\n", cur_status, max_outv_sum, record_pts_sz); break;   //  search end break for this point
				}
				//cur_pts_sz = pts_i_recall + 1;
				//Goto next status search
				//cur_status++;
				
				//for (int pts_i = 0; pts_i < cur_pts_sz; pts_i++) {
				//	int pts_c = cur_pts[pts_i].px, pts_r = cur_pts[pts_i].py;
				//	int pts_direction = cur_pts[pts_i].direction, pts_size = cur_pts[pts_i].size;
				//	for (int i = 0; i < pts_size; i++) {
				//		pix_status.at<int>(pts_r, pts_c) = cur_status;
				//		pts_c += incx1[pts_direction]; pts_r += incy1[pts_direction];
				//	}
				//}
				
			}
			if (max_outv_sum > 32 * 120) { //get the contour from record_pts
				std::vector<cv::Point> line_sequence; line_sequence.clear();
				for (int pts_i = 0; pts_i < record_pts_sz; pts_i++) {
					line_sequence.push_back(cv::Point(record_pts[pts_i].px, record_pts[pts_i].py));
				}
				contour_loop_set.push_back(line_sequence);
				for (int pts_i = 0; pts_i < record_pts_sz; pts_i++) {
					int pts_c = record_pts[pts_i].px, pts_r = record_pts[pts_i].py;
					int pts_len = record_pts[pts_i].size;
					int pts_direction = cur_pts[pts_i].direction;
					for (int i = 0; i < pts_len; i++) {
						for (long m = 0; m < BM; ++m)
						{
							for (long n = 0; n < BN; ++n)
							{
								pix_used.at<int>(pts_r - BM / 2 + m, pts_c - BN / 2 + n) = 1;
							}
						}
						pts_c += incx1[pts_direction]; pts_r += incy1[pts_direction];
					}
				}
			}

		}
	}
}

float ContourSelectInstance::distance_minimum(float x1_start, float y1_start, float x2_start, float y2_start, float xci1_start, float yci1_start, float xci2_start, float yci2_start) {
	float min_gap_ext = 1024.0;
	float gap_girth = sqrt((x1_start - xci1_start) * (x1_start - xci1_start) + (y1_start - yci1_start) * (y1_start - yci1_start));
	if (gap_girth < min_gap_ext) {
		min_gap_ext = gap_girth;
	}
	gap_girth = sqrt((x1_start - xci2_start) * (x1_start - xci2_start) + (y1_start - yci2_start) * (y1_start - yci2_start));
	if (gap_girth < min_gap_ext) {
		min_gap_ext = gap_girth;
	}
	gap_girth = sqrt((x2_start - xci1_start) * (x2_start - xci1_start) + (y2_start - yci1_start) * (y2_start - yci1_start));
	if (gap_girth < min_gap_ext) {
		min_gap_ext = gap_girth;
	}
	gap_girth = sqrt((x2_start - xci2_start) * (x2_start - xci2_start) + (y2_start - yci2_start) * (y2_start - yci2_start));
	if (gap_girth < min_gap_ext) {
		min_gap_ext = gap_girth;
	}
	return min_gap_ext;
}

void ContourSelectInstance::contours_combined() {
	int c2_len = contour_loop_set.size();
	std::vector<int> used_flag; used_flag.clear();
	std::vector<float> con_girth; con_girth.clear();
	int max_girth = 0; int maxid = 0;
	for (int ci = 0; ci < c2_len; ci++) {
		used_flag.push_back(0);
		std::vector<cv::Point> contour_near = contour_loop_set[ci];
		int con_size = contour_near.size(); float girth = 0.0;
		for (int cn = 0; cn < con_size; cn += 2) {
			int x1 = contour_near[cn].x, y1 = contour_near[cn].y;
			int x2 = contour_near[cn + 1].x, y2 = contour_near[cn + 1].y;
			if (x1 == x2) girth += abs(y1 - y2);
			else if (y1 == y2) girth += abs(x1 - x2);
			else if (abs(x1 - x2) == abs(y1 - y2)) girth += 1.4141 * abs(x1 - x2);
		}
		con_girth.push_back(girth);
		if (girth > max_girth) {
			max_girth = girth;
			maxid = ci;
		}
	}
	//std::vector<cv::Point> contour_object;
	std::vector<cv::Point> contour_loop; 	contour_loop.clear();
	std::vector<cv::Point> max_contor = contour_loop_set[maxid];
	for (int ci = 0; ci < max_contor.size(); ci++)	contour_loop.push_back(cv::Point(max_contor[ci].x, max_contor[ci].y));
	float min_gap_girth = 1024; int con_mode_sel = 0;
	used_flag[maxid] = 1;
	float gap_between_line = 0.0;
	max_object_ssim = 0.0;
	int cur_num = 1;
	while (1) {
		//see if loop
		int loop_size = contour_loop.size(); float contour_girth = 0.0;
		for (int i = 0; i < loop_size; i += 2) {
			float x1i_start = contour_loop[i].x, y1i_start = contour_loop[i].y;
			float x2i_start = contour_loop[i + 1].x, y2i_start = contour_loop[i + 1].y;
			float distance_i_len = sqrt((x1i_start - x2i_start) * (x1i_start - x2i_start) + (y1i_start - y2i_start) * (y1i_start - y2i_start));
			contour_girth += distance_i_len;
		}
		float x1_start = contour_loop[0].x, y1_start = contour_loop[0].y;
		float x2_start = contour_loop[1].x, y2_start = contour_loop[1].y;
		float x1_end = contour_loop[loop_size - 1].x, y1_end = contour_loop[loop_size - 1].y;
		float x2_end = contour_loop[loop_size - 2].x, y2_end = contour_loop[loop_size - 2].y;
		int con_mode = 1;
		float gap_girth = sqrt((x1_start - x1_end) * (x1_start - x1_end) + (y1_start - y1_end) * (y1_start - y1_end));
		float gap_girth1 = sqrt((x1_start - x2_end) * (x1_start - x2_end) + (y1_start - y2_end) * (y1_start - y2_end));
		if (gap_girth1 < gap_girth) {
			gap_girth = gap_girth1; con_mode = 2;
		}
		gap_girth1 = sqrt((x2_start - x1_end) * (x2_start - x1_end) + (y2_start - y1_end) * (y2_start - y1_end));
		if (gap_girth1 < gap_girth) {
			gap_girth = gap_girth1; con_mode = 3;
		}
		gap_girth1 = sqrt((x2_start - x2_end) * (x2_start - x2_end) + (y2_start - y2_end) * (y2_start - y2_end));
		if (gap_girth1 < gap_girth) {
			gap_girth = gap_girth1; con_mode = 4;
		}

		min_gap_girth = gap_girth + gap_between_line;
		float new_object_ssim = 1.0 - min_gap_girth / (min_gap_girth + contour_girth);
		if (new_object_ssim > max_object_ssim) {
			max_object_ssim = new_object_ssim;
			contour_object.clear();
			for (int ni = 0; ni < loop_size; ni++) contour_object.push_back(contour_loop[ni]);
		}
		if (cur_num == 0) break;
		//find and connect next nearest contour
		float min_gap_ext = min_gap_ceil; int near_ci = -1; int rev_mode = 0;
		for (int ci = 0; ci < c2_len; ci++) {
			//分别从每一条线段添加
			if (used_flag[ci] == 0) {
				std::vector<cv::Point> contour_near = contour_loop_set[ci];
				int len_ci = contour_near.size();

				float xci1_start = contour_near[0].x, yci1_start = contour_near[0].y;
				float xci2_start = contour_near[1].x, yci2_start = contour_near[1].y;
				//(x1_start,y1_start)---(x2_start, y2_start)   ------------>   (xci1_start,yci1_start)---(xci2_start, yci2_start)
				float gap_girth1 = distance_minimum(x1_start, y1_start, x2_start, y2_start, xci1_start, yci1_start, xci2_start, yci2_start);
				if (gap_girth1 < min_gap_ext) {
					min_gap_ext = gap_girth1;
					near_ci = ci;
					rev_mode = 1;
				}
				//(x1_end,y1_end)---(x2_end, y2_end)   ------------>   (xci1_start,yci1_start)---(xci2_start, yci2_start)
				float gap_girth2 = distance_minimum(x1_end, y1_end, x2_end, y2_end, xci1_start, yci1_start, xci2_start, yci2_start);
				if (gap_girth2 < min_gap_ext) {
					min_gap_ext = gap_girth2;
					near_ci = ci;
					rev_mode = 2;
				}
				float xci1_end = contour_near[len_ci - 1].x, yci1_end = contour_near[len_ci - 1].y;
				float xci2_end = contour_near[len_ci - 2].x, yci2_end = contour_near[len_ci - 2].y;
				//(x1_start,y1_start)---(x2_start, y2_start)   ------------>   (xci1_end,yci1_end)---(xci2_end, yci2_end)
				float gap_girth3 = distance_minimum(x1_start, y1_start, x2_start, y2_start, xci1_end, yci1_end, xci2_end, yci2_end);
				if (gap_girth3 < min_gap_ext) {
					min_gap_ext = gap_girth3;
					near_ci = ci;
					rev_mode = 3;
				}
				//(x1_end,y1_end)---(x2_end, y2_end)   ------------>   (xci1_end,yci1_end)---(xci2_end, yci2_end)
				float gap_girth4 = distance_minimum(x1_end, y1_end, x2_end, y2_end, xci1_end, yci1_end, xci2_end, yci2_end);
				if (gap_girth4 < min_gap_ext) {
					min_gap_ext = gap_girth4;
					near_ci = ci;
					rev_mode = 4;
				}
			}
		}
		if (near_ci >= 0) {
			cur_num++;
			//printf("Line point num and nearest ci : %d, %d, %d.\n", cur_num, contour_con_num, near_ci);
			used_flag[near_ci] = 1;
			gap_between_line += min_gap_ext;
			std::vector<cv::Point> contour_nearest = contour_loop_set[near_ci];
			std::vector<cv::Point> contour_combined; contour_combined.clear();
			int nearest_size = contour_nearest.size();
			if (rev_mode == 1) {
				//search the nearest neighbour points
				int x1 = contour_loop[0].x, y1 = contour_loop[0].y, x2 = contour_loop[1].x, y2 = contour_loop[1].y;
				int inc_x = 0; if (x2 > x1) inc_x = 1; if (x1 < x2) inc_x = -1;
				int inc_y = 0; if (y2 > y1) inc_y = 1; if (y1 < y2) inc_y = -1;

				int x1n = contour_nearest[0].x, y1n = contour_nearest[0].y, x2n = contour_nearest[1].x, y2n = contour_nearest[1].y;
				int inc_xn = 0; if (x2n > x1n) inc_xn = 1; if (x1n < x2n) inc_xn = -1;
				int inc_yn = 0; if (y2n > y1n) inc_yn = 1; if (y1n < y2n) inc_yn = -1;
				int newx = x1, newy = y1, new_xn = x1n, new_yn = y1n;
				int loop = 16;
				while (loop-- > 0) {
					int inc_flag_min = 0, inc_flagn_min = 0;
					float distance_gap = sqrt((newx - new_xn) * (newx - new_xn) + (newy - new_yn) * (newy - new_yn));
					for (int inc_flag = -1; inc_flag <= 1; inc_flag++) {
						int stepx = newx + inc_x * inc_flag, stepy = newy + inc_y * inc_flag;
						for (int inc_nflag = -1; inc_nflag <= 1; inc_nflag++) {
							int stepxn = new_xn + inc_xn * inc_nflag, stepyn = new_yn + inc_yn * inc_nflag;
							float distance_gap_new = sqrt((stepx - stepxn) * (stepx - stepxn) + (stepy - stepyn) * (stepy - stepyn));
							if (distance_gap_new < distance_gap) {
								distance_gap_new = distance_gap;
								inc_flag_min = inc_flag; inc_flagn_min = inc_nflag;
							}
						}
					}
					if ((inc_flag_min == 0) && (inc_flagn_min == 0)) break;
					newx = newx + inc_x * inc_flag_min; newy = newy + inc_y * inc_flag_min;
					new_xn = new_xn + inc_xn * inc_flagn_min; new_yn = new_yn + inc_yn * inc_flagn_min;
					if ((newx == x2) && (newy == y2)) {
						break;
					}
					if ((new_xn == x2n) && (newy == new_yn)) {
						break;
					}
				}
				//combine the new line
				for (int ni = loop_size - 1; ni > 0; ni--) contour_combined.push_back(contour_loop[ni]);
				contour_combined.push_back(cv::Point(newx, newy));
				contour_combined.push_back(cv::Point(newx, newy)); contour_combined.push_back(cv::Point(new_xn, new_yn));  //add new line
				contour_combined.push_back(cv::Point(new_xn, new_yn));
				for (int ni = 1; ni < nearest_size; ni++) contour_combined.push_back(contour_nearest[ni]);
			}
			else if (rev_mode == 2) {
				//search the nearest neighbour points
				int x1 = contour_loop[loop_size - 1].x, y1 = contour_loop[loop_size - 1].y, x2 = contour_loop[loop_size - 2].x, y2 = contour_loop[loop_size - 2].y;
				int inc_x = 0; if (x2 > x1) inc_x = 1; if (x1 < x2) inc_x = -1;
				int inc_y = 0; if (y2 > y1) inc_y = 1; if (y1 < y2) inc_y = -1;

				int x1n = contour_nearest[0].x, y1n = contour_nearest[0].y, x2n = contour_nearest[1].x, y2n = contour_nearest[1].y;
				int inc_xn = 0; if (x2n > x1n) inc_xn = 1; if (x1n < x2n) inc_xn = -1;
				int inc_yn = 0; if (y2n > y1n) inc_yn = 1; if (y1n < y2n) inc_yn = -1;
				int newx = x1, newy = y1, new_xn = x1n, new_yn = y1n;
				int loop = 16;
				while (loop-- > 0) {
					int inc_flag_min = 0, inc_flagn_min = 0;
					float distance_gap = sqrt((newx - new_xn) * (newx - new_xn) + (newy - new_yn) * (newy - new_yn));
					for (int inc_flag = -1; inc_flag <= 1; inc_flag++) {
						int stepx = newx + inc_x * inc_flag, stepy = newy + inc_y * inc_flag;
						for (int inc_nflag = -1; inc_nflag <= 1; inc_nflag++) {
							int stepxn = new_xn + inc_xn * inc_nflag, stepyn = new_yn + inc_yn * inc_nflag;
							float distance_gap_new = sqrt((stepx - stepxn) * (stepx - stepxn) + (stepy - stepyn) * (stepy - stepyn));
							if (distance_gap_new < distance_gap) {
								distance_gap_new = distance_gap;
								inc_flag_min = inc_flag; inc_flagn_min = inc_nflag;
							}
						}
					}
					if ((inc_flag_min == 0) && (inc_flagn_min == 0)) break;
					newx = newx + inc_x * inc_flag_min; newy = newy + inc_y * inc_flag_min;
					new_xn = new_xn + inc_xn * inc_flagn_min; new_yn = new_yn + inc_yn * inc_flagn_min;
					if ((newx == x2) && (newy == y2)) {
						break;
					}
					if ((new_xn == x2n) && (newy == new_yn)) {
						break;
					}
				}
				//combine the new line
				for (int ni = 0; ni < loop_size - 1; ni++) contour_combined.push_back(contour_loop[ni]);
				contour_combined.push_back(cv::Point(newx, newy));
				contour_combined.push_back(cv::Point(newx, newy)); contour_combined.push_back(cv::Point(new_xn, new_yn));  //add new line
				contour_combined.push_back(cv::Point(new_xn, new_yn));
				for (int ni = 1; ni < nearest_size; ni++) contour_combined.push_back(contour_nearest[ni]);
			}
			else if (rev_mode == 3) {
				//search the nearest neighbour points
				int x1 = contour_loop[0].x, y1 = contour_loop[0].y, x2 = contour_loop[1].x, y2 = contour_loop[1].y;
				int inc_x = 0; if (x2 > x1) inc_x = 1; if (x1 < x2) inc_x = -1;
				int inc_y = 0; if (y2 > y1) inc_y = 1; if (y1 < y2) inc_y = -1;

				int x1n = contour_nearest[nearest_size - 1].x, y1n = contour_nearest[nearest_size - 1].y, x2n = contour_nearest[nearest_size - 2].x, y2n = contour_nearest[nearest_size - 2].y;
				int inc_xn = 0; if (x2n > x1n) inc_xn = 1; if (x1n < x2n) inc_xn = -1;
				int inc_yn = 0; if (y2n > y1n) inc_yn = 1; if (y1n < y2n) inc_yn = -1;
				int newx = x1, newy = y1, new_xn = x1n, new_yn = y1n;
				int loop = 16;
				while (loop-- > 0) {
					int inc_flag_min = 0, inc_flagn_min = 0;
					float distance_gap = sqrt((newx - new_xn) * (newx - new_xn) + (newy - new_yn) * (newy - new_yn));
					for (int inc_flag = -1; inc_flag <= 1; inc_flag++) {
						int stepx = newx + inc_x * inc_flag, stepy = newy + inc_y * inc_flag;
						for (int inc_nflag = -1; inc_nflag <= 1; inc_nflag++) {
							int stepxn = new_xn + inc_xn * inc_nflag, stepyn = new_yn + inc_yn * inc_nflag;
							float distance_gap_new = sqrt((stepx - stepxn) * (stepx - stepxn) + (stepy - stepyn) * (stepy - stepyn));
							if (distance_gap_new < distance_gap) {
								distance_gap_new = distance_gap;
								inc_flag_min = inc_flag; inc_flagn_min = inc_nflag;
							}
						}
					}
					if ((inc_flag_min == 0) && (inc_flagn_min == 0)) break;
					newx = newx + inc_x * inc_flag_min; newy = newy + inc_y * inc_flag_min;
					new_xn = new_xn + inc_xn * inc_flagn_min; new_yn = new_yn + inc_yn * inc_flagn_min;
					if ((newx == x2) && (newy == y2)) {
						break;
					}
					if ((new_xn == x2n) && (newy == new_yn)) {
						break;
					}
				}
				//combine the new line
				for (int ni = loop_size - 1; ni > 0; ni--) contour_combined.push_back(contour_loop[ni]);
				contour_combined.push_back(cv::Point(newx, newy));
				contour_combined.push_back(cv::Point(newx, newy)); contour_combined.push_back(cv::Point(new_xn, new_yn));  //add new line
				contour_combined.push_back(cv::Point(new_xn, new_yn));
				for (int ni = nearest_size - 2; ni >= 0; ni--) contour_combined.push_back(contour_nearest[ni]);
			}
			else if (rev_mode == 4) {
				//search the nearest neighbour points
				int x1 = contour_loop[loop_size - 1].x, y1 = contour_loop[loop_size - 1].y, x2 = contour_loop[loop_size - 2].x, y2 = contour_loop[loop_size - 2].y;
				int inc_x = 0; if (x2 > x1) inc_x = 1; if (x1 < x2) inc_x = -1;
				int inc_y = 0; if (y2 > y1) inc_y = 1; if (y1 < y2) inc_y = -1;

				int x1n = contour_nearest[nearest_size - 1].x, y1n = contour_nearest[nearest_size - 1].y, x2n = contour_nearest[nearest_size - 2].x, y2n = contour_nearest[nearest_size - 2].y;
				int inc_xn = 0; if (x2n > x1n) inc_xn = 1; if (x1n < x2n) inc_xn = -1;
				int inc_yn = 0; if (y2n > y1n) inc_yn = 1; if (y1n < y2n) inc_yn = -1;
				int newx = x1, newy = y1, new_xn = x1n, new_yn = y1n;
				int loop = 16;
				while (loop-- > 0) {
					int inc_flag_min = 0, inc_flagn_min = 0;
					float distance_gap = sqrt((newx - new_xn) * (newx - new_xn) + (newy - new_yn) * (newy - new_yn));
					for (int inc_flag = -1; inc_flag <= 1; inc_flag++) {
						int stepx = newx + inc_x * inc_flag, stepy = newy + inc_y * inc_flag;
						for (int inc_nflag = -1; inc_nflag <= 1; inc_nflag++) {
							int stepxn = new_xn + inc_xn * inc_nflag, stepyn = new_yn + inc_yn * inc_nflag;
							float distance_gap_new = sqrt((stepx - stepxn) * (stepx - stepxn) + (stepy - stepyn) * (stepy - stepyn));
							if (distance_gap_new < distance_gap) {
								distance_gap_new = distance_gap;
								inc_flag_min = inc_flag; inc_flagn_min = inc_nflag;
							}
						}
					}
					if ((inc_flag_min == 0) && (inc_flagn_min == 0)) break;
					newx = newx + inc_x * inc_flag_min; newy = newy + inc_y * inc_flag_min;
					new_xn = new_xn + inc_xn * inc_flagn_min; new_yn = new_yn + inc_yn * inc_flagn_min;
					if ((newx == x2) && (newy == y2)) {
						break;
					}
					if ((new_xn == x2n) && (newy == new_yn)) {
						break;
					}
				}
				//combine the new line
				for (int ni = 0; ni < loop_size - 1; ni++) contour_combined.push_back(contour_loop[ni]);
				contour_combined.push_back(cv::Point(newx, newy));
				contour_combined.push_back(cv::Point(newx, newy)); contour_combined.push_back(cv::Point(new_xn, new_yn));  //add new line
				contour_combined.push_back(cv::Point(new_xn, new_yn));
				for (int ni = nearest_size - 2; ni >= 0; ni--) contour_combined.push_back(contour_nearest[ni]);
			}
			contour_loop.clear();
			for (int ni = 0; ni < contour_combined.size(); ni++) contour_loop.push_back(cv::Point(contour_combined[ni].x, contour_combined[ni].y));
		}
		else break;

	}

}

int ContourSelectInstance::extend_new_contours(const cv::Mat& amps)
{
	int height = amps.rows, width = amps.cols;
	const long first_row = std::max(M / 2, (int)ceil(roi_y_min * height));
	const long first_col = std::max(N / 2, (int)ceil(roi_x_min * width));
	const long last_row = std::min(height - M / 2, (int)ceil(roi_y_max * height));
	const long last_col = std::min(width - N / 2, (int)ceil(roi_x_max * width));
	cv::Size processSize = cv::Size(amps.cols, amps.rows);
	cv::Mat pix_cur_set = cv::Mat::zeros(processSize, CV_32F);
	int back_cons_len = cons_len;
	for (int cid = 0; cid < back_cons_len; cid++) {
		struct pt_list* list_head = contours_pool + cid;
		int cur_pts_sz = list_head->len;
		bool extend_ret_neg = list_head->tail_extend_en, extend_ret_pos = list_head->head_extend_en;
		
		bool extend_ret = extend_ret_neg || extend_ret_pos;
		if ( extend_ret ) {
			//printf("###extend_new_contours for contours:%d start\n", cid);
			int zero_offset = MAX_PT_RECORD / 2;
			struct pt_struct* pt_head_cur = list_head->pt_head, *pt_tail_cur = list_head->pt_head + cur_pts_sz - 1;
			int positive_offset = zero_offset, negative_offset = zero_offset;
			int loop_cnt = 0; const int max_loop = 2;
			memset(record_pts, 0, MAX_PT_RECORD * sizeof(struct pt_struct));
			memcpy(record_pts + zero_offset, pt_head_cur + 4, sizeof(struct pt_struct)); memset(record_pts[zero_offset].mask, 0, 8 * sizeof(bool));
			memcpy(record_pts + zero_offset - 1, pt_head_cur + 3, sizeof(struct pt_struct)); memset(record_pts[zero_offset - 1].mask, 0, 8 * sizeof(bool));
			memcpy(record_pts + zero_offset - 2, pt_head_cur + 2, sizeof(struct pt_struct)); memset(record_pts[zero_offset - 2].mask, 0, 8 * sizeof(bool));
			memcpy(record_pts + zero_offset - 3, pt_head_cur + 1, sizeof(struct pt_struct)); memset(record_pts[zero_offset - 3].mask, 0, 8 * sizeof(bool));
			memcpy(record_pts + zero_offset - 4, pt_head_cur + 0, sizeof(struct pt_struct)); memset(record_pts[zero_offset - 4].mask, 0, 8 * sizeof(bool));
			while (extend_ret_neg) {
				extend_ret_neg = extend_band_double(amps, record_pts, negative_offset, true, pix_cur_set);
				if (extend_ret_neg) negative_offset = negative_offset - slide_maximum_len / 2 + 1;
				loop_cnt++;
				if (loop_cnt >= max_loop) break;
			}
			int extend_pts_num = 0;
			for (int pts_i = zero_offset - slide_maximum_len * max_loop; pts_i <= zero_offset; pts_i++) {
				int pts_len = record_pts[pts_i].size;
				if (pts_len > 0) extend_pts_num++;
			}
			if (extend_pts_num >= 1 * slide_maximum_len) {
				//printf("	extend_new_contours for contours:%d, find new contours %d, pts size:%d\n", cid, cons_len, extend_pts_num);
				std::vector<cv::Point> line_sequence; line_sequence.clear();
				cv::Mat distribute2d = cv::Mat::zeros(processSize, CV_32S);
				struct pt_list* new_list_head = contours_pool + cons_len;
				new_list_head->pt_head = (struct pt_struct*)malloc(extend_pts_num * sizeof(struct pt_struct));
				new_list_head->len = extend_pts_num;
				new_list_head->selection = false; new_list_head->sel_in = -1; new_list_head->sel_out = -1;
				new_list_head->branch_sz = 0; new_list_head->head_extend_en = false; new_list_head->tail_extend_en = true;
				int pts_ptr_idx = 0;
				for (int pts_i = zero_offset - slide_maximum_len * max_loop; pts_i <= zero_offset; pts_i++) { //set used flag
					int pts_c = record_pts[pts_i].px, pts_r = record_pts[pts_i].py;
					int pts_direction = record_pts[pts_i].direction;
					int pts_len = record_pts[pts_i].size;
					if (pts_len > 0) {
						line_sequence.push_back(cv::Point(record_pts[pts_i].px, record_pts[pts_i].py));
						for (int i = 0; i < pts_len; i++) {
							pts_c += incx1[pts_direction]; pts_r += incy1[pts_direction];
							pix_cur_set.at<float>(pts_r, pts_c) = 0.0;
							distribute2d.at<int>(pts_r, pts_c) = 1;
						}
						memcpy(new_list_head->pt_head + pts_ptr_idx, record_pts + pts_i, sizeof(struct pt_struct));
						pts_ptr_idx++;
					}
				}
				contour_loop_set.push_back(line_sequence);
				struct crossx_struct* cross_ptr = crossx_pool + cross_len;
				cross_ptr->listA = cid; cross_ptr->listB = cons_len;
				cross_ptr->A_ptr = 3; cross_ptr->B_ptr = extend_pts_num - 2; cross_ptr->used_status = 0;
				int branch_len = list_head->branch_sz, branch_len_new = new_list_head->branch_sz;
				int* branch_idx = list_head->branch_info, *branch_idx_new = new_list_head->branch_info;
				int* branch_status = list_head->b_recall_status, *branch_status_new = new_list_head->b_recall_status;
				branch_idx[branch_len] = cross_len; branch_idx_new[branch_len_new] = cross_len;
				branch_status[branch_len] = 0;  branch_status_new[branch_len] = 0;
				list_head->tail_extend_en = false;  list_head->branch_sz++;
				new_list_head->branch_sz++; cross_len++; cons_len++;
				//printf("	Add new cross point idx:%d, bridged between %d and %d, index %d , %d\n", cross_len, cross_ptr->listA, cross_ptr->listB, cross_ptr->A_ptr, cross_ptr->B_ptr);
				if (cons_len > MAX_POOL_SZ) {
					printf("Error contour number bigger than maximum value\n"); while (1);
				}
				if (cross_len > 2 * MAX_POOL_SZ) {
					printf("Error cross X number bigger than maximum value\n"); while (1);
				}
				pts_distributed.push_back(distribute2d);
				list_head->head_extend_en = false;
			}

			loop_cnt = 0;
			memset(record_pts, 0, MAX_PT_RECORD * sizeof(struct pt_struct));
			memcpy(record_pts + zero_offset, pt_tail_cur - 4, sizeof(struct pt_struct));     memset(record_pts[zero_offset].mask, 0, 8 * sizeof(bool));
			memcpy(record_pts + zero_offset + 1, pt_tail_cur - 3, sizeof(struct pt_struct)); memset(record_pts[zero_offset + 1].mask, 0, 8 * sizeof(bool));
			memcpy(record_pts + zero_offset + 2, pt_tail_cur - 2, sizeof(struct pt_struct)); memset(record_pts[zero_offset + 2].mask, 0, 8 * sizeof(bool));
			memcpy(record_pts + zero_offset + 3, pt_tail_cur - 1, sizeof(struct pt_struct)); memset(record_pts[zero_offset + 3].mask, 0, 8 * sizeof(bool));
			memcpy(record_pts + zero_offset + 4, pt_tail_cur + 0, sizeof(struct pt_struct)); memset(record_pts[zero_offset + 4].mask, 0, 8 * sizeof(bool));
			while (extend_ret_pos) {
				extend_ret_pos = extend_band_double(amps, record_pts, positive_offset, false, pix_cur_set);
				if (extend_ret_pos)	positive_offset = positive_offset + slide_maximum_len / 2 - 1;
				loop_cnt++;
				if (loop_cnt >= max_loop) break;
			}
			
			extend_pts_num = 0;
			for (int pts_i = zero_offset; pts_i <= zero_offset + slide_maximum_len * max_loop; pts_i++) {
				int pts_len = record_pts[pts_i].size;
				if (pts_len > 0) extend_pts_num++;
			}
			if (extend_pts_num >= 1 * slide_maximum_len) {
				//printf("	extend_new_contours for contours:%d, find new contours %d, points size:%d\n", cid, cons_len, extend_pts_num);
				std::vector<cv::Point> line_sequence; line_sequence.clear();
				cv::Mat distribute2d = cv::Mat::zeros(processSize, CV_32S);
				struct pt_list* new_list_head = contours_pool + cons_len;
				new_list_head->pt_head = (struct pt_struct*)malloc(extend_pts_num * sizeof(struct pt_struct));
				new_list_head->len = extend_pts_num;
				new_list_head->selection = false; new_list_head->sel_in = -1; new_list_head->sel_out = -1;
				new_list_head->branch_sz = 0; new_list_head->tail_extend_en = false; new_list_head->head_extend_en = true;
				int pts_ptr_idx = 0;
				for (int pts_i = zero_offset; pts_i <= zero_offset + slide_maximum_len * max_loop; pts_i++) { //set used flag
					int pts_c = record_pts[pts_i].px, pts_r = record_pts[pts_i].py;
					int pts_direction = record_pts[pts_i].direction;
					int pts_len = record_pts[pts_i].size;
					if (pts_len > 0) {
						line_sequence.push_back(cv::Point(record_pts[pts_i].px, record_pts[pts_i].py));
						for (int i = 0; i < pts_len; i++) {
							pts_c += incx1[pts_direction]; pts_r += incy1[pts_direction];
							pix_cur_set.at<float>(pts_r, pts_c) = 0.0;
							distribute2d.at<int>(pts_r, pts_c) = 1;
						}
						memcpy(new_list_head->pt_head + pts_ptr_idx, record_pts + pts_i, sizeof(struct pt_struct));
						pts_ptr_idx++;
					}
				}
				contour_loop_set.push_back(line_sequence);
				struct crossx_struct* cross_ptr = crossx_pool + cross_len;
				cross_ptr->listA = cid; cross_ptr->listB = cons_len;
				cross_ptr->A_ptr = cur_pts_sz - 2; cross_ptr->B_ptr = 3; cross_ptr->used_status = 0;
				int branch_len = list_head->branch_sz, branch_len_new = new_list_head->branch_sz;
				int* branch_idx = list_head->branch_info, *branch_idx_new = new_list_head->branch_info;
				int* branch_status = list_head->b_recall_status, *branch_status_new = new_list_head->b_recall_status;
				branch_idx[branch_len] = cross_len; branch_idx_new[branch_len_new] = cross_len;
				branch_status[branch_len] = 0;  branch_status_new[branch_len] = 0;
				list_head->head_extend_en = false;  list_head->branch_sz++;
				new_list_head->branch_sz++; cross_len++; cons_len++;
				//printf("	Add new cross point idx:%d, bridged between %d and %d, index %d , %d\n", cross_len, cross_ptr->listA, cross_ptr->listB, cross_ptr->A_ptr, cross_ptr->B_ptr);
				if (cons_len > MAX_POOL_SZ) {
					printf("Error contour number bigger than maximum value\n"); while (1);
				}
				if (cross_len > 2 * MAX_POOL_SZ) {
					printf("Error cross X number bigger than maximum value\n"); while (1);
				}
				pts_distributed.push_back(distribute2d);
				list_head->tail_extend_en = false;
			}
			//printf("###extend_new_contours for contours:%d end\n", cid);
		}
	}

	return cons_len - back_cons_len;
}

void ContourSelectInstance::contours_search_intersection(int cid) {
	//head branch
	struct pt_list* list_head = contours_pool + cid;
	struct pt_struct* pts_ptr = list_head->pt_head;
	int* branch_idx = list_head->branch_info; int* branch_status = list_head->b_recall_status;
	int pts_len = list_head->len;
	struct pt_struct* pts_tail2, *pts_head2;
	const int max_extend = 5;

	if (list_head->head_extend_en) {
		int intersection_num = 0;
		for (int fid = 0; fid < cons_len; fid++) {
			if (cid != fid) {
				cv::Mat fmat = pts_distributed[fid];
				int extend_cnt = 0; int pts_tail_idx = pts_len - 1;
				while (extend_cnt <= max_extend) {
					pts_tail2 = pts_ptr + pts_tail_idx;
					int pts_r = pts_tail2->py, pts_c = pts_tail2->px, pts_d = pts_tail2->direction;
					int pts_r2 = pts_r + incy1[pts_d], pts_c2 = pts_c + incx1[pts_d];

					if ((fmat.at<int>(pts_r, pts_c) == 1) || (fmat.at<int>(pts_r2, pts_c2) == 1)) { //found new cross point
						struct crossx_struct* cross_ptr = crossx_pool + cross_len;
						struct pt_list* list_head_f = contours_pool + fid;
						struct pt_struct* pts_ptr_f = list_head_f->pt_head;
						int* branch_idx_f = list_head_f->branch_info; int* branch_status_f = list_head_f->b_recall_status;
						int pts_len_f = list_head_f->len; int BP = -1;
						for (int lfi = 0; lfi < pts_len_f; lfi++) {
							struct pt_struct* pts_crs = pts_ptr_f + lfi;
							int crs_r = pts_crs->py, crs_c = pts_crs->px, crs_d = pts_crs->direction;
							int crs_r2 = crs_r + incy1[crs_d], crs_c2 = crs_c + incx1[crs_d];
							if ((crs_r == pts_r) && (crs_c == pts_c)) { BP = lfi; break; }
							if ((crs_r == pts_r2) && (crs_c == pts_c2)) { BP = lfi; break; }
							if ((crs_r2 == pts_r) && (crs_c2 == pts_c)) { BP = lfi; break; }
							if ((crs_r2 == pts_r2) && (crs_c2 == pts_c2)) { BP = lfi; break; }
						}
						if (BP >= 0) {
							cross_ptr->listA = cid; cross_ptr->listB = fid;
							cross_ptr->A_ptr = pts_tail_idx; cross_ptr->B_ptr = BP; cross_ptr->used_status = 0;
							int branch_len = list_head->branch_sz, branch_len_f = list_head_f->branch_sz;
							branch_idx[branch_len] = cross_len; branch_idx_f[branch_len_f] = cross_len;
							branch_status[branch_len] = 0;  branch_status_f[branch_len] = 0;
							list_head->head_extend_en = false;  list_head->branch_sz++;
							list_head_f->branch_sz++; cross_len++;
							if (cross_len > 2 * MAX_POOL_SZ) {
								printf("Error cross X number bigger than maximum value\n"); while (1);
							}
							printf("Found cross point idx:%d, bridged between %d and %d, index %d , %d\n", cross_len, cid, fid, pts_tail_idx, BP);
							intersection_num++;
							break;
						}
					}
					extend_cnt++;
					pts_tail_idx--;
				}
				if (intersection_num > 0) break;
			}
		}
	}
	if(list_head->tail_extend_en) {
		int intersection_num = 0;
		for (int fid = 0; fid < cons_len; fid++) {
			if (cid != fid) {
				cv::Mat fmat = pts_distributed[fid];
				int pts_head_idx = 0; int extend_cnt = 0;
				while (extend_cnt <= max_extend) {
					pts_head2 = pts_ptr + pts_head_idx;
					int pts_r = pts_head2->py, pts_c = pts_head2->px, pts_d = pts_head2->direction;
					int pts_r2 = pts_r + incy1[pts_d], pts_c2 = pts_c + incx1[pts_d];

					if ((fmat.at<int>(pts_r, pts_c) == 1) || (fmat.at<int>(pts_r2, pts_c2) == 1)) { //found new cross point
						struct crossx_struct* cross_ptr = crossx_pool + cross_len;
						struct pt_list* list_head_f = contours_pool + fid;
						struct pt_struct* pts_ptr_f = list_head_f->pt_head;
						int* branch_idx_f = list_head_f->branch_info;
						int* branch_status_f = list_head_f->b_recall_status;
						int pts_len_f = list_head_f->len; int BP = -1;
						for (int lfi = 0; lfi < pts_len_f; lfi++) {
							struct pt_struct* pts_crs = pts_ptr_f + lfi;
							int crs_r = pts_crs->py, crs_c = pts_crs->px, crs_d = pts_crs->direction;
							int crs_r2 = crs_r + incy1[crs_d], crs_c2 = crs_c + incx1[crs_d];
							if ((crs_r == pts_r) && (crs_c == pts_c)) { BP = lfi; break; }
							if ((crs_r == pts_r2) && (crs_c == pts_c2)) { BP = lfi; break; }
							if ((crs_r2 == pts_r) && (crs_c2 == pts_c)) { BP = lfi; break; }
							if ((crs_r2 == pts_r2) && (crs_c2 == pts_c2)) { BP = lfi; break; }
						}
						if (BP >= 0) {
							cross_ptr->listA = cid; cross_ptr->listB = fid;
							cross_ptr->A_ptr = pts_head_idx; cross_ptr->B_ptr = BP; cross_ptr->used_status = 0;
							int branch_len = list_head->branch_sz, branch_len_f = list_head_f->branch_sz;
							branch_idx[branch_len] = cross_len; branch_idx_f[branch_len_f] = cross_len;
							branch_status[branch_len] = 0;  branch_status_f[branch_len] = 0;
							list_head->tail_extend_en = false;  list_head->branch_sz++;
							list_head_f->branch_sz++; cross_len++;
							intersection_num++;
							if (cross_len > 2 * MAX_POOL_SZ) {
								printf("Error cross X number bigger than maximum value\n"); while (1);
							}
							printf("Found cross point index %d, link between %d and %d, index %d , %d\n", cross_len, cid, fid, pts_head_idx, BP);
							break;
						}
					}

					extend_cnt++;
					pts_head_idx++;
				}
				if (intersection_num > 0) break;
			}
		}
	}
}

void ContourSelectInstance::contours_maximum_combined(const cv::Mat& amps) {

	if (cons_len > 0)  //connect two lines
	for (int cid = 0; cid < cons_len; cid++) {//find cross point for all contours
		//head branch
		contours_search_intersection(cid);
	}
	int ext_len = extend_new_contours(amps);
	if (ext_len > 0) {
		for (int cid = cons_len - ext_len; cid < cons_len; cid++) {
			//head branch
			contours_search_intersection(cid);
		}
	}
	struct pt_list* contours_opts = (struct pt_list*)malloc(cons_len * sizeof(struct pt_list));
	memcpy(contours_opts, contours_pool, cons_len * sizeof(struct pt_list));
	printf( "Current contours size:%d, mask num:%d\n", cons_len, pts_distributed.size() );
	
	int max_object_pts = -100;
	if (cons_len > 0) { //try branches and connect lines
		for (int cid = 0; cid < cons_len; cid++) {
			struct pt_list* start_list_head = contours_pool + cid;
			struct pt_struct* pts_head = start_list_head->pt_head;
			int pts_len = start_list_head->len;
			int* branchs_head = start_list_head->branch_info;
			start_list_head->selection = true;
			int cur_list = cid, pre_list = -1;
			struct pt_list* pre_list_head = NULL, *cur_list_head = start_list_head;
			int branch_sz;
			while (1) {
				int cur_sel_in = cur_list_head->sel_in, cur_sel_out = cur_list_head->sel_out;
				int branch_id = 0, cross_id = -1; bool extend_or_recall = false;
				struct crossx_struct* cur_branch_ptr = NULL;
				branch_sz = cur_list_head->branch_sz;
				while (branch_id < branch_sz) {
					cross_id = cur_list_head->branch_info[branch_id];
					cur_branch_ptr = crossx_pool + cross_id;
					int ptr_branch_recall = cur_list_head->b_recall_status[branch_id];
					if (ptr_branch_recall == 0) { //check branch's recall status
						if (cur_branch_ptr->used_status == 0) { //check branch's used status
							int first_list = cur_branch_ptr->listA;
							int second_list = cur_branch_ptr->listB;
							cur_list_head->sel_out = cross_id;
							if (cur_list == first_list) {
								pre_list = first_list;
								cur_list = second_list;
								pre_list_head = cur_list_head;
								cur_list_head = contours_pool + second_list;
							}
							else if (cur_list == second_list) {
								pre_list = second_list;
								cur_list = first_list;
								pre_list_head = cur_list_head;
								cur_list_head = contours_pool + first_list;
							}
							else {
								printf("Found error cross point strcture.\n"); while (1);
							}
							
							if (cur_list_head->selection) { //发现环路闭合, 不能扩展, 不连接交点
								float pts_len = 0;
								struct pt_list* list_next = cur_list_head, *list_pre = NULL; int cur_list_idx = cur_list;
								cur_sel_in = cross_id;
								cur_sel_out = list_next->sel_out;
								std::vector<cv::Point> contour_pts; contour_pts.clear();
								struct crossx_struct* cur_branch_in = crossx_pool + cur_sel_in, *cur_branch_out = crossx_pool + cur_sel_out;
								while (1) {
									int in_num = 0, out_num = 0;
									if (cur_branch_in->listA == cur_list_idx)	in_num = cur_branch_in->A_ptr;
									else if (cur_branch_in->listB == cur_list_idx)  in_num = cur_branch_in->B_ptr;

									if (cur_branch_out->listA == cur_list_idx) {
										out_num = cur_branch_out->A_ptr;
										cur_list_idx = cur_branch_out->listB;
										list_pre = list_next;
										list_next = contours_pool + cur_list_idx;
									}
									else if (cur_branch_out->listB == cur_list_idx) {
										out_num = cur_branch_out->B_ptr;
										cur_list_idx = cur_branch_out->listA;
										list_pre = list_next;
										list_next = contours_pool + cur_list_idx;
									}

									pts_len = pts_len + abs(in_num - out_num);
									if (list_next == cur_list_head) {
										//printf("Loop over all\n"); 
										break;
									}
									if ((list_next->sel_in < 0) || (list_next->sel_out < 0)) {
										printf("Found loop with no branch in or branch out\n"); while (1);
									}
									cur_sel_in = list_next->sel_in;
									cur_branch_in = crossx_pool + cur_sel_in;
									cur_sel_out = list_next->sel_out;
									cur_branch_out = crossx_pool + cur_sel_out;
								}
								if (pts_len > max_object_pts) {//record the maximum new points list
									memcpy(contours_opts, contours_pool, cons_len * sizeof(struct pt_list));
									for (int i_cons = 0; i_cons < cons_len; i_cons++) {
										struct pt_list* contours_cy = contours_opts + i_cons;
										contours_cy->selection = 0; contours_cy->sel_in = -1; contours_cy->sel_out = -1;
									}
									list_next = cur_list_head; cur_list_idx = cur_list;
									struct pt_list* list_next_copy = contours_opts + cur_list_idx;
									memcpy(list_next_copy, list_next, sizeof(struct pt_list));
									list_next_copy->sel_in = list_pre->sel_out;
									while (1) {	
										cur_sel_out = list_next->sel_out;
										struct crossx_struct* cur_branch_out = crossx_pool + cur_sel_out;
										if (cur_branch_out->listA == cur_list_idx) {
											cur_list_idx = cur_branch_out->listB;
											list_next = contours_pool + cur_list_idx;
										}
										else if (cur_branch_out->listB == cur_list_idx) {
											cur_list_idx = cur_branch_out->listA;
											list_next = contours_pool + cur_list_idx;
										}
										if (list_next == cur_list_head) break;
										list_next_copy = contours_opts + cur_list_idx;
										memcpy(list_next_copy, list_next, sizeof(struct pt_list));
									}
									max_object_pts = pts_len;
								}
								cur_list = pre_list;
								cur_list_head = pre_list_head;
								cur_list_head->b_recall_status[branch_id] = 1;
								cur_list_head->sel_out = -1;
								branch_id++;
								continue;
							}
							else { // 连接没有环路闭合，自然扩展， 记录交点in cross & out cross.
								cur_list_head->sel_in = cross_id;
								cur_list_head->selection = true;
								cur_branch_ptr->used_status = 1;
								pre_list_head->b_recall_status[branch_id] = 1;
								extend_or_recall = true;
								break;
							}
						}
						
					} else if (ptr_branch_recall == 1) {//recall status
						cur_list_head->b_recall_status[branch_id] = 2;
					}

					branch_id++;
				}
				if ( !extend_or_recall ) {   //recall shorten size
					cur_list_head->selection = false;
					for (int bid_r = 0; bid_r < branch_sz; bid_r++)
						cur_list_head->b_recall_status[bid_r] = 0;    //clear recall status for outside old line

					cur_sel_in = cur_list_head->sel_in;
					if (cur_sel_in < 0) {  //do not have input cross point
						break;
					} else {
						cur_list_head->sel_in = -1;
						struct crossx_struct* ptr_branch_in = crossx_pool + cur_sel_in;
						ptr_branch_in->used_status = 0;
						int listA = ptr_branch_in->listA, listB = ptr_branch_in->listB;
						if (cur_list == listA) {
							cur_list = listB;
							cur_list_head = contours_pool + listB;
						}
						else {
							cur_list = listA;
							cur_list_head = contours_pool + listA;
						}
						cur_list_head->sel_out = -1;
					}
				}
			}
			start_list_head->selection = false;
			start_list_head->sel_in = -1; start_list_head->sel_out = -1;
			branch_sz = start_list_head->branch_sz;
			for (int bid_r = 0; bid_r < branch_sz; bid_r++)
				start_list_head->b_recall_status[bid_r] = 0;
		}
		max_object_girth = 0.0;
		max_object_area = 0.0;
		if (max_object_pts > 32) {
			printf("Detect loop circle object girth:%d.\n", max_object_pts);
			contours_object_opt.push_back(contours_opts);
			for (int cid = 0; cid < cons_len; cid++) {
				struct pt_list* pt_list_st = contours_opts + cid;
				int branch_in = pt_list_st->sel_in, branch_out = pt_list_st->sel_out;
				if ((branch_in < 0) || (branch_out < 0)) continue;
				contour_object.clear();
				struct pt_list* list_next = pt_list_st, *list_pre = NULL;
				int cur_list_idx = cid;
				while (1) {
					branch_in = list_next->sel_in;
					branch_out = list_next->sel_out;
					struct pt_struct* pt_list_hd = list_next->pt_head;
					if ( (branch_in >= 0) && (branch_out >= 0) ) {
						struct crossx_struct* in_crossx = crossx_pool + branch_in, *out_crossx = crossx_pool + branch_out;
						int cur_ptr_in = in_crossx->A_ptr, cur_ptr_out = out_crossx->A_ptr;
						if (in_crossx->listB == cur_list_idx) {
							cur_ptr_in = in_crossx->B_ptr;
						}
						if (out_crossx->listB == cur_list_idx) {
							cur_ptr_out = out_crossx->B_ptr;
						}
						//int cur_ptr_min = std::min(cur_ptr_in, cur_ptr_out), cur_ptr_max = std::max(cur_ptr_in, cur_ptr_out);
						if (cur_ptr_in < cur_ptr_out) {
							for (int cur_ptr = cur_ptr_in; cur_ptr <= cur_ptr_out; cur_ptr++) {
								struct pt_struct* cur_pt = pt_list_hd + cur_ptr;
								if (cur_pt->direction % 2 == 0) max_object_girth += 2;
								else max_object_girth += 2 * sqrt(2);
								contour_object.push_back(cv::Point(cur_pt->px, cur_pt->py));
							}
						}
						else {
							for (int cur_ptr = cur_ptr_in; cur_ptr >= cur_ptr_out; cur_ptr--) {
								struct pt_struct* cur_pt = pt_list_hd + cur_ptr;
								if (cur_pt->direction % 2 == 0) max_object_girth += 2;
								else max_object_girth += 2 * sqrt(2);
								contour_object.push_back(cv::Point(cur_pt->px, cur_pt->py));
							}
						}
						if (out_crossx->listA == cur_list_idx) {
							cur_list_idx = out_crossx->listB;
							list_next = contours_opts + cur_list_idx;
						}
						else if (out_crossx->listB == cur_list_idx) {
							cur_list_idx = out_crossx->listA;
							list_next = contours_opts + cur_list_idx;
						}
						if (list_next == pt_list_st) break;
					}
					else {
						printf("Found dead end for loop list\n");
						while (1);
					}
				}
				if (contour_object.size() > 0) max_object_area = cv::contourArea(contour_object);
				break;
			}
		}
	}

	free(contours_opts);
	
}
