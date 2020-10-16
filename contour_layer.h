// techdpnn computerbis pleased to support the open source community by making ncnn available.
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

#include <float.h>
#include <stdio.h>
#include <stdlib.h>        
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "struct_defines.h"
//#define pt_sz_min 3
//#define pt_recall_step 2
//#define slide_maximum_len 16
//#define slide_recall_len 3
//#define th_filter_sz 7
//#define MAX_PT_POOL (8 * slide_maximum_len)
//#define MAX_PT_RECORD (8 * slide_maximum_len)
//const int incx1[8] = { 1, 1, 0, -1, -1, -1,  0,  1 };
//const int incy1[8] = { 0, 1, 1,  1,  0, -1, -1, -1 };
//struct pt_struct
//{
//	int px, py;
//	int direction;int direction_flag;
//	bool mask[8];
//	int size;
//	int recall_flag;
//};
class ContourSelectInstance
{
public:
	ContourSelectInstance();
	~ContourSelectInstance();
	int load_model(const char * path);
	int save_model(const char * path);
	float distance_minimum(float x1_start, float y1_start, float x2_start, float y2_start, float xci1_start, float yci1_start, float xci2_start, float yci2_start);
	void set_threshold_b(float bin_th) { bin_threshold = bin_th; }
	void set_length_threshold(float min_length) { minLength = min_length; }
	void set_gap_threshold(float min_gap) { min_gap_ceil = min_gap;  }
	float get_object_ssim() { return max_object_ssim; }
	void maximum_length_contours(const cv::Mat& amps);
	void maximum_value_contours(const cv::Mat& amps);
	void maximum_band_contours(const cv::Mat& amps);
	void maximum_density_depth(const cv::Mat& amps);
	void contours_combined();
	inline float square(float v) { return (v * v); }
	std::vector<std::vector<cv::Point>> get_contour_loop()	{	return contour_loop_set;	}
	std::vector<cv::Point> get_contour_object() { return contour_object;  }
	float get_sum_gray_vert(const cv::Mat& amps, int r, int c, int m, int n);
	float get_sum_gray_horz(const cv::Mat& amps, int r, int c, int m, int n);
	float get_sum_gray_band(const cv::Mat& amps, int r, int c, int direction0, int direction1, int direction2);
	float get_sum_gray_gauss(const cv::Mat& amps, int r, int c, int direction0, int direction1, int direction2);
	bool extend_band_double(const cv::Mat& amps, struct pt_struct* cur_pts, int zero_offset, bool reves_enable, cv::Mat& pix_used);
	void maximum_density_double(const cv::Mat& amps);
	void contours_maximum_combined(const cv::Mat& amps);
	int extend_new_contours(const cv::Mat& amps);
	void contours_search_intersection(int cid);
public:
	float roi_x_min;
	float roi_x_max;
	float roi_y_min;
	float roi_y_max;
	float bin_threshold;
	float minLength;
	float min_gap_ceil;
	float max_object_ssim;
	float max_object_girth;
	float max_object_area;
protected:
	struct pt_struct record_pts[MAX_PT_RECORD];
	struct pt_struct cur_pts[MAX_PT_POOL];
	struct pt_struct cur_pts_max[MAX_PT_POOL];
private:
	std::vector<std::vector<cv::Point>> contour_loop_set;      //loop contours
	std::vector<cv::Point> contour_object;                     //object contours
	struct pt_list contours_pool[MAX_POOL_SZ];
	struct crossx_struct crossx_pool[MAX_POOL_SZ * 2];
	std::vector<struct pt_list*> contours_object_opt;
	int cons_len;
	int cross_len;
	std::vector< cv::Mat> pts_distributed;
	const int M = 23, N = 23;
	const int KM = 13, KN = 13;
	const int BM = 5, BN = 5;
};
