//
//    Copyright (c) 2009-2011
//      Patrick Sudowe	<sudowe@umic.rwth-aachen.de>
//      RWTH Aachen University, Germany
//
//    This file is part of groundHOG.
//
//    GroundHOG is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    GroundHOG is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with groundHOG.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef __CUDAHOG_H__
#define __CUDAHOG_H__

#include <string>
#include <vector>
#include <climits>

#ifndef uchar
#define uchar unsigned char
#endif
#ifndef ushort
#define ushort unsigned short
#endif

namespace cudaHOG {

#include "roi.h"

class Feature
{
public:
	float target;
	std::vector<float> values;
};

class Detection
{
public:
	Detection()
		: x(INT_MAX), y(INT_MAX), scale(0.f), score(0.f) {};
	Detection(int nx, int ny, float nscale, float nscore) {
		x = nx; y = ny; scale = nscale; score = nscore; };

public:
	int x;
	int y;
	float scale;
	float score;
};

typedef std::vector<Detection> DetectionList;
typedef std::vector<DetectionList> MultiDetectionList;

// forward declaration - avoid including other headers in interface
class Parameters;
class ModelParameters;

using namespace std;
class cudaHOGManager
{
public:
	// construct without model - just for training
	cudaHOGManager();
	// construct with SVM model file - for detection
	cudaHOGManager(const std::string& fnModel);
	~cudaHOGManager();

	int read_params_file(std::string& fnParams);
	int load_svm_models();
	int set_active_model(std::string& modelIdentifier);

// Preparation
	// initialize SVM model for classification
	// call multiple times for multi-class (and multi-view)
	int add_svm_model(const std::string& fnModel);

	// Transfer an image to the GPU device - preparing for computations
	int prepare_image(uchar* pImgData, ushort width, ushort height);

	// Prepare camera & groundplane information
	// set the parameters directly
	int set_camera(float *R, float *K, float *t);
	int set_groundplane(float *n, float* d);
	// - alternatively -
	// read camera & groundplane parameters from a file
	int read_camera_data(const string& fnCamera);
	// specify were to find the camera data files - to use with test_images
	int set_camera_data_path(const string& cameraDataPath, char* m_cameraFilenameFormat="camera.%05d");

	// Set groundplane homography directly, without camera parameter
	int set_groundplane_homography(const string& fnHomography);

	// specify the minimum and maximum object height
	void set_groundplane_corridor(float minimum_height, float maximum_height);
	void set_roi_x_center_constraint(int nPixels);
	void set_valid_object_height(int minimum_pixel_height, int maximum_pixel_height);

	void set_detector_params(float start_scale=1.0f , float scale_step=1.05f);

	// pre-compute region-of-interest for each scale
	int prepare_roi_by_groundplane();
	// set the ROI for each scale directly - instead of using groundplane
	int set_roi_external(std::vector<ROI>& roi);

	// Release the image - freeing GPU memory
	int release_image();

// Detection
	// Run detection on prepared image - returns final results after NMS
	int test_image(DetectionList& detections);
	int test_image(MultiDetectionList& detections);

	// Run detection on a set of images - final results are dumped to files
	// pass two vectors of same length, input files & output files
	int test_images(const vector<string>& fnsImages, const vector<string>& fnsOutput,
						int* cntBlocks=NULL, int* cntSVM=NULL, double* timings=NULL);

// Training
	int dump_features(vector<string>& pos, vector<string>& neg, bool bSampled, bool bPadding,
					std::string& fnFeatures);
	int dump_hard_features(vector<string>& neg, string& fnFeatures, string& fnOutput);

protected:
	int compute_features(vector<string>& examples, int& count, float target, bool pPadding,
										vector<Feature>& features);
	int compute_features_sampled(vector<string>& fnImageList, int& count,
									vector<Feature>& features);

	int compute_roi_one_scale(float scale, ModelParameters& params,
							 int& min_x, int& min_y, int& max_x, int& max_y);

	int compute_homography();

	int features_to_file(vector<Feature>& features, const string& fnOutput);
	int features_from_file(vector<Feature>& features, const string& fnFeatures);

private:
	bool bImagePrepared;
	bool bWithModel;
	bool bValidCamera, bValidGroundPlane, bValidHomography;
	ushort imgWidth, imgHeight;

	float m_Homography[3][3];
	float m_ProjectedNormal[3];

	float cam_K[3][3];
	float cam_R[3][3];
	float cam_t[3];
	float GP_n[3];
	float GP_d;
	float m_h_w_min;
	float m_h_w_max;
	int   m_roi_center_pixels;
	int   m_minimum_pixel_height, m_maximum_pixel_height;
	string m_cameraDataPath;
	char* m_cameraFilenameFormat;

	ModelParameters* m_pActiveModel;
};

} // namespace cudaHOG
#endif	//__CUDAHOG_H__
