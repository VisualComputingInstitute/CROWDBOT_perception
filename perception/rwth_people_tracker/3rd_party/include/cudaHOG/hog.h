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

#include "detections.h"
#include "roi.h"
#include "cudaHOG.h"

namespace cudaHOG {

extern int hog_initialize();
extern int hog_finalize();

extern int hog_transfer_image(const unsigned char* h_pImg, int width, int height);
extern int hog_release_image();

extern int hog_process_image(int width, int height, float scale,
							int padX, int padY, ROI* roi, int* cntBlocks, int* cntSVM, MultiDetectionList& detections);
extern int hog_process_image_multiscale(int width, int height, std::vector<ROI>& roi, int* cntBlocks, int* cntSVM,
							double* timings, MultiDetectionList& detections);

extern int hog_get_descriptor(int width, int height, int bPad,
						int featureX, int featureY, float scale,
						ModelParameters& params,
						float* h_pDescriptor);

}	// end of namespace cudaHOG
