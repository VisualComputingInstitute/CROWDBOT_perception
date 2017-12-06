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

#ifndef __SVM_H__
#define __SVM_H__

#include "cudaHOG.h"
#include "detections.h"

namespace cudaHOG {

int svm_initialize();
int svm_finalize();

int svm_add_model(const char* fnModel);

int svm_evaluate(float* d_pBlocks, int nBlockX, int nBlockY,
				int padX, int padY, int minX, int minY, float scale,
				int* cnt,
				MultiDetectionList& detections);

}
#endif
