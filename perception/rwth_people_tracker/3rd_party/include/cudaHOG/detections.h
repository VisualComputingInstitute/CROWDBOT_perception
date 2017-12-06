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

#ifndef __DETECTIONS_H__
#define __DETECTIONS_H__

#include <vector>
#include <string>

#include "cudaHOG.h"

namespace cudaHOG {

void dump_DetectionList(DetectionList& det, const std::string& fn);
void dump_MultiDetectionList(MultiDetectionList& det, const std::string& fn);

}	// end of namespace cudaHOG
#endif
