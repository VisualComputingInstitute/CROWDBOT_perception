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


#ifndef __blocks_h__
#define __blocks_h__

extern "C++" __host__ int prepareGaussWeights();
extern "C++" __global__ void testGaussWeights(float* d_pOutput);

extern "C++" __host__ int prepareBilinearWeights();
extern "C++" __global__ void testBilinearWeights(float* d_pOutput);

extern "C++" __host__ int blocks_finalize();

extern "C++" __host__ int compute_blocks(dim3 grid,
						int width, int height, float2* d_pGradMag,
						float* d_pBlocks);
#endif
