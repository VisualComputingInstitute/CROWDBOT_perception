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

//extern "C++" __global__ void d_compute_gradients(int width, int height, float2* d_pGradMag);
extern __host__ int compute_gradients(int paddedWidth, int paddedHeight,
										int padX, int padY,
										int min_x, int min_y, int max_x, int max_y,
										float2* d_pGradMag);


int prepare_image(const unsigned char* h_pImg, int width, int height);
int destroy_image();
int test_prepared_image(float scale, int origwidth, int origheight, int padX, int padY);

