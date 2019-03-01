/*
 * Copyright (C) by
 *   MetraLabs GmbH (MLAB), GERMANY
 *  and
 *   Neuroinformatics and Cognitive Robotics Labs (NICR) at TU Ilmenau, GERMANY
 * All rights reserved.
 *
 * Redistribution and modification of this code is strictly prohibited.
 *
 * IN NO EVENT SHALL "MLAB" OR "NICR" BE LIABLE TO ANY PARTY FOR DIRECT,
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
 * THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF "MLAB" OR
 * "NICR" HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * "MLAB" AND "NICR" SPECIFICALLY DISCLAIM ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND "MLAB" AND "NICR" HAVE NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS OR MODIFICATIONS.
 */

/**
 * @file PanoramaCameraIntrinsic.h
 *    normalized and unnormalized intrinsic parameters for panorama cameras
 *
 * @author Michael Volkhardt
 * @date   2011/09/22
 */

#ifndef _MIRA_CAMERA_PANORAMACAMERAINTRINSIC_H_
#define _MIRA_CAMERA_PANORAMACAMERAINTRINSIC_H_

#include <vector>

//#include <math/Eigen.h>
//#include <geometry/Size.h>

//#include <cameraparameters/CameraParametersExports.h>

namespace mira {
namespace camera {

//////////////////////////////////////////////////////////////////////////////

/**
 * Describes the intrinsic parameters of an panorama camera.
 * For each physical camera just one instance of this class must be used since
 * the intrinsic camera parameters remain constant.
 * Uses a Cylindrical Equidistant Projection (CEP)
 *
 * The model is defined by 4 parameters
 * minPhi and maxPhi define the horizontal viewing angle and forward direction
 * of the camera model. Usually, maxPhi is 2*Pi larger than minPhi.
 *
 * The minHeight and maxHeight define the vertical viewing angle of the camera model.
 * Both parameters are given in unit sphere coordinates
 */
//class MIRA_CAMERAPARAMETERS_EXPORT PanoramaCameraIntrinsicNormalized {
class  PanoramaCameraIntrinsicNormalized {
public:

	/// Constructor.
	PanoramaCameraIntrinsicNormalized();

	/**
	 * Sets all intrinsic parameters, such as the
	 *
	 * The parameters must be specified TODO copy from above
	 */
	PanoramaCameraIntrinsicNormalized(float iMinPhi, float iMaxPhi,
			float iMinHeight, float iMaxHeight);

public:
	template<typename Reflector>
	void reflect(Reflector& r) {
		r.property("MinPhi", minPhi, "minPhi of image [0..2*MPI]");
		r.property("MaxPhi", maxPhi, "maxPhi of image [0..2*MPI]");
		r.property("MinHeight", minHeight,
		           "min height of image in unit sphere world coordinates"
		           "usually [-1..1.]");
		r.property("MaxHeight", maxHeight,
		           "max height of image in unit sphere world coordinates"
		           "usually [-1..1.]");
	}

public:
	float minPhi; ///< min phi angle of image in range [0..2 PI]
	float maxPhi; ///< max phi angle of image in range [0..2 PI],
	float minHeight; ///< min height of image in unit sphere coordinates
	float maxHeight; ///< max height of image in unit sphere coordinates
};

///////////////////////////////////////////////////////////////////////////////

/**
 * Describes the intrinsic parameters of an panorama camera.
 * For each physical camera just one instance of this class must be used since
 * the intrinsic camera parameters remain constant.
 * Uses a Cylindrical Equidistant Projection (CEP)
 *
 * The model is defined by 6 parameters
 * minPhi and maxPhi define the horizontal viewing angle and forward direction
 * of the camera model. Usually, maxPhi is 2*Pi larger than minPhi.
 *
 * The minHeight and maxHeight define the vertical viewing angle of the camera model.
 * Both parameters are given in unit sphere coordinates
 *
 * The width and height parameters define the width and height of the resulting image.
 */

class PanoramaCameraIntrinsic {
//class MIRA_CAMERAPARAMETERS_EXPORT PanoramaCameraIntrinsic {
public:

	/// Constructor.
	PanoramaCameraIntrinsic();

	/**
	 * Sets all intrinsic parameters
	 */
	PanoramaCameraIntrinsic(float iMinPhi, float iMaxPhi, float iMinHeight,
                            float iMaxHeight, float iWidth, float iHeight)
        :minPhi(iMinPhi),maxPhi(iMaxPhi), minHeight(iMinHeight), maxHeight(iMaxHeight),width(iWidth),height(iHeight)
    {
    }

//	PanoramaCameraIntrinsic(
//			const PanoramaCameraIntrinsicNormalized& iNormalizedIntrinsic,
//			const Size2i& iImageSize);

public:

	template<typename Reflector>
	void reflect(Reflector& r) {
		r.property("MinPhi", minPhi, "minPhi of image [0..2*MPI]");
		r.property("MaxPhi", maxPhi, "maxPhi of image [0..2*MPI]");
		r.property("MinHeight", minHeight,
		           "min height of image in unit sphere world coordinates"
		           "usually in range [-1..1.]");
		r.property("MaxHeight", maxHeight,
		           "max height of image in unit sphere world coordinates"
		           "usually in range[-1..1.]");

		r.property("width", width, "height of image in pixel coord");
		r.property("height", height, "width of image in pixel coord");
	}

public:
	float minPhi; ///< min phi angle of image in range [0..2 PI]
	float maxPhi; ///< max phi angle of image in range [0..2 PI]
	float minHeight; ///< min height of image in unit sphere coordinates
	float maxHeight; ///< max height of image in unit sphere coordinates
	float width; ///< width of image in pixel
	float height; ///< height of image in pixel
};

///////////////////////////////////////////////////////////////////////////////

}
} // namespace

// hard code the camera parameter
float minPhi =  -1.775;
float maxPhi =  1.775;
/*float maxHeight = 1.400;
float minHeight = -1.400;
float iwidth = 1280;
float iheight = 800;*/
float maxHeight = 1.300;
float minHeight = -1.300;
float iwidth = 640;
float iheight = 480;
mira::camera::PanoramaCameraIntrinsic panorama_intrinsic(minPhi, maxPhi, minHeight, maxHeight, iwidth, iheight);


#endif
