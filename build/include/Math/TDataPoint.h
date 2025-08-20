// @(#)root/mathcore:$Id$
// Authors: C. Gumpert    09/2011
/**********************************************************************
 *                                                                    *
 * Copyright (c) 2011 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
//
// Header file for TDataPointclass
//

#ifndef ROOT_Math_TDataPoint
#define ROOT_Math_TDataPoint

// ROOT include(s)
#include "RtypesCore.h"

#include <cassert>
#include <math.h>

namespace ROOT {
namespace Math {

/// \brief class representing a data point
///
/// This class can be used for describing data points in a high-dimensional space.
/// The (positive) dimension is specified by the first template parameter. The second
/// template parameter can be used to tweak the precision of the stored coordinates. By
/// default all coordinates are stored with 4 byte float precision. In addition to the
/// coordinates a weight can be assigned to each data point allowing the representation
/// of fields in high dimensions.
/// Basic functionality for accessing/modifying the coordinates/weight are provided
/// as well as a comparison method and the basic euclidean metric.
template <unsigned int K, typename _val_type = float>
class TDataPoint {
public:
   typedef _val_type value_type;
   enum {
      kDimension = K // the dimensionality of this data point
   };
   static UInt_t Dimension() { return kDimension; }
   /// standard constructor
   /// sets the weight to 1 and initialises all coordinates with 0
   TDataPoint()
   {
      // at least one dimension
      assert(kDimension > 0);

      for (UInt_t k = 0; k < K; ++k)
         m_vCoordinates[k] = 0;
   }
#ifndef __MAKECINT__
   /// constructor initialising the data point from an array
   ///
   /// Input: pData   - array with kDimension coordinates
   ///        fWeight - weight (default = 1)
   template <typename _coord_typ>
   TDataPoint(const _coord_typ *pData, _val_type fWeight = 1)
   {
      // at least one dimension
      assert(kDimension > 0);
      // fill coordinates
      for (unsigned int i = 0; i < kDimension; ++i)
         m_vCoordinates[i] = pData[i];
      m_fWeight = fWeight;
   }
   /// euclidean distance
   ///
   /// returns the euclidean distance to the given data point
   ///
   /// Input: rPoint - data point of same dimensionality
   template <typename _val>
   value_type Distance(const TDataPoint<K, _val> &rPoint) const
   {
      _val_type fDist2 = 0;
      for (unsigned int i = 0; i < kDimension; ++i)
         fDist2 += pow(GetCoordinate(i) - rPoint.GetCoordinate(i), 2);

      return sqrt(fDist2);
   }
#endif
   /// returns the coordinate at the given axis
   ///
   /// Input: iAxis - axis in the range of [0...kDimension-1]
   value_type GetCoordinate(unsigned int iAxis) const
   {
      assert(iAxis < kDimension);
      return m_vCoordinates[iAxis];
   }
   value_type GetWeight() const { return m_fWeight; }
   /// compares two points at a given axis
   ///
   ///  returns: this_point.at(iAxis) < rPoint.at(iAxis)
   ///
   /// Input: rPoint - second point to compare to (of same dimensionality)
   ///        iAxis  - axis in the range of [0...kDimension-1]
   Bool_t Less(TDataPoint &rPoint, unsigned int iAxis) const
   {
      assert(iAxis < kDimension);
      return (m_vCoordinates[iAxis] < rPoint.GetCoordinate(iAxis));
   }
   /// sets the coordinate along one axis
   ///
   /// Input: iAxis  - axis in the range of [0...kDimension-1]
   ///        fValue - new coordinate
   void SetCoordinate(unsigned int iAxis, _val_type fValue)
   {
      assert(iAxis < kDimension);
      m_vCoordinates[iAxis] = fValue;
   }
   void SetWeight(float fWeight) { m_fWeight = fWeight; }

private:
   value_type m_vCoordinates[K]; ///< coordinates
   value_type m_fWeight = 1;     ///< weight at this point
};

// some typedef definitions
typedef TDataPoint<1, Float_t> TDataPoint1F;
typedef TDataPoint<2, Float_t> TDataPoint2F;
typedef TDataPoint<3, Float_t> TDataPoint3F;
typedef TDataPoint<1, Double_t> TDataPoint1D;
typedef TDataPoint<2, Double_t> TDataPoint2D;
typedef TDataPoint<3, Double_t> TDataPoint3D;

} // namespace Math
} // namespace ROOT

#endif // ROOT_Math_TDataPoint
