// @(#)root/mathcore:$Id$
// Authors: C. Gumpert    09/2011
/**********************************************************************
 *                                                                    *
 * Copyright (c) 2011 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
//
// Header file for TDataPointN class
//

#ifndef ROOT_Math_TDataPointN
#define ROOT_Math_TDataPointN

// STL include(s)
#include <cassert>

// ROOT include(s)
#include "RtypesCore.h"

namespace ROOT {
namespace Math {

template <typename _val_type = float>
class TDataPointN {
private:
   static UInt_t kDimension;

public:
   typedef _val_type value_type;

   static UInt_t Dimension() { return kDimension; }
   static void SetDimension(UInt_t dim)
   {
      assert(dim > 0);
      kDimension = dim;
   }

   TDataPointN()
   {
      m_vCoordinates = new _val_type[kDimension];
      for (UInt_t k = 0; k < kDimension; ++k)
         m_vCoordinates[k] = 0;
   }
#ifndef __MAKECINT__
   template <typename _coord_typ>
   TDataPointN(const _coord_typ *pData, value_type fWeight = 1)
   {
      // fill coordinates
      m_vCoordinates = new _val_type[kDimension];
      for (unsigned int i = 0; i < kDimension; ++i)
         m_vCoordinates[i] = pData[i];
   }
   TDataPointN(TDataPointN const &) = delete;
#endif
   virtual ~TDataPointN() { delete[] m_vCoordinates; }

#ifndef __MAKECINT__
   template <typename _val>
   _val_type Distance(const TDataPointN<_val> &rPoint) const
   {
      _val_type fDist2 = 0;
      for (unsigned int i = 0; i < kDimension; ++i)
         fDist2 += pow(GetCoordinate(i) - rPoint.GetCoordinate(i), 2);

      return sqrt(fDist2);
   }
#endif
   _val_type GetCoordinate(unsigned int iAxis) const
   {
      assert(iAxis < kDimension);
      return m_vCoordinates[iAxis];
   }
   _val_type GetWeight() const { return m_fWeight; }
   bool Less(TDataPointN &rPoint, unsigned int iAxis) const
   {
      assert(iAxis < kDimension);
      return (m_vCoordinates[iAxis] < rPoint.GetCoordinate(iAxis));
   }
   void SetCoordinate(unsigned int iAxis, value_type fValue)
   {
      assert(iAxis < kDimension);
      m_vCoordinates[iAxis] = fValue;
   }
   void SetWeight(float fWeight) { m_fWeight = fWeight; }

private:
   value_type *m_vCoordinates = nullptr;
   value_type m_fWeight = 1;
};

template <typename _val_type>
UInt_t TDataPointN<_val_type>::kDimension = 0;

} // namespace Math
} // namespace ROOT

#endif // ROOT_Math_TDataPointN
