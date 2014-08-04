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

//STL include(s)
#include <assert.h>

//ROOT include(s)
#include "Rtypes.h"

namespace ROOT
{
namespace Math
{


template<typename _val_type = float>
class TDataPointN
{
private:
   static UInt_t kDimension;

public:
   typedef _val_type value_type;

   static UInt_t Dimension() {return kDimension;}
   static void SetDimension(UInt_t dim) {assert(dim>0);kDimension=dim;}

   TDataPointN();
#ifndef __MAKECINT__
   template<typename _coord_typ>
   TDataPointN(const _coord_typ* pData,value_type fWeight = 1);
   template<typename _val>
   TDataPointN(const TDataPointN<_val>&);
#endif
   virtual ~TDataPointN();

#ifndef __MAKECINT__
   template<typename _val>
   _val_type   Distance(const TDataPointN<_val>& rPoint) const;
#endif
   _val_type   GetCoordinate(unsigned int iAxis) const;
   _val_type   GetWeight() const {return m_fWeight;}
   bool        Less(TDataPointN& rPoint,unsigned int iAxis) const;
   void        SetCoordinate(unsigned int iAxis,value_type fValue);
   void        SetWeight(float fWeight) {m_fWeight = fWeight;}

private:
   value_type*   m_vCoordinates;
   value_type    m_fWeight;
};


}//namespace Math
}//namespace ROOT


#include "TDataPointN.icc"

#endif // ROOT_Math_TDataPointN
