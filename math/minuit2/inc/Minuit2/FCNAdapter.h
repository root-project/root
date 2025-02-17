// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FCNAdapter
#define ROOT_Minuit2_FCNAdapter

#include "Minuit2/FCNBase.h"

#include <ROOT/RSpan.hxx>

#include <vector>

namespace ROOT {

namespace Minuit2 {

/**


template wrapped class for adapting to FCNBase signature

@author Lorenzo Moneta

@ingroup Minuit

*/

template <class Function>
class FCNAdapter : public FCNBase {

public:
   FCNAdapter(const Function &f, double up = 1.) : fFunc(f), fUp(up) {}

   double operator()(std::vector<double> const& v) const override { return fFunc.operator()(&v[0]); }
   double operator()(const double *v) const { return fFunc.operator()(v); }
   double Up() const override { return fUp; }

   void SetErrorDef(double up) override { fUp = up; }

private:
   const Function &fFunc;
   double fUp;
};

} // end namespace Minuit2

} // end namespace ROOT

#endif // ROOT_Minuit2_FCNAdapter
