// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef MN_GaussFcn2_H_
#define MN_GaussFcn2_H_

#include "Minuit2/FCNBase.h"

#include <vector>

namespace ROOT {

namespace Minuit2 {

class GaussFcn2 : public FCNBase {

public:
   GaussFcn2(std::span<const double> meas, std::span<const double> pos, std::span<const double> mvar)
      : fMeasurements(meas.begin(), meas.end()),
        fPositions(pos.begin(), pos.end()),
        fMVariances(mvar.begin(), mvar.end()),
        fMin(0.)
   {
      Init();
   }

   virtual void Init();

   double Up() const override { return 1.; }
   double operator()(std::vector<double> const &) const override;
   double ErrorDef() const override { return Up(); }

   std::vector<double> Measurements() const { return fMeasurements; }
   std::vector<double> Positions() const { return fPositions; }
   std::vector<double> Variances() const { return fMVariances; }

private:
   std::vector<double> fMeasurements;
   std::vector<double> fPositions;
   std::vector<double> fMVariances;
   double fMin;
};

} // namespace Minuit2

} // namespace ROOT

#endif // MN_GaussFcn2_H_
