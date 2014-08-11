// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnMachinePrecision
#define ROOT_Minuit2_MnMachinePrecision

#include <math.h>

namespace ROOT {

   namespace Minuit2 {


/**
    determines the relative floating point arithmetic precision. The
    SetPrecision() method can be used to override Minuit's own determination,
    when the user knows that the {FCN} function Value is not calculated to
    the nominal machine accuracy.
 */

class MnMachinePrecision {

public:

  MnMachinePrecision();

  ~MnMachinePrecision() {}

  MnMachinePrecision(const MnMachinePrecision& prec) : fEpsMac(prec.fEpsMac), fEpsMa2(prec.fEpsMa2) {}

  MnMachinePrecision& operator=(const MnMachinePrecision& prec) {
    fEpsMac = prec.fEpsMac;
    fEpsMa2 = prec.fEpsMa2;
    return *this;
  }

  /// eps returns the smallest possible number so that 1.+eps > 1.
  double Eps() const {return fEpsMac;}

  /// eps2 returns 2*sqrt(eps)
  double Eps2() const {return fEpsMa2;}

  /// override Minuit's own determination
  void SetPrecision(double prec) {
    fEpsMac = prec;
    fEpsMa2 = 2.*sqrt(fEpsMac);
  }

private:

  double fEpsMac;
  double fEpsMa2;
};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_MnMachinePrecision
