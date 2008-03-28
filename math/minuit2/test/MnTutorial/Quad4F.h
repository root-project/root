// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/FCNBase.h"

namespace ROOT {

   namespace Minuit2 {


class Quad4F : public FCNBase {

public:

  Quad4F() {}

  ~Quad4F() {}

  double operator()(const std::vector<double>& par) const {

    double x = par[0];
    double y = par[1];
    double z = par[2];
    double w = par[3];

    return ( (1./70.)*(21*x*x + 20*y*y + 19*z*z - 14*x*z - 20*y*z) + w*w );
  }

  double Up() const {return 1.;}

private:

};

  }  // namespace Minuit2

}  // namespace ROOT
