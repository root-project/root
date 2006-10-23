// @(#)root/minuit2:$Name:  $:$Id: MnFcn.h,v 1.2 2006/07/03 15:48:06 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnFcn
#define ROOT_Minuit2_MnFcn

#include "Minuit2/MnConfig.h"
#include "Minuit2/MnMatrix.h"

#include <vector>

namespace ROOT {

   namespace Minuit2 {


class FCNBase;
/**
   Wrapper class to FCNBase interface used internally by Minuit.
   Apply conversion from calling the function from a Minuit Vector (MnAlgebraicVector) to a std::vector  for 
   the function coordinates
 */
class MnFcn {

public:

  MnFcn(const FCNBase& fcn) : fFCN(fcn), fNumCall(0) {}

  virtual ~MnFcn();

  virtual double operator()(const MnAlgebraicVector&) const;
  unsigned int NumOfCalls() const {return fNumCall;}

  //
  //forward interface
  //
  double ErrorDef() const;
  double Up() const;

  const FCNBase& Fcn() const {return fFCN;}

private:

  const FCNBase& fFCN;

protected:

  mutable int fNumCall;
};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_MnFcn
