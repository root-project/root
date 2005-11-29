// @(#)root/minuit2:$Name:  $:$Id: FunctionMinimum.h,v 1.13.6.4 2005/11/29 11:08:34 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_FunctionMinimum
#define ROOT_Minuit2_FunctionMinimum

#include "Minuit2/BasicFunctionMinimum.h"

namespace ROOT {

   namespace Minuit2 {


/** result of the minimization; 
    both internal and external (MnUserParameterState) representation available
    for the parameters at the Minimum
 */

class FunctionMinimum {

public:

  class MnReachedCallLimit {};
  class MnAboveMaxEdm {};

public:
  
  FunctionMinimum(const MinimumSeed& seed, double up) : fData(MnRefCountedPointer<BasicFunctionMinimum>(new BasicFunctionMinimum(seed, up))) {}
  
  FunctionMinimum(const MinimumSeed& seed, const std::vector<MinimumState>& states, double up) : fData(MnRefCountedPointer<BasicFunctionMinimum>(new BasicFunctionMinimum(seed, states, up))) {}
  
  FunctionMinimum(const MinimumSeed& seed, const std::vector<MinimumState>& states, double up, MnReachedCallLimit) : fData(MnRefCountedPointer<BasicFunctionMinimum>(new BasicFunctionMinimum(seed, states, up, BasicFunctionMinimum::MnReachedCallLimit()))) {}
  
  FunctionMinimum(const MinimumSeed& seed, const std::vector<MinimumState>& states, double up, MnAboveMaxEdm) : fData(MnRefCountedPointer<BasicFunctionMinimum>(new BasicFunctionMinimum(seed, states, up, BasicFunctionMinimum::MnAboveMaxEdm()))) {}

  FunctionMinimum(const FunctionMinimum& min) : fData(min.fData) {}
  
  FunctionMinimum& operator=(const FunctionMinimum& min) {
    fData = min.fData;
    return *this;
  }
  
  ~FunctionMinimum() {}
  
  // why not
  void Add(const MinimumState& state) {fData->Add(state);}

  const MinimumSeed& Seed() const {return fData->Seed();}
  const std::vector<MinimumState>& States() const {return fData->States();}

// user representation of state at Minimum
  const MnUserParameterState& UserState() const {
    return fData->UserState();
  }
  const MnUserParameters& UserParameters() const {
    return fData->UserParameters();
  }
  const MnUserCovariance& UserCovariance() const {
    return fData->UserCovariance();
  }

// forward interface of last state
  const MinimumState& State() const {return fData->State();}
  const MinimumParameters& Parameters() const {return fData->Parameters();}
  const MinimumError& Error() const {return fData->Error();}
  const FunctionGradient& Grad() const {return fData->Grad();}
  double Fval() const {return fData->Fval();}
  double Edm() const {return fData->Edm();}
  int NFcn() const {return fData->NFcn();}  
  
  double Up() const {return fData->Up();}
  bool IsValid() const {return fData->IsValid();}
  bool HasValidParameters() const {return fData->HasValidParameters();}
  bool HasValidCovariance() const {return fData->HasValidCovariance();}
  bool HasAccurateCovar() const {return fData->HasAccurateCovar();}
  bool HasPosDefCovar() const {return fData->HasPosDefCovar();}
  bool HasMadePosDefCovar() const {return fData->HasMadePosDefCovar();}
  bool HesseFailed() const {return fData->HesseFailed();}
  bool HasCovariance() const {return fData->HasCovariance();}
  bool IsAboveMaxEdm() const {return fData->IsAboveMaxEdm();}
  bool HasReachedCallLimit() const {return fData->HasReachedCallLimit();}

private:

  MnRefCountedPointer<BasicFunctionMinimum> fData;
};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_FunctionMinimum
