// @(#)root/minuit2:$Name:  $:$Id: MnPrint.h,v 1.2 2006/04/12 16:30:30 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnPrint
#define ROOT_Minuit2_MnPrint

#include "Minuit2/MnConfig.h"

//#define DEBUG
//#define WARNINGMSG

#include <iostream>

#ifdef DEBUG
#ifndef WARNINGMSG
#define WARNINGMSG
#endif
#endif


namespace ROOT {

   namespace Minuit2 {


/**
    define ostream operators for output 
*/

class FunctionMinimum;
std::ostream& operator<<(std::ostream&, const FunctionMinimum&);

class MinimumState;
std::ostream& operator<<(std::ostream&, const MinimumState&);

class LAVector;
std::ostream& operator<<(std::ostream&, const LAVector&);

class LASymMatrix;
std::ostream& operator<<(std::ostream&, const LASymMatrix&);

class MnUserParameters;
std::ostream& operator<<(std::ostream&, const MnUserParameters&);

class MnUserCovariance;
std::ostream& operator<<(std::ostream&, const MnUserCovariance&);

class MnGlobalCorrelationCoeff;
std::ostream& operator<<(std::ostream&, const MnGlobalCorrelationCoeff&);

class MnUserParameterState;
std::ostream& operator<<(std::ostream&, const MnUserParameterState&);

class MnMachinePrecision;
std::ostream& operator<<(std::ostream&, const MnMachinePrecision&);

class MinosError;
std::ostream& operator<<(std::ostream&, const MinosError&);

class ContoursError;
std::ostream& operator<<(std::ostream&, const ContoursError&);

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_MnPrint
