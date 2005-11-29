// @(#)root/minuit2:$Name:  $:$Id: MnMinos.h,v 1.7.2.4 2005/11/29 11:08:34 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnMinos
#define ROOT_Minuit2_MnMinos

#include "Minuit2/MnStrategy.h"

#include <utility>

namespace ROOT {

   namespace Minuit2 {


class FCNBase;
class FunctionMinimum;
class MinosError;
class MnCross;


/** API class for Minos Error analysis (asymmetric errors);
    minimization has to be done before and Minimum must be valid;
    possibility to ask only for one side of the Minos Error;
 */

class MnMinos {

public:

  /// construct from FCN + Minimum
  MnMinos(const FCNBase& fcn, const FunctionMinimum& min) : 
    fFCN(fcn), fMinimum(min), fStrategy(MnStrategy(1)) {} 

  /// construct from FCN + Minimum + strategy
  MnMinos(const FCNBase& fcn, const FunctionMinimum& min, unsigned int stra) : 
    fFCN(fcn), fMinimum(min), fStrategy(MnStrategy(stra)) {} 

  /// construct from FCN + Minimum + strategy
  MnMinos(const FCNBase& fcn, const FunctionMinimum& min, const MnStrategy& stra) : fFCN(fcn), fMinimum(min), fStrategy(stra) {} 

  ~MnMinos() {}
  
  /// returns the negative (pair.first) and the positive (pair.second) 
  /// Minos Error of the Parameter
  std::pair<double,double> operator()(unsigned int, unsigned int maxcalls = 0) const;

  /// calculate one side (negative or positive Error) of the Parameter
  double Lower(unsigned int, unsigned int maxcalls = 0) const;
  double Upper(unsigned int, unsigned int maxcalls = 0) const;

  MnCross Loval(unsigned int, unsigned int maxcalls = 0) const;
  MnCross Upval(unsigned int, unsigned int maxcalls = 0) const;

  /// ask for MinosError (Lower + Upper)
  /// can be printed via std::cout  
  MinosError Minos(unsigned int, unsigned int maxcalls = 0) const;
  
private:
  
  const FCNBase& fFCN;
  const FunctionMinimum& fMinimum;
  MnStrategy fStrategy;
};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_MnMinos
