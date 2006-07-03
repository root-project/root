// @(#)root/minuit2:$Name:  $:$Id: MnPosDef.h,v 1.1 2005/11/29 14:42:18 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnPosDef
#define ROOT_Minuit2_MnPosDef

namespace ROOT {

   namespace Minuit2 {


class MinimumState;
class MinimumError;
class MnMachinePrecision;

/**
   Force the covariance matrix to be positive defined
   by adding extra terms in the diagonal
 */
class MnPosDef {

public:
  
  MnPosDef() {}
  
  ~MnPosDef() {}
  
  MinimumState operator()(const MinimumState&, const MnMachinePrecision&) const;
  MinimumError operator()(const MinimumError&, const MnMachinePrecision&) const;
private:

};

  }  // namespace Minuit2

}  // namespace ROOT

#endif  // ROOT_Minuit2_MnPosDef
