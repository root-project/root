// @(#)root/minuit2:$Name:  $:$Id: TFitterFumili.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TFitterFumili_H_
#define ROOT_TFitterFumili_H_


#ifndef ROOT_TVirtualFitter
#include "TVirtualFitter.h"
#endif

#include "TFitterMinuit.h"

/**
   TVirtualFitter implementation for new Fumili  
*/


class TFitterFumili : public TFitterMinuit {
   
public:

  TFitterFumili();

  TFitterFumili(Int_t maxpar);

  virtual ~TFitterFumili() { } 

public:



  //virtual FunctionMinimum Minimize(  int nfcn = 0, double edmval = 0.1) const;
  

protected: 

  void CreateMinimizer(EMinimizerType ); 

  void CreateChi2FCN(); 

  void CreateChi2ExtendedFCN(); 

  void CreateUnbinLikelihoodFCN() {}

  void CreateBinLikelihoodFCN();
   
private:

  

  ClassDef(TFitterFumili,1)
};

R__EXTERN TFitterFumili* gFumili2;


#endif //ROOT_TFitterFumili_H_
