// @(#)root/tmva $Id$   
// Author: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : ROCCurve                                                              *
 *                                                                                *
 * Description:                                                                   *
 *      This is class to compute ROC Integral (AUC)                               *
 *                                                                                *
 * Authors :                                                                      *
 *      Omar Zapata     <Omar.Zapata@cern.ch>    - UdeA/ITM Colombia              *
 *      Lorenzo Moneta  <Lorenzo.Moneta@cern.ch> - CERN, Switzerland              *
 *      Sergei Gleyzer  <Sergei.Gleyzer@cern.ch> - U of Florida & CERN            *
 *                                                                                *
 * Copyright (c) 2015:                                                            *
 *      CERN, Switzerland                                                         *
 *      UdeA/ITM, Colombia                                                        *
 *      U. of Florida, USA                                                        *
 **********************************************************************************/
#ifndef ROOT_TMVA_ROCCurve
#define ROOT_TMVA_ROCCurve
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>

class TList;
class TTree;
class TString;
class TH1;
class TH2;
class TH2F;
class TSpline;
class TSpline1;

namespace TMVA {

  class MsgLogger;


  class ROCCurve {
    
  public:
    ROCCurve( const std::vector<Float_t> & mvaS, const std::vector<Bool_t> & mvat);
    
    ~ROCCurve();
    

    Double_t GetROCIntegral();
    
  private:
    std::vector<Float_t> fMvaS;
    std::vector<Float_t> fMvaB;

    mutable MsgLogger* fLogger;   //! message logger
    MsgLogger& Log() const { return *fLogger; }                       

  };
}
#endif
