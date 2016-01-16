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

#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_ROCCurve
#include "TMVA/ROCCurve.h"
#endif
#ifndef ROOT_TMVA_Config
#include "TMVA/Config.h"
#endif
#ifndef ROOT_TMVA_Version
#include "TMVA/Version.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

#include<vector>
#include <cassert>

using namespace std;


TMVA::ROCCurve::ROCCurve(const std::vector<Float_t> & mva, const std::vector<Bool_t> & mvat) :
   fLogger ( new TMVA::MsgLogger("ROCCurve") )
{
   assert(mva.size() == mvat.size() );
   for(UInt_t i=0;i<mva.size();i++)
   {
      if(mvat[i] ) fMvaS.push_back(mva[i]);
      else fMvaB.push_back(mva[i]);
   }

}



////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::ROCCurve::~ROCCurve() {
   delete fLogger;
}
      
////////////////////////////////////////////////////////////////////////////////
/// ROC Integral (AUC)

Double_t TMVA::ROCCurve::GetROCIntegral(){
  
  Float_t integral=0;
  int ndivisions = 40;
  std::vector<Float_t> vec_epsilon_s(1);
  vec_epsilon_s.push_back(0);
  
  std::vector<Float_t>  vec_epsilon_b(1);
  vec_epsilon_b.push_back(0);
  
  Float_t epsilon_s = 0.0;
  Float_t epsilon_b = 0.0;

  for(Float_t i=-1.0;i<1.0;i+=(1.0/ndivisions))
  {
      Float_t acounter = 0.0;
      Float_t bcounter = 0.0;
      Float_t ccounter = 0.0;
      Float_t dcounter = 0.0;
      
      for(UInt_t j=0;j<fMvaS.size();j++)
      {
        if(fMvaS[j] > i) acounter++;
        else            bcounter++;
	
        if(fMvaB[j] > i) ccounter++;
        else            dcounter++;
      }
      
      if(acounter != 0 || bcounter != 0)
      {
	epsilon_s = 1.0*bcounter/(acounter+bcounter);
      }
      vec_epsilon_s.push_back(epsilon_s);
      
      if(ccounter != 0 || dcounter != 0)
      {
	epsilon_b = 1.0*dcounter/(ccounter+dcounter);
      }
      vec_epsilon_b.push_back(epsilon_b);      
  }
  vec_epsilon_s.push_back(1.0);
  vec_epsilon_b.push_back(1.0);
  for(UInt_t i=0;i<vec_epsilon_s.size()-1;i++)
  {
      integral += 0.5*(vec_epsilon_s[i+1]-vec_epsilon_s[i])*(vec_epsilon_b[i]+vec_epsilon_b[i+1]);
  }
   return integral;
}







