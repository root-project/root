// @(#)root/tmva $Id$
// Author: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer and Simon Pfreundschuh

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

/*! \class TMVA::ROCCurve
\ingroup TMVA

*/
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
#ifndef ROOT_TGraph
#include "TGraph.h"
#endif

#include<vector>
#include <cassert>

using namespace std;

////////////////////////////////////////////////////////////////////////////////

TMVA::ROCCurve::ROCCurve(const std::vector<Float_t> & mva, const std::vector<Bool_t> & mvat) :
   fLogger ( new TMVA::MsgLogger("ROCCurve") ),fGraph(NULL)
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
   if(fGraph) delete fGraph;
}

////////////////////////////////////////////////////////////////////////////////
/// ROC Integral (AUC)

Double_t TMVA::ROCCurve::GetROCIntegral(){

  Float_t integral=0;
  int ndivisions = 40;
  fEpsilonSig.push_back(0);
  fEpsilonBgk.push_back(0);

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
      fEpsilonSig.push_back(epsilon_s);

      if(ccounter != 0 || dcounter != 0)
      {
   epsilon_b = 1.0*dcounter/(ccounter+dcounter);
      }
      fEpsilonBgk.push_back(epsilon_b);
  }
  fEpsilonSig.push_back(1.0);
  fEpsilonBgk.push_back(1.0);
  for(UInt_t i=0;i<fEpsilonSig.size()-1;i++)
  {
      integral += 0.5*(fEpsilonSig[i+1]-fEpsilonSig[i])*(fEpsilonBgk[i]+fEpsilonBgk[i+1]);
  }
   return integral;
}


////////////////////////////////////////////////////////////////////////////////

TGraph* TMVA::ROCCurve::GetROCCurve(const UInt_t points)
{
   const UInt_t ndivisions = points - 1;
   fEpsilonSig.resize(points);
   fEpsilonBgk.resize(points);
   // Fixed values.
   fEpsilonSig[0] = 0.0;
   fEpsilonSig[ndivisions] = 1.0;
   fEpsilonBgk[0] = 1.0;
   fEpsilonBgk[ndivisions] = 0.0;

   for (UInt_t i = 1; i < ndivisions; i++) {
      Float_t threshold = -1.0 + i * 2.0 / (Float_t) ndivisions;
      Float_t true_positives = 0.0;
      Float_t false_positives = 0.0;
      Float_t true_negatives = 0.0;
      Float_t false_negatives = 0.0;

      for (UInt_t j=0; j<fMvaS.size(); j++) {
         if(fMvaS[j] > threshold)
         true_positives += 1.0;
         else
         false_negatives += 1.0;

         if(fMvaB[j] > threshold)
         false_positives += 1.0;
         else
         true_negatives += 1.0;
      }

      fEpsilonSig[ndivisions - i] = 0.0;
      if ((true_positives > 0.0) || (false_negatives > 0.0))
         fEpsilonSig[ndivisions - i] =
         true_positives / (true_positives + false_negatives);

      fEpsilonBgk[ndivisions - i] =0.0;
      if ((true_negatives > 0.0) || (false_positives > 0.0))
         fEpsilonBgk[ndivisions - i] =
         true_negatives / (true_negatives + false_positives);
   }
   if(!fGraph)    fGraph=new TGraph(fEpsilonSig.size(),&fEpsilonSig[0],&fEpsilonBgk[0]);
   return fGraph;
}
