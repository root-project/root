// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableTransformBase                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <iomanip>

#include "TMath.h"
#include "TVectorD.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"

#include "TMVA/VariableTransformBase.h"
#include "TMVA/Ranking.h"
#include "TMVA/Config.h"
#include "TMVA/Tools.h"
#include "TMVA/Version.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

ClassImp(TMVA::VariableTransformBase)

//_______________________________________________________________________
TMVA::VariableTransformBase::VariableTransformBase( DataSetInfo& dsi,
                                                    Types::EVariableTransform tf,
                                                    const TString& trfName )
   : TObject(),
     fDsi(dsi),
     fTransformedEvent(0),
     fBackTransformedEvent(0),
     fVariableTransform(tf),
     fEnabled( kTRUE ),
     fCreated( kFALSE ),
     fNormalise( kFALSE ),
     fTransformName(trfName),
     fTMVAVersion(TMVA_VERSION_CODE),
     fLogger( 0 )
{
   // standard constructor
   fLogger = new MsgLogger(this, kINFO);
   for (UInt_t ivar = 0; ivar < fDsi.GetNVariables(); ivar++) {
      fVariables.push_back( VariableInfo( fDsi.GetVariableInfo(ivar) ) );
   }
   for (UInt_t itgt = 0; itgt < fDsi.GetNTargets(); itgt++) {
      fTargets.push_back( VariableInfo( fDsi.GetTargetInfo(itgt) ) );
   }
}

//_______________________________________________________________________
TMVA::VariableTransformBase::~VariableTransformBase()
{
   if (fTransformedEvent!=0)     delete fTransformedEvent;
   if (fBackTransformedEvent!=0) delete fBackTransformedEvent;
   // destructor
   delete fLogger;
}

//_______________________________________________________________________
void TMVA::VariableTransformBase::CalcNorm( const std::vector<Event*>& events ) 
{
   // method to calculate minimum, maximum, mean, and RMS for all
   // variables used in the MVA

   if (!IsCreated()) return;

   const UInt_t nvars = GetNVariables();
   const UInt_t ntgts = GetNTargets();

   UInt_t nevts = events.size();

   TVectorD x2( nvars+ntgts ); x2 *= 0;
   TVectorD x0( nvars+ntgts ); x0 *= 0;   

   Double_t sumOfWeights = 0;
   for (UInt_t ievt=0; ievt<nevts; ievt++) {
      const Event* ev = events[ievt];

      Double_t weight = ev->GetWeight();
      sumOfWeights += weight;
      for (UInt_t ivar=0; ivar<nvars; ivar++) {
         Double_t x = ev->GetValue(ivar);
         if (ievt==0) {
            Variables().at(ivar).SetMin(x);
            Variables().at(ivar).SetMax(x);
         } 
         else {
            UpdateNorm( ivar,  x );
         }
         x0(ivar) += x*weight;
         x2(ivar) += x*x*weight;
      }
      for (UInt_t itgt=0; itgt<ntgts; itgt++) {
         Double_t x = ev->GetTarget(itgt);
         if (ievt==0) {
            Targets().at(itgt).SetMin(x);
            Targets().at(itgt).SetMax(x);
         } 
         else {
            UpdateNorm( nvars+itgt,  x );
         }
         x0(nvars+itgt) += x*weight;
         x2(nvars+itgt) += x*x*weight;
      }
   }

   if (sumOfWeights <= 0) {
      Log() << kFATAL << " the sum of event weights calcualted for your input is == 0"
            << " or exactly: " << sumOfWeights << " there is obviously some problem..."<< Endl;
   } 

   // set Mean and RMS
   for (UInt_t ivar=0; ivar<nvars; ivar++) {
      Double_t mean = x0(ivar)/sumOfWeights;
      
      Variables().at(ivar).SetMean( mean ); 
      if (x2(ivar)/sumOfWeights - mean*mean < 0) {
         Log() << kFATAL << " the RMS of your input variable " << ivar 
               << " evaluates to an imaginary number: sqrt("<< x2(ivar)/sumOfWeights - mean*mean
               <<") .. sometimes related to a problem with outliers and negative event weights"
               << Endl;
      }
      Variables().at(ivar).SetRMS( TMath::Sqrt( x2(ivar)/sumOfWeights - mean*mean) );
   }
   for (UInt_t itgt=0; itgt<ntgts; itgt++) {
      Double_t mean = x0(nvars+itgt)/sumOfWeights;
      Targets().at(itgt).SetMean( mean ); 
      if (x2(nvars+itgt)/sumOfWeights - mean*mean < 0) {
         Log() << kFATAL << " the RMS of your target variable " << itgt 
               << " evaluates to an imaginary number: sqrt(" << x2(nvars+itgt)/sumOfWeights - mean*mean
               <<") .. sometimes related to a problem with outliers and negative event weights"
               << Endl;
      }
      Targets().at(itgt).SetRMS( TMath::Sqrt( x2(nvars+itgt)/sumOfWeights - mean*mean) );
   }

   Log() << kVERBOSE << "Set minNorm/maxNorm for variables to: " << Endl;
   Log() << std::setprecision(3);
   for (UInt_t ivar=0; ivar<GetNVariables(); ivar++)
      Log() << "    " << Variables().at(ivar).GetInternalName()
              << "\t: [" << Variables().at(ivar).GetMin() << "\t, " << Variables().at(ivar).GetMax() << "\t] " << Endl;
   Log() << kVERBOSE << "Set minNorm/maxNorm for targets to: " << Endl;
   Log() << std::setprecision(3);
   for (UInt_t itgt=0; itgt<GetNTargets(); itgt++)
      Log() << "    " << Targets().at(itgt).GetInternalName()
              << "\t: [" << Targets().at(itgt).GetMin() << "\t, " << Targets().at(itgt).GetMax() << "\t] " << Endl;
   Log() << std::setprecision(5); // reset to better value       
}

//_______________________________________________________________________
std::vector<TString>* TMVA::VariableTransformBase::GetTransformationStrings( Int_t /*cls*/ ) const
{
   // default transformation output
   // --> only indicate that transformation occurred
   std::vector<TString>* strVec = new std::vector<TString>;
   for (UInt_t ivar=0; ivar<GetNVariables(); ivar++) {
      strVec->push_back( Variables()[ivar].GetLabel() + "_[transformed]");
   }

   return strVec;   
}

//_______________________________________________________________________
void TMVA::VariableTransformBase::UpdateNorm ( Int_t ivar,  Double_t x ) 
{
   // update min and max of a given variable (target) and a given transformation method
   Int_t nvars = fDsi.GetNVariables();
   if( ivar < nvars ){
      if (x < Variables().at(ivar).GetMin()) Variables().at(ivar).SetMin(x);
      if (x > Variables().at(ivar).GetMax()) Variables().at(ivar).SetMax(x);
   }else{
      if (x < Targets().at(ivar-nvars).GetMin()) Targets().at(ivar-nvars).SetMin(x);
      if (x > Targets().at(ivar-nvars).GetMax()) Targets().at(ivar-nvars).SetMax(x);
   }
}

