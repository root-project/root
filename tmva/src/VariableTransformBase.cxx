// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

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
     fLogger( new MsgLogger(this, kINFO) )
{
   // standard constructor

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

   // set Mean and RMS
   for (UInt_t ivar=0; ivar<nvars; ivar++) {
      Double_t mean = x0(ivar)/sumOfWeights;
      Variables().at(ivar).SetMean( mean ); 
      Variables().at(ivar).SetRMS( TMath::Sqrt( x2(ivar)/sumOfWeights - mean*mean) );
   }
   for (UInt_t itgt=0; itgt<ntgts; itgt++) {
      Double_t mean = x0(nvars+itgt)/sumOfWeights;
      Targets().at(itgt).SetMean( mean ); 
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

// TODO

// //_______________________________________________________________________
// void TMVA::VariableTransformBase::WriteVarsToStream( std::ostream& o, const TString& prefix ) const 
// {
//    // write the list of variables (name, min, max) for a given data
//    // transformation method to the stream
//    o << prefix << "NVar " << GetNVariables() << endl;
//    std::vector<VariableInfo>::const_iterator varIt = Variables().begin();
//    for (; varIt!=Variables().end(); varIt++) { o << prefix; varIt->WriteToStream(o); }
// }

// //_______________________________________________________________________
// void TMVA::VariableTransformBase::ReadVarsFromStream( std::istream& istr ) 
// {
//    // Read the variables (name, min, max) for a given data
//    // transformation method from the stream. In the stream we only
//    // expect the limits which will be set

//    TString dummy;
//    UInt_t readNVar;
//    istr >> dummy >> readNVar;

//    if (readNVar!=Variables().size()) {
//       Log() << kFATAL << "You declared "<< Variables().size() << " variables in the Reader"
//               << " while there are " << readNVar << " variables declared in the file"
//               << Endl;
//    }

//    // we want to make sure all variables are read in the order they are defined
//    VariableInfo varInfo;
//    std::vector<VariableInfo>::iterator varIt = Variables().begin();
//    int varIdx = 0;
//    for (; varIt!=Variables().end(); varIt++, varIdx++) {
//       varInfo.ReadFromStream(istr);
//       if (varIt->GetExpression() == varInfo.GetExpression()) {
//          varInfo.SetExternalLink((*varIt).GetExternalLink());
//          (*varIt) = varInfo;
//       } 
//       else {
//          Log() << kINFO << "The definition (or the order) of the variables found in the input file is"  << Endl;
//          Log() << kINFO << "is not the same as the one declared in the Reader (which is necessary for" << Endl;
//          Log() << kINFO << "the correct working of the classifier):" << Endl;
//          Log() << kINFO << "   var #" << varIdx <<" declared in Reader: " << varIt->GetExpression() << Endl;
//          Log() << kINFO << "   var #" << varIdx <<" declared in file  : " << varInfo.GetExpression() << Endl;
//          Log() << kFATAL << "The expression declared to the Reader needs to be checked (name or order are wrong)" << Endl;
//       }
//    }
// }
