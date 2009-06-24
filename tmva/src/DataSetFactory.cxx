// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DataSetFactory                                                        *
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
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <assert.h>

#include <vector>
#include <iomanip>
#include <iostream>

#include "TMVA/DataSetFactory.h"

#include "TEventList.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TRandom3.h"
#include "TMatrixF.h"
#include "TVectorF.h"
#include "TMath.h"
#include "TROOT.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_Configurable
#include "TMVA/Configurable.h"
#endif
#ifndef ROOT_TMVA_VariableIdentityTransform
#include "TMVA/VariableIdentityTransform.h"
#endif
#ifndef ROOT_TMVA_VariableDecorrTransform
#include "TMVA/VariableDecorrTransform.h"
#endif
#ifndef ROOT_TMVA_VariablePCATransform
#include "TMVA/VariablePCATransform.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif
#ifndef ROOT_TMVA_DataSetInfo
#include "TMVA/DataSetInfo.h"
#endif
#ifndef ROOT_TMVA_DataInputHandler
#include "TMVA/DataInputHandler.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif

using namespace std;

TMVA::DataSetFactory* TMVA::DataSetFactory::fgInstance = 0;

namespace TMVA {
   // calculate the largest common divider
   // this function is not happy if numbers are negative!
   Int_t LargestCommonDivider(Int_t a, Int_t b) 
   {
      if (a<b) {Int_t tmp = a; a=b; b=tmp; } // achieve a>=b
      if (b==0) return a;
      Int_t fullFits = a/b;
      return LargestCommonDivider(b,a-b*fullFits);
   }
}

//_______________________________________________________________________
TMVA::DataSetFactory::DataSetFactory() :
   fVerbose(kFALSE),
   fVerboseLevel(TString("Info")),
   fCurrentTree(0),
   fCurrentEvtIdx(0),
   fInputFormulas(0),
   fLogger( new MsgLogger("DataSetFactory", kINFO) )
{
   // constructor
}

//_______________________________________________________________________
TMVA::DataSetFactory::~DataSetFactory() 
{
   // destructor
   std::vector<TTreeFormula*>::const_iterator formIt;

   for (formIt = fInputFormulas.begin()    ; formIt!=fInputFormulas.end()    ; formIt++) if (*formIt) delete *formIt;
   for (formIt = fTargetFormulas.begin()   ; formIt!=fTargetFormulas.end()   ; formIt++) if (*formIt) delete *formIt;
   for (formIt = fCutFormulas.begin()      ; formIt!=fCutFormulas.end()      ; formIt++) if (*formIt) delete *formIt;
   for (formIt = fWeightFormula.begin()    ; formIt!=fWeightFormula.end()    ; formIt++) if (*formIt) delete *formIt;
   for (formIt = fSpectatorFormulas.begin(); formIt!=fSpectatorFormulas.end(); formIt++) if (*formIt) delete *formIt;

   delete fLogger;
}

//_______________________________________________________________________
TMVA::DataSet* TMVA::DataSetFactory::CreateDataSet( TMVA::DataSetInfo& dsi, TMVA::DataInputHandler& dataInput ) 
{
   // steering the creation of a new dataset

   // build the first dataset from the data input
   DataSet * ds = BuildInitialDataSet( dsi, dataInput );

   if (ds->GetNEvents() > 1) {
      CalcMinMax(ds,dsi);
      
      // from the the final dataset build the correlation matrix
      for (UInt_t cl = 0; cl< dsi.GetNClasses(); cl++) {
         const TString className = dsi.GetClassInfo(cl)->GetName();
         dsi.SetCorrelationMatrix( className, CalcCorrelationMatrix( ds, cl ) );
         dsi.PrintCorrelationMatrix( className );
      }
      Log() << kINFO << " " << Endl;
   }
   return ds;
}

//_______________________________________________________________________
TMVA::DataSet* TMVA::DataSetFactory::BuildDynamicDataSet( TMVA::DataSetInfo& dsi ) 
{
   Log() << kDEBUG << "Build DataSet consisting of one Event with dynamically changing variables" << Endl;
   DataSet* ds = new DataSet(dsi);

   // create a DataSet with one Event which uses dynamic variables (pointers to variables)
   dsi.AddClass( "data" );
   dsi.GetClassInfo( "data" )->SetNumber(0);

   std::vector<VariableInfo>& varinfos = dsi.GetVariableInfos();
   std::vector<Float_t*>* evdyn = new std::vector<Float_t*>(0);
   std::vector<VariableInfo>::iterator it = varinfos.begin();
   for (;it!=varinfos.end();it++) evdyn->push_back( (Float_t*)(*it).GetExternalLink() );
   TMVA::Event * ev = new Event((const std::vector<Float_t*>*&)evdyn);
   std::vector<Event*>* newEventVector = new std::vector<Event*>;
   newEventVector->push_back(ev);
   ds->SetEventCollection(newEventVector, Types::kTraining);
   ds->SetCurrentType( Types::kTraining );
   ds->SetCurrentEvent( 0 );

   Log() << kINFO << "... created" << Endl;

   return ds;
}

//_______________________________________________________________________
void TMVA::DataSetFactory::InitOptions( TMVA::DataSetInfo& dsi, 
                                        std::vector< std::pair< Int_t, Int_t > >& nTrainTestEvents, 
                                        TString& normMode, UInt_t& splitSeed, 
                                        TString& splitMode ) 
{
   // the dataset splitting
   Configurable splitSpecs( dsi.GetSplitOptions() );
   splitSpecs.SetConfigName("DataSetFactory");
   splitSpecs.SetConfigDescription( "Configuration options given in the \"PrepareForTrainingAndTesting\" call; these options define the creation of the data sets used for training and expert validation by TMVA" );

   splitMode = "Random";    // the splitting mode
   splitSpecs.DeclareOptionRef( splitMode, "SplitMode",
                                "Method of picking training and testing events (default: random)" );
   splitSpecs.AddPreDefVal(TString("Random"));
   splitSpecs.AddPreDefVal(TString("Alternate"));
   splitSpecs.AddPreDefVal(TString("Block"));

   splitSeed = 100;
   splitSpecs.DeclareOptionRef( splitSeed, "SplitSeed",
                                "Seed for random event shuffling" );   

   normMode = "NumEvents";  // the weight normalisation modes
   splitSpecs.DeclareOptionRef( normMode, "NormMode",
                                "Overall renormalisation of event-by-event weights (NumEvents: average weight of 1 per event, independently for signal and background; EqualNumEvents: average weight of 1 per event for signal, and sum of weights for background equal to sum of weights for signal)" );
   splitSpecs.AddPreDefVal(TString("None"));
   splitSpecs.AddPreDefVal(TString("NumEvents"));
   splitSpecs.AddPreDefVal(TString("EqualNumEvents"));

   // the number of events
   nTrainTestEvents.resize( dsi.GetNClasses() );
   for (UInt_t cl = 0; cl < dsi.GetNClasses(); cl++) {
      nTrainTestEvents.at(cl).first  = 0;
      nTrainTestEvents.at(cl).second = 0;
      TString clName = dsi.GetClassInfo(cl)->GetName();
      TString titleTrain =  TString().Format("Number of training events of class %s (default: 0 = all)",clName.Data()).Data();
      TString titleTest  =  TString().Format("Number of test events of class %s (default: 0 = all)",clName.Data()).Data();
      splitSpecs.DeclareOptionRef( nTrainTestEvents.at(cl).first , TString("nTrain_")+clName, titleTrain );
      splitSpecs.DeclareOptionRef( nTrainTestEvents.at(cl).second, TString("nTest_")+clName, titleTest  );
   }

   splitSpecs.DeclareOptionRef( fVerbose, "V", "Verbosity (default: true)" );

   splitSpecs.DeclareOptionRef( fVerboseLevel=TString("Info"), "VerboseLevel", "VerboseLevel (Debug/Verbose/Info)" );
   splitSpecs.AddPreDefVal(TString("Debug"));
   splitSpecs.AddPreDefVal(TString("Verbose"));
   splitSpecs.AddPreDefVal(TString("Info"));

   splitSpecs.ParseOptions();
   splitSpecs.CheckForUnusedOptions();

   // output logging verbosity
   if (Verbose()) fLogger->SetMinType( kVERBOSE );   
   if (fVerboseLevel.CompareTo("Debug")   ==0) fLogger->SetMinType( kDEBUG );
   if (fVerboseLevel.CompareTo("Verbose") ==0) fLogger->SetMinType( kVERBOSE );
   if (fVerboseLevel.CompareTo("Info")    ==0) fLogger->SetMinType( kINFO );

   // put all to upper case
   splitMode.ToUpper(); normMode.ToUpper();
}

//_______________________________________________________________________
void TMVA::DataSetFactory::BuildEventVector( TMVA::DataSetInfo& dsi, 
                                             TMVA::DataInputHandler& dataInput, 
                                             std::vector< std::vector< Event* > >& tmpEventVector, 
                                             std::vector<Double_t>& sumOfWeights, 
                                             std::vector<Double_t>& nTempEvents, 
                                             std::vector<Double_t>& renormFactor,
                                             std::vector< std::vector< std::pair< Long64_t, Types::ETreeType > > >& userDefinedEventTypes ) 
{
   // build event vector
   tmpEventVector.resize(dsi.GetNClasses());

   // create the type, weight and boostweight branches
   const UInt_t nvars    = dsi.GetNVariables();
   const UInt_t ntgts    = dsi.GetNTargets();
   const UInt_t nvis     = dsi.GetNSpectators();
   //   std::vector<Float_t> fmlEval(nvars+ntgts+1+1+nvis);     // +1+1 for results of evaluation of cut and weight ttreeformula  

   // the sum of weights should be renormalised to the number of events
   renormFactor.assign( dsi.GetNClasses(), -1 );


   // number of signal and background events passing cuts
   std::vector< Int_t >    nInitialEvents( dsi.GetNClasses() );
   std::vector< Int_t >    nEvBeforeCut(   dsi.GetNClasses() );
   std::vector< Int_t >    nEvAfterCut(    dsi.GetNClasses() );
   std::vector< Float_t >  nWeEvBeforeCut( dsi.GetNClasses() );
   std::vector< Float_t >  nWeEvAfterCut(  dsi.GetNClasses() );
   std::vector< Double_t > nNegWeights(    dsi.GetNClasses() );
   std::vector< Float_t* > varAvLength(    dsi.GetNClasses() );

   Bool_t haveArrayVariable = kFALSE;
   Bool_t *varIsArray = new Bool_t[nvars];

   for (size_t i=0; i<varAvLength.size(); i++) {
      varAvLength[i] = new Float_t[nvars];
      for (UInt_t ivar=0; ivar<nvars; ivar++) {
         //varIsArray[ivar] = kFALSE;
         varAvLength[i][ivar] = 0;
      }
   }

   // if we work with chains we need to remember the current tree
   // if the chain jumps to a new tree we have to reset the formulas
   for (UInt_t cl=0; cl<dsi.GetNClasses(); cl++) {

      Log() << kINFO << "Create training and testing trees: looping over class " << dsi.GetClassInfo(cl)->GetName() 
            << "..." << Endl;

      // info output for weights
      const TString tmpWeight = dsi.GetClassInfo(cl)->GetWeight();
      if (tmpWeight!="") {
         Log() << kINFO << "Weight expression for class \"" << dsi.GetClassInfo(cl)->GetName() << "\": \""
               << tmpWeight << "\"" << Endl; 
      }
      else {
         Log() << kINFO << "No weight expression defined for class \"" << dsi.GetClassInfo(cl)->GetName() 
               << "\"" << Endl; 
      }
      
      // used for chains only
      TString currentFileName("");
      
      std::vector<TreeInfo>::const_iterator treeIt(dataInput.begin(dsi.GetClassInfo(cl)->GetName()));
      for (;treeIt!=dataInput.end(dsi.GetClassInfo(cl)->GetName()); treeIt++) {

         // read first the variables
         std::vector<Float_t> vars(nvars);
         std::vector<Float_t> tgts(ntgts);
         std::vector<Float_t> vis(nvis);
         TreeInfo currentInfo = *treeIt;
         
         Bool_t isChain = (TString("TChain") == currentInfo.GetTree()->ClassName());
         currentInfo.GetTree()->LoadTree(0);
         ChangeToNewTree( currentInfo, dsi );

         // count number of events in tree before cut
         nInitialEvents.at(cl) += currentInfo.GetTree()->GetEntries();
         
         std::vector< std::pair< Long64_t, Types::ETreeType > >& userEvType = userDefinedEventTypes.at(cl);
         if (userEvType.size() == 0 || userEvType.back().second != currentInfo.GetTreeType()) {
            userEvType.push_back( std::pair< Long64_t, Types::ETreeType >(tmpEventVector.at(cl).size()-1, currentInfo.GetTreeType()) );
         }

         // loop over events in ntuple
         for (Long64_t evtIdx = 0; evtIdx < currentInfo.GetTree()->GetEntries(); evtIdx++) {
            currentInfo.GetTree()->LoadTree(evtIdx);
            
            // may need to reload tree in case of chains
            if (isChain) {
               if (currentInfo.GetTree()->GetTree()->GetDirectory()->GetFile()->GetName() != currentFileName) {
                  currentFileName = currentInfo.GetTree()->GetTree()->GetDirectory()->GetFile()->GetName();
                  ChangeToNewTree( currentInfo, dsi );
               }
            }
            currentInfo.GetTree()->GetEntry(evtIdx);
            Int_t sizeOfArrays = 1;
            Int_t prevArrExpr = 0;
            
            // ======= evaluate all formulas =================

            // first we check if some of the formulas are arrays
            for (UInt_t ivar=0; ivar<nvars; ivar++) {
               Int_t ndata = fInputFormulas[ivar]->GetNdata();
               varAvLength[cl][ivar] += ndata;
               if (ndata == 1) continue;
               haveArrayVariable = kTRUE;
               varIsArray[ivar] = kTRUE;
               if (sizeOfArrays == 1) {
                  sizeOfArrays = ndata;
                  prevArrExpr = ivar;
               } 
               else if (sizeOfArrays!=ndata) {
                  Log() << kERROR << "ERROR while preparing training and testing trees:" << Endl;
                  Log() << "   multiple array-type expressions of different length were encountered" << Endl;
                  Log() << "   location of error: event " << evtIdx 
                        << " in tree " << currentInfo.GetTree()->GetName()
                        << " of file " << currentInfo.GetTree()->GetCurrentFile()->GetName() << Endl;
                  Log() << "   expression " << fInputFormulas[ivar]->GetTitle() << " has " 
                        << ndata << " entries, while" << Endl;
                  Log() << "   expression " << fInputFormulas[prevArrExpr]->GetTitle() << " has "
                        << fInputFormulas[prevArrExpr]->GetNdata() << " entries" << Endl;
                  Log() << kFATAL << "Need to abort" << Endl;
               }
            }

            // now we read the information
            for (Int_t idata = 0;  idata<sizeOfArrays; idata++) {
               Bool_t containsNaN = kFALSE;

               TTreeFormula* formula = 0;

               // the cut expression
               Float_t cutVal = 1;
               formula = fCutFormulas[cl];
               if (formula) {
                  Int_t ndata = formula->GetNdata();
                  cutVal = (ndata==1 ? 
                            formula->EvalInstance(0) :
                            formula->EvalInstance(idata));
                  if (TMath::IsNaN(cutVal)) {
                     containsNaN = kTRUE;
                     Log() << kWARNING << "Cut expression resolves to infinite value (NaN): " 
                           << formula->GetTitle() << Endl;
                  }
               }
               
               // the input variable
               for (UInt_t ivar=0; ivar<nvars; ivar++) {
                  formula = fInputFormulas[ivar];
                  Int_t ndata = formula->GetNdata();               
                  vars[ivar] = (ndata == 1 ? 
                                formula->EvalInstance(0) : 
                                formula->EvalInstance(idata));
                  if (TMath::IsNaN(vars[ivar])) {
                     containsNaN = kTRUE;
                     Log() << kWARNING << "Input expression resolves to infinite value (NaN): " 
                           << formula->GetTitle() << Endl;
                  }
               }

               // the targets
               for (UInt_t itrgt=0; itrgt<ntgts; itrgt++) {
                  formula = fTargetFormulas[itrgt];
                  Int_t ndata = formula->GetNdata();               
                  tgts[itrgt] = (ndata == 1 ? 
                                 formula->EvalInstance(0) : 
                                 formula->EvalInstance(idata));
                  if (TMath::IsNaN(tgts[itrgt])) {
                     containsNaN = kTRUE;
                     Log() << kWARNING << "Target expression resolves to infinite value (NaN): " 
                           << formula->GetTitle() << Endl;
                  }
               }

               // the spectators
               for (UInt_t itVis=0; itVis<nvis; itVis++) {
                  formula = fSpectatorFormulas[itVis];
                  Int_t ndata = formula->GetNdata();               
                  vis[itVis] = (ndata == 1 ? 
                                formula->EvalInstance(0) : 
                                formula->EvalInstance(idata));
                  if (TMath::IsNaN(vis[itVis])) {
                     containsNaN = kTRUE;
                     Log() << kWARNING << "Spectator expression resolves to infinite value (NaN): " 
                           << formula->GetTitle() << Endl;
                  }
               }


               // the weight
               Float_t weight = currentInfo.GetWeight(); // multiply by tree weight
               formula = fWeightFormula[cl];
               if (formula!=0) {
                  Int_t ndata = formula->GetNdata();
                  weight *= (ndata == 1 ?
                             formula->EvalInstance() :
                             formula->EvalInstance(idata));
                  if (TMath::IsNaN(weight)) {
                     containsNaN = kTRUE;
                     Log() << kWARNING << "Weight expression resolves to infinite value (NaN): " 
                           << formula->GetTitle() << Endl;
                  }
               }
            
               // Count the events before rejection due to cut or NaN value
               // (weighted and unweighted)
               nEvBeforeCut.at(cl) ++;
               if (!TMath::IsNaN(weight))
                  nWeEvBeforeCut.at(cl) += weight;

               // apply the cut
               // skip rest if cut is not fulfilled
               if (cutVal<0.5) continue;

               // global flag if negative weights exist -> can be used by classifiers who may 
               // require special data treatment (also print warning)
               if (weight < 0) nNegWeights.at(cl)++;

               // now read the event-values (variables and regression targets)

               if (containsNaN) {
                  Log() << kWARNING << "Event " << evtIdx;
                  if (sizeOfArrays>1) Log() << kWARNING << " rejected" << Endl;
                  continue;
               }

               // Count the events after rejection due to cut or NaN value
               // (weighted and unweighted)
               nEvAfterCut.at(cl) ++;
               nWeEvAfterCut.at(cl) += weight;

               // event accepted, fill temporary ntuple
               tmpEventVector.at(cl).push_back(new Event(vars, tgts , vis, cl , weight));

               // --------------- this is to keep <Event>->IsSignal() working. TODO: this should be removed on the long run
               ClassInfo* ci = dsi.GetClassInfo("Signal");
               if( ci == 0 ) tmpEventVector.at(cl).back()->SetSignalClass( 0 );
               else          tmpEventVector.at(cl).back()->SetSignalClass( ci->GetNumber()   );
               // ---------------

               // add up weights
               sumOfWeights.at(cl) += weight;
               nTempEvents.at(cl)  += 1;
            }
         }
         
         currentInfo.GetTree()->ResetBranchAddresses();
      }

      // compute renormalisation factors
      renormFactor.at(cl) = nTempEvents.at(cl)/sumOfWeights.at(cl);
   }

   // for output, check maximum class name length
   Int_t maxL = dsi.GetClassNameMaxLength();
   
   Log() << kINFO << "Number of events in input trees (after possible flattening of arrays):" << Endl;
   for (UInt_t cl = 0; cl < dsi.GetNClasses(); cl++) {
      Log() << kINFO << "    " 
            << setiosflags(ios::left) << std::setw(maxL) << dsi.GetClassInfo(cl)->GetName() 
            << "      -- number of events       : "
            << std::setw(5) << nEvBeforeCut.at(cl) 
            << "  / sum of weights: " << std::setw(5) << nWeEvBeforeCut.at(cl) << Endl;
   }

   for (UInt_t cl = 0; cl < dsi.GetNClasses(); cl++) {
      Log() << kINFO << "    " << std::setw(maxL) << dsi.GetClassInfo(cl)->GetName() 
            <<" tree -- total number of entries: " 
            << std::setw(5) << dataInput.GetEntries(dsi.GetClassInfo(cl)->GetName()) << Endl;
   }

   Log() << kINFO << "Preselection:" << Endl;
   if (dsi.HasCuts()) {
      for (UInt_t cl = 0; cl< dsi.GetNClasses(); cl++) {
         Log() << kINFO << "    " << setiosflags(ios::left) << std::setw(maxL) << dsi.GetClassInfo(cl)->GetName() 
               << " requirement: \"" << dsi.GetClassInfo(cl)->GetCut() << "\"" << Endl;
         Log() << kINFO << "    " 
               << setiosflags(ios::left) << std::setw(maxL) << dsi.GetClassInfo(cl)->GetName() 
               << "      -- number of events passed: "
               << std::setw(5) << nEvAfterCut.at(cl)
               << "  / sum of weights: " << std::setw(5) << nWeEvAfterCut.at(cl) << Endl;
         Log() << kINFO << "    " 
               << setiosflags(ios::left) << std::setw(maxL) << dsi.GetClassInfo(cl)->GetName() 
               << "      -- efficiency             : "
               << std::setw(6) << nWeEvAfterCut.at(cl)/nWeEvBeforeCut.at(cl) << Endl;
      }
   }
   else Log() << kINFO << "    No preselection cuts applied on event classes" << Endl;

   delete[] varIsArray;
   for (size_t i=0; i<varAvLength.size(); i++)
      delete[] varAvLength[i];

}

//_______________________________________________________________________
TMVA::DataSet* TMVA::DataSetFactory::MixEvents( TMVA::DataSetInfo& dsi, 
                                                std::vector< std::vector< TMVA::Event* > >& tmpEventVector, 
                                                std::vector< std::pair< Int_t, Int_t > >& nTrainTestEvents,
                                                const TString& splitMode, UInt_t splitSeed, 
                                                std::vector<Double_t>& renormFactor,
                                                std::vector< std::vector< std::pair< Long64_t, Types::ETreeType > > >& userDefinedEventTypes )
{
   // create a dataset from the datasetinfo object
   DataSet* ds = new DataSet(dsi);
   
   typedef std::vector<Event*>::size_type EVTVSIZE;

   std::vector<EVTVSIZE> origSize(dsi.GetNClasses());

   Log() << kVERBOSE << "Number of available training events:" << Endl;
   for ( UInt_t cl = 0; cl<dsi.GetNClasses(); cl++ ) {
      origSize.at(cl) = tmpEventVector.at(cl).size();
      Log() << kVERBOSE << "  " << dsi.GetClassInfo(cl)->GetName() << "    : " << origSize.at(cl) << Endl;
   }

   std::vector< std::vector< EVTVSIZE > > finalNEvents( dsi.GetNClasses() );
   for (UInt_t cl = 0; cl < dsi.GetNClasses(); cl++) {
      finalNEvents.at(cl).resize(2); // resize: training and test
      finalNEvents[cl][0] = nTrainTestEvents.at(cl).first;
      finalNEvents[cl][1] = nTrainTestEvents.at(cl).second;
   }

   // loop over all classes
   for ( UInt_t cl = 0; cl<dsi.GetNClasses(); cl++) { 
      
      if (finalNEvents.at(cl).at(0)>origSize.at(cl)) // training
         Log() << kFATAL << "More training events requested than available for the " << dsi.GetClassInfo(cl)->GetName() << Endl;
      
      if (finalNEvents.at(cl).at(1)>origSize.at(cl)) // testing
         Log() << kFATAL << "More testing events requested than available for the " << dsi.GetClassInfo(cl)->GetName() << Endl;
      
      if (finalNEvents.at(cl).at(0)>origSize.at(cl) || finalNEvents.at(cl).at(1)>origSize.at(cl) ) // training and testing
         Log() << kFATAL << "More testing and training events requested than available for the " << dsi.GetClassInfo(cl)->GetName() << Endl;

      if (finalNEvents.at(cl).at(0)==0 || finalNEvents.at(cl).at(1)==0) {   // if events requested for training or testing are 0 (== all)
         if (finalNEvents.at(cl).at(0)==0 && finalNEvents.at(cl).at(1)==0) { // if both, training and testing are 0
            finalNEvents.at(cl).at(0) = finalNEvents.at(cl).at(1) = origSize.at(cl)/2;  // use half of the events for training, the other half for testing
         }
         else if (finalNEvents.at(cl).at(1)==0) { // if testing is chosen "all"
            finalNEvents.at(cl).at(1)  = origSize.at(cl) - finalNEvents.at(cl).at(0); // take the remaining events (not training) for testing
         } 
         else {          // the other way around
            finalNEvents.at(cl).at(0)  = origSize.at(cl) - finalNEvents.at(cl).at(1); // take the remaining events (not testing) for training
         }
      }
   }

   TRandom3 rndm( splitSeed ); 

   // create event-lists for mixing
   std::vector< std::vector< TEventList* > > evtList(dsi.GetNClasses());
   for (UInt_t cl = 0; cl < dsi.GetNClasses(); cl++) {
      evtList.at(cl).resize(2);
      evtList.at(cl).at(0) = new TEventList();
      evtList.at(cl).at(1) = new TEventList();
   }

   std::vector< std::vector< std::vector<EVTVSIZE>::size_type > > userDefN(dsi.GetNClasses());  // class/training-testing/<size>

   for ( UInt_t cl = 0; cl<dsi.GetNClasses(); cl++ ) { // loop over the different classes
   
      const std::vector<EVTVSIZE>::size_type size = origSize.at(cl);

      userDefN[cl].resize(2,0); // 0 training, 1 testing

      if (splitMode == "RANDOM") {

         Log() << kINFO << "Randomly shuffle events in training and testing trees for " << dsi.GetClassInfo(cl)->GetName() << Endl;

         // the index array
         std::vector<EVTVSIZE> idxArray(size);
         std::vector<Char_t>   allPickedIdxArray(size);
         allPickedIdxArray.assign( size, Char_t(kFALSE) );

         // search for all events of which the trees have been defined as training or testing by the user
         std::vector< std::pair< Long64_t, Types::ETreeType > >::iterator it = userDefinedEventTypes[cl].begin();
         for(;it!=userDefinedEventTypes[cl].end(); it++)
            it = userDefinedEventTypes[cl].begin();
         Types::ETreeType currentType = Types::kMaxTreeType;
         // loop through the idxArray
         
         for (EVTVSIZE i = 0; i < size; i++) {
            // if i is larger than the eventnumber of the current entry of the user defined types
            if (it!=userDefinedEventTypes[cl].end() && Long64_t(i) >= (*it).first) {
               // then take the treetype as currentType and increse the iterator by one to point at the next entry
               currentType = (*it).second;
               it++;
            }
            // now things depending on the current tree type (the tree type of the event)
            // if (currentType == Types::kMaxTreeType ) ===> do nothing
            if (currentType == Types::kTraining || currentType == Types::kTesting) {
               Int_t tp = (currentType == Types::kTraining?0:1);
               evtList[cl][tp]->Enter(Long64_t(i));       // add the eventnumber of the picked event to the TEventList
               (userDefN[cl][tp])++;                      // one more has been picked
               allPickedIdxArray[i] = Char_t(kTRUE);              // mark as picked

               if (finalNEvents.at(cl).at(0) < userDefN[cl][tp]) 
                  Log() << kFATAL << "More " << (currentType == Types::kTraining?"training":"testing")  << " events [" << userDefN[cl][tp] << "] requested than available for the "
                        << dsi.GetClassInfo(cl)->GetName() << " [" << finalNEvents[cl][0] << "] by having specified too many " 
                        << (currentType == Types::kTraining?"training":"testing") << "input trees." << Endl;
            }
         }


         for (EVTVSIZE i=0; i<size; i++) { idxArray.at(i)=i; }
      
         for (Int_t itype=0; itype<2; itype++) {  // training (0) and then testing (1)

            // the selected events
            std::vector<Char_t> thisPickedIdxArray(size);
            thisPickedIdxArray.assign( size, Char_t(kFALSE) );


            EVTVSIZE pos = 0;
            for (EVTVSIZE i=0; i<finalNEvents.at(cl).at(itype); i++) {
               // throw random positions until one is found where the event hasn't been picked yet
               do { 
                  pos = EVTVSIZE(size * rndm.Rndm()); 
               } while (allPickedIdxArray.at(idxArray.at(pos)) == Char_t(kTRUE) );
               // pick the found event
               thisPickedIdxArray.at(idxArray.at(pos)) = Char_t(kTRUE);
               allPickedIdxArray .at(idxArray.at(pos)) = Char_t(kTRUE);
            }
            // write all for this class and this event type picked events into the according TEventList
            for (EVTVSIZE i=0; i<size; i++) if (thisPickedIdxArray.at(i)==Char_t(kTRUE)) evtList.at(cl).at(itype)->Enter(Long64_t(i)); 
         }
      } 
      else if (splitMode == "ALTERNATE") {
         Log() << kINFO << "Pick alternating training and test events from input tree for " 
               << dsi.GetClassInfo(cl)->GetName() << Endl;
      
         Int_t ntrain = finalNEvents.at(cl).at(0);   // training
         Int_t ntest  = finalNEvents.at(cl).at(1);   // testing

         UInt_t lcd       = LargestCommonDivider(ntrain,ntest);
         UInt_t trainfrac = ntrain/lcd;
         UInt_t modulo    = (ntrain+ntest)/lcd;

         for (EVTVSIZE i=0; i<finalNEvents.at(cl).at(0)+finalNEvents.at(cl).at(1); i++) {
            Bool_t isTrainingEvent = (i%modulo)<trainfrac;
            evtList.at(cl).at(isTrainingEvent ? 0:1)->Enter( i );
         }
      }
      else if (splitMode == "BLOCK") {
         Log() << kINFO << "Pick block-wise training and test events from input tree for " 
               << dsi.GetClassInfo(cl)->GetName() << Endl;
      
         for (EVTVSIZE i=0; i<finalNEvents.at(cl).at(0); i++)     // training events
            evtList.at(cl).at(0)->Enter( i );                     // write them into the training-eventlist of that class
         for (EVTVSIZE i=0; i<finalNEvents.at(cl).at(1); i++)     // test events 
            evtList.at(cl).at(1)->Enter( i + finalNEvents.at(cl).at(0));  // write them into test-eventlist of that class

      }
      else Log() << kFATAL << "Unknown type: " << splitMode << Endl;
   }

   // merge signal and background trees, and renormalise the event weights in this step   
   for (Int_t itreeType=0; itreeType<2; itreeType++) {

      Log() << kINFO << "Create internal " << (itreeType == 0 ? "training" : "testing") << " tree" << Endl;        

      std::vector<Event*>* newEventVector = new std::vector<Event*>();
      // hand the event vector over to the dataset, which will have to take care of destroying it
      ds->SetEventCollection(newEventVector, (itreeType==0? Types::kTraining : Types::kTesting) ); 

      EVTVSIZE newVectSize = 0;
      for (UInt_t cl = 0; cl < dsi.GetNClasses(); cl++) {
         newVectSize += evtList.at(cl).at(itreeType)->GetN();
      }
      newEventVector->reserve( newVectSize );

      for ( UInt_t cl=0; cl<dsi.GetNClasses(); cl++) {

         // renormalise only if non-trivial renormalisation factor
         for (EVTVSIZE ievt=0; ievt<tmpEventVector.at(cl).size(); ievt++) {
            if (!evtList.at(cl).at(itreeType)->Contains(Long64_t(ievt))) continue;

            newEventVector->push_back(tmpEventVector.at(cl)[ievt] );
            newEventVector->back()->ScaleWeight( renormFactor.at(cl) );

            ds->IncrementNClassEvents( itreeType, cl );
         }
      }
   }

   for (UInt_t cl = 0; cl < dsi.GetNClasses(); cl++) {
      tmpEventVector.at(cl).clear(); 
      tmpEventVector.at(cl).resize(0);
   }

   for (UInt_t cl = 0; cl < dsi.GetNClasses(); cl++) {
      delete evtList.at(cl).at(0);
      delete evtList.at(cl).at(1);
   }
   return ds;
}

//_______________________________________________________________________
TMVA::DataSet* TMVA::DataSetFactory::BuildInitialDataSet( DataSetInfo& dsi, DataInputHandler& dataInput ) 
{
   // if no entries, than create a DataSet with one Event which uses dynamic variables (pointers to variables)
   if (dataInput.GetEntries()==0) return BuildDynamicDataSet( dsi );
   // ------------------------------------------------------------------------------------

   // register the classes in the datasetinfo-object. Information comes from the trees in the dataInputHandler-object
   std::vector< TString >* classList = dataInput.GetClassList();
   for (std::vector<TString>::iterator it = classList->begin(); it< classList->end(); it++) {
      dsi.AddClass( (*it) );
   }
   delete classList;

   std::vector< std::pair< Int_t, Int_t > > nTrainTestEvents;
   std::vector< std::vector< std::pair< Long64_t, Types::ETreeType > > > userDefinedEventTypes( dsi.GetNClasses() ); // class/automatically growing/startindex+treetype

   TString normMode;
   TString splitMode;
   UInt_t splitSeed;
   InitOptions( dsi, nTrainTestEvents, normMode, splitSeed, splitMode );

   // ======= build event-vector =================================
   
   std::vector< std::vector< Event* > > tmpEventVector;

   std::vector<Double_t> sumOfWeights( dsi.GetNClasses() );
   std::vector<Double_t> nTempEvents ( dsi.GetNClasses() );
   std::vector<Double_t> renormFactor( dsi.GetNClasses() );
   BuildEventVector( dsi, dataInput, tmpEventVector, sumOfWeights, nTempEvents, renormFactor, userDefinedEventTypes );

   // ============================================================
   // create training and test tree
   // ============================================================

   Log() << kINFO << "Prepare training and Test samples:" << Endl;

   // ============================================================
   // renormalisation
   // ============================================================

   // print rescaling info
   if (normMode == "NONE") {
      Log() << kINFO << "No weight renormalisation applied: use original event weights" << Endl;
      renormFactor.assign( dsi.GetNClasses(), 1.0 );
   }
   else if (normMode == "NUMEVENTS") {
      Log() << kINFO << "Weight renormalisation mode: \"NumEvents\": renormalise the different classes" << Endl;
      Log() << kINFO << "... weights independently so that Sum[i=1..N_j]{w_i} = N_j, j=0,1,2..." << Endl;
      Log() << kINFO << "... (note that N_j is the sum of training and test events)" << Endl;
      for (UInt_t cl=0; cl<dsi.GetNClasses(); cl++) { 
         Log() << kINFO << "Rescale " << dsi.GetClassInfo(cl)->GetName() << " event weights by factor: " << renormFactor.at(cl) << Endl;
      }
   }
   else if (normMode == "EQUALNUMEVENTS") {
      Log() << kINFO << "Weight renormalisation mode: \"EqualNumEvents\": renormalise weights of events of classes" << Endl;
      Log() << kINFO << "   so that Sum[i=1..N_j]{w_i} = N_classA, j=classA, classB, ..." << Endl;
      Log() << kINFO << "   (note that N_j is the sum of training and test events)" << Endl;

      for (UInt_t cl = 1; cl < dsi.GetNClasses(); cl++ ) {
         renormFactor.at(cl) *= nTempEvents.at(0)/nTempEvents.at(cl);
      }
      for (UInt_t cl=0; cl<dsi.GetNClasses(); cl++) { 
         Log() << kINFO << "Rescale " << dsi.GetClassInfo(cl)->GetName() << " event weights by factor: " << renormFactor.at(cl) << Endl;
      }
   }
   else {
      Log() << kFATAL << "<PrepareForTrainingAndTesting> Unknown NormMode: " << normMode << Endl;
   }
   dsi.SetNormalization( normMode );

   // ============= now the events have to be mixed and put into training- and test-eventcollections =============
   
   DataSet* ds = MixEvents( dsi, tmpEventVector, nTrainTestEvents, splitMode, splitSeed, renormFactor, userDefinedEventTypes );
   
   Int_t maxL = dsi.GetClassNameMaxLength();
   Log() << kINFO << "Collected:" << Endl;
   for (UInt_t cl = 0; cl < dsi.GetNClasses(); cl++) {
      Log() << kINFO << "    " 
            << setiosflags(ios::left) << std::setw(maxL) << dsi.GetClassInfo(cl)->GetName() 
            << " training entries: " << ds->GetNClassEvents( 0, cl ) << Endl;
      Log() << kINFO << "    " 
            << setiosflags(ios::left) << std::setw(maxL) << dsi.GetClassInfo(cl)->GetName() 
            << " testing  entries: " << ds->GetNClassEvents( 1, cl ) << Endl;      
   }

   return ds;
}

//_______________________________________________________________________
void TMVA::DataSetFactory::ChangeToNewTree( TreeInfo& tinfo, const DataSetInfo & dsi )
{ 
   // While the data gets copied into the local training and testing
   // trees, the input tree can change (for intance when changing from
   // signal to background tree, or using TChains as input) The
   // TTreeFormulas, that hold the input expressions need to be
   // reassociated with the new tree, which is done here

   TTree *tr = tinfo.GetTree()->GetTree();

   tr->SetBranchStatus("*",1);

   Bool_t hasDollar = false;

   // 1) the input variable formulas
   Log() << kDEBUG << "transform input variables" << Endl;
   std::vector<TTreeFormula*>::const_iterator formIt;
   for (formIt = fInputFormulas.begin(); formIt!=fInputFormulas.end(); formIt++) if (*formIt) delete *formIt;
   fInputFormulas.clear();
   TTreeFormula* ttf = 0;

   for (UInt_t i=0; i<dsi.GetNVariables(); i++) {
      ttf = new TTreeFormula( Form( "Formula%s", dsi.GetVariableInfo(i).GetInternalName().Data() ),
                              dsi.GetVariableInfo(i).GetExpression().Data(), tr );
      fInputFormulas.push_back( ttf );
      if (ttf->GetNcodes() == 0)
         Log() << kFATAL << "Expression: " << dsi.GetVariableInfo(i).GetExpression() 
               << " does not appear to depend on any TTree variable --> please check spelling" << Endl;
      if (dsi.GetVariableInfo(i).GetExpression().Contains( "$" ) ) hasDollar = true;
   }

   //
   // targets
   //
   Log() << kDEBUG << "transform regression targets" << Endl;
   for (formIt = fTargetFormulas.begin(); formIt!=fTargetFormulas.end(); formIt++) if (*formIt) delete *formIt;
   fTargetFormulas.clear();
   for (UInt_t i=0; i<dsi.GetNTargets(); i++) {
      ttf = new TTreeFormula( Form( "Formula%s", dsi.GetTargetInfo(i).GetInternalName().Data() ),
                              dsi.GetTargetInfo(i).GetExpression().Data(), tr );
      fTargetFormulas.push_back( ttf );
      if (ttf->GetNcodes() == 0) {
         Log() << kFATAL << "Expression: " << dsi.GetTargetInfo(i).GetExpression() 
               << " does not appear to depend on any TTree variable --> please check spelling" << Endl;
      }
      if (dsi.GetTargetInfo(i).GetExpression().Contains( "$" ) ) hasDollar = true;
   }

   //
   // spectators
   //
   Log() << kDEBUG << "transform spectator variables" << Endl;
   for (formIt = fSpectatorFormulas.begin(); formIt!=fSpectatorFormulas.end(); formIt++) if (*formIt) delete *formIt;
   fSpectatorFormulas.clear();
   for (UInt_t i=0; i<dsi.GetNSpectators(); i++) {
      ttf = new TTreeFormula( Form( "Formula%s", dsi.GetSpectatorInfo(i).GetInternalName().Data() ),
                              dsi.GetSpectatorInfo(i).GetExpression().Data(), tr );
      fSpectatorFormulas.push_back( ttf );
      if (ttf->GetNcodes() == 0) {
         Log() << kFATAL << "Expression: " << dsi.GetSpectatorInfo(i).GetExpression() 
               << " does not appear to depend on any TTree variable --> please check spelling" << Endl;
      }
      if (dsi.GetSpectatorInfo(i).GetExpression().Contains( "$" ) ) hasDollar = true;
   }

   //
   // the cuts (one per class, if non-existent: formula pointer = 0)
   //
   Log() << kDEBUG << "transform cuts" << Endl;
   for (formIt = fCutFormulas.begin(); formIt!=fCutFormulas.end(); formIt++) if (*formIt) delete *formIt;
   fCutFormulas.clear();
   for (UInt_t clIdx=0; clIdx<dsi.GetNClasses(); clIdx++) {
      const TCut& tmpCut = dsi.GetClassInfo(clIdx)->GetCut();
      const TString tmpCutExp(tmpCut.GetTitle());
      ttf = 0;
      if (tmpCutExp!="") {
         ttf = new TTreeFormula( Form("CutClass%i",clIdx), tmpCutExp, tr );
      }
      fCutFormulas.push_back( ttf );
      if (ttf && ttf->GetNcodes() == 0 )
         Log() << kFATAL << "Class \"" << dsi.GetClassInfo(clIdx)->GetName()
               << "\" cut \"" << dsi.GetClassInfo(clIdx)->GetCut()
               << "\" does not appear to depend on any TTree variable --> please check spelling" << Endl;
      if (TString(tmpCut.GetTitle()).Contains( "$" ) ) hasDollar = true;
   }

   //
   // the weights (one per class, if non-existent: formula pointer = 0)
   //
   Log() << kDEBUG << "transform weights" << Endl;
   for (formIt = fWeightFormula.begin(); formIt!=fWeightFormula.end(); formIt++) if (*formIt) delete *formIt;
   fWeightFormula.clear();
   for (UInt_t clIdx=0; clIdx<dsi.GetNClasses(); clIdx++) {
      const TString tmpWeight = dsi.GetClassInfo(clIdx)->GetWeight();

      if (dsi.GetClassInfo(clIdx)->GetName() != tinfo.GetClassName() ) { // if the tree is of another class
         fWeightFormula.push_back( 0 );
         continue; 
      }

      ttf = 0;
      if (tmpWeight!="") {
         ttf = new TTreeFormula( "FormulaWeight", tmpWeight, tr );
         if (ttf && ttf->GetNcodes() == 0) {
            ttf = 0;
            Log() << kFATAL << "Class " << dsi.GetClassInfo(clIdx)->GetName() << " weight '" 
                  << dsi.GetClassInfo(clIdx)->GetWeight()
                  << "' does not appear to depend on any TTree variable --> please check spelling" << Endl;
         }
         if (tmpWeight.Contains( "$" )) hasDollar = true;
      }
      else {
         ttf = 0;
      }
      fWeightFormula.push_back( ttf );
   }
   Log() << kDEBUG << "enable branches" << Endl;
   // now enable only branches that are needed in any input formula, target, cut, weight

   if (!hasDollar) {
      tr->SetBranchStatus("*",0);
      Log() << kDEBUG << "enable branches: input variables" << Endl;
      // input vars
      for (formIt = fInputFormulas.begin(); formIt!=fInputFormulas.end(); formIt++) {
         ttf = *formIt;
         // 	 Log() << kDEBUG << "ttf: " << ttf->GetName() << Endl;
         // 	 Log() << kDEBUG << "ttf->GetNcodes(): " << ttf->GetNcodes() << Endl;
         for (Int_t bi = 0; bi<ttf->GetNcodes(); bi++) {
            // 	    Log() << kDEBUG << "bi: " << bi << Endl;
            // 	    Log() << kDEBUG << "ttf->GetLeaf(bi): " << ttf->GetLeaf(bi) << Endl;
            // 	    Log() << kDEBUG << "ttf->GetLeafInfo(bi): " << ttf->GetLeafInfo(bi) << Endl;
            //	 Log() << kDEBUG << "ttf->GetLeaf(bi)->GetBranch(): " << ttf->GetLeaf(bi)->GetBranch() << Endl;
            //	 Log() << kDEBUG << "ttf->GetLeaf(bi)->GetBranch()->GetName(): " << ttf->GetLeaf(bi)->GetBranch()->GetName() << Endl;
            tr->SetBranchStatus( ttf->GetLeaf(bi)->GetBranch()->GetName(), 1 );
            //	    tr->GetListOfLeaves()->Print();
         }
      }
      // targets
      Log() << kDEBUG << "enable branches: targets" << Endl;
      for (formIt = fTargetFormulas.begin(); formIt!=fTargetFormulas.end(); formIt++) {
         ttf = *formIt;
         for (Int_t bi = 0; bi<ttf->GetNcodes(); bi++)
            tr->SetBranchStatus( ttf->GetLeaf(bi)->GetBranch()->GetName(), 1 );
      }
      // spectators
      Log() << kDEBUG << "enable branches: spectators" << Endl;
      for (formIt = fSpectatorFormulas.begin(); formIt!=fSpectatorFormulas.end(); formIt++) {
         ttf = *formIt;
         for (Int_t bi = 0; bi<ttf->GetNcodes(); bi++)
            tr->SetBranchStatus( ttf->GetLeaf(bi)->GetBranch()->GetName(), 1 );
      }
      // cuts
      Log() << kDEBUG << "enable branches: cuts" << Endl;
      for (formIt = fCutFormulas.begin(); formIt!=fCutFormulas.end(); formIt++) {
         ttf = *formIt;
         if (!ttf) continue;
         for (Int_t bi = 0; bi<ttf->GetNcodes(); bi++)
            tr->SetBranchStatus( ttf->GetLeaf(bi)->GetBranch()->GetName(), 1 );
      }
      // weights
      Log() << kDEBUG << "enable branches: weights" << Endl;
      for (formIt = fWeightFormula.begin(); formIt!=fWeightFormula.end(); formIt++) {
         ttf = *formIt;
         if (!ttf) continue;
         for (Int_t bi = 0; bi<ttf->GetNcodes(); bi++)
            tr->SetBranchStatus( ttf->GetLeaf(bi)->GetBranch()->GetName(), 1 );
      }
   }
   Log() << kDEBUG << "tree initialized" << Endl;
   return;
}

//_______________________________________________________________________
void TMVA::DataSetFactory::CalcMinMax( DataSet* ds, TMVA::DataSetInfo& dsi )
{
   // compute covariance matrix
   const UInt_t nvar  = ds->GetNVariables();
   const UInt_t ntgts = ds->GetNTargets();
   const UInt_t nvis  = ds->GetNSpectators();

   Float_t *min = new Float_t[nvar];
   Float_t *max = new Float_t[nvar];
   Float_t *tgmin = new Float_t[ntgts];
   Float_t *tgmax = new Float_t[ntgts];
   Float_t *vmin  = new Float_t[nvis];
   Float_t *vmax  = new Float_t[nvis];

   for (UInt_t ivar=0; ivar<nvar ; ivar++) {   min[ivar] = 1e30;   max[ivar] = -1e30; }
   for (UInt_t ivar=0; ivar<ntgts; ivar++) { tgmin[ivar] = 1e30; tgmax[ivar] = -1e30; }
   for (UInt_t ivar=0; ivar<nvis;  ivar++) {  vmin[ivar] = 1e30;  vmax[ivar] = -1e30; }

   // perform event loop

   for (Int_t i=0; i<ds->GetNEvents(); i++) {
      Event * ev = ds->GetEvent(i);
      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         Double_t v = ev->GetValue(ivar);
         if (v<min[ivar]) min[ivar] = v;
         if (v>max[ivar]) max[ivar] = v;
      }
      for (UInt_t itgt=0; itgt<ntgts; itgt++) {
         Double_t v = ev->GetTarget(itgt);
         if (v<tgmin[itgt]) tgmin[itgt] = v;
         if (v>tgmax[itgt]) tgmax[itgt] = v;
      }
      for (UInt_t ivis=0; ivis<nvis; ivis++) {
         Double_t v = ev->GetSpectator(ivis);
         if (v<vmin[ivis]) vmin[ivis] = v;
         if (v>vmax[ivis]) vmax[ivis] = v;
      }
   }

   for (UInt_t ivar=0; ivar<nvar; ivar++) {
      dsi.GetVariableInfo(ivar).SetMin(min[ivar]);
      dsi.GetVariableInfo(ivar).SetMax(max[ivar]);
   }
   for (UInt_t ivar=0; ivar<ntgts; ivar++) {
      dsi.GetTargetInfo(ivar).SetMin(tgmin[ivar]);
      dsi.GetTargetInfo(ivar).SetMax(tgmax[ivar]);
   }
   for (UInt_t ivar=0; ivar<nvis; ivar++) {
      dsi.GetSpectatorInfo(ivar).SetMin(vmin[ivar]);
      dsi.GetSpectatorInfo(ivar).SetMax(vmax[ivar]);
   }
   delete [] min;
   delete [] max;
   delete [] tgmin;
   delete [] tgmax;
   delete [] vmin;
   delete [] vmax;
}

//_______________________________________________________________________
TMatrixD* TMVA::DataSetFactory::CalcCorrelationMatrix( DataSet* ds, const UInt_t classNumber )
{
   // computes correlation matrix for variables "theVars" in tree;
   // "theType" defines the required event "type" 
   // ("type" variable must be present in tree)

   // first compute variance-covariance
   TMatrixD* mat = CalcCovarianceMatrix( ds, classNumber );

   // now the correlation
   UInt_t nvar = ds->GetNVariables(), ivar, jvar;

   for (ivar=0; ivar<nvar; ivar++) {
      for (jvar=0; jvar<nvar; jvar++) {
         if (ivar != jvar) {
            Double_t d = (*mat)(ivar, ivar)*(*mat)(jvar, jvar);
            if (d > 0) (*mat)(ivar, jvar) /= sqrt(d);
            else {
               Log() << kWARNING << "<GetCorrelationMatrix> Zero variances for variables "
                     << "(" << ivar << ", " << jvar << ") = " << d                   
                     << Endl;
               (*mat)(ivar, jvar) = 0;
            }
         }
      }
   }

   for (ivar=0; ivar<nvar; ivar++) (*mat)(ivar, ivar) = 1.0;

   return mat;
}

//_______________________________________________________________________
TMatrixD* TMVA::DataSetFactory::CalcCovarianceMatrix( DataSet * ds, const UInt_t classNumber )
{
   // compute covariance matrix

   UInt_t nvar = ds->GetNVariables();
   UInt_t ivar = 0, jvar = 0;

   TMatrixD* mat = new TMatrixD( nvar, nvar );

   // init matrices
   TVectorD vec(nvar);
   TMatrixD mat2(nvar, nvar);      
   for (ivar=0; ivar<nvar; ivar++) {
      vec(ivar) = 0;
      for (jvar=0; jvar<nvar; jvar++) mat2(ivar, jvar) = 0;
   }

   // perform event loop
   Double_t ic = 0;
   for (Int_t i=0; i<ds->GetNEvents(); i++) {

      // fill the event
      Event * ev = ds->GetEvent(i);

      if (ev->GetClass() == classNumber ) {

         Double_t weight = ev->GetWeight();
         ic += weight; // count used events

         for (ivar=0; ivar<nvar; ivar++) {

            Double_t xi = ev->GetVal(ivar);
            vec(ivar) += xi*weight;
            mat2(ivar, ivar) += (xi*xi*weight);

            for (jvar=ivar+1; jvar<nvar; jvar++) {
               Double_t xj =  ev->GetVal(jvar);
               mat2(ivar, jvar) += (xi*xj*weight);
               mat2(jvar, ivar) = mat2(ivar, jvar); // symmetric matrix
            }
         }
      }
   }

   // variance-covariance
   for (ivar=0; ivar<nvar; ivar++) {
      for (jvar=0; jvar<nvar; jvar++) {
         (*mat)(ivar, jvar) = mat2(ivar, jvar)/ic - vec(ivar)*vec(jvar)/(ic*ic);
      }
   }

   return mat;
}

