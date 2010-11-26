// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Eckhard von Toerne, Helge Voss

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
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch> - CERN, Switzerland           *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Eckhard von Toerne <evt@physik.uni-bonn.de>  - U. of Bonn, Germany        *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2009:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <assert.h>

#include <map>
#include <vector>
#include <iomanip>
#include <iostream>

#include <algorithm>
#include <functional>
#include <numeric>

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
   if(dsi.GetNClasses()==0){
      dsi.AddClass( "data" );
      dsi.GetClassInfo( "data" )->SetNumber(0);
   }
   
   std::vector<Float_t*>* evdyn = new std::vector<Float_t*>(0);

   std::vector<VariableInfo>& varinfos = dsi.GetVariableInfos();
   std::vector<VariableInfo>::iterator it = varinfos.begin();
   for (;it!=varinfos.end();it++) evdyn->push_back( (Float_t*)(*it).GetExternalLink() );

   std::vector<VariableInfo>& spectatorinfos = dsi.GetSpectatorInfos();
   it = spectatorinfos.begin();
   for (;it!=spectatorinfos.end();it++) evdyn->push_back( (Float_t*)(*it).GetExternalLink() );

   TMVA::Event * ev = new Event((const std::vector<Float_t*>*&)evdyn, varinfos.size());
   std::vector<Event*>* newEventVector = new std::vector<Event*>;
   newEventVector->push_back(ev);

   ds->SetEventCollection(newEventVector, Types::kTraining);
   ds->SetCurrentType( Types::kTraining );
   ds->SetCurrentEvent( 0 );

   return ds;
}


//_______________________________________________________________________
TMVA::DataSet* TMVA::DataSetFactory::BuildInitialDataSet( DataSetInfo& dsi, DataInputHandler& dataInput ) 
{
   // if no entries, than create a DataSet with one Event which uses dynamic variables (pointers to variables)
   if (dataInput.GetEntries()==0) return BuildDynamicDataSet( dsi );
   // ------------------------------------------------------------------------------------

   // register the classes in the datasetinfo-object
   // information comes from the trees in the dataInputHandler-object
   std::vector< TString >* classList = dataInput.GetClassList();
   for (std::vector<TString>::iterator it = classList->begin(); it< classList->end(); it++) {
      dsi.AddClass( (*it) );
   }
   delete classList;

   TString normMode;
   TString splitMode;
   TString mixMode;
   UInt_t splitSeed;

   // ======= build event-vector tentative new ordering =================================
   
   TMVA::EventVectorOfClassesOfTreeType tmpEventVector;
   TMVA::NumberPerClassOfTreeType       nTrainTestEvents;

   InitOptions     ( dsi, nTrainTestEvents, normMode, splitSeed, splitMode , mixMode );
   BuildEventVector( dsi, dataInput, tmpEventVector );
      
   DataSet* ds = MixEvents( dsi, tmpEventVector, nTrainTestEvents, splitMode, mixMode, normMode, splitSeed);

   const Bool_t showCollectedOutput = kFALSE;
   if (showCollectedOutput) {
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
      Log() << kINFO << " " << Endl;
   }

   return ds;
}

//_______________________________________________________________________
Bool_t TMVA::DataSetFactory::CheckTTreeFormula( TTreeFormula* ttf, const TString& expression, Bool_t& hasDollar )
{ 
   // checks a TTreeFormula for problems
   Bool_t worked = kTRUE;
      
   if( ttf->GetNdim() <= 0 )
      Log() << kFATAL << "Expression " << expression.Data() << " could not be resolved to a valid formula. " << Endl;
   //    if( ttf->GetNcodes() == 0 ){
   //       Log() << kWARNING << "Expression: " << expression.Data() << " does not appear to depend on any TTree variable --> please check spelling" << Endl;
   //       worked = kFALSE;
   //    }
   if( ttf->GetNdata() == 0 ){
      Log() << kWARNING << "Expression: " << expression.Data() 
            << " does not provide data for this event. "
            << "This event is not taken into account. --> please check if you use as a variable "
            << "an entry of an array which is not filled for some events "
            << "(e.g. arr[4] when arr has only 3 elements)." << Endl;
      Log() << kWARNING << "If you want to take the event into account you can do something like: "
            << "\"Alt$(arr[4],0)\" where in cases where arr doesn't have a 4th element, "
            << " 0 is taken as an alternative." << Endl;
      worked = kFALSE;
   }
   if( expression.Contains("$") ) hasDollar = kTRUE;
   return worked;
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

   Bool_t hasDollar = kFALSE;

   // 1) the input variable formulas
   Log() << kDEBUG << "transform input variables" << Endl;
   std::vector<TTreeFormula*>::const_iterator formIt, formItEnd;
   for (formIt = fInputFormulas.begin(), formItEnd=fInputFormulas.end(); formIt!=formItEnd; formIt++) if (*formIt) delete *formIt;
   fInputFormulas.clear();
   TTreeFormula* ttf = 0;

   for (UInt_t i=0; i<dsi.GetNVariables(); i++) {
      ttf = new TTreeFormula( Form( "Formula%s", dsi.GetVariableInfo(i).GetInternalName().Data() ),
                              dsi.GetVariableInfo(i).GetExpression().Data(), tr );
      CheckTTreeFormula( ttf, dsi.GetVariableInfo(i).GetExpression(), hasDollar );
      fInputFormulas.push_back( ttf );
   }

   //
   // targets
   //
   Log() << kDEBUG << "transform regression targets" << Endl;
   for (formIt = fTargetFormulas.begin(), formItEnd = fTargetFormulas.end(); formIt!=formItEnd; formIt++) if (*formIt) delete *formIt;
   fTargetFormulas.clear();
   for (UInt_t i=0; i<dsi.GetNTargets(); i++) {
      ttf = new TTreeFormula( Form( "Formula%s", dsi.GetTargetInfo(i).GetInternalName().Data() ),
                              dsi.GetTargetInfo(i).GetExpression().Data(), tr );
      CheckTTreeFormula( ttf, dsi.GetTargetInfo(i).GetExpression(), hasDollar );
      fTargetFormulas.push_back( ttf );
   }

   //
   // spectators
   //
   Log() << kDEBUG << "transform spectator variables" << Endl;
   for (formIt = fSpectatorFormulas.begin(), formItEnd = fSpectatorFormulas.end(); formIt!=formItEnd; formIt++) if (*formIt) delete *formIt;
   fSpectatorFormulas.clear();
   for (UInt_t i=0; i<dsi.GetNSpectators(); i++) {
      ttf = new TTreeFormula( Form( "Formula%s", dsi.GetSpectatorInfo(i).GetInternalName().Data() ),
                              dsi.GetSpectatorInfo(i).GetExpression().Data(), tr );
      CheckTTreeFormula( ttf, dsi.GetSpectatorInfo(i).GetExpression(), hasDollar );
      fSpectatorFormulas.push_back( ttf );
   }

   //
   // the cuts (one per class, if non-existent: formula pointer = 0)
   //
   Log() << kDEBUG << "transform cuts" << Endl;
   for (formIt = fCutFormulas.begin(), formItEnd = fCutFormulas.end(); formIt!=formItEnd; formIt++) if (*formIt) delete *formIt;
   fCutFormulas.clear();
   for (UInt_t clIdx=0; clIdx<dsi.GetNClasses(); clIdx++) {
      const TCut& tmpCut = dsi.GetClassInfo(clIdx)->GetCut();
      const TString tmpCutExp(tmpCut.GetTitle());
      ttf = 0;
      if (tmpCutExp!="") {
         ttf = new TTreeFormula( Form("CutClass%i",clIdx), tmpCutExp, tr );
         Bool_t worked = CheckTTreeFormula( ttf, tmpCutExp, hasDollar );
         if( !worked ){
            Log() << kWARNING << "Please check class \"" << dsi.GetClassInfo(clIdx)->GetName()
                  << "\" cut \"" << dsi.GetClassInfo(clIdx)->GetCut() << Endl;
         }
      }
      fCutFormulas.push_back( ttf );
   }

   //
   // the weights (one per class, if non-existent: formula pointer = 0)
   //
   Log() << kDEBUG << "transform weights" << Endl;
   for (formIt = fWeightFormula.begin(), formItEnd = fWeightFormula.end(); formIt!=formItEnd; formIt++) if (*formIt) delete *formIt;
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
         Bool_t worked = CheckTTreeFormula( ttf, tmpWeight, hasDollar );
         if( !worked ){
            Log() << kWARNING << "Please check class \"" << dsi.GetClassInfo(clIdx)->GetName()
                  << "\" weight \"" << dsi.GetClassInfo(clIdx)->GetWeight() << Endl;
         }
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
         for (Int_t bi = 0; bi<ttf->GetNcodes(); bi++) {
            tr->SetBranchStatus( ttf->GetLeaf(bi)->GetBranch()->GetName(), 1 );
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

   for (UInt_t ivar=0; ivar<nvar ; ivar++) {   min[ivar] = FLT_MAX;   max[ivar] = -FLT_MAX; }
   for (UInt_t ivar=0; ivar<ntgts; ivar++) { tgmin[ivar] = FLT_MAX; tgmax[ivar] = -FLT_MAX; }
   for (UInt_t ivar=0; ivar<nvis;  ivar++) {  vmin[ivar] = FLT_MAX;  vmax[ivar] = -FLT_MAX; }

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
      if( TMath::Abs(max[ivar]-min[ivar]) <= FLT_MIN )
         Log() << kFATAL << "Variable " << dsi.GetVariableInfo(ivar).GetExpression().Data() << " is constant. Please remove the variable." << Endl;
   }
   for (UInt_t ivar=0; ivar<ntgts; ivar++) {
      dsi.GetTargetInfo(ivar).SetMin(tgmin[ivar]);
      dsi.GetTargetInfo(ivar).SetMax(tgmax[ivar]);
      if( TMath::Abs(tgmax[ivar]-tgmin[ivar]) <= FLT_MIN )
         Log() << kFATAL << "Target " << dsi.GetTargetInfo(ivar).GetExpression().Data() << " is constant. Please remove the variable." << Endl;
   }
   for (UInt_t ivar=0; ivar<nvis; ivar++) {
      dsi.GetSpectatorInfo(ivar).SetMin(vmin[ivar]);
      dsi.GetSpectatorInfo(ivar).SetMax(vmax[ivar]);
      //       if( TMath::Abs(vmax[ivar]-vmin[ivar]) <= FLT_MIN )
      //          Log() << kWARNING << "Spectator variable " << dsi.GetSpectatorInfo(ivar).GetExpression().Data() << " is constant." << Endl;
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

      Event * ev = ds->GetEvent(i);
      if (ev->GetClass() != classNumber ) continue;

      Double_t weight = ev->GetWeight();
      ic += weight; // count used events
      
      for (ivar=0; ivar<nvar; ivar++) {
         
         Double_t xi = ev->GetValue(ivar);
         vec(ivar) += xi*weight;
         mat2(ivar, ivar) += (xi*xi*weight);
         
         for (jvar=ivar+1; jvar<nvar; jvar++) {
            Double_t xj =  ev->GetValue(jvar);
            mat2(ivar, jvar) += (xi*xj*weight);
         }
      }
   }

   for (ivar=0; ivar<nvar; ivar++)
      for (jvar=ivar+1; jvar<nvar; jvar++)
         mat2(jvar, ivar) = mat2(ivar, jvar); // symmetric matrix


   // variance-covariance
   for (ivar=0; ivar<nvar; ivar++) {
      for (jvar=0; jvar<nvar; jvar++) {
         (*mat)(ivar, jvar) = mat2(ivar, jvar)/ic - vec(ivar)*vec(jvar)/(ic*ic);
      }
   }

   return mat;
}

// --------------------------------------- new versions

//_______________________________________________________________________
void TMVA::DataSetFactory::InitOptions( TMVA::DataSetInfo& dsi, 
                                        TMVA::NumberPerClassOfTreeType& nTrainTestEvents, 
                                        TString& normMode, UInt_t& splitSeed, 
                                        TString& splitMode,
                                        TString& mixMode  ) 
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

   mixMode = "SameAsSplitMode";    // the splitting mode
   splitSpecs.DeclareOptionRef( mixMode, "MixMode",
                                "Method of mixing events of differnt classes into one dataset (default: SameAsSplitMode)" );
   splitSpecs.AddPreDefVal(TString("SameAsSplitMode"));
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

   // initialization
   nTrainTestEvents.insert( TMVA::NumberPerClassOfTreeType::value_type( Types::kTraining, TMVA::NumberPerClass( dsi.GetNClasses() ) ) );
   nTrainTestEvents.insert( TMVA::NumberPerClassOfTreeType::value_type( Types::kTesting,  TMVA::NumberPerClass( dsi.GetNClasses() ) ) );

   // fill in the numbers
   for (UInt_t cl = 0; cl < dsi.GetNClasses(); cl++) {
      nTrainTestEvents[Types::kTraining].at(cl)  = 0;
      nTrainTestEvents[Types::kTesting].at(cl)   = 0;

      TString clName = dsi.GetClassInfo(cl)->GetName();
      TString titleTrain =  TString().Format("Number of training events of class %s (default: 0 = all)",clName.Data()).Data();
      TString titleTest  =  TString().Format("Number of test events of class %s (default: 0 = all)",clName.Data()).Data();

      splitSpecs.DeclareOptionRef( nTrainTestEvents[Types::kTraining].at(cl) , TString("nTrain_")+clName, titleTrain );
      splitSpecs.DeclareOptionRef( nTrainTestEvents[Types::kTesting].at(cl)  , TString("nTest_")+clName , titleTest  );
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
   splitMode.ToUpper(); mixMode.ToUpper(); normMode.ToUpper();
   // adjust mixmode if same as splitmode option has been set
   Log() << kINFO << "Splitmode is: \"" << splitMode << "\" the mixmode is: \"" << mixMode << "\"" << Endl;
   if (mixMode=="SAMEASSPLITMODE") mixMode = splitMode;
   else if (mixMode!=splitMode) 
      Log() << kINFO << "DataSet splitmode="<<splitMode
            <<" differs from mixmode="<<mixMode<<Endl;
}


//_______________________________________________________________________
void  TMVA::DataSetFactory::BuildEventVector( TMVA::DataSetInfo& dsi, 
                                              TMVA::DataInputHandler& dataInput, 
                                              TMVA::EventVectorOfClassesOfTreeType& tmpEventVector )
{
   // build empty event vectors
   // distributes events between kTraining/kTesting/kMaxTreeType
   
   tmpEventVector.insert( std::make_pair(Types::kTraining   ,TMVA::EventVectorOfClasses(dsi.GetNClasses() ) ) );
   tmpEventVector.insert( std::make_pair(Types::kTesting    ,TMVA::EventVectorOfClasses(dsi.GetNClasses() ) ) );
   tmpEventVector.insert( std::make_pair(Types::kMaxTreeType,TMVA::EventVectorOfClasses(dsi.GetNClasses() ) ) );


   // create the type, weight and boostweight branches
   const UInt_t nvars    = dsi.GetNVariables();
   const UInt_t ntgts    = dsi.GetNTargets();
   const UInt_t nvis     = dsi.GetNSpectators();
   //   std::vector<Float_t> fmlEval(nvars+ntgts+1+1+nvis);     // +1+1 for results of evaluation of cut and weight ttreeformula  

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

      Log() << kINFO << "Create training and testing trees -- looping over class \"" 
            << dsi.GetClassInfo(cl)->GetName() << "\" ..." << Endl;

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
         
//          std::vector< std::pair< Long64_t, Types::ETreeType > >& userEvType = userDefinedEventTypes.at(cl);
//          if (userEvType.size() == 0 || userEvType.back().second != currentInfo.GetTreeType()) {
//             userEvType.push_back( std::make_pair< Long64_t, Types::ETreeType >(tmpEventVector.at(cl).size(), currentInfo.GetTreeType()) );
//          }

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
               tmpEventVector.find(currentInfo.GetTreeType())->second.at(cl).push_back(new Event(vars, tgts , vis, cl , weight));

            }
         }
         
         currentInfo.GetTree()->ResetBranchAddresses();
      }

//       // compute renormalisation factors
//       renormFactor.at(cl) = nTempEvents.at(cl)/sumOfWeights.at(cl); --> will be done in dedicated member function
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
TMVA::DataSet*  TMVA::DataSetFactory::MixEvents( DataSetInfo& dsi, 
                                                 TMVA::EventVectorOfClassesOfTreeType& tmpEventVector, 
                                                 TMVA::NumberPerClassOfTreeType& nTrainTestEvents,
                                                 const TString& splitMode,
                                                 const TString& mixMode, 
                                                 const TString& normMode, 
                                                 UInt_t splitSeed)
{
   // Select and distribute unassigned events to kTraining and kTesting
   Bool_t emptyUndefined  = kTRUE;

//    // check if the vectors of all classes are empty
   for( Int_t cls = 0, clsEnd = dsi.GetNClasses(); cls < clsEnd; ++cls ){
      emptyUndefined &= tmpEventVector[Types::kMaxTreeType].at(cls).empty();
   }

   TMVA::RandomGenerator rndm( splitSeed );
   
   // ==== splitting of undefined events to kTraining and kTesting

   // if splitMode contains "RANDOM", then shuffle the undefined events
   if (splitMode.Contains( "RANDOM" ) && !emptyUndefined ) {
      Log() << kDEBUG << "randomly shuffling events which are not yet associated to testing or training"<<Endl;
      // random shuffle the undefined events of each class
      for( UInt_t cls = 0; cls < dsi.GetNClasses(); ++cls ){
         std::random_shuffle(tmpEventVector[Types::kMaxTreeType].at(cls).begin(), 
                             tmpEventVector[Types::kMaxTreeType].at(cls).end(),
                             rndm );
      }
   }

   // check for each class the number of training and testing events, the requested number and the available number
   Log() << kDEBUG << "SPLITTING ========" << Endl;
   for( UInt_t cls = 0; cls < dsi.GetNClasses(); ++cls ){
      Log() << kDEBUG << "---- class " << cls << Endl;
      Log() << kDEBUG << "check number of training/testing events, requested and available number of events and for class " << cls << Endl;

      // check if enough or too many events are already in the training/testing eventvectors of the class cls
      EventVector& eventVectorTraining = tmpEventVector.find( Types::kTraining    )->second.at(cls);
      EventVector& eventVectorTesting  = tmpEventVector.find( Types::kTesting     )->second.at(cls);
      EventVector& eventVectorUndefined= tmpEventVector.find( Types::kMaxTreeType )->second.at(cls);
      
      Int_t alreadyAvailableTraining   = eventVectorTraining.size();
      Int_t alreadyAvailableTesting    = eventVectorTesting.size();
      Int_t availableUndefined         = eventVectorUndefined.size();

      Int_t requestedTraining          = nTrainTestEvents.find( Types::kTraining )->second.at(cls);
      Int_t requestedTesting           = nTrainTestEvents.find( Types::kTesting  )->second.at(cls);
      
      Log() << kDEBUG << "availableTraining  " << alreadyAvailableTraining << Endl;
      Log() << kDEBUG << "availableTesting   " << alreadyAvailableTesting << Endl;
      Log() << kDEBUG << "availableUndefined " << availableUndefined << Endl;
      Log() << kDEBUG << "requestedTraining  " << requestedTraining << Endl;
      Log() << kDEBUG << "requestedTesting  " << requestedTesting << Endl;
      //
      // nomenclature r=available training
      //              s=available testing 
      //              u=available undefined
      //              R= requested training
      //              S= requested testing
      //              nR = used for selection of training events
      //              nS = used for selection of test events
      //              we have: nR + nS = r+s+u
      //              free events: Nfree = u-Thet(R-r)-Thet(S-s)
      //              nomenclature: Thet(x) = x,  if x>0 else 0;
      //              nR = max(R,r) + 0.5 * Nfree
      //              nS = max(S,s) + 0.5 * Nfree
      //              nR +nS = R+S + u-R+r-S+s = u+r+s= ok! for R>r
      //              nR +nS = r+S + u-S+s = u+r+s= ok! for r>R

      //EVT three different cases might occur here
      //
      // Case a
      // requestedTraining and requestedTesting >0 
      // free events: Nfree = u-Thet(R-r)-Thet(S-s)
      //              nR = Max(R,r) + 0.5 * Nfree
      //              nS = Max(S,s) + 0.5 * Nfree
      // 
      // Case b
      // exactly one of requestedTraining or requestedTesting >0
      // assume training R >0
      //    nR  = max(R,r) 
      //    nS  = s+u+r-nR
      //    and  s=nS
      //
      //Case c: 
      // requestedTraining=0, requestedTesting=0 
      // Nfree = u-|r-s|
      // if NFree >=0
      //    R = Max(r,s) + 0.5 * Nfree = S
      // else if r>s 
      //    R = r; S=s+u
      // else
      //    R = r+u; S=s
      //
      // Next steps:
      // Determination of Event numbers R,S, nR, nS
      // distribute undefined events according to nR, nS
      // finally determine actual sub samples from nR and nS to be used in training / testing
      //
      // implementation of case C)
      int useForTesting,useForTraining;
      if( (requestedTraining == 0) && (requestedTesting == 0)){ 
         // 0 means automatic distribution of events
         Log() << kDEBUG << "requested 0" << Endl;         
         // try to get the same number of events in training and testing for this class (balance)
         Int_t NFree = availableUndefined - TMath::Abs(alreadyAvailableTraining - alreadyAvailableTesting);
         if (NFree >=0){
            requestedTraining = TMath::Max(alreadyAvailableTraining,alreadyAvailableTesting) + NFree/2;
            requestedTesting  = availableUndefined+alreadyAvailableTraining+alreadyAvailableTesting - requestedTraining; // the rest
         } else if (alreadyAvailableTraining > alreadyAvailableTesting){ //r>s
            requestedTraining = alreadyAvailableTraining;
            requestedTesting  = alreadyAvailableTesting +availableUndefined;
         }
         else {
            requestedTraining = alreadyAvailableTraining+availableUndefined;
            requestedTesting  = alreadyAvailableTesting;            
         }
         useForTraining = requestedTraining; 
         useForTesting  = requestedTesting; 
      }
      else if ((requestedTesting == 0)){ // case B)
         useForTraining = TMath::Max(requestedTraining,alreadyAvailableTraining);
         useForTesting= availableUndefined+alreadyAvailableTraining+alreadyAvailableTesting - useForTraining; // the rest
         requestedTesting = useForTesting;
      }
      else if ((requestedTraining == 0)){ // case B)
         useForTesting = TMath::Max(requestedTesting,alreadyAvailableTesting);
         useForTraining= availableUndefined+alreadyAvailableTraining+alreadyAvailableTesting - useForTesting; // the rest
         requestedTraining = useForTraining;
      }
      else{ // case A
         int NFree = availableUndefined-TMath::Max(requestedTraining-alreadyAvailableTraining,0)-TMath::Max(requestedTesting-alreadyAvailableTesting,0);
         if (NFree <0) NFree = 0;
         useForTraining = TMath::Max(requestedTraining,alreadyAvailableTraining) + NFree/2;
         useForTesting= availableUndefined+alreadyAvailableTraining+alreadyAvailableTesting - useForTraining; // the rest
      }
      Log() << kDEBUG << "determined event sample size to select training sample from="<<useForTraining<<Endl;
      Log() << kDEBUG << "determined event sample size to select test sample from="<<useForTesting<<Endl;
      

      // associate undefined events 
      if( splitMode == "ALTERNATE" ){
         Log() << kDEBUG << "split 'ALTERNATE'" << Endl;
	 Int_t nTraining = alreadyAvailableTraining;
	 Int_t nTesting  = alreadyAvailableTesting;
         for( EventVector::iterator it = eventVectorUndefined.begin(), itEnd = eventVectorUndefined.end(); it != itEnd; ){
	    ++nTraining;
	    if( nTraining <= requestedTraining ){
	       eventVectorTraining.insert( eventVectorTraining.end(), (*it) );
	       ++it;
	    }
            if( it != itEnd ){
	       ++nTesting;
               eventVectorTesting.insert( eventVectorTesting.end(), (*it) );
               ++it;
            }
         }
      }else{
         Log() << kDEBUG << "split '" << splitMode << "'" << Endl;

	 // test if enough events are available
	 Log() << kDEBUG << "availableundefined : " << availableUndefined << Endl;
	 Log() << kDEBUG << "useForTraining     : " << useForTraining << Endl;
	 Log() << kDEBUG << "useForTesting      : " << useForTesting  << Endl;
	 Log() << kDEBUG << "alreadyAvailableTraining      : " << alreadyAvailableTraining  << Endl;
	 Log() << kDEBUG << "alreadyAvailableTesting       : " << alreadyAvailableTesting  << Endl;

	 if( availableUndefined<(useForTraining-alreadyAvailableTraining) ||
	     availableUndefined<(useForTesting -alreadyAvailableTesting ) || 
	     availableUndefined<(useForTraining+useForTesting-alreadyAvailableTraining-alreadyAvailableTesting ) ){
	    Log() << kFATAL << "More events requested than available!" << Endl;
	 }

	 // select the events
         if (useForTraining>alreadyAvailableTraining){
            eventVectorTraining.insert(  eventVectorTraining.end() , eventVectorUndefined.begin(), eventVectorUndefined.begin()+ useForTraining- alreadyAvailableTraining );
            eventVectorUndefined.erase( eventVectorUndefined.begin(), eventVectorUndefined.begin() + useForTraining- alreadyAvailableTraining);
         }
         if (useForTesting>alreadyAvailableTesting){
            eventVectorTesting.insert(  eventVectorTesting.end() , eventVectorUndefined.begin(), eventVectorUndefined.begin()+ useForTesting- alreadyAvailableTesting );
         }
      }
      eventVectorUndefined.clear();      
      // finally shorten the event vectors to the requested size by removing random events
      if (splitMode.Contains( "RANDOM" )){
         UInt_t sizeTraining  = eventVectorTraining.size();
         if( sizeTraining > UInt_t(requestedTraining) ){
           std::vector<UInt_t> indicesTraining( sizeTraining );
            // make indices
            std::generate( indicesTraining.begin(), indicesTraining.end(), TMVA::Increment<UInt_t>(0) );
            // shuffle indices
            std::random_shuffle( indicesTraining.begin(), indicesTraining.end(), rndm );
            // erase indices of not needed events
            indicesTraining.erase( indicesTraining.begin()+sizeTraining-UInt_t(requestedTraining), indicesTraining.end() );
            // delete all events with the given indices
            for( std::vector<UInt_t>::iterator it = indicesTraining.begin(), itEnd = indicesTraining.end(); it != itEnd; ++it ){
               delete eventVectorTraining.at( (*it) ); // delete event
               eventVectorTraining.at( (*it) ) = NULL; // set pointer to NULL
            }
            // now remove and erase all events with pointer==NULL
            eventVectorTraining.erase( std::remove( eventVectorTraining.begin(), eventVectorTraining.end(), (void*)NULL ), eventVectorTraining.end() );
         }

         UInt_t sizeTesting   = eventVectorTesting.size();
         if( sizeTesting > UInt_t(requestedTesting) ){
            std::vector<UInt_t> indicesTesting( sizeTesting );
            // make indices
            std::generate( indicesTesting.begin(), indicesTesting.end(), TMVA::Increment<UInt_t>(0) );
            // shuffle indices
            std::random_shuffle( indicesTesting.begin(), indicesTesting.end(), rndm );
            // erase indices of not needed events
            indicesTesting.erase( indicesTesting.begin()+sizeTesting-UInt_t(requestedTesting), indicesTesting.end() );
            // delete all events with the given indices
            for( std::vector<UInt_t>::iterator it = indicesTesting.begin(), itEnd = indicesTesting.end(); it != itEnd; ++it ){
               delete eventVectorTesting.at( (*it) ); // delete event
               eventVectorTesting.at( (*it) ) = NULL; // set pointer to NULL
            }
            // now remove and erase all events with pointer==NULL
            eventVectorTesting.erase( std::remove( eventVectorTesting.begin(), eventVectorTesting.end(), (void*)NULL ), eventVectorTesting.end() );
         }
      }
      else { // erase at end
	 if( eventVectorTraining.size() < UInt_t(requestedTraining) )
	    Log() << kWARNING << "DataSetFactory/requested number of training samples larger than size of eventVectorTraining.\n"
		  << "There is probably an issue. Please contact the TMVA developers." << Endl;
         std::for_each( eventVectorTraining.begin()+requestedTraining, eventVectorTraining.end(), DeleteFunctor<Event>() );
         eventVectorTraining.erase(eventVectorTraining.begin()+requestedTraining,eventVectorTraining.end());

	 if( eventVectorTesting.size() < UInt_t(requestedTesting) )
	    Log() << kWARNING << "DataSetFactory/requested number of testing samples larger than size of eventVectorTesting.\n"
		  << "There is probably an issue. Please contact the TMVA developers." << Endl;
         std::for_each( eventVectorTesting.begin()+requestedTesting, eventVectorTesting.end(), DeleteFunctor<Event>() );
         eventVectorTesting.erase(eventVectorTesting.begin()+requestedTesting,eventVectorTesting.end());
      }
   }

   TMVA::DataSetFactory::RenormEvents( dsi, tmpEventVector, normMode );

   Int_t trainingSize = 0;
   Int_t testingSize  = 0;

   // sum up number of training and testing events
   for( UInt_t cls = 0; cls < dsi.GetNClasses(); ++cls ){
      trainingSize += tmpEventVector[Types::kTraining].at(cls).size();
      testingSize  += tmpEventVector[Types::kTesting].at(cls).size();
   }

   // --- collect all training (testing) events into the training (testing) eventvector

   // create event vectors reserve enough space
   EventVector* trainingEventVector = new EventVector();
   EventVector* testingEventVector  = new EventVector();

   trainingEventVector->reserve( trainingSize );
   testingEventVector->reserve( testingSize );


   // collect the events

   // mixing of kTraining and kTesting data sets
   Log() << kDEBUG << " MIXING ============= " << Endl;

   if( mixMode == "ALTERNATE" ){
      // Inform user if he tries to use alternate mixmode for 
      // event classes with different number of events, this works but the alternation stops at the last event of the smaller class
      for( UInt_t cls = 1; cls < dsi.GetNClasses(); ++cls ){
         if (tmpEventVector[Types::kTraining].at(cls).size() != tmpEventVector[Types::kTraining].at(0).size()){
            Log() << kINFO << "Training sample: You are trying to mix events in alternate mode although the classes have different event numbers. This works but the alternation stops at the last event of the smaller class."<<Endl;
         }
         if (tmpEventVector[Types::kTesting].at(cls).size() != tmpEventVector[Types::kTesting].at(0).size()){
            Log() << kINFO << "Testing sample: You are trying to mix events in alternate mode although the classes have different event numbers. This works but the alternation stops at the last event of the smaller class."<<Endl;
         }
      }
      typedef EventVector::iterator EvtVecIt;
      EvtVecIt itEvent, itEventEnd;

      // insert first class
      Log() << kDEBUG << "insert class 0 into training and test vector" << Endl;
      trainingEventVector->insert( trainingEventVector->end(), tmpEventVector[Types::kTraining].at(0).begin(), tmpEventVector[Types::kTraining].at(0).end() );
      testingEventVector->insert( testingEventVector->end(),   tmpEventVector[Types::kTesting].at(0).begin(),  tmpEventVector[Types::kTesting].at(0).end() );
      
      // insert other classes
      EvtVecIt itTarget;
      for( UInt_t cls = 1; cls < dsi.GetNClasses(); ++cls ){
         Log() << kDEBUG << "insert class " << cls << Endl;
         // training vector
         itTarget = trainingEventVector->begin() - 1; // start one before begin
         // loop over source 
         for( itEvent = tmpEventVector[Types::kTraining].at(cls).begin(), itEventEnd = tmpEventVector[Types::kTraining].at(cls).end(); itEvent != itEventEnd; ++itEvent ){
//            if( std::distance( itTarget, trainingEventVector->end()) < Int_t(cls+1) ) {
            if( (trainingEventVector->end() - itTarget) < Int_t(cls+1) ) {
               itTarget = trainingEventVector->end();
               trainingEventVector->insert( itTarget, itEvent, itEventEnd ); // fill in the rest without mixing
               break;
            }else{ 
               itTarget += cls+1;
               trainingEventVector->insert( itTarget, (*itEvent) ); // fill event
            }
         }
         // testing vector
         itTarget = testingEventVector->begin() - 1;
         // loop over source 
         for( itEvent = tmpEventVector[Types::kTesting].at(cls).begin(), itEventEnd = tmpEventVector[Types::kTesting].at(cls).end(); itEvent != itEventEnd; ++itEvent ){
//             if( std::distance( itTarget, testingEventVector->end()) < Int_t(cls+1) ) {
            if( ( testingEventVector->end() - itTarget ) < Int_t(cls+1) ) {
               itTarget = testingEventVector->end();
               testingEventVector->insert( itTarget, itEvent, itEventEnd ); // fill in the rest without mixing
               break;
            }else{ 
               itTarget += cls+1;
               testingEventVector->insert( itTarget, (*itEvent) ); // fill event
            }
         }
      }

      // debugging output: classnumbers of all events in training and testing vectors
      //       std::cout << std::endl;
      //       std::cout << "TRAINING VECTOR" << std::endl;
      //       std::transform( trainingEventVector->begin(), trainingEventVector->end(), ostream_iterator<Int_t>(std::cout, "|"), std::mem_fun(&TMVA::Event::GetClass) );
      
      //       std::cout << std::endl;
      //       std::cout << "TESTING VECTOR" << std::endl;
      //       std::transform( testingEventVector->begin(), testingEventVector->end(), ostream_iterator<Int_t>(std::cout, "|"), std::mem_fun(&TMVA::Event::GetClass) );
      //       std::cout << std::endl;

   }else{ 
      for( UInt_t cls = 0; cls < dsi.GetNClasses(); ++cls ){
         trainingEventVector->insert( trainingEventVector->end(), tmpEventVector[Types::kTraining].at(cls).begin(), tmpEventVector[Types::kTraining].at(cls).end() );
         testingEventVector->insert ( testingEventVector->end(),  tmpEventVector[Types::kTesting].at(cls).begin(),  tmpEventVector[Types::kTesting].at(cls).end()  );
      }
   }

   //    std::cout << "trainingEventVector " << trainingEventVector->size() << std::endl;
   //    std::cout << "testingEventVector  " << testingEventVector->size() << std::endl;

   //    std::transform( trainingEventVector->begin(), trainingEventVector->end(), ostream_iterator<Int_t>(std::cout, "> \n"), std::mem_fun(&TMVA::Event::GetNVariables) );
   //    std::transform( testingEventVector->begin(), testingEventVector->end(), ostream_iterator<Int_t>(std::cout, "> \n"), std::mem_fun(&TMVA::Event::GetNVariables) );

   // delete the tmpEventVector (but not the events therein)
   tmpEventVector[Types::kTraining].clear();
   tmpEventVector[Types::kTesting].clear();

   tmpEventVector[Types::kMaxTreeType].clear();

   if (mixMode == "RANDOM") {
      Log() << kDEBUG << "shuffling events"<<Endl;

      //       std::cout << "before" << std::endl;
      //       std::for_each( trainingEventVector->begin(), trainingEventVector->begin()+10, std::bind2nd(std::mem_fun(&TMVA::Event::Print),std::cout) );
      
      std::random_shuffle( trainingEventVector->begin(), trainingEventVector->end(), rndm );
      std::random_shuffle( testingEventVector->begin(),  testingEventVector->end(),  rndm  );

      //       std::cout << "after" << std::endl;
      //       std::for_each( trainingEventVector->begin(), trainingEventVector->begin()+10, std::bind2nd(std::mem_fun(&TMVA::Event::Print),std::cout) );
   }

   Log() << kDEBUG << "trainingEventVector " << trainingEventVector->size() << Endl;
   Log() << kDEBUG << "testingEventVector  " << testingEventVector->size() << Endl;

   // create dataset
   DataSet* ds = new DataSet(dsi);

   Log() << kINFO << "Create internal training tree" << Endl;        
   ds->SetEventCollection(trainingEventVector, Types::kTraining ); 
   Log() << kINFO << "Create internal testing tree" << Endl;        
   ds->SetEventCollection(testingEventVector,  Types::kTesting  ); 


   return ds;
   
}



//_______________________________________________________________________
void  TMVA::DataSetFactory::RenormEvents( TMVA::DataSetInfo& dsi, 
                                          TMVA::EventVectorOfClassesOfTreeType& tmpEventVector, 
                                          const TString&        normMode )
{
   // ============================================================
   // renormalisation
   // ============================================================



   // print rescaling info
   if (normMode == "NONE") {
      Log() << kINFO << "No weight renormalisation applied: use original event weights" << Endl;
      return;
   }

   // ---------------------------------
   // compute sizes and sums of weights
   Int_t trainingSize = 0;
   Int_t testingSize  = 0;

   ValuePerClass trainingSumWeightsPerClass( dsi.GetNClasses() );
   ValuePerClass testingSumWeightsPerClass( dsi.GetNClasses() );

   NumberPerClass trainingSizePerClass( dsi.GetNClasses() );
   NumberPerClass testingSizePerClass( dsi.GetNClasses() );

   Double_t trainingSumWeights = 0;
   Double_t testingSumWeights  = 0;

   for( UInt_t cls = 0, clsEnd = dsi.GetNClasses(); cls < clsEnd; ++cls ){
      trainingSizePerClass.at(cls) = tmpEventVector[Types::kTraining].at(cls).size();
      testingSizePerClass.at(cls)  = tmpEventVector[Types::kTesting].at(cls).size();

      trainingSize += trainingSizePerClass.back();
      testingSize  += testingSizePerClass.back();

      // the functional solution
      // sum up the weights in Double_t although the individual weights are Float_t to prevent rounding issues in addition of floating points
      //
      // accumulate --> does what the name says
      //     begin() and end() denote the range of the vector to be accumulated
      //     Double_t(0) tells accumulate the type and the starting value
      //     compose_binary creates a BinaryFunction of ...
      //         std::plus<Double_t>() knows how to sum up two doubles
      //         null<Double_t>() leaves the first argument (the running sum) unchanged and returns it
      //         std::mem_fun knows how to call a member function (type and member-function given as argument) and return the result
      //
      // all together sums up all the event-weights of the events in the vector and returns it
      trainingSumWeightsPerClass.at(cls) = std::accumulate( tmpEventVector[Types::kTraining].at(cls).begin(),
                                                            tmpEventVector[Types::kTraining].at(cls).end(),
                                                            Double_t(0),
                                                            compose_binary( std::plus<Double_t>(),
                                                                            null<Double_t>(),
                                                                            std::mem_fun(&TMVA::Event::GetOriginalWeight) ) );

      testingSumWeightsPerClass.at(cls)  = std::accumulate( tmpEventVector[Types::kTesting].at(cls).begin(),
                                                            tmpEventVector[Types::kTesting].at(cls).end(),
                                                            Double_t(0),
                                                            compose_binary( std::plus<Double_t>(),
                                                                            null<Double_t>(),
                                                                            std::mem_fun(&TMVA::Event::GetOriginalWeight) ) );


      trainingSumWeights += trainingSumWeightsPerClass.at(cls);
      testingSumWeights  += testingSumWeightsPerClass.at(cls);
   }

   // ---------------------------------
   // compute renormalization factors

   ValuePerClass renormFactor( dsi.GetNClasses() );

   if (normMode == "NUMEVENTS") {
      Log() << kINFO << "Weight renormalisation mode: \"NumEvents\": renormalise independently the ..." << Endl;
      Log() << kINFO << "... class weights so that Sum[i=1..N_j]{w_i} = N_j, j=0,1,2..." << Endl;
      Log() << kINFO << "... (note that N_j is the sum of training and test events)" << Endl;

      for( UInt_t cls = 0, clsEnd = dsi.GetNClasses(); cls < clsEnd; ++cls ){
         renormFactor.at(cls) = ( (trainingSizePerClass.at(cls) + testingSizePerClass.at(cls))/
                                  (trainingSumWeightsPerClass.at(cls) + testingSumWeightsPerClass.at(cls)) );
      }
   }
   else if (normMode == "EQUALNUMEVENTS") {
      Log() << kINFO << "Weight renormalisation mode: \"EqualNumEvents\": renormalise class weights ..." << Endl;
      Log() << kINFO << "... so that Sum[i=1..N_j]{w_i} = N_classA, j=classA, classB, ..." << Endl;
      Log() << kINFO << "... (note that N_j is the sum of training and test events)" << Endl;

      for (UInt_t cls = 0, clsEnd = dsi.GetNClasses(); cls < clsEnd; ++cls ) {
         renormFactor.at(cls) = Float_t(trainingSizePerClass.at(cls)+testingSizePerClass.at(cls))/
            (trainingSumWeightsPerClass.at(cls)+testingSumWeightsPerClass.at(cls));
      }
      // normalize to size of first class
      UInt_t referenceClass = 0;
      for (UInt_t cls = 0, clsEnd = dsi.GetNClasses(); cls < clsEnd; ++cls ) {
         if( cls == referenceClass ) continue;
         renormFactor.at(cls) *= Float_t(trainingSizePerClass.at(referenceClass)+testingSizePerClass.at(referenceClass) )/
            Float_t( trainingSizePerClass.at(cls)+testingSizePerClass.at(cls) );
      }
   }
   else {
      Log() << kFATAL << "<PrepareForTrainingAndTesting> Unknown NormMode: " << normMode << Endl;
   }

   // ---------------------------------
   // now apply the normalization factors
   Int_t maxL = dsi.GetClassNameMaxLength();
   for (UInt_t cls = 0, clsEnd = dsi.GetNClasses(); cls<clsEnd; ++cls) { 
      Log() << kINFO << "--> Rescale " << setiosflags(ios::left) << std::setw(maxL) 
            << dsi.GetClassInfo(cls)->GetName() << " event weights by factor: " << renormFactor.at(cls) << Endl;
      std::for_each( tmpEventVector[Types::kTraining].at(cls).begin(), 
                     tmpEventVector[Types::kTraining].at(cls).end(),
                     std::bind2nd(std::mem_fun(&TMVA::Event::ScaleWeight),renormFactor.at(cls)) );
      std::for_each( tmpEventVector[Types::kTesting].at(cls).begin(), 
                     tmpEventVector[Types::kTesting].at(cls).end(),
                     std::bind2nd(std::mem_fun(&TMVA::Event::ScaleWeight),renormFactor.at(cls)) );
   }



      
   // ---------------------------------
   // for information purposes
   dsi.SetNormalization( normMode );

   // ============================
   // print out the result
   // (same code as before --> this can be done nicer )
   //

   Log() << kINFO << "Number of training and testing events after rescaling:" << Endl;
   Log() << kINFO << "------------------------------------------------------" << Endl;
   trainingSumWeights = 0;
   testingSumWeights  = 0;
   for( UInt_t cls = 0, clsEnd = dsi.GetNClasses(); cls < clsEnd; ++cls ){

      trainingSumWeightsPerClass.at(cls) = (std::accumulate( tmpEventVector[Types::kTraining].at(cls).begin(),  // accumulate --> start at begin
                                                             tmpEventVector[Types::kTraining].at(cls).end(),    //    until end()
                                                             Double_t(0),                                       // values are of type double
                                                             compose_binary( std::plus<Double_t>(),             // define addition for doubles
                                                                             null<Double_t>(),                  // take the argument, don't do anything and return it
                                                                             std::mem_fun(&TMVA::Event::GetOriginalWeight) ) )); // take the value from GetOriginalWeight

      testingSumWeightsPerClass.at(cls)  = std::accumulate( tmpEventVector[Types::kTesting].at(cls).begin(),
                                                            tmpEventVector[Types::kTesting].at(cls).end(),
                                                            Double_t(0),
                                                            compose_binary( std::plus<Double_t>(),
                                                                            null<Double_t>(),
                                                                            std::mem_fun(&TMVA::Event::GetOriginalWeight) ) );


      trainingSumWeights += trainingSumWeightsPerClass.at(cls);
      testingSumWeights  += testingSumWeightsPerClass.at(cls);

      // output statistics
      Log() << kINFO << setiosflags(ios::left) << std::setw(maxL) 
            << dsi.GetClassInfo(cls)->GetName() << " -- " 
            << "training entries            : " << trainingSizePerClass.at(cls) 
            <<  " (" << "sum of weights: " << trainingSumWeightsPerClass.at(cls) << ")" << Endl;
      Log() << kINFO << setiosflags(ios::left) << std::setw(maxL) 
            << dsi.GetClassInfo(cls)->GetName() << " -- " 
            << "testing entries             : " << testingSizePerClass.at(cls) 
            <<  " (" << "sum of weights: " << testingSumWeightsPerClass.at(cls) << ")" << Endl;
      Log() << kINFO << setiosflags(ios::left) << std::setw(maxL) 
            << dsi.GetClassInfo(cls)->GetName() << " -- " 
            << "training and testing entries: " 
            << (trainingSizePerClass.at(cls)+testingSizePerClass.at(cls)) 
            << " (" << "sum of weights: " 
            << (trainingSumWeightsPerClass.at(cls)+testingSumWeightsPerClass.at(cls)) << ")" << Endl;
   }

}



