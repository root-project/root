// @(#)root/tmva $Id: MethodRuleFit.h,v 1.11 2006/11/23 17:43:38 rdm Exp $
// Author: Andreas Hoecker, Fredrik Tegenfeldt, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodRuleFit                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Friedman's RuleFit method                                                 * 
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker    <Andreas.Hocker@cern.ch>     - CERN, Switzerland       *
 *      Fredrik Tegenfeldt <Fredrik.Tegenfeldt@cern.ch> - Iowa State U., USA      *
 *      Helge Voss         <Helge.Voss@cern.ch>         - MPI-KP Heidelberg, Ger. *
 *      Kai Voss           <Kai.Voss@cern.ch>           - U. of Victoria, Canada  *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodRuleFit
#define ROOT_TMVA_MethodRuleFit

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodRuleFit                                                        //
//                                                                      //
// J Friedman's RuleFit method                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_TMatrixD
#include "TMatrixD.h"
#endif
#ifndef ROOT_TMVA_TVectorD
#include "TVectorD.h"
#endif
#ifndef ROOT_TMVA_DecisionTree
#include "TMVA/DecisionTree.h"
#endif
#ifndef ROOT_TMVA_SeparationBase
#include "TMVA/SeparationBase.h"
#endif
#ifndef ROOT_TMVA_GiniIndex
#include "TMVA/GiniIndex.h"
#endif
#ifndef ROOT_TMVA_CrossEntropy
#include "TMVA/CrossEntropy.h"
#endif
#ifndef ROOT_TMVA_MisClassificationError
#include "TMVA/MisClassificationError.h"
#endif
#ifndef ROOT_TMVA_SdivSqrtSplusB
#include "TMVA/SdivSqrtSplusB.h"
#endif
#ifndef ROOT_TMVA_RULEFIT_H
#include "TMVA/RuleFit.h"
#endif

namespace TMVA {

   class MethodRuleFit : public MethodBase {

   public:

      MethodRuleFit( TString jobName,
                     TString methodTitle, 
                     DataSet& theData,
                     TString theOption = "",
                     TDirectory* theTargetDir = 0 );

      MethodRuleFit( DataSet& theData,
                     TString theWeightFile,
                     TDirectory* theTargetDir = NULL );

      virtual ~MethodRuleFit( void );

      // training method
      virtual void Train( void );

      using MethodBase::WriteWeightsToStream;
      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      virtual void WriteWeightsToStream( ostream& o ) const;

      // read weights from file
      virtual void ReadWeightsFromStream( istream& istr );

      // calculate the MVA value
      //      virtual Double_t GetMvaValue(Event *e);
      virtual Double_t GetMvaValue();

      // write method specific histos to target file
      virtual void WriteMonitoringHistosToFile( void ) const;

      // ranking of input variables
      const Ranking* CreateRanking();

      // get training event in a std::vector
      const std::vector<TMVA::Event*>         &GetTrainingEvents() const    { return fEventSample; }
      const std::vector<TMVA::DecisionTree*>  &GetForest() const            { return fForest; }
      Int_t                                    GetNTrees() const            { return fNTrees; }
      Double_t                                 GetSampleFraction() const    { return fSampleFraction; }
      Double_t                                 GetSubSampleFraction() const { return fSubSampleFraction; }
      const SeparationBase                    *GetSeparationBase() const    { return fSepType; }
      Int_t                                    GetNCuts() const             { return fNCuts; }

      TDirectory*      GetMethodBaseDir() const { return BaseDir(); }

   protected:
      // initialize rulefit
      void InitRuleFit( void );

      // copy all training events into a stl::vector
      void InitEventSample( void );

      // initialise monitor ntuple
      void InitMonitorNtuple();

      // build a decision tree
      void BuildTree( DecisionTree *dt, std::vector< Event *> & el );

      // make a forest of decision trees
      void MakeForest();

      void MakeForestRnd();

      //
      std::vector< Event *>        fEventSample;   // the complete training sample
      std::vector<DecisionTree *>  fForest;        // the forest
      Int_t                        fNTrees;        // number of trees in forest
      Double_t                     fSampleFraction;// fraction of events used for traing each tree
      Double_t                     fSubSampleFraction; // fraction of subsamples used for the fitting
      SeparationBase              *fSepType;       // the separation used in node splitting
      Int_t                        fNodeMinEvents; // min number of events in node - NOT USED NOW!
      Double_t                     fMinFracNEve;   // min fraction of number events
      Double_t                     fMaxFracNEve;   // ditto max
      Int_t                        fNCuts;         // grid used in cut applied in node splitting
      RuleFit                      fRuleFit;       // RuleFit instance

   private:

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();

      Double_t                     fSignalFraction; // scalefactor for bkg events to modify initial s/b fraction in training data

      // ntuple
      TTree                       *fMonitorNtuple;  // pointer to monitor rule ntuple
      Double_t                     fNTImportance;   // ntuple: rule importance
      Double_t                     fNTCoefficient;  // ntuple: rule coefficient
      Double_t                     fNTSupport;      // ntuple: rule support
      Int_t                        fNTNcuts;        // ntuple: rule number of cuts
      Double_t                     fNTPtag;         // ntuple: rule P(tag)
      Double_t                     fNTPss;          // ntuple: rule P(tag s, true s)
      Double_t                     fNTPsb;          // ntuple: rule P(tag s, true b)
      Double_t                     fNTPbs;          // ntuple: rule P(tag b, true s)
      Double_t                     fNTPbb;          // ntuple: rule P(tag b, true b)
      Double_t                     fNTSSB;          // ntuple: rule S/(S+B)
      Int_t                        fNTType;         // ntuple: rule type (+1->signal, -1->bkg)


      // options
      Double_t                     fGDTau;          // gradient directed path: def threshhold fraction [0..1]
      Double_t                     fGDTauMin;       // gradient directed path: min threshhold fraction [0..1]
      Double_t                     fGDTauMax;       // gradient directed path: max threshhold fraction [0..1]
      UInt_t                       fGDNTau;         // gradient directed path: N(tau)
      UInt_t                       fGDTauScan;      // gradient directed path: number of points to scan
      Double_t                     fGDPathStep;     // gradient directed path: step size in path
      Int_t                        fGDNPathSteps;   // gradient directed path: number of steps
      Double_t                     fGDErrScale;     // gradient directed path: stop 
      Double_t                     fMinimp;         // rule/linear: minimum importance
      //
      TString                      fSepTypeS;        // forest generation: separation type - see DecisionTree
      TString                      fPruneMethodS;    // forest generation: prune method - see DecisionTree
      TMVA::DecisionTree::EPruneMethod fPruneMethod; // forest generation: method used for pruning - see DecisionTree 
      Double_t                     fPruneStrength;   // forest generation: prune strength - see DecisionTree
      //
      TString                      fModelTypeS;     // rule ensemble: which model (rule,linear or both)
      Double_t                     fRuleMinDist;    // rule min distance - see RuleEnsemble
      Double_t                     fLinQuantile;    // quantile cut to remove outliers - see RuleEnsemble

      ClassDef(MethodRuleFit,0)  // Friedman's RuleFit method
   };

} // namespace TMVA

#endif // MethodRuleFit_H
