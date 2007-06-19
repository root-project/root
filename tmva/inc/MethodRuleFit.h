// @(#)root/tmva $Id: MethodRuleFit.h,v 1.12 2007/04/19 06:53:01 brun Exp $
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
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
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

      Bool_t                                   UseBoost() const { return fUseBoost; }

      // accessors
      RuleFit                                 *GetRuleFitPtr() { return &fRuleFit; }
      const RuleFit                           *GetRuleFitConstPtr() const { return &fRuleFit; }
      TDirectory*                              GetMethodBaseDir() const     { return BaseDir(); }
      const std::vector<TMVA::Event*>         &GetTrainingEvents() const    { return fEventSample; }
      const std::vector<TMVA::DecisionTree*>  &GetForest() const            { return fForest; }
      Int_t                                    GetNTrees() const            { return fNTrees; }
      Double_t                                 GetTreeEveFrac() const       { return fTreeEveFrac; }
      //      Double_t                                 GetSubSampleFraction() const { return fSubSampleFraction; }
      const SeparationBase                    *GetSeparationBaseConst() const { return fSepType; }
      SeparationBase                          *GetSeparationBase() const { return fSepType; }
      TMVA::DecisionTree::EPruneMethod         GetPruneMethod() const       { return fPruneMethod; }
      Double_t                                 GetPruneStrength() const     { return fPruneStrength; }
      Double_t                                 GetMinFracNEve() const       { return fMinFracNEve; }
      Double_t                                 GetMaxFracNEve() const       { return fMaxFracNEve; }
      Int_t                                    GetNCuts() const             { return fNCuts; }
      //
      Int_t                                    GetGDNPathSteps() const      { return fGDNPathSteps; }
      Double_t                                 GetGDPathStep() const        { return fGDPathStep; }
      Double_t                                 GetGDErrScale() const        { return fGDErrScale; }
      Double_t                                 GetGDPathEveFrac() const     { return fGDPathEveFrac; }
      Double_t                                 GetGDValidEveFrac() const    { return fGDValidEveFrac; }
      //
      Double_t                                 GetLinQuantile() const       { return fLinQuantile; }

      const TString                            GetRFWorkDir() const         { return fRFWorkDir; }
      Int_t                                    GetRFNrules() const          { return fRFNrules; }
      Int_t                                    GetRFNendnodes() const       { return fRFNendnodes; }

   protected:

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      virtual void MakeClassSpecific( std::ostream&, const TString& ) const;

      virtual void MakeClassRuleCuts( std::ostream& ) const;

      virtual void MakeClassLinear( std::ostream& ) const;

      // get help message text
      void GetHelpMessage() const;

      // initialize rulefit
      void InitRuleFit( void );

      // copy all training events into a stl::vector
      void InitEventSample( void );

      // initialize monitor ntuple
      void InitMonitorNtuple();

      // build a decision tree
      //      void BuildTree( DecisionTree *dt, std::vector< Event *> & el );

      // make a forest of decision trees
      //      void MakeForest();

      //      void MakeForestRnd();

      void TrainTMVARuleFit();
      void TrainJFRuleFit();
      //

   private:

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();

      RuleFit                      fRuleFit;       // RuleFit instance
      std::vector< Event *>        fEventSample;   // the complete training sample
      Double_t                     fSignalFraction; // scalefactor for bkg events to modify initial s/b fraction in training data

      // ntuple
      TTree                       *fMonitorNtuple;  // pointer to monitor rule ntuple
      Double_t                     fNTImportance;   // ntuple: rule importance
      Double_t                     fNTCoefficient;  // ntuple: rule coefficient
      Double_t                     fNTSupport;      // ntuple: rule support
      Int_t                        fNTNcuts;        // ntuple: rule number of cuts
      Int_t                        fNTNvars;        // ntuple: rule number of vars
      Double_t                     fNTPtag;         // ntuple: rule P(tag)
      Double_t                     fNTPss;          // ntuple: rule P(tag s, true s)
      Double_t                     fNTPsb;          // ntuple: rule P(tag s, true b)
      Double_t                     fNTPbs;          // ntuple: rule P(tag b, true s)
      Double_t                     fNTPbb;          // ntuple: rule P(tag b, true b)
      Double_t                     fNTSSB;          // ntuple: rule S/(S+B)
      Int_t                        fNTType;         // ntuple: rule type (+1->signal, -1->bkg)

      // options
      TString                      fRuleFitModuleS;// which rulefit module to use
      Bool_t                       fUseRuleFitJF;  // if true interface with J.Friedmans RuleFit module
      TString                      fRFWorkDir;     // working directory from Friedmans module
      Int_t                        fRFNrules;      // max number of rules (only Friedmans module)
      Int_t                        fRFNendnodes;   // max number of rules (only Friedmans module)
      std::vector<DecisionTree *>  fForest;        // the forest
      Int_t                        fNTrees;        // number of trees in forest
      Double_t                     fTreeEveFrac;   // fraction of events used for traing each tree
      SeparationBase              *fSepType;       // the separation used in node splitting
      Double_t                     fMinFracNEve;   // min fraction of number events
      Double_t                     fMaxFracNEve;   // ditto max
      Int_t                        fNCuts;         // grid used in cut applied in node splitting
      TString                      fSepTypeS;        // forest generation: separation type - see DecisionTree
      TString                      fPruneMethodS;    // forest generation: prune method - see DecisionTree
      TMVA::DecisionTree::EPruneMethod fPruneMethod; // forest generation: method used for pruning - see DecisionTree 
      Double_t                     fPruneStrength;   // forest generation: prune strength - see DecisionTree
      TString                      fForestTypeS;     // forest generation: how the trees are generated
      Bool_t                       fUseBoost;        // use boosted events for forest generation
      //
      Double_t                     fGDPathEveFrac; //  GD path: fraction of subsamples used for the fitting
      Double_t                     fGDValidEveFrac; // GD path: fraction of subsamples used for the fitting
      Double_t                     fGDTau;          // GD path: def threshhold fraction [0..1]
      Double_t                     fGDTauPrec;      // GD path: precision of estimated tau
      Double_t                     fGDTauMin;       // GD path: min threshhold fraction [0..1]
      Double_t                     fGDTauMax;       // GD path: max threshhold fraction [0..1]
      UInt_t                       fGDTauScan;      // GD path: number of points to scan
      Double_t                     fGDPathStep;     // GD path: step size in path
      Int_t                        fGDNPathSteps;   // GD path: number of steps
      Double_t                     fGDErrScale;     // GD path: stop 
      Double_t                     fMinimp;         // rule/linear: minimum importance
      //
      TString                      fModelTypeS;     // rule ensemble: which model (rule,linear or both)
      Double_t                     fRuleMinDist;    // rule min distance - see RuleEnsemble
      Double_t                     fLinQuantile;    // quantile cut to remove outliers - see RuleEnsemble

      ClassDef(MethodRuleFit,0)  // Friedman's RuleFit method
   };

} // namespace TMVA

#endif // MethodRuleFit_H
