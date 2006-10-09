// @(#)root/tmva $Id: MethodRuleFit.h,v 1.21 2006/10/03 17:49:10 tegen Exp $
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
 *      MPI-KP Heidelberg, Germany                                                * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 * $Id: MethodRuleFit.h,v 1.21 2006/10/03 17:49:10 tegen Exp $    
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

      // write weights to file
      virtual void WriteWeightsToStream( ostream& o ) const;

      // read weights from file
      virtual void ReadWeightsFromStream( istream& istr );

      // calculate the MVA value
      //      virtual Double_t GetMvaValue(Event *e);
      virtual Double_t GetMvaValue();

      // write method specific histos to target file
      virtual void WriteHistosToFile( void ) const;

      // ranking of input variables
      const Ranking* CreateRanking();

      // get training event in a std::vector
      const std::vector<TMVA::Event*>         &GetTrainingEvents() const   { return fEventSample; }
      const std::vector<TMVA::DecisionTree*>  &GetForest() const           { return fForest; }
      const Int_t                              GetNTrees() const           { return fNTrees; }
      const Double_t                           GetSampleFraction() const   { return fSampleFraction; }
      const SeparationBase                    *GetSeparationBase() const   { return fSepType; }
      const Int_t                              GetNCuts() const            { return fNCuts; }

   protected:
      // initialize rulefit
      void InitRuleFit( void );

      // copy all training events into a stl::vector
      void InitEventSample( void );

      //MethodBDT                   *fBDT;

      // initialise monitor ntuple
      void InitMonitorNtuple();

      // build a decision tree
      void BuildTree( DecisionTree *dt, std::vector< Event *> & el );

      // make a forest of decision trees
      void MakeForest();

      //
      std::vector< Event *>        fEventSample;   // the complete training sample
      std::vector<DecisionTree *>  fForest;        // the forest
      Int_t                        fNTrees;        // number of trees in forest
      Double_t                     fSampleFraction;// fraction of events used for traing each tree
      SeparationBase              *fSepType;       // the separation used in node splitting
      Int_t                        fNodeMinEvents; // min number of events in node
      Int_t                        fNCuts;         // grid used in cut applied in node splitting
      RuleFit                      fRuleFit;       // RuleFit instance

   private:

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();

      Double_t                     fSignalFraction; // scalefactor for bkg events to modify initial s/b fraction in training data

      // ntuple
      TTree                       *fMonitorNtuple;
      Double_t                     fNTImportance;
      Double_t                     fNTCoefficient;
      Double_t                     fNTSupport;
      Int_t                        fNTNcuts;
      Double_t                     fNTPtag;
      Double_t                     fNTPss;
      Double_t                     fNTPsb;
      Double_t                     fNTPbs;
      Double_t                     fNTPbb;
      Double_t                     fNTSSB;
      Int_t                        fNTType;


      // options
      Double_t                     fGDTau;
      Double_t                     fGDPathStep;
      Int_t                        fGDNPathSteps;
      Double_t                     fGDErrNsigma;
      Double_t                     fMinimp;
      TString                      fSepTypeS;
      TString                      fModelTypeS;
      Double_t                     fRuleMaxDist;   // rule max distance - see RuleEnsemble

      ClassDef(MethodRuleFit,0)  // Friedman's RuleFit method
	 };

} // namespace TMVA

#endif // MethodRuleFit_H
