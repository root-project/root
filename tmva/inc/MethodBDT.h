// @(#)root/tmva $Id: MethodBDT.h,v 1.9 2006/05/22 09:06:25 helgevoss Exp $ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodBDT  (Boosted Decision Trees)                                   *
 *                                                                                *
 * Description:                                                                   *
 *      Analysis of Boosted Decision Trees                                        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodBDT
#define ROOT_TMVA_MethodBDT

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodBDT                                                            //
//                                                                      //
// Analysis of Boosted Decision Trees                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include "TH2.h"
#include "TTree.h"
#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_BinarySearchTree
#include "TMVA/BinarySearchTree.h"
#endif
#ifndef ROOT_TMVA_DecisionTree
#include "TMVA/DecisionTree.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
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

namespace TMVA {

  class MethodBDT : public MethodBase {

  public:
    // MethodBDT (Boosted Decision Trees) options:
    // format and syntax of option string: "nTrees:BoostType:SeparationType:
    //                                      nEventsMin:dummy:
    //                                      nCuts:SignalFraction"
    // nTrees:          number of trees in the forest to be created
    // BoostType:       the boosting type for the trees in the forest (AdaBoost e.t.c..)
    // SeparationType   the separation criterion applied in the node splitting
    // nEventsMin:      the minimum number of events in a node (leaf criteria, stop splitting)
    // dummy:           a dummy variable, just to keep backward compatible
    // nCuts:  the number of steps in the optimisation of the cut for a node
    // SignalFraction:  scale parameter of the number of Bkg events  
    //                  applied to the training sample to simulate different initial purity
    //                  of your data sample. 
    //
    // known SeparationTypes are:
    //    - MisClassificationError
    //    - GiniIndex
    //    - CrossEntropy
    // known BoostTypes are:
    //    - AdaBoost
    //    - Bagging

    // constructor for training and reading
    MethodBDT( TString jobName, 
	       vector<TString>* theVariables, 
	       TTree* theTree , 
	       TString theOption = "100:AdaBoost:GiniIndex:10:0:20:-1",
	       TDirectory* theTargetDir = 0 );

    // constructor for calculating BDT-MVA using previously generatad decision trees
    MethodBDT( vector<TString> *theVariables, 
	       TString theWeightFile,  
	       TDirectory* theTargetDir = NULL );
  
    virtual ~MethodBDT( void );
    
    // write all Events from the Tree into a vector of Events, that are 
    // more easily manipulated 
    virtual void InitEventSample();

    // training method
    virtual void Train( void );

    // write weights to file
    virtual void WriteWeightsToFile( void );
  
    // read weights from file
    virtual void ReadWeightsFromFile( void );

    // write method specific histos to target file
    virtual void WriteHistosToFile( void ) ;

    // calculate the MVA value
    virtual Double_t GetMvaValue( Event *e );

    // apply the boost algorithm to a tree in the collection 
    virtual Double_t Boost( std::vector<Event*>, DecisionTree *dt, Int_t iTree );

  protected:

  private:

    // boosting algorithm (adaptive boosting)
    Double_t AdaBoost(std::vector<Event*>, DecisionTree *dt );
    Double_t                        fAdaBoostBeta; // parameter in AdaBoost
 
    //--> not used: Double_t EpsilonBoost(std::vector<Event*>, DecisionTree *dt );

    // boosting as a random re-weighting
    Double_t Bagging(std::vector<Event*>, Int_t iTree);
  
    std::vector<Event*>             fEventSample; // the training events
 
    Int_t                           fNTrees;      // number of decision trees requested
    std::vector<DecisionTree*>      fForest;      // the collection of decision trees
    std::vector<double>             fBoostWeights;// the weights applied in the individual boosts
    TString                         fBoostType;   // string specifying the boost type

    //options for the decision Tree
    SeparationBase                 *fSepType;       // the separation used in node splitting
    Int_t                           fNodeMinEvents; // min number of events in node 
    Double_t                        fDummyOpt;      // dummy option (for backward compatibility)
  
    Int_t                           fNCuts;          // grid used in cut applied in node splitting
    Double_t                        fSignalFraction; // scalefactor for bkg events to modify initial s/b fraction in training data

    // Init used in the various constructors
    void InitBDT( void );

    //some histograms for monitoring
    TH1F*                           fBoostWeightHist;//weights applied in boosting
    TH2F*                           fErrFractHist;   //error fraction vs tree number
    TTree*                          fMonitorNtuple;  //monitoring ntuple
    Int_t                           fITree      ;    //ntuple var: ith tree
    Double_t                        fBoostWeight;    //ntuple var: boost weight
    Double_t                        fErrorFraction;  //ntuple var: misclassification error fraction 
    Int_t                           fNnodes;         //ntuple var: nNodes

    ClassDef(MethodBDT,0)  // Analysis of Boosted Decision Trees 
  };

} // namespace TMVA

#endif
