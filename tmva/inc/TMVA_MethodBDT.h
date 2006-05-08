// @(#)root/tmva $Id: TMVA_MethodBDT.h,v 1.9 2006/05/02 23:27:40 helgevoss Exp $ 
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodBDT  (Boosted Decision Trees)                              *
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
 * File and Version Information:                                                  *
 * $Id: TMVA_MethodBDT.h,v 1.9 2006/05/02 23:27:40 helgevoss Exp $ 
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodBDT
#define ROOT_TMVA_MethodBDT

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_MethodBDT                                                       //
//                                                                      //
// Analysis of Boosted Decision Trees                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>

#ifndef ROOT_TMVA_MethodBase
#include "TMVA_MethodBase.h"
#endif
#ifndef ROOT_TMVA_BinarySearchTree
#include "TMVA_BinarySearchTree.h"
#endif
#ifndef ROOT_TMVA_DecisionTree
#include "TMVA_DecisionTree.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA_Event.h"
#endif
#ifndef ROOT_TMVA_SeparationBase
#include "TMVA_SeparationBase.h"
#endif
#ifndef ROOT_TMVA_GiniIndex
#include "TMVA_GiniIndex.h"
#endif
#ifndef ROOT_TMVA_CrossEntropy
#include "TMVA_CrossEntropy.h"
#endif
#ifndef ROOT_TMVA_MisClassificationError
#include "TMVA_MisClassificationError.h"
#endif
#ifndef ROOT_TMVA_SdivSqrtSplusB
#include "TMVA_SdivSqrtSplusB.h"
#endif

class TMVA_MethodBDT : public TMVA_MethodBase {

 public:

  // the option String defines:  nTrees:SeparationType:BoostType:
  //                              nEventsMin:minNodePurity:maxNodePurity:
  //                              nCuts:IntervalCut?:
  // known GetSeparationTypes are
  //    MisClassificationError,
  //    GiniIndex, 
  //    CrossEntropy;
  // known BoostTypes are
  //    AdaBoost
  //    EpsilonBoost
  //    Bagging       (call it whatever you want, it's just random weights)
  // nEventsMin: the minimum Number of events in a node (leaf criteria)
  // minSeparationGain: 
  // nCuts:  the number of steps in the optimisation of the cut for a node
  // signalFraction:  scale the #bkg such that the #sig/#bkg = signalFraction 
  TMVA_MethodBDT( TString jobName, 
		  vector<TString>* theVariables, 
		  TTree* theTree , 
		  TString theOption = "100:AdaBoost:GiniIndex:10:0.002:20:-1",
		  TDirectory* theTargetDir = 0 );

  TMVA_MethodBDT( vector<TString> *theVariables, 
		  TString theWeightFile,  
		  TDirectory* theTargetDir = NULL );
  
  virtual ~TMVA_MethodBDT( void );
    
  // write all Events from the Tree into a vector of TMVA_Events, that are 
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
  virtual Double_t GetMvaValue( TMVA_Event *e );

  // boost 
  void Boost( std::vector<TMVA_Event*>, TMVA_DecisionTree *dt, Int_t iTree );

 protected:

 private:

  void AdaBoost(std::vector<TMVA_Event*>, TMVA_DecisionTree *dt );
  Double_t                        fAdaBoostBeta;
  void EpsilonBoost(std::vector<TMVA_Event*>, TMVA_DecisionTree *dt );
  void Bagging(std::vector<TMVA_Event*>, Int_t iTree);
  
  std::vector<TMVA_Event*>        fEventSample;

  Int_t                           fNTrees;
  std::vector<TMVA_DecisionTree*> fForest;
  TString                         fBoostType;

  //options for the decision Tree
  TMVA_SeparationBase            *fSepType;
  Int_t                           fNodeMinEvents;
  Double_t                        fNodeMinSepGain;
  
  Int_t                           fNCuts;
  Double_t                        fSignalFraction; 

  void InitBDT( void );

  ClassDef(TMVA_MethodBDT,0)  //Analysis of Boosted Decision Trees 
};

#endif
