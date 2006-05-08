// @(#)root/tmva $Id: TMVA_DecisionTree.h,v 1.4 2006/04/29 23:55:41 andreas.hoecker Exp $
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_DecisionTree                                                     *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation of a Decision Tree                                         *
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
 * $Id: TMVA_DecisionTree.h,v 1.4 2006/04/29 23:55:41 andreas.hoecker Exp $    
 **********************************************************************************/

#ifndef ROOT_TMVA_DecisionTree
#define ROOT_TMVA_DecisionTree

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_DecisionTree                                                    //
//                                                                      //
// Implementation of a Decision Tree                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_DecisionTreeNode
#include "TMVA_DecisionTreeNode.h"
#endif
#ifndef ROOT_TMVA_BinarySearchTree
#include "TMVA_BinarySearchTree.h"
#endif
#ifndef ROOT_TMVA_SeparationBase
#include "TMVA_SeparationBase.h"
#endif

using std::vector;

class TMVA_DecisionTree : public TMVA_BinaryTree {
  
 public:

  // the constructur needed for the "reading" of the decision tree from weight files
  TMVA_DecisionTree( void );

  // the constructur needed for constructing the decision tree via training with events
  TMVA_DecisionTree( TMVA_SeparationBase *sepType,Int_t minSize, Double_t mnsep, 
		     Int_t nCuts);
  virtual ~TMVA_DecisionTree( void );
  
  void BuildTree( vector<TMVA_Event*> & eventSample, 
		  TMVA_DecisionTreeNode *node = NULL );

  Double_t TrainNode( vector<TMVA_Event*> & eventSample,  TMVA_DecisionTreeNode *node );

  // returns: 1 = Signal (right),  -1 = Bkg (left)
  Int_t CheckEvent( TMVA_Event* ); 

 private:

  // calculate the Purity out of the number of sig and bkg events collected
  // from individual samples.

  Double_t SamplePurity(vector<TMVA_Event*> eventSample);
  
  Int_t     fNvars;
  Int_t     fNCuts; // ! fNCuts * fNCuts  different cuts are scanned.
  TMVA_SeparationBase *fSepType;

  Double_t  fMinSize;
  Double_t  fMinSepGain;

  Bool_t    fUseSearchTree;
  
  ClassDef(TMVA_DecisionTree,0) //Implementation of a Decision Tree
};

#endif 
