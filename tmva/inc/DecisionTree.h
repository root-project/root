// @(#)root/tmva $Id: DecisionTree.h,v 1.2 2006/05/23 13:03:15 brun Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DecisionTree                                                          *
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
 **********************************************************************************/

#ifndef ROOT_TMVA_DecisionTree
#define ROOT_TMVA_DecisionTree

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// DecisionTree                                                         //
//                                                                      //
// Implementation of a Decision Tree                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_DecisionTreeNode
#include "TMVA/DecisionTreeNode.h"
#endif
#ifndef ROOT_TMVA_BinarySearchTree
#include "TMVA/BinarySearchTree.h"
#endif
#ifndef ROOT_TMVA_SeparationBase
#include "TMVA/SeparationBase.h"
#endif

using std::vector;

namespace TMVA {

   class DecisionTree : public BinaryTree {
  
   public:

      // the constructur needed for the "reading" of the decision tree from weight files
      DecisionTree( void );

      // the constructur needed for constructing the decision tree via training with events
      DecisionTree( SeparationBase *sepType,Int_t minSize, Int_t nCuts);
      virtual ~DecisionTree( void );
  
      // building of a tree by recursivly splitting the nodes 
      Int_t BuildTree( vector<TMVA::Event*> & eventSample, 
                       DecisionTreeNode *node = NULL );

      // determine the way how a node is split (which variable, which cut value)
      Double_t TrainNode( vector<TMVA::Event*> & eventSample,  DecisionTreeNode *node );

      // returns: 1 = Signal (right),  -1 = Bkg (left)
      Double_t CheckEvent( Event* ); 

   private:

      // calculate the Purity out of the number of sig and bkg events collected
      // from individual samples.

      //calculates the purity S/(S+B) of a given event sample
      Double_t SamplePurity(vector<Event*> eventSample);
  
      Int_t     fNvars; // number of variables used to separate S and B
      Int_t     fNCuts; // number of grid point in variable cut scans
      SeparationBase *fSepType; // the separation crition

      Double_t  fMinSize;  // min number of events in node
      Double_t  fMinSepGain;// min number of separation gain to perform node splitting

      Bool_t    fUseSearchTree; //cut scan done with binary trees or simple event loop.
  
      ClassDef(DecisionTree,0) //Implementation of a Decision Tree
         };

} // namespace TMVA

#endif 
