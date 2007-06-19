// @(#)root/tmva $Id: BinarySearchTree.h,v 1.11 2007/04/19 06:53:01 brun Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : BinarySearchTree                                                      *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      BinarySearchTree incl. volume Search method                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_BinarySearchTree
#define ROOT_TMVA_BinarySearchTree

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// BinarySearchTree                                                     //
//                                                                      //
// A simple Binary search tree including volume search method           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Riostream.h"
#include <vector>
#include "time.h"

#ifndef ROOT_TMVA_Volume
#include "TMVA/Volume.h"
#endif
#ifndef ROOT_TMVA_BinaryTree
#include "TMVA/BinaryTree.h"
#endif
#ifndef ROOT_TMVA_BinarySearchTreeNode
#include "TMVA/BinarySearchTreeNode.h"
#endif

class TString;
class TTree;

// -----------------------------------------------------------------------------
// the binary search tree

using std::vector;

namespace TMVA {

   class DataSet;
   class Event;
   class MethodBase;
   
   class BinarySearchTree : public BinaryTree {
      
   public:
      
      // constructor
      BinarySearchTree( void );
    
      // copy constructor
      BinarySearchTree (const BinarySearchTree &b);

      // destructor
      virtual ~BinarySearchTree( void );
    
      virtual Node * CreateNode() { return new BinarySearchTreeNode(); }

      virtual Node * CreateNode(const std::vector<TMVA::VariableInfo> * vI) { 
         TMVA::Event *ev = new TMVA::Event(*vI, kFALSE);
         return new BinarySearchTreeNode(ev);
      }

      // Searches for a node with the specified data 
      // by calling  the private, recursive, function for searching
      BinarySearchTreeNode* Search( Event * event ) const;
    
      // Adds an item to the tree, 
      void Insert( Event * );
    
      //get sum of weights of the nodes;
      Double_t GetSumOfWeights( void ) const;
    
      //set the periode (number of variables)
      inline void SetPeriode( Int_t p )      { fPeriod = p; }
      // return periode (number of variables)
      inline UInt_t  GetPeriode( void ) const { return fPeriod; }

      // counts events (weights) within a given volume 
      Double_t SearchVolume( Volume*, std::vector<const TMVA::BinarySearchTreeNode*>* events = 0 );
    
      // create the search tree from the events in a TTree
      // using the variables specified in the calling Method
      Double_t Fill( const TMVA::MethodBase& callingMethod, TTree* theTree, Int_t theType );
    
      // Create the search tree from the event collection 
      // using ONLY the variables specified in "theVars"
      Double_t Fill( vector<TMVA::Event*>, vector<Int_t> theVars, Int_t theType = -1 );
    
      // create the search tree from the events in a TTree
      // using ALL the variables specified included in the Event
      Double_t Fill( vector<TMVA::Event*> theTree, Int_t theType = -1 );

      void CalcStatistics(TMVA::Node *n = 0);

      // access to mean for signal and background for each variable
      Float_t Mean(Types::ESBType sb, UInt_t var ) { return fMeans[sb==Types::kSignal?0:1][var]; }

      // access to RMS for signal and background for each variable
      Float_t RMS(Types::ESBType sb, UInt_t var ) { return fRMS[sb==Types::kSignal?0:1][var]; }

      // access to Minimum for signal and background for each variable
      Float_t Min(Types::ESBType sb, UInt_t var ) { return fMin[sb==Types::kSignal?0:1][var]; }

      // access to Maximum for signal and background for each variable
      Float_t Max(Types::ESBType sb, UInt_t var ) { return fMax[sb==Types::kSignal?0:1][var]; }

      Int_t GetMemSize() { return sizeof(*this) + GetRoot()->GetMemSize(); }


   private:

      // add a new  node to the tree (as daughter) 
      void       Insert( Event*, Node* );
      // recursively search the nodes for Event
      BinarySearchTreeNode*      Search( Event*, Node *) const ;
    
      //check of Event variables lie with the volumde
      Bool_t   InVolume    (const std::vector<Float_t>&, Volume* ) const;
      //
      void     DestroyNode ( BinarySearchTreeNode* );
      // recursive search through daughter nodes in weight counting
      Double_t SearchVolume( Node*, Volume*, Int_t, 
                             std::vector<const TMVA::BinarySearchTreeNode*>* events );
      UInt_t fPeriod;            // periode (number of event variables)
      UInt_t fCurrentDepth;      // internal variable, counting the depth of the tree during insertion    
      Bool_t fStatisticsIsValid; // flag if last stat calculation is still valid, set to false if new node is insert

      std::vector<Float_t>        fMeans[2];    // mean for signal and background for each variable
      std::vector<Float_t>        fRMS[2];      // RMS for signal and background for each variable
      std::vector<Float_t>        fMin[2];      // RMS for signal and background for each variable
      std::vector<Float_t>        fMax[2];      // RMS for signal and background for each variable
      std::vector<Double_t>       fSum[2];      // Sum for signal and background for each variable
      std::vector<Double_t>       fSumSq[2];    // Squared Sum for signal and background for each variable
      Double_t                    fNEventsW[2]; // Number of events per class, taking into account event weights
      Double_t                    fSumOfWeights;// Total number of events (weigthed) counted during filling
                                                // should be the same as fNEventsW[0]+fNEventsW[1].. used as a check

      ClassDef(BinarySearchTree,0) // Binary search tree including volume search method  
   };
  
} // namespace TMVA

#endif
