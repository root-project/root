// @(#)root/tmva $Id: BinarySearchTree.h,v 1.13 2006/09/29 23:27:15 andreas.hoecker Exp $    
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
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

#include "TTree.h"

#ifndef ROOT_TMVA_Volume
#include "TMVA/Volume.h"
#endif
#ifndef ROOT_TMVA_BinaryTree
#include "TMVA/BinaryTree.h"
#endif

class TString;

// -----------------------------------------------------------------------------
// the binary search tree

using std::vector;

namespace TMVA {

   class DataSet;
   class Event;

   class BinarySearchTree : public BinaryTree {
      
   public:
  
      // constructor
      BinarySearchTree( void );
      // destructor
      virtual ~BinarySearchTree( void );
  
      // counts events (weights) within a given volume 
      Double_t SearchVolume( Volume*, std::vector<TMVA::Event*>* events = 0 );
  
      // create the search tree from the events in a TTree
      // using the variables specified in "theVars"
      Int_t Fill( const DataSet& ds, TTree* theTree, Int_t theType = -1, 
                  Types::PreprocessingMethod corr = Types::kNone, Types::SBType type = Types::kSignal );

      // create the search tree from the event collection 
      // using ONLY the variables specified in "theVars"
      Int_t Fill( vector<TMVA::Event*>, vector<Int_t> theVars, Int_t theType = -1 );

      // create the search tree from the events in a TTree
      // using ALL the variables specified included in the Event
      Int_t Fill( vector<TMVA::Event*> theTree, Int_t theType = -1 );
  
   private:

      //check of Event variables lie with the volumde
      Bool_t   InVolume    ( TMVA::Event*, Volume* ) const;
      //
      void     DestroyNode ( Node* );
      // recursive search through daughter nodes in weight counting
      Double_t SearchVolume( Node*, Volume*, Int_t, 
                             std::vector<Event*>* events );
      Int_t    fPeriode;   // the number of variables defining the periode of ordering
  
      ClassDef(BinarySearchTree,0) // Binary search tree including volume search method  
         };

   // -----------------------------------------------------------------------------

} // namespace TMVA

#endif
