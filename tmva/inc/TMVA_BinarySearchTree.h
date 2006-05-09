// @(#)root/tmva $Id: TMVA_BinarySearchTree.h,v 1.2 2006/05/08 20:56:16 brun Exp $    
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * class  : TMVA_BinarySearchTree                                                 *
 *                                                                                *
 * Description:                                                                   *
 *      BinarySearchTree incl. volume Search method                               *
 *                                                                                *
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
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_BinarySearchTree
#define ROOT_TMVA_BinarySearchTree

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_BinarySearchTree                                                //
//                                                                      //
// Binary search tree including volume search method                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Riostream.h"
#include <vector>
#include "time.h"
#ifndef ROOT_TMVA_Volume
#include "TMVA_Volume.h"
#endif
#ifndef ROOT_TMVA_BinaryTree
#include "TMVA_BinaryTree.h"
#endif

class TTree;
class TString;
class TMVA_Event;

// -----------------------------------------------------------------------------
// the binary search tree

using std::vector;

class TMVA_BinarySearchTree : public TMVA_BinaryTree {
      
 public:
  
  TMVA_BinarySearchTree( void );
  virtual ~TMVA_BinarySearchTree( void );
  
  //counts events (weights) within a given volume
  Double_t SearchVolume( TMVA_Volume*, std::vector<TMVA_Event*>* events = 0 );
  
  //  creator function (why the hell was this "static" Andreas ???
  void Fill( TTree*, vector<TString>*, Int_t&, Int_t theType = -1 );
  void Fill( vector<TMVA_Event*>, vector<Int_t>, Int_t&, Int_t theType = -1 );
  void Fill( vector<TMVA_Event*> theTree, Int_t theType = -1 );
  
 private:
  
  Bool_t   InVolume    ( TMVA_Event*, TMVA_Volume* ) const;
  void     DestroyNode ( TMVA_Node* );
  Double_t SearchVolume( TMVA_Node*, TMVA_Volume*, Int_t, 
                         std::vector<TMVA_Event*>* events );
  Int_t    fPeriode;
  Int_t    fDbgcount; 
  
  ClassDef(TMVA_BinarySearchTree,0) //Binary search tree including volume search method  
};

  // -----------------------------------------------------------------------------

#endif
