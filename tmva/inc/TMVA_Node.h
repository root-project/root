// @(#)root/tmva $Id: TMVA_Node.h,v 1.10 2006/05/02 23:27:40 helgevoss Exp $    
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: TMVA_Node, TMVA_NodeID                                                *
 *                                                                                *
 * Description:                                                                   *
 *      Node for the BinarySearch or Decision Trees                               *
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
 * File and Version Information:                                                  *
 * $Id: TMVA_Node.h,v 1.10 2006/05/02 23:27:40 helgevoss Exp $    
 **********************************************************************************/

#ifndef ROOT_TMVA_Node
#define ROOT_TMVA_Node

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_Node                                                            //
//                                                                      //
// Node for the BinarySearch or Decision Trees                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <vector>
#include "Riostream.h"
#include "Rtypes.h"

class TMVA_Event;

// a class used to identify a Node; (needed for recursive reading from text file)
// (currently it is NOT UNIQUE... but could eventually made it

class TMVA_NodeID {

 public:

  TMVA_NodeID( Int_t d=0, std::string p="Root" ) : fDepth( d ), fPos( p ) {}
  ~TMVA_NodeID() {}

  void SetDepth(const Int_t d){fDepth=d;}
  Int_t GetDepth() const {return fDepth;}
  void SetPos(const std::string s) {fPos=s;}
  std::string GetPos() const {return fPos;}

 private:

  Int_t        fDepth;
  std::string  fPos;
};


// a node in the tree structure
class TMVA_Node {

  friend ostream& operator << (ostream& os, const TMVA_Node& node);
  friend ostream& operator << (ostream& os, const TMVA_Node* node);

public:

  TMVA_Node( TMVA_Event* e =NULL, Bool_t o=kFALSE ) : fEvent( e ), fLeft( NULL ), 
    fRight( NULL ), fParent ( NULL ), fSelector( -1 ), fEventOwnership ( o ) {}

  TMVA_Node( TMVA_Node* p ) : fEvent( NULL ), fLeft( NULL ), 
    fRight( NULL ), fParent ( p ), fSelector( -1 ), fEventOwnership (kFALSE) {}

  virtual ~TMVA_Node ();

  virtual Bool_t GoesRight( const TMVA_Event* ) const;
  virtual Bool_t GoesLeft ( const TMVA_Event* ) const;
  virtual Bool_t EqualsMe ( const TMVA_Event* ) const;

  inline TMVA_Node* GetLeft  () const { return fLeft;   }
  inline TMVA_Node* GetRight () const { return fRight;  }
  inline TMVA_Node* GetParent() const { return fParent; }

  inline void SetParent(TMVA_Node* p) { fParent = p;} // Set the parent pointer of the node.
  inline void SetLeft  (TMVA_Node* l) { fLeft   = l;}     // Set the left pointer of the node.
  inline void SetRight (TMVA_Node* r) { fRight  = r;}   // Set the right pointer of the node.
  
  inline void SetSelector( const Short_t i) { fSelector = i; }
  inline void SetSelector( const Int_t i  ) { fSelector = Short_t(i); }
  inline Short_t GetSelector() const { return fSelector; }
  inline void SetData( TMVA_Event* e ) { fEvent = e; }
  inline TMVA_Event* GetData() const { return fEvent; }

  Int_t  CountMeAndAllDaughters() const;
  void   Print( ostream& os ) const;

  virtual void PrintRec( ostream& os, const Int_t depth=0, const std::string pos="root" ) const;
  virtual TMVA_NodeID ReadRec( ifstream& is, TMVA_NodeID nodeID, TMVA_Node* parent=NULL );

  Bool_t      GetEventOwnership( void           ) { return fEventOwnership; }
  void        SetEventOwnership( const Bool_t b ) { fEventOwnership = b; }

 private:
  TMVA_Event* fEvent;

  TMVA_Node*  fLeft;    // pointers to the two "daughter" nodes
  TMVA_Node*  fRight;   // pointers to the two "daughter" nodes
  TMVA_Node*  fParent;  // the previous (parent) node

  Short_t     fSelector;
  Bool_t      fEventOwnership;
  
  ClassDef(TMVA_Node,0) //Node for the BinarySearch or Decision Trees
};



#endif

