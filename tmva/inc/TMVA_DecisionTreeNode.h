// @(#)root/tmva $Id: TMVA_DecisionTreeNode.h,v 1.1 2006/05/08 12:46:30 brun Exp $    
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_DecisionTreeNode                                                 *
 *                                                                                *
 * Description:                                                                   *
 *      Node for the Decision Tree                                                *
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

#ifndef ROOT_TMVA_DecisionTreeNode
#define ROOT_TMVA_DecisionTreeNode

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_DecisionTreeNode                                                //
//                                                                      //
// Node for the Decision Tree                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#ifndef ROOT_TMVA_Node
#include "TMVA_Node.h"
#endif

using std::string;

class TMVA_DecisionTreeNode: public TMVA_Node {

 public:

  TMVA_DecisionTreeNode (TMVA_Event* e =NULL);
  TMVA_DecisionTreeNode (TMVA_Node* p); 

  virtual ~TMVA_DecisionTreeNode(){}

  virtual Bool_t GoesRight( const TMVA_Event* ) const;
  virtual Bool_t GoesLeft ( const TMVA_Event* ) const;

  void  SetCutMin ( Double_t c ) { fCutMin  = c; }
  void  SetCutMax ( Double_t c ) { fCutMax  = c; }
  void  SetCutType( Bool_t t   ) { fCutType = t; }

  Double_t GetCutMin ( void ) const { return fCutMin;  }
  Double_t GetCutMax ( void ) const { return fCutMax;  }

  // kTRUE: Cuts select signal, kFALSE: Cuts select bkg
  Bool_t    GetCutType( void ) const { return fCutType; }

  // 1 signal node, -1 bkg leave, 0 intermediate Node
  void  SetNodeType( Int_t t ) { fNodeType = t;} 
  Int_t GetNodeType( void ) const { return fNodeType; }

  void     SetSoverSB( Double_t ssb ){ fSoverSB =ssb ; }
  Double_t GetSoverSB( void ) const  { return fSoverSB; }

  void     SetSeparationIndex( Double_t sep ){ fSeparationIndex =sep ; }
  Double_t GetSeparationIndex( void ) const  { return fSeparationIndex; }

  void     SetSeparationGain( Double_t sep ){ fSeparationGain =sep ; }
  Double_t GetSeparationGain( void ) const  { return fSeparationGain; }

  void     SetNEvents( Double_t nev ){ fNEvents =nev ; }
  Double_t GetNEvents( void ) const  { return fNEvents; }

  virtual void        PrintRec( ostream&  os, const Int_t depth=0, const string pos="root" ) const;
  virtual TMVA_NodeID ReadRec ( ifstream& is, TMVA_NodeID nodeID, TMVA_Node* parent=NULL );
  
 private:
  
  Double_t fCutMin;
  Double_t fCutMax;
  Bool_t   fCutType;
  
  Double_t fSoverSB;
  Double_t fSeparationIndex;
  Double_t fSeparationGain;
  Double_t fNEvents;
  Int_t    fNodeType;
  
  ClassDef(TMVA_DecisionTreeNode,0) //Node for the Decision Tree 
  
};

#endif 
