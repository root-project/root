// @(#)root/tmva $Id: TMVA_Event.h,v 1.7 2006/05/03 19:45:38 helgevoss Exp $     
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_Event                                                            *
 *                                                                                *
 * Description:                                                                   *
 *       Event: variables of an event as used for the Binary Tree                 *
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
 * $Id: TMVA_Event.h,v 1.7 2006/05/03 19:45:38 helgevoss Exp $    
 **********************************************************************************/

#ifndef ROOT_TMVA_Event
#define ROOT_TMVA_Event

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_Event                                                           //
//                                                                      //
// Variables of an event as used for the Binary Tree                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <vector>
#include "Riostream.h"
#include "TVector.h"
#include "TTree.h"

// an Event coordinate
// simple enough for any event with one weight and 1 to n characterising variables,
// but could also be inherited from extened if needed 

class TMVA_Event {

  friend ostream& operator << (ostream& os, const TMVA_Event& event);
  friend ostream& operator << (ostream& os, const TMVA_Event* event);

 public:
    
  TMVA_Event() : fWeight( 1 ), fType( -1 ) {};
  TMVA_Event( std::vector<Double_t> &v, Double_t w = 1 , Int_t t=-1) 
    : fVar( v ), fWeight( w ), fType( t ) {}
  TMVA_Event( TTree* tree, Int_t ievt, std::vector<TString>* fInputVars );

  virtual ~TMVA_Event() {}
  inline const std::vector<Double_t> &GetData() const  { return fVar; }
  const Double_t                     &GetData( Int_t i ) const;
  inline Int_t    GetEventSize() const         { return fVar.size(); }
  inline void     Insert( Double_t v ) { fVar.push_back(v); }
  inline void     SetWeight( Double_t w ) { fWeight = w; }
  inline Double_t GetWeight() const  { return fWeight; }
  inline Int_t    GetType() const    { return fType;   }
  inline Int_t    GetType2() const   { return fType ? fType : -1 ; }
  inline void     SetType( Int_t t ) { fType = t; }

  void Print(ostream& os) const;

  TMVA_Event* Read(std::ifstream& is);

 private:

  std::vector<Double_t>  fVar;
  Double_t               fWeight;
  Int_t                  fType;
  
  ClassDef(TMVA_Event,0) //Variables of an event as used for the Binary Tree
};

#endif

