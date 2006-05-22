// @(#)root/tmva $Id: Event.h,v 1.6 2006/05/22 08:04:39 andreas.hoecker Exp $     
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Event                                                                 *
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
 **********************************************************************************/

#ifndef ROOT_TMVA_Event
#define ROOT_TMVA_Event

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Event                                                                //
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
// categorised as signal or background
// but could also be inherited from and extened if needed 

namespace TMVA {

  class Event {
    
    friend ostream& operator << (ostream& os, const Event& event);
    friend ostream& operator << (ostream& os, const Event* event);
    
  public:
    
    // default constructor
    Event() : fWeight( 1 ), fType( -1 ) {};
    // constructor specifying the event variables
    Event( std::vector<Double_t> &v, Double_t w = 1 , Int_t t=-1) 
      : fVar( v ), fWeight( w ), fType( t ) {}
    // constructor reading the Event from the ROOT tree
    Event( TTree* tree, Int_t ievt, std::vector<TString>* fInputVars );

    // destructor
    virtual ~Event() {}
    // return reference to the event variables as a STL vector
    inline const std::vector<Double_t> &GetData() const  { return fVar; }
    // return reference to "i-th" event variable
    const Double_t                     &GetData( Int_t i ) const;
    // return the number of the event variabels
    inline Int_t    GetEventSize() const         { return fVar.size(); }
    // add an event variable
    inline void     Insert( Double_t v ) { fVar.push_back(v); }
    // set an event weight
    inline void     SetWeight( Double_t w ) { fWeight = w; }
    //return the event weight
    inline Double_t GetWeight() const  { return fWeight; }
    // return the event type (signal = 1, bkg = 0);
    inline Int_t    GetType() const    { return fType;   }
    // return alternative type variables (signal = 1, bkg = -1)
    inline Int_t    GetType2() const   { return fType ? fType : -1 ; }
    // set the event type (signal = 1, bkg = 0);
    inline void     SetType( Int_t t ) { fType = t; }

    void Print(ostream& os) const;

    Event* Read(std::ifstream& is);

  private:

    std::vector<Double_t>  fVar;     // the vector of event variables
    Double_t               fWeight;  // event weight
    Int_t                  fType;    // event type (sigal=1 bkg = 0)
  
    ClassDef(Event,0); //Variables of an event as used for the Binary Tree
  };

} // namespace TMVA

#endif

