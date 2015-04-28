// @(#)root/tmva $Id$    
// Author: Andrzej Zemla

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SVEvent                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Marcin Wolter  <Marcin.Wolter@cern.ch> - IFJ PAN, Krakow, Poland          *
 *      Andrzej Zemla  <azemla@cern.ch>        - IFJ PAN, Krakow, Poland          *
 *      (IFJ PAN: Henryk Niewodniczanski Inst. Nucl. Physics, Krakow, Poland)     *   
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      PAN, Krakow, Poland                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TMVA/SVEvent.h"
#include "TMVA/Event.h"

#include <iostream>

ClassImp(TMVA::SVEvent)

TMVA::SVEvent::SVEvent()
   : fDataVector(0),
     fCweight(1.), 
     fAlpha(0),
     fAlpha_p(0),
     fErrorCache(0),
     fNVar(0),
     fTypeFlag(0),
     fIdx(0),
     fNs(0),
     fIsShrinked(0),
     fLine(0),
     fTarget(0)   
{
}

//_______________________________________________________________________
TMVA::SVEvent::SVEvent( const Event* event, Float_t C_par, Bool_t isSignal )
   : fDataVector(event->GetValues()),
     fCweight(C_par*event->GetWeight()),
     fAlpha(0),
     fAlpha_p(0),
     fErrorCache(0),
     fNVar    ( event->GetNVariables() ),
     fTypeFlag( isSignal ? -1 : 1 ),
     fIdx     ( isSignal ? -1 : 1 ),
     fNs(0),
     fIsShrinked(0),
     fLine(0),
     fTarget((event->GetNTargets() > 0 ? event->GetTarget(0) : 0))
{
   // constructor
}

//_______________________________________________________________________
TMVA::SVEvent::SVEvent( const std::vector<Float_t>* svector, Float_t alpha, Int_t typeFlag, UInt_t ns )
   : fDataVector(*svector),
     fCweight(-1.),
     fAlpha(alpha),
     fAlpha_p(0),
     fErrorCache(-1.),
     fNVar(svector->size()),
     fTypeFlag(typeFlag),
     fIdx(-1),
     fNs(ns),
     fIsShrinked(0),
     fLine(0),
     fTarget(0)
{
   // constructor
}

//_______________________________________________________________________
TMVA::SVEvent::SVEvent( const std::vector<Float_t>* svector, Float_t alpha, Float_t alpha_p,Int_t typeFlag)
   : fDataVector(*svector),
     fCweight(-1.),
     fAlpha(alpha),
     fAlpha_p(alpha_p),
     fErrorCache(-1.),
     fNVar(svector->size()),
     fTypeFlag(typeFlag),
     fIdx(-1),
     fNs(0),
     fIsShrinked(0),
     fLine(0),
     fTarget(0)
{
   // constructor
}

//_______________________________________________________________________
TMVA::SVEvent::~SVEvent()
{
   // destructor
   if (fLine != 0) {
      delete fLine; 
      fLine = 0;
   }
}

//_______________________________________________________________________
void TMVA::SVEvent::Print( std::ostream& os ) const
{
   // printout 
   os << "type::" << fTypeFlag <<" target::"<< fTarget << " alpha::" << fAlpha <<" alpha_p::"<< fAlpha_p<< " values::" ;
   for (UInt_t j =0; j < fDataVector.size();j++) os<<fDataVector.at(j)<<" ";
   os << std::endl;
}

//_______________________________________________________________________
void TMVA::SVEvent::PrintData()
{
   // printout 
   for (UInt_t i = 0; i < fNVar; i++) std::cout<<fDataVector.at(i)<<" ";
   std::cout<<std::endl;
}
