// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Event                                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

#include "TMVA/Event.h"
#include "TMVA/Tools.h"
#include <iostream>
#include <iomanip>
 
Int_t TMVA::Event::fgCount = 0;
std::vector<Float_t*>* TMVA::Event::fgValuesDynamic = 0;

//____________________________________________________________
TMVA::Event::Event() 
   : fValues(),
     fTargets(),
     fSpectators(),
     fClass(1),
     fWeight(1.0),
     fBoostWeight(1.0),
     fDynamic(kFALSE),
     fSignalClass( 100 ) // TODO: remove this.. see "IsSignal"
{
   // copy constructor
   fgCount++; 
}

//____________________________________________________________
TMVA::Event::Event( const std::vector<Float_t>& ev,
                    const std::vector<Float_t>& tg,
                    UInt_t cls,
                    Float_t weight,
                    Float_t boostweight )
   : fValues(ev),
     fTargets(tg),
     fSpectators(0),
     fClass(cls),
     fWeight(weight),
     fBoostWeight(boostweight),
     fDynamic(kFALSE),
     fSignalClass( 100 ) // TODO: remove this.. see "IsSignal"
{
   // constructor
   fgCount++;
}

//____________________________________________________________
TMVA::Event::Event( const std::vector<Float_t>& ev,
                    const std::vector<Float_t>& tg,
                    const std::vector<Float_t>& vi,
                    UInt_t cls,
                    Float_t weight,
                    Float_t boostweight )
   : fValues(ev),
     fTargets(tg),
     fSpectators(vi),
     fClass(cls),
     fWeight(weight),
     fBoostWeight(boostweight),
     fDynamic(kFALSE),
     fSignalClass( 100 ) // TODO: remove this.. see "IsSignal"
{
   // constructor
   fgCount++;
}

//____________________________________________________________
TMVA::Event::Event( const std::vector<Float_t>& ev,
                    UInt_t cls,
                    Float_t weight,
                    Float_t boostweight )
   : fValues(ev),
     fTargets(0),
     fSpectators(0),
     fClass(cls),
     fWeight(weight),
     fBoostWeight(boostweight),
     fDynamic(kFALSE),
     fSignalClass( 100 ) // TODO: remove this.. see "IsSignal"
{
   // constructor
   fgCount++;
}

//____________________________________________________________
TMVA::Event::Event( const std::vector<Float_t*>*& evdyn )
   : fValues(evdyn->size()),
     fTargets(0),
     fSpectators(0),
     fClass(0),
     fWeight(0),
     fBoostWeight(0),
     fDynamic(true),
     fSignalClass( 100 ) // TODO: remove this.. see "IsSignal" ... !!!!!! NOT CLEAR TO ME WHAT VALUE TO SET HERE...
{

   fgValuesDynamic = (std::vector<Float_t*>*) evdyn;
   // constructor for single events
   fgCount++;
}

//____________________________________________________________
TMVA::Event::Event( const Event& event ) 
   : fValues(event.fValues),
     fTargets(event.fTargets),
     fSpectators(event.fSpectators),
     fClass(event.fClass),
     fWeight(event.fWeight),
     fBoostWeight(event.fBoostWeight),
     fDynamic(event.fDynamic),
     fSignalClass( event.fSignalClass ) // TODO: remove this.. see "IsSignal"
{
   // copy constructor
   fgCount++; 
}

//____________________________________________________________
TMVA::Event::~Event() 
{
   // Event destructor
   fgCount--;;
   if (fDynamic && fgCount==0) TMVA::Event::ClearDynamicVariables();
}
 
//____________________________________________________________
void TMVA::Event::ClearDynamicVariables() 
{ 
   // clear global variable
   if (fgValuesDynamic != 0) { 
      fgValuesDynamic->clear();
      delete fgValuesDynamic;
      fgValuesDynamic = 0;
   }
} 

//____________________________________________________________
void TMVA::Event::CopyVarValues( const Event& other )
{
   // copies only the variable values
   fValues      = other.fValues;
   fClass       = other.fClass;
   fWeight      = other.fWeight;
   fBoostWeight = other.fBoostWeight;
   fSignalClass = other.fSignalClass;      // TODO: remove this.. see "IsSignal"
}

//____________________________________________________________
Float_t TMVA::Event::GetVal( UInt_t ivar ) const 
{ 
   // return value of i'th variable
   return ( fDynamic ?( *(*fgValuesDynamic)[ivar] ) : fValues[ivar] ); 
}

//____________________________________________________________
const std::vector<Float_t>& TMVA::Event::GetValues() const 
{  
   // return va;lue vector
   if (fDynamic) {
      fValues.clear();
      for (std::vector<Float_t*>::const_iterator it = fgValuesDynamic->begin(); 
           it != fgValuesDynamic->end(); it++) { 
         Float_t val = *(*it); 
         fValues.push_back( val ); 
      }
   }
   return fValues;
}

//____________________________________________________________
void TMVA::Event::SetVal( UInt_t ivar, Float_t val ) 
{
   // set variable ivar to val
   if ((fDynamic ?( (*fgValuesDynamic).size() ) : fValues.size())<=ivar)
      (fDynamic ?( (*fgValuesDynamic).resize(ivar+1) ) : fValues.resize(ivar+1));
   (fDynamic ?( *(*fgValuesDynamic)[ivar] ) : fValues[ivar])=val;
}

//____________________________________________________________
void TMVA::Event::Print( std::ostream& o ) const
{
   // print method
   if (!fDynamic) {
      o << fValues.size() << " variables: ";
      for (UInt_t ivar=0; ivar<fValues.size(); ivar++)
         o << " " << std::setw(10) << GetValue(ivar);
      o << ", weight = " << GetWeight();
      o << std::setw(10) << "class: " << std::setw(10) << GetClass() << " , ";
      o << fTargets.size() << " targets: ";
      for (UInt_t ivar=0; ivar<fTargets.size(); ivar++)
         o << " " << std::setw(10) << GetTarget(ivar);
      o << fSpectators.size() << " spectators: ";
      for (UInt_t ivar=0; ivar<fSpectators.size(); ivar++)
         o << " " << std::setw(10) << GetSpectator(ivar);
      o << std::endl;
   }
   else {
      o << "|dynamic variables|" << std::endl;
   }
}

//_______________________________________________________________________
ostream& TMVA::operator << ( ostream& os, const TMVA::Event& event )
{ 
   // Outputs the data of an event
   
   event.Print(os);
   return os;
}

//_______________________________________________________________________
ostream& TMVA::operator << ( ostream& os, const TMVA::Event* event )
{
   // Outputs the data of an event
   return os << *event;
}
