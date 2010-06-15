// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

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
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch> - CERN, Switzerland           *
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
#include "assert.h"
#include <iomanip>
#include <cassert>
#include "TCut.h"
 
Int_t TMVA::Event::fgCount = 0;
std::vector<Float_t*>* TMVA::Event::fgValuesDynamic = 0;

//____________________________________________________________
TMVA::Event::Event() 
   : fValues(),
     fTargets(),
     fSpectators(),
     fVariableArrangement(0),
     fClass(1),
     fWeight(1.0),
     fBoostWeight(1.0),
     fDynamic(kFALSE)
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
     fVariableArrangement(0),
     fClass(cls),
     fWeight(weight),
     fBoostWeight(boostweight),
     fDynamic(kFALSE)
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
     fVariableArrangement(0),
     fClass(cls),
     fWeight(weight),
     fBoostWeight(boostweight),
     fDynamic(kFALSE)
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
     fVariableArrangement(0),
     fClass(cls),
     fWeight(weight),
     fBoostWeight(boostweight),
     fDynamic(kFALSE)
{
   // constructor
   fgCount++;
}

//____________________________________________________________
TMVA::Event::Event( const std::vector<Float_t*>*& evdyn, UInt_t nvar )
   : fValues(nvar),
     fTargets(0),
     fSpectators(evdyn->size()-nvar),
     fVariableArrangement(0),
     fClass(0),
     fWeight(0),
     fBoostWeight(0),
     fDynamic(true)
{
   // constructor for single events
   fgValuesDynamic = (std::vector<Float_t*>*) evdyn;
   fgCount++;
}

//____________________________________________________________
TMVA::Event::Event( const Event& event ) 
   : fValues(event.fValues),
     fTargets(event.fTargets),
     fSpectators(event.fSpectators),
     fVariableArrangement(event.fVariableArrangement),
     fClass(event.fClass),
     fWeight(event.fWeight),
     fBoostWeight(event.fBoostWeight),
     fDynamic(event.fDynamic)
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
void TMVA::Event::SetVariableArrangement( std::vector<UInt_t>* const m ) const {
   // set the variable arrangement

   // mapping from global variable index (the position in the vector)
   // to the new index in the subset of variables used by the
   // composite classifier
   fVariableArrangement = m;
}



//____________________________________________________________
void TMVA::Event::CopyVarValues( const Event& other )
{
   // copies only the variable values
   fValues      = other.fValues;
   fTargets     = other.fTargets;
   fSpectators  = other.fSpectators;

   fVariableArrangement = other.fVariableArrangement;

   fClass       = other.fClass;
   fWeight      = other.fWeight;
   fBoostWeight = other.fBoostWeight;
}

//____________________________________________________________
Float_t TMVA::Event::GetValue( UInt_t ivar ) const 
{ 
   // return value of i'th variable
   Float_t retval;
   if (fVariableArrangement==0) {
      retval = fDynamic ?( *(*fgValuesDynamic)[ivar] ) : fValues[ivar]; 
   } 
   else {
      UInt_t mapIdx = (*fVariableArrangement)[ivar];
      if (fDynamic) {
         retval = *(*fgValuesDynamic)[mapIdx];
      } 
      else {
         retval = ( mapIdx<fValues.size() ) ? fValues[mapIdx] : fSpectators[mapIdx-fValues.size()];
      }
   }
   return retval;
}

//____________________________________________________________
Float_t TMVA::Event::GetSpectator( UInt_t ivar) const 
{
   // return spectator content
   if (fDynamic) return *(fgValuesDynamic->at(GetNVariables()+ivar));
   else          return fSpectators.at(ivar);
}

//____________________________________________________________
const std::vector<Float_t>& TMVA::Event::GetValues() const 
{  
   // return value vector
   if (fVariableArrangement!=0) {
      assert(0);
   }
   if (fDynamic) {
//       if (fgValuesDynamic->size()-GetNSpectators() != fValues.size()) {
//          std::cout << "ERROR Event::GetValues() is trying to change the size of the variable vector, exiting ..." << std::endl;
//          assert(0);
//       }
      fValues.clear();
      for (std::vector<Float_t*>::const_iterator it = fgValuesDynamic->begin(); 
           it != fgValuesDynamic->end()-GetNSpectators(); it++) { 
         Float_t val = *(*it); 
         fValues.push_back( val ); 
      }
   }
   return fValues;
}

//____________________________________________________________
UInt_t TMVA::Event::GetNVariables() const 
{
   // accessor to the number of variables 

   // if variables have to arranged (as it is the case for the
   // composite classifier) the number of the variables changes

   if (fVariableArrangement==0) return fValues.size();
   else                         return fVariableArrangement->size();
}

//____________________________________________________________
UInt_t TMVA::Event::GetNTargets() const 
{
   // accessor to the number of targets
   return fTargets.size();
}

//____________________________________________________________
UInt_t TMVA::Event::GetNSpectators() const 
{
   // accessor to the number of spectators 

   // if variables have to arranged (as it is the case for the
   // composite classifier) the number of the variables changes

   if (fVariableArrangement==0) return fSpectators.size();
   else                         return fValues.size()-fVariableArrangement->size();
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
   o << *this << std::endl;
}

//_____________________________________________________________
void TMVA::Event::SetTarget( UInt_t itgt, Float_t value ) 
{ 
   // set the target value (dimension itgt) to value

   if (fTargets.size() <= itgt) fTargets.resize( itgt+1 );
   fTargets.at(itgt) = value;
}

//_____________________________________________________________
void TMVA::Event::SetSpectator( UInt_t ivar, Float_t value ) 
{ 
   // set spectator value (dimension ivar) to value

   if (fSpectators.size() <= ivar) fSpectators.resize( ivar+1 );
   fSpectators.at(ivar) = value;
}

//_______________________________________________________________________
ostream& TMVA::operator << ( ostream& os, const TMVA::Event& event )
{ 
   // Outputs the data of an event
   os << "Variables [" << event.fValues.size() << "]:";
   for (UInt_t ivar=0; ivar<event.fValues.size(); ++ivar)
      os << " " << std::setw(10) << event.GetValue(ivar);
   os << ", targets [" << event.fTargets.size() << "]:";
   for (UInt_t ivar=0; ivar<event.fTargets.size(); ++ivar)
      os << " " << std::setw(10) << event.GetTarget(ivar);
   os << ", spectators ["<< event.fSpectators.size() << "]:";
   for (UInt_t ivar=0; ivar<event.fSpectators.size(); ++ivar)
      os << " " << std::setw(10) << event.GetSpectator(ivar);
   os << ", weight: " << event.GetWeight();
   os << ", class: " << event.GetClass();
   return os;
}
