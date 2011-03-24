// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss, Jan Therhaag

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
 *      Jan Therhaag       <Jan.Therhaag@cern.ch>     - U of Bonn, Germany        *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      U. of Bonn, Germany                                                       *
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

//____________________________________________________________
TMVA::Event::Event()
   : fValues(),
     fValuesDynamic(0),
     fTargets(),
     fSpectators(),
     fClass(0),
     fWeight(1.0),
     fBoostWeight(1.0),
     fDynamic(kFALSE)
{
   // copy constructor
}

//____________________________________________________________
TMVA::Event::Event( const std::vector<Float_t>& ev,
                    const std::vector<Float_t>& tg,
                    UInt_t cls,
                    Double_t weight,
                    Double_t boostweight )
   : fValues(ev),
     fValuesDynamic(0),
     fTargets(tg),
     fSpectators(0),
     fClass(cls),
     fWeight(weight),
     fBoostWeight(boostweight),
     fDynamic(kFALSE)
{
   // constructor
}

//____________________________________________________________
TMVA::Event::Event( const std::vector<Float_t>& ev,
                    const std::vector<Float_t>& tg,
                    const std::vector<Float_t>& vi,
                    UInt_t cls,
                    Double_t weight,
                    Double_t boostweight )
   : fValues(ev),
     fValuesDynamic(0),
     fTargets(tg),
     fSpectators(vi),
     fClass(cls),
     fWeight(weight),
     fBoostWeight(boostweight),
     fDynamic(kFALSE)
{
   // constructor
}

//____________________________________________________________
TMVA::Event::Event( const std::vector<Float_t>& ev,
                    UInt_t cls,
                    Double_t weight,
                    Double_t boostweight )
   : fValues(ev),
     fValuesDynamic(0),
     fTargets(0),
     fSpectators(0),
     fClass(cls),
     fWeight(weight),
     fBoostWeight(boostweight),
     fDynamic(kFALSE)
{
   // constructor
}

//____________________________________________________________
TMVA::Event::Event( const std::vector<Float_t*>*& evdyn, UInt_t nvar )
   : fValues(nvar),
     fValuesDynamic(0),
     fTargets(0),
     fSpectators(evdyn->size()-nvar),
     fClass(0),
     fWeight(0),
     fBoostWeight(0),
     fDynamic(true)
{
   // constructor for single events
   fValuesDynamic = (std::vector<Float_t*>*) evdyn;
}

//____________________________________________________________
TMVA::Event::Event( const Event& event ) 
   : fValues(event.fValues),
     fValuesDynamic(event.fValuesDynamic),
     fTargets(event.fTargets),
     fSpectators(event.fSpectators),
     fClass(event.fClass),
     fWeight(event.fWeight),
     fBoostWeight(event.fBoostWeight),
     fDynamic(event.fDynamic)
{
   // copy constructor
   if (event.fDynamic){
      fValues.clear();
      UInt_t nvar = event.GetNVariables();
      UInt_t idx=0;
      std::vector<Float_t*>::iterator itDyn=event.fValuesDynamic->begin(), itDynEnd=event.fValuesDynamic->end();
      for (; itDyn!=itDynEnd && idx<nvar; ++itDyn){
	 Float_t value=*(*itDyn);
	 fValues.push_back( value );
	 ++idx;
      }
      fSpectators.clear();
      for (; itDyn!=itDynEnd; ++itDyn){
	 Float_t value=*(*itDyn);
	 fSpectators.push_back( value );
	 ++idx;
      }

      fDynamic=kFALSE;
      fValuesDynamic=NULL;
   }
}

//____________________________________________________________
TMVA::Event::~Event()
{
   // Event destructor
}

//____________________________________________________________
void TMVA::Event::CopyVarValues( const Event& other )
{
   // copies only the variable values
   fValues      = other.fValues;
   fTargets     = other.fTargets;
   fSpectators  = other.fSpectators;
   if (other.fDynamic){
      UInt_t nvar = other.GetNVariables();
      fValues.clear();
      UInt_t idx=0;
      std::vector<Float_t*>::iterator itDyn=other.fValuesDynamic->begin(), itDynEnd=other.fValuesDynamic->end();
      for (; itDyn!=itDynEnd && idx<nvar; ++itDyn){
	 Float_t value=*(*itDyn);
	 fValues.push_back( value );
	 ++idx;
      }
      fSpectators.clear();
      for (; itDyn!=itDynEnd; ++itDyn){
	 Float_t value=*(*itDyn);
	 fSpectators.push_back( value );
	 ++idx;
      }
   }
   fDynamic     = kFALSE;
   fValuesDynamic = NULL;

   fClass       = other.fClass;
   fWeight      = other.fWeight;
   fBoostWeight = other.fBoostWeight;
}

//____________________________________________________________
Float_t TMVA::Event::GetValue( UInt_t ivar ) const
{
   // return value of i'th variable
   Float_t retval;

   if (fDynamic){
      retval = *((*fValuesDynamic).at(ivar));
   }
   else{
      retval = fValues.at(ivar);
   }

   return retval;
}

//____________________________________________________________
Float_t TMVA::Event::GetSpectator( UInt_t ivar) const 
{
   // return spectator content
   if (fDynamic) return *(fValuesDynamic->at(GetNVariables()+ivar));
   else          return fSpectators.at(ivar);
}

//____________________________________________________________
const std::vector<Float_t>& TMVA::Event::GetValues() const
{
   // return value vector
   if (fDynamic) {
      fValues.clear();
      for (std::vector<Float_t*>::const_iterator it = fValuesDynamic->begin(), itEnd=fValuesDynamic->end()-GetNSpectators(); 
           it != itEnd; ++it) { 
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
   return fValues.size();
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

   return fSpectators.size();
}


//____________________________________________________________
void TMVA::Event::SetVal( UInt_t ivar, Float_t val ) 
{
   // set variable ivar to val
   if ((fDynamic ?( (*fValuesDynamic).size() ) : fValues.size())<=ivar)
      (fDynamic ?( (*fValuesDynamic).resize(ivar+1) ) : fValues.resize(ivar+1));

   (fDynamic ?( *(*fValuesDynamic)[ivar] ) : fValues[ivar])=val;
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
