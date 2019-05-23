// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Volume                                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::Volume
\ingroup TMVA
Volume for BinarySearchTree

volume element: variable space between upper and lower bonds of
nvar-dimensional variable space
*/

#include "TMVA/Volume.h"

#include "TMVA/MsgLogger.h"
#include "TMVA/Tools.h"
#include "TMVA/Types.h"

#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////
/// constructor specifying the volume by std::vectors of doubles

TMVA::Volume::Volume( std::vector<Double_t>* l, std::vector<Double_t>* u )
   : fLower( l ),
     fUpper( u ),
     fOwnerShip (kFALSE){
     }

////////////////////////////////////////////////////////////////////////////////
/// constructor specifying the volume by std::vectors of floats

TMVA::Volume::Volume( std::vector<Float_t>* l, std::vector<Float_t>* u )
{
   fLower = new std::vector<Double_t>( l->size() );
   fUpper = new std::vector<Double_t>( u->size() );
   fOwnerShip = kTRUE;

   for (UInt_t ivar=0; ivar<l->size(); ivar++) {
      (*fLower)[ivar] = Double_t((*l)[ivar]);
      (*fUpper)[ivar] = Double_t((*u)[ivar]);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// constructor specifying the volume by c-style arrays of doubles

TMVA::Volume::Volume( Double_t* l, Double_t* u, Int_t nvar )
{
   fLower = new std::vector<Double_t>( nvar );
   fUpper = new std::vector<Double_t>( nvar );
   fOwnerShip = kTRUE;

   for (int ivar=0; ivar<nvar; ivar++) {
      (*fLower)[ivar] = l[ivar];
      (*fUpper)[ivar] = u[ivar];
   }
}

////////////////////////////////////////////////////////////////////////////////
/// constructor specifying the volume by c-style arrays of floats

TMVA::Volume::Volume( Float_t* l, Float_t* u, Int_t nvar )
{
   fLower = new std::vector<Double_t>( nvar );
   fUpper = new std::vector<Double_t>( nvar );
   fOwnerShip = kTRUE;

   for (int ivar=0; ivar<nvar; ivar++) {
      (*fLower)[ivar] = Double_t(l[ivar]);
      (*fUpper)[ivar] = Double_t(u[ivar]);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// simple constructors for 1 dimensional values (double)

TMVA::Volume::Volume( Double_t l, Double_t u )
{
   fLower = new std::vector<Double_t>(1);
   fUpper = new std::vector<Double_t>(1);
   fOwnerShip = kTRUE;
   (*fLower)[0] = l;
   (*fUpper)[0] = u;
}

////////////////////////////////////////////////////////////////////////////////
/// simple constructors for 1 dimensional values (float)

TMVA::Volume::Volume( Float_t l, Float_t u )
{
   fLower = new std::vector<Double_t>(1);
   fUpper = new std::vector<Double_t>(1);
   fOwnerShip = kTRUE;
   (*fLower)[0] = Double_t(l);
   (*fUpper)[0] = Double_t(u);
}

////////////////////////////////////////////////////////////////////////////////
/// copy constructor

TMVA::Volume::Volume( Volume& V )
{
   fLower = new std::vector<Double_t>( *V.fLower );
   fUpper = new std::vector<Double_t>( *V.fUpper );
   fOwnerShip = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// assignment operator

TMVA::Volume& TMVA::Volume::operator=( const Volume& V )
{
   if (fOwnerShip) {
      if (fLower) delete fLower;
      if (fUpper) delete fUpper;
      fLower = new std::vector<Double_t>( *V.fLower );
      fUpper = new std::vector<Double_t>( *V.fUpper );
   }
   else {
      fLower = V.fLower;
      fUpper = V.fUpper;
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::Volume::~Volume( void )
{
   // delete volume boundaries only if owned by the volume
   if (fOwnerShip) this->Delete();
}

////////////////////////////////////////////////////////////////////////////////
/// delete array of volume bondaries

void TMVA::Volume::Delete( void )
{
   if (NULL != fLower) { delete fLower; fLower = NULL; }
   if (NULL != fUpper) { delete fUpper; fUpper = NULL; }
}

////////////////////////////////////////////////////////////////////////////////
/// "scale" the volume by multiplying each upper and lower boundary by "f"

void TMVA::Volume::Scale( Double_t f )
{
   gTools().Scale(*fLower,f);
   gTools().Scale(*fUpper,f);
}

////////////////////////////////////////////////////////////////////////////////
/// "scale" the volume by symmetrically blowing up the interval in each dimension

void TMVA::Volume::ScaleInterval( Double_t f )
{
   for (UInt_t ivar=0; ivar<fLower->size(); ivar++) {
      Double_t lo = 0.5*((*fLower)[ivar]*(1.0 + f) + (*fUpper)[ivar]*(1.0 - f));
      Double_t up = 0.5*((*fLower)[ivar]*(1.0 - f) + (*fUpper)[ivar]*(1.0 + f));
      (*fLower)[ivar] = lo;
      (*fUpper)[ivar] = up;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// printout of the volume boundaries

void TMVA::Volume::Print( void ) const
{
   MsgLogger fLogger( "Volume" );
   for (UInt_t ivar=0; ivar<fLower->size(); ivar++) {
      fLogger << kINFO << "... Volume: var: " << ivar << "\t(fLower, fUpper) = ("
              << (*fLower)[ivar] << "\t " << (*fUpper)[ivar] <<")"<< Endl;
   }
}

