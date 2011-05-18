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

#include <stdexcept>

#include "TMVA/Volume.h"
#include "TMVA/Tools.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

//_______________________________________________________________________
//                                                                      
// Volume                                                               //
//                                                                      //
// Volume for BinarySearchTree                                          //
//                                                                      //
// volume element: variable space beteen upper and lower bonds of       //
// nvar-dimensional variable space                                      //
//_______________________________________________________________________

TMVA::Volume::Volume( std::vector<Double_t>* l, std::vector<Double_t>* u ) 
   : fLower( l ), 
     fUpper( u ),
     fOwnerShip (kFALSE){
   // constructor specifying the volume by std::vectors of doubles
}

TMVA::Volume::Volume( std::vector<Float_t>* l, std::vector<Float_t>* u ) 
{
   // constructor specifying the volume by std::vectors of floats
   fLower = new std::vector<Double_t>( l->size() );
   fUpper = new std::vector<Double_t>( u->size() );
   fOwnerShip = kTRUE;
  
   for (UInt_t ivar=0; ivar<l->size(); ivar++) {
      (*fLower)[ivar] = Double_t((*l)[ivar]);
      (*fUpper)[ivar] = Double_t((*u)[ivar]);
   }  
}

TMVA::Volume::Volume( Double_t* l, Double_t* u, Int_t nvar ) 
{
   // constructor specifiying the volume by c-style arrays of doubles
   fLower = new std::vector<Double_t>( nvar );
   fUpper = new std::vector<Double_t>( nvar );
   fOwnerShip = kTRUE;

   for (int ivar=0; ivar<nvar; ivar++) {
      (*fLower)[ivar] = l[ivar];
      (*fUpper)[ivar] = u[ivar];
   }  
}

TMVA::Volume::Volume( Float_t* l, Float_t* u, Int_t nvar ) 
{
   // constructor specifiying the volume by c-style arrays of floats
   fLower = new std::vector<Double_t>( nvar );
   fUpper = new std::vector<Double_t>( nvar );
   fOwnerShip = kTRUE;

   for (int ivar=0; ivar<nvar; ivar++) {
      (*fLower)[ivar] = Double_t(l[ivar]);
      (*fUpper)[ivar] = Double_t(u[ivar]);
   }  
}

TMVA::Volume::Volume( Double_t l, Double_t u ) 
{
   // simple constructors for 1 dimensional values (double)
   fLower = new std::vector<Double_t>(1);
   fUpper = new std::vector<Double_t>(1);
   fOwnerShip = kTRUE;
   (*fLower)[0] = l;
   (*fUpper)[0] = u;
}

TMVA::Volume::Volume( Float_t l, Float_t u ) 
{
   // simple constructors for 1 dimensional values (float)
   fLower = new std::vector<Double_t>(1);
   fUpper = new std::vector<Double_t>(1);
   fOwnerShip = kTRUE;
   (*fLower)[0] = Double_t(l);
   (*fUpper)[0] = Double_t(u);
}

TMVA::Volume::Volume( Volume& V ) 
{ 
   // copy constructor
   fLower = new std::vector<Double_t>( *V.fLower );
   fUpper = new std::vector<Double_t>( *V.fUpper );  
   fOwnerShip = kTRUE;
}

TMVA::Volume::~Volume( void )
{
   // destructor
   // delete volume boundaries only if owend by the volume
   if (fOwnerShip) this->Delete();
} 

void TMVA::Volume::Delete( void )
{
   // delete array of volume bondaries
   if (NULL != fLower) { delete fLower; fLower = NULL; }
   if (NULL != fUpper) { delete fUpper; fUpper = NULL; }
}

void TMVA::Volume::Scale( Double_t f )
{
   // "scale" the volume by multiplying each upper and lower boundary by "f" 
   gTools().Scale(*fLower,f);
   gTools().Scale(*fUpper,f);
}

void TMVA::Volume::ScaleInterval( Double_t f ) 
{ 
   // "scale" the volume by symmetrically blowing up the interval in each dimension
   for (UInt_t ivar=0; ivar<fLower->size(); ivar++) {
      Double_t lo = 0.5*((*fLower)[ivar]*(1.0 + f) + (*fUpper)[ivar]*(1.0 - f));
      Double_t up = 0.5*((*fLower)[ivar]*(1.0 - f) + (*fUpper)[ivar]*(1.0 + f));
      (*fLower)[ivar] = lo;
      (*fUpper)[ivar] = up;
   }
}

void TMVA::Volume::Print( void ) const 
{
   // printout of the volume boundaries
   MsgLogger fLogger( "Volume" );
   for (UInt_t ivar=0; ivar<fLower->size(); ivar++) {
      fLogger << kINFO << "... Volume: var: " << ivar << "\t(fLower, fUpper) = (" 
              << (*fLower)[ivar] << "\t " << (*fUpper)[ivar] <<")"<< Endl;   
   }
}

