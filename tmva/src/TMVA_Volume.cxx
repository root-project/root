// @(#)root/tmva $Id: TMVA_Volume.cpp,v 1.10 2006/05/03 19:45:38 helgevoss Exp $
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_Volume                                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
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
 * File and Version Information:                                                  *
 * $Id: TMVA_Volume.cpp,v 1.10 2006/05/03 19:45:38 helgevoss Exp $        
 **********************************************************************************/

#include "TMVA_Volume.h"
#include "TMVA_Tools.h"
#include "Riostream.h"
#include <stdexcept>

//_______________________________________________________________________
//                                                                      
// Volume for BinarySearchTree                                          
//                                                                      
//_______________________________________________________________________

TMVA_Volume::TMVA_Volume( std::vector<Double_t>* l, std::vector<Double_t>* u ) 
  : Lower( l ), 
    Upper( u ),
    fOwnerShip (kFALSE)
{}

TMVA_Volume::TMVA_Volume( std::vector<Float_t>* l, std::vector<Float_t>* u ) 
{
  Lower = new std::vector<Double_t>( l->size() );
  Upper = new std::vector<Double_t>( u->size() );
  fOwnerShip = kTRUE;
  
  for (UInt_t ivar=0; ivar<l->size(); ivar++) {
    (*Lower)[ivar] = Double_t((*l)[ivar]);
    (*Upper)[ivar] = Double_t((*u)[ivar]);
  }  
}

TMVA_Volume::TMVA_Volume( Double_t* l, Double_t* u, Int_t nvar ) 
{
  Lower = new std::vector<Double_t>( nvar );
  Upper = new std::vector<Double_t>( nvar );
  fOwnerShip = kTRUE;

  for (int ivar=0; ivar<nvar; ivar++) {
    (*Lower)[ivar] = l[ivar];
    (*Upper)[ivar] = u[ivar];
  }  
}

TMVA_Volume::TMVA_Volume( Float_t* l, Float_t* u, Int_t nvar ) 
{
  Lower = new std::vector<Double_t>( nvar );
  Upper = new std::vector<Double_t>( nvar );
  fOwnerShip = kTRUE;

  for (int ivar=0; ivar<nvar; ivar++) {
    (*Lower)[ivar] = Double_t(l[ivar]);
    (*Upper)[ivar] = Double_t(u[ivar]);
  }  
}

TMVA_Volume::TMVA_Volume( Double_t l, Double_t u ) 
{
  Lower = new std::vector<Double_t>(1);
  Upper = new std::vector<Double_t>(1);
  fOwnerShip = kTRUE;
  (*Lower)[0] = l;
  (*Upper)[0] = u;
}

TMVA_Volume::TMVA_Volume( Float_t l, Float_t u ) 
{
  Lower = new std::vector<Double_t>(1);
  Upper = new std::vector<Double_t>(1);
  fOwnerShip = kTRUE;
  (*Lower)[0] = Double_t(l);
  (*Upper)[0] = Double_t(u);
}

// copy constructor
TMVA_Volume::TMVA_Volume( TMVA_Volume& V ) 
{ 
  Lower = new std::vector<Double_t>( *V.Lower );
  Upper = new std::vector<Double_t>( *V.Upper );  
  fOwnerShip = kTRUE;
}

// don't delete the input vectors, if not explicitly requested
TMVA_Volume::~TMVA_Volume( void )
{
  if (fOwnerShip) this->Delete();
} 

void TMVA_Volume::Delete( void )
{
  if (NULL != Lower) { delete Lower; Lower = NULL; }
  if (NULL != Upper) { delete Upper; Upper = NULL; }
}

void TMVA_Volume::Scale( Double_t f )
{
  TMVA_Tools::Scale(*Lower,f);
  TMVA_Tools::Scale(*Upper,f);
}

void TMVA_Volume::ScaleInterval( Double_t f ) 
{ 
  for (UInt_t ivar=0; ivar<Lower->size(); ivar++) {
    Double_t lo = 0.5*((*Lower)[ivar]*(1.0 + f) + (*Upper)[ivar]*(1.0 - f));
    Double_t up = 0.5*((*Lower)[ivar]*(1.0 - f) + (*Upper)[ivar]*(1.0 + f));
    (*Lower)[ivar] = lo;
    (*Upper)[ivar] = up;
  }
}

void TMVA_Volume::Print( void ) const 
{
  for (UInt_t ivar=0; ivar<Lower->size(); ivar++) 
    cout << "... TMVA_Volume: var: " << ivar << "\t(Lower, Upper) = (" 
	 << (*Lower)[ivar] << "\t " << (*Upper)[ivar] <<")"<< endl;
}

