// @(#)root/tmva $Id$
// Author: S. Jadach, Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamVect                                                           *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Auxiliary class PDEFoamVect of n-dimensional vector, with dynamic         *
 *      allocation used for the cartesian geometry of the PDEFoam cells           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      S. Jadach        - Institute of Nuclear Physics, Cracow, Poland           *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      Alexander Voigt  - TU Dresden, Germany                                    *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::PDEFoamVect
\ingroup TMVA

*/
#include "TMVA/PDEFoamVect.h"

#include "Rtypes.h"
#include "TObject.h"

#include <iostream>
#include <iomanip>

using namespace std;

//#define SW2 std::setw(12)

ClassImp(TMVA::PDEFoamVect);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor for streamer

TMVA::PDEFoamVect::PDEFoamVect()
: TObject(),
   fDim(0),
   fCoords(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// User constructor creating n-dimensional vector
/// and allocating dynamically array of components

TMVA::PDEFoamVect::PDEFoamVect(Int_t n)
   : TObject(),
     fDim(n),
     fCoords(0)
{
   if (n>0) {
      fCoords = new Double_t[fDim];
      for (Int_t i=0; i<n; i++) *(fCoords+i)=0.0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TMVA::PDEFoamVect::PDEFoamVect(const PDEFoamVect &vect)
   : TObject(),
     fDim(vect.fDim),
     fCoords(vect.fCoords)
{
   Error( "PDEFoamVect", "COPY CONSTRUCTOR NOT IMPLEMENTED" );
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TMVA::PDEFoamVect::~PDEFoamVect()
{
   delete [] fCoords; //  free(fCoords)
   fCoords=0;
}

//////////////////////////////////////////////////////////////////////////////
//                     Overloading operators                                //
//////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// substitution operator

TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator =(const PDEFoamVect& vect)
{
   if (&vect == this) return *this;
   if (fDim != vect.fDim)
      Error("PDEFoamVect", "operator=Dims. are different: %d and %d \n ", fDim, vect.fDim);
   if (fDim != vect.fDim) {  // cleanup
      delete [] fCoords;
      fCoords = new Double_t[fDim];
   }
   fDim = vect.fDim;
   for(Int_t i=0; i<fDim; i++)
      fCoords[i] = vect.fCoords[i];
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// [] is for access to elements as in ordinary matrix like a[j]=b[j]
/// (Perhaps against some strict rules but rather practical.)
/// Range protection is built in, consequently for substitution
/// one should use rather use a=b than explicit loop!

Double_t &TMVA::PDEFoamVect::operator[](Int_t n)
{
   if ((n<0) || (n>=fDim)) {
      Error(  "PDEFoamVect","operator[], out of range \n");
   }
   return fCoords[n];
}

////////////////////////////////////////////////////////////////////////////////
/// unary multiplication operator *=

TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator*=(const Double_t &x)
{
   for(Int_t i=0;i<fDim;i++)
      fCoords[i] = fCoords[i]*x;
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// unary addition operator +=; adding vector c*=x,

TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator+=(const PDEFoamVect& shift)
{
   if(fDim != shift.fDim){
      Error("PDEFoamVect", "operator+, different dimensions= %d %d \n", fDim, shift.fDim);
   }
   for(Int_t i=0;i<fDim;i++)
      fCoords[i] = fCoords[i] + shift.fCoords[i];
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// unary subtraction operator -=

TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator-=(const PDEFoamVect& shift)
{
   if(fDim != shift.fDim) {
      Error("PDEFoamVect", "operator+, different dimensions= %d %d \n", fDim, shift.fDim);
   }
   for(Int_t i=0;i<fDim;i++)
      fCoords[i] = fCoords[i] - shift.fCoords[i];
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// addition operator +; sum of 2 vectors: c=a+b, a=a+b,
/// NEVER USE IT, VERY SLOW!!!

TMVA::PDEFoamVect TMVA::PDEFoamVect::operator+(const PDEFoamVect &p2)
{
   PDEFoamVect temp(fDim);
   temp  = (*this);
   temp += p2;
   return temp;
}

////////////////////////////////////////////////////////////////////////////////
/// subtraction operator -; difference of 2 vectors; c=a-b, a=a-b,
/// NEVER USE IT, VERY SLOW!!!

TMVA::PDEFoamVect TMVA::PDEFoamVect::operator-(const PDEFoamVect &p2)
{
   PDEFoamVect temp(fDim);
   temp  = (*this);
   temp -= p2;
   return temp;
}

////////////////////////////////////////////////////////////////////////////////
/// Loading in ordinary double prec. vector, sometimes can be useful

TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator =(Double_t Vect[])
{
   for(Int_t i=0; i<fDim; i++)
      fCoords[i] = Vect[i];
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Loading in double prec. number, sometimes can be useful

TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator =(Double_t x)
{
   if(fCoords != 0) {
      for(Int_t i=0; i<fDim; i++)
         fCoords[i] = x;
   }
   return *this;
}

//////////////////////////////////////////////////////////////////////////////
//                          OTHER METHODS                                   //
//////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Printout of all vector components

void TMVA::PDEFoamVect::Print(Option_t *option) const
{
   streamsize wid = std::cout.width(); // saving current field width
   if(!option) Error( "Print ", "No option set \n");
   std::cout << "(";
   for(Int_t i=0; i<fDim-1; i++)
      std::cout << std::setw(12) << *(fCoords+i) << ",";
   std::cout << std::setw(12) << *(fCoords+fDim-1);
   std::cout << ")";
   std::cout.width(wid);
}
