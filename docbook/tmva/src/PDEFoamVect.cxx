
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
 *      Alexander Voigt  - CERN, Switzerland                                      *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <iostream>
#include <iomanip>

#ifndef ROOT_TMVA_PDEFoamVect
#include "TMVA/PDEFoamVect.h"
#endif

using namespace std;

//#define SW2 std::setw(12)

ClassImp(TMVA::PDEFoamVect)

//_____________________________________________________________________
TMVA::PDEFoamVect::PDEFoamVect()
   : TObject(),
     fDim(0),
     fCoords(0)
{
   // Default constructor for streamer
}

//_____________________________________________________________________
TMVA::PDEFoamVect::PDEFoamVect(Int_t n)
   : TObject(),
     fDim(n),
     fCoords(0)
{
   // User constructor creating n-dimensional vector
   // and allocating dynamically array of components

   if (n>0) {
      fCoords = new Double_t[fDim];
      for (Int_t i=0; i<n; i++) *(fCoords+i)=0.0;
   }
}

//_____________________________________________________________________
TMVA::PDEFoamVect::PDEFoamVect(const PDEFoamVect &vect)
   : TObject(),
     fDim(vect.fDim),
     fCoords(vect.fCoords)
{
   // Copy constructor
   Error( "PDEFoamVect", "COPY CONSTRUCTOR NOT IMPLEMENTED" );
}

//_____________________________________________________________________
TMVA::PDEFoamVect::~PDEFoamVect()
{
   // Destructor
   delete [] fCoords; //  free(fCoords)
   fCoords=0;
}

//////////////////////////////////////////////////////////////////////////////
//                     Overloading operators                                //
//////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________
TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator =(const PDEFoamVect& Vect)
{
   // substitution operator

   if (&Vect == this) return *this;
   if( fDim != Vect.fDim )
      Error( "PDEFoamVect","operator=Dims. are different: %d and %d \n ",fDim,Vect.fDim);
   if( fDim != Vect.fDim ) {  // cleanup
      delete [] fCoords;
      fCoords = new Double_t[fDim];
   }
   fDim=Vect.fDim;
   for(Int_t i=0; i<fDim; i++)
      fCoords[i] = Vect.fCoords[i];
   return *this;
}

//_____________________________________________________________________
Double_t &TMVA::PDEFoamVect::operator[](Int_t n)
{
   // [] is for access to elements as in ordinary matrix like a[j]=b[j]
   // (Perhaps against some strict rules but rather practical.)
   // Range protection is built in, consequently for substitution
   // one should use rather use a=b than explicit loop!

   if ((n<0) || (n>=fDim)) {
      Error(  "PDEFoamVect","operator[], out of range \n");
   }
   return fCoords[n];
}

//_____________________________________________________________________
TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator*=(const Double_t &x)
{
   // unary multiplication operator *=

   for(Int_t i=0;i<fDim;i++)
      fCoords[i] = fCoords[i]*x;
   return *this;
}

//_____________________________________________________________________
TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator+=(const PDEFoamVect& Shift)
{
   // unary addition operator +=; adding vector c*=x,
   if( fDim != Shift.fDim){
      Error(  "PDEFoamVect","operator+, different dimensions= %d %d \n",fDim,Shift.fDim);
   }
   for(Int_t i=0;i<fDim;i++)
      fCoords[i] = fCoords[i]+Shift.fCoords[i];
   return *this;
}

//_____________________________________________________________________
TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator-=(const PDEFoamVect& Shift)
{
   // unary subtraction operator -=
   if( fDim != Shift.fDim) {
      Error(  "PDEFoamVect","operator+, different dimensions= %d %d \n",fDim,Shift.fDim);
   }
   for(Int_t i=0;i<fDim;i++)
      fCoords[i] = fCoords[i]-Shift.fCoords[i];
   return *this;
}

//_____________________________________________________________________
TMVA::PDEFoamVect TMVA::PDEFoamVect::operator+(const PDEFoamVect &p2)
{
   // addition operator +; sum of 2 vectors: c=a+b, a=a+b,
   // NEVER USE IT, VERY SLOW!!!
   PDEFoamVect temp(fDim);
   temp  = (*this);
   temp += p2;
   return temp;
}

//_____________________________________________________________________
TMVA::PDEFoamVect TMVA::PDEFoamVect::operator-(const PDEFoamVect &p2)
{
   // subtraction operator -; difference of 2 vectors; c=a-b, a=a-b,
   // NEVER USE IT, VERY SLOW!!!
   PDEFoamVect temp(fDim);
   temp  = (*this);
   temp -= p2;
   return temp;
}

//_____________________________________________________________________
TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator =(Double_t Vect[])
{
   // Loading in ordinary double prec. vector, sometimes can be useful
   for(Int_t i=0; i<fDim; i++)
      fCoords[i] = Vect[i];
   return *this;
}

//_____________________________________________________________________
TMVA::PDEFoamVect& TMVA::PDEFoamVect::operator =(Double_t x)
{
   // Loading in double prec. number, sometimes can be useful
   if(fCoords != 0) {
      for(Int_t i=0; i<fDim; i++)
         fCoords[i] = x;
   }
   return *this;
}

//////////////////////////////////////////////////////////////////////////////
//                          OTHER METHODS                                   //
//////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________
void TMVA::PDEFoamVect::Print(Option_t *option) const
{
   streamsize wid = cout.width(); // saving current field width
   // Printout of all vector components
   if(!option) Error( "Print ", "No option set \n");
   cout << "(";
   for(Int_t i=0; i<fDim-1; i++) 
      cout << std::setw(12) << *(fCoords+i) << ",";
   cout << std::setw(12) << *(fCoords+fDim-1);
   cout << ")";
   cout.width(wid);
}
