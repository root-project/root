// @(#)root/foam:$Id$
// Author: S. Jadach <mailto:Stanislaw.jadach@ifj.edu.pl>, P.Sawicki <mailto:Pawel.Sawicki@ifj.edu.pl>

//_____________________________________________________________________________
//                                                                            //
// Auxiliary class TFoamVect of n-dimensional vector, with dynamic allocation //
// used for the cartesian geometry of the TFoam  cells                        //
//                                                                            //
//_____________________________________________________________________________

#include "Riostream.h"
#include "TSystem.h"
#include "TFoamVect.h"


ClassImp(TFoamVect);

//_____________________________________________________________________________
TFoamVect::TFoamVect()
{
// Default constructor for streamer

   fDim    =0;
   fCoords =0;
   fNext   =0;
   fPrev   =0;
}

//______________________________________________________________________________
TFoamVect::TFoamVect(Int_t n)
{
// User constructor creating n-dimensional vector
// and allocating dynamically array of components

   Int_t i;
   fNext=0;
   fPrev=0;
   fDim=n;
   fCoords = 0;
   if (n>0) {
      fCoords = new Double_t[fDim];
      if(gDebug) {
         if(fCoords == 0)
            Error("TFoamVect", "Constructor failed to allocate\n");
      }
      for (i=0; i<n; i++) *(fCoords+i)=0.0;
   }
   if(gDebug) Info("TFoamVect", "USER CONSTRUCTOR TFoamVect(const Int_t)\n ");
}

//___________________________________________________________________________
TFoamVect::TFoamVect(const TFoamVect &Vect): TObject(Vect)
{
// Copy constructor

   fNext=0;
   fPrev=0;
   fDim=Vect.fDim;
   fCoords = 0;
   if(fDim>0)  fCoords = new Double_t[fDim];
   if(gDebug) {
      if(fCoords == 0) {
         Error("TFoamVect", "Constructor failed to allocate fCoords\n");
      }
   }
   for(Int_t i=0; i<fDim; i++)
      fCoords[i] = Vect.fCoords[i];
   Error("TFoamVect","+++++ NEVER USE Copy constructor !!!!!\n ");
}

//___________________________________________________________________________
TFoamVect::~TFoamVect()
{
// Destructor
   if(gDebug) Info("TFoamVect"," DESTRUCTOR TFoamVect~ \n");
   delete [] fCoords; //  free(fCoords)
   fCoords=0;
}


//////////////////////////////////////////////////////////////////////////////
//                     Overloading operators                                //
//////////////////////////////////////////////////////////////////////////////

//____________________________________________________________________________
TFoamVect& TFoamVect::operator =(const TFoamVect& Vect)
{
// substitution operator

   Int_t i;
   if (&Vect == this) return *this;
   if( fDim != Vect.fDim )
      Error("TFoamVect","operator=Dims. are different: %d and %d \n ",fDim,Vect.fDim);
   if( fDim != Vect.fDim ) {  // cleanup
      delete [] fCoords;
      fCoords = new Double_t[fDim];
   }
   fDim=Vect.fDim;
   for(i=0; i<fDim; i++)
      fCoords[i] = Vect.fCoords[i];
   fNext=Vect.fNext;
   fPrev=Vect.fPrev;
   if(gDebug)  Info("TFoamVect", "SUBSITUTE operator =\n ");
   return *this;
}

//______________________________________________________________________
Double_t &TFoamVect::operator[](Int_t n)
{
// [] is for access to elements as in ordinary matrix like a[j]=b[j]
// (Perhaps against some strict rules but rather practical.)
// Range protection is built in, consequently for substitution
// one should use rather use a=b than explicit loop!

   if ((n<0) || (n>=fDim)) {
      Error( "TFoamVect","operator[], out of range \n");
   }
   return fCoords[n];
}

//______________________________________________________________________
TFoamVect& TFoamVect::operator*=(const Double_t &x)
{
// unary multiplication operator *=

   for(Int_t i=0;i<fDim;i++)
      fCoords[i] = fCoords[i]*x;
   return *this;
}

//_______________________________________________________________________
TFoamVect& TFoamVect::operator+=(const TFoamVect& Shift)
{
// unary addition operator +=; adding vector c*=x,
   if( fDim != Shift.fDim){
      Error( "TFoamVect","operator+, different dimensions= %d %d \n",fDim,Shift.fDim);
   }
   for(Int_t i=0;i<fDim;i++)
      fCoords[i] = fCoords[i]+Shift.fCoords[i];
   return *this;
}

//________________________________________________________________________
TFoamVect& TFoamVect::operator-=(const TFoamVect& Shift)
{
// unary subtraction operator -=
   if( fDim != Shift.fDim) {
      Error( "TFoamVect","operator+, different dimensions= %d %d \n",fDim,Shift.fDim);
   }
   for(Int_t i=0;i<fDim;i++)
      fCoords[i] = fCoords[i]-Shift.fCoords[i];
   return *this;
}

//_________________________________________________________________________
TFoamVect TFoamVect::operator+(const TFoamVect &p2)
{
// addition operator +; sum of 2 vectors: c=a+b, a=a+b,
// NEVER USE IT, VERY SLOW!!!
   TFoamVect temp(fDim);
   temp  = (*this);
   temp += p2;
   return temp;
}

//__________________________________________________________________________
TFoamVect TFoamVect::operator-(const TFoamVect &p2)
{
// subtraction operator -; difference of 2 vectors; c=a-b, a=a-b,
// NEVER USE IT, VERY SLOW!!!
   TFoamVect temp(fDim);
   temp  = (*this);
   temp -= p2;
   return temp;
}

//___________________________________________________________________________
TFoamVect& TFoamVect::operator =(Double_t Vect[])
{
// Loading in ordinary double prec. vector, sometimes can be useful
   Int_t i;
   for(i=0; i<fDim; i++)
      fCoords[i] = Vect[i];
   return *this;
}

//____________________________________________________________________________
TFoamVect& TFoamVect::operator =(Double_t x)
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

//_____________________________________________________________________________
void TFoamVect::Print(Option_t *option) const
{
// Printout of all vector components on "cout"
   if(!option) Error("Print ", "No option set \n");
   Int_t i;
   Int_t pr = cout.precision(7); 
   cout << "(";
   for(i=0; i<fDim-1; i++) cout  << setw(12) << *(fCoords+i) << ",";
   cout  << setw(12) << *(fCoords+fDim-1);
   cout << ")";
   cout.precision(pr);
}
//______________________________________________________________________________
void TFoamVect::PrintList(void)
{
// Printout of all member vectors in the list starting from "this"
   Long_t i=0;
   if(this == 0) return;
   TFoamVect *current=this;
   while(current != 0) {
      cout<<"vec["<<i<<"]=";
      current->Print("1");
      cout<<endl;
      current = current->fNext;
      i++;
   }
}

///////////////////////////////////////////////////////////////////////////////
//                End of Class TFoamVect                                        //
///////////////////////////////////////////////////////////////////////////////
