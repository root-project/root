// $Id: TFoamVect.cxx,v 1.2 2005/04/04 10:59:34 psawicki Exp $

/////////////////////////////////////////////////////////////////////////////
//                                                                         //
// Auxiliary class TFoamVect of n-dimensional vector, with dynamic allocation //
// used for the cartesian geometry of the TFoam  cells                     //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////

#include"Riostream.h"
#include "TFoamVect.h"

#define SW2 setprecision(7) << setw(12)

ClassImp(TFoamVect);

//_____________________________________________________________________________
TFoamVect::TFoamVect(){
// Default constructor for streamer

  fDim    =0;
  fCoords =NULL;
  fNext   =NULL;
  fPrev   =NULL;
}

//______________________________________________________________________________
TFoamVect::TFoamVect(const Int_t n){
// User constructor creating n-densional vector
// and allocating dynamicaly array of components

  Int_t i;
  fNext=NULL;
  fPrev=NULL;
  fDim=n;
  fCoords = NULL;
  if (n>0){
    fCoords = new Double_t[fDim];
    if(fchat) {
      if(fCoords == NULL)
	Error("TFoamVect", "Constructor failed to allocate\n");
    }
    for (i=0; i<n; i++) *(fCoords+i)=0.0;
  }
  if(fchat) Info("TFoamVect", "USER CONSTRUCTOR TFoamVect(const Int_t)\n ");
}

//___________________________________________________________________________
TFoamVect::TFoamVect(const TFoamVect &Vect): TObject(Vect)
{
// Copy constructor

  fNext=NULL;
  fPrev=NULL;
  fDim=Vect.fDim;
  fCoords = NULL;
  if(fDim>0)  fCoords = new Double_t[fDim];
  if(fchat) {
    if(fCoords == NULL){ 
      Error("TFoamVect", "Constructor failed to allocate fCoords\n"); 
    }
  }
  for(Int_t i=0; i<fDim; i++)
    fCoords[i] = Vect.fCoords[i];
  Error("TFoamVect","+++++ NEVER USE Copy constructor !!!!!\n ");

}

//___________________________________________________________________________
TFoamVect::~TFoamVect(){
// Destructor
  if(fchat) Info("TFoamVect"," DESTRUCTOR TFoamVect~ \n");
  delete [] fCoords; //  free(fCoords)
  fCoords=NULL;
}


//////////////////////////////////////////////////////////////////////////////
//                     Overloading operators                                //
//////////////////////////////////////////////////////////////////////////////

//____________________________________________________________________________
TFoamVect& TFoamVect::operator =(const TFoamVect& Vect){
// substitution operator

  Int_t i;
  if (&Vect == this) return *this;
  if( fDim != Vect.fDim )
    Error("TFoamVect","operator=Dims. are different: %d and %d \n ",fDim,Vect.fDim);
  if( fDim != Vect.fDim ){  // cleanup
    delete [] fCoords;
    fCoords = new Double_t[fDim];
  }
  fDim=Vect.fDim;
  for(i=0; i<fDim; i++)
    fCoords[i] = Vect.fCoords[i];
  fNext=Vect.fNext;
  fPrev=Vect.fPrev;
  if(fchat)  Info("TFoamVect", "SUBSITUTE operator =\n ");
  return *this;
}

//______________________________________________________________________
Double_t &TFoamVect::operator[](Int_t n){
// [] is for access to elements as in ordinary matrix like a[j]=b[j]
// (Perhaps against some strict rules but rather practical.)
// Range protection is built in, consequently for substitution
// one should use rather use a=b than explicit loop!

  if ((n<0) || (n>=fDim)){
    Error( "TFoamVect","operator[], out of range \n");
  }
  return fCoords[n];
}

//______________________________________________________________________
TFoamVect& TFoamVect::operator*=(const Double_t &x){
// unary multiplication operator *=

  for(Int_t i=0;i<fDim;i++) 
    fCoords[i] = fCoords[i]*x;
  return *this;
}

//_______________________________________________________________________
TFoamVect& TFoamVect::operator+=(const TFoamVect& Shift){
// unary addition operator +=; adding vector c*=x,
  if( fDim != Shift.fDim){
    Error( "TFoamVect","operator+, different dimensions= %d %d \n",fDim,Shift.fDim);
  }
  for(Int_t i=0;i<fDim;i++) 
    fCoords[i] = fCoords[i]+Shift.fCoords[i];
  return *this;
}

//________________________________________________________________________
TFoamVect& TFoamVect::operator-=(const TFoamVect& Shift){
// unary subtraction operator -=
  if( fDim != Shift.fDim){
    Error( "TFoamVect","operator+, different dimensions= %d %d \n",fDim,Shift.fDim);
  }
  for(Int_t i=0;i<fDim;i++) 
    fCoords[i] = fCoords[i]-Shift.fCoords[i];
  return *this;
}

//_________________________________________________________________________
TFoamVect TFoamVect::operator+(const TFoamVect &p2){
// addition operator +; sum of 2 vectors: c=a+b, a=a+b,
// NEVER USE IT, VERY SLOW!!!
  TFoamVect Temp(fDim);
  Temp  = (*this);
  Temp += p2;
  return Temp;
}

//__________________________________________________________________________
TFoamVect TFoamVect::operator-(const TFoamVect &p2){
// subtraction operator -; difference of 2 vectors; c=a-b, a=a-b, 
// NEVER USE IT, VERY SLOW!!!
  TFoamVect Temp(fDim);
  Temp  = (*this);
  Temp -= p2;
  return Temp;
}

//___________________________________________________________________________
TFoamVect& TFoamVect::operator =(Double_t Vect[]){
// Loading in ordinary double prec. vector, sometimes can be useful
  Int_t i;
  for(i=0; i<fDim; i++)
    fCoords[i] = Vect[i];
  return *this;
}

//____________________________________________________________________________
TFoamVect& TFoamVect::operator =(Double_t x){
// Loading in double prec. number, sometimes can be useful
  if(fCoords != NULL){
    for(Int_t i=0; i<fDim; i++)
      fCoords[i] = x;
  }
  return *this;
}
//////////////////////////////////////////////////////////////////////////////
//                          OTHER METHODS                                   //
//////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________________
void TFoamVect::PrintCoord(){
// Printout of all vector components on "cout"
  Int_t i;
  cout << "(";
  for(i=0; i<fDim-1; i++) cout  << SW2 << *(fCoords+i) << ",";
  cout  << SW2 << *(fCoords+fDim-1);
  cout << ")";
}
//______________________________________________________________________________
void TFoamVect::PrintList(void){
// Printout of all member vectors in the list starting from "this"
  Long_t i=0;
  if(this == NULL) return;
  TFoamVect *current=this;
  while(current != NULL){
    cout<<"vec["<<i<<"]=";
    current->PrintCoord();
    cout<<endl;
    current = current->fNext;
    i++;
  }
}

//______________________________________________________________________________
const Int_t &TFoamVect::GetDim(void){
// Getter returning vector dimension
  return fDim;
}
///////////////////////////////////////////////////////////////////////////////
//                End of Class TFoamVect                                        //
///////////////////////////////////////////////////////////////////////////////
