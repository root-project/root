// @(#)root/foam:$Name:  $:$Id: TFoamCell.cxx,v 1.2 2005/04/08 16:29:48 rdm Exp $
// Authors: S. Jadach and P.Sawicki

//_________________________________________________________________________________
//
// Class TFoamCell  used in TFoam
// ==============================
// Objects of this class are hyper-rectangular cells organized in the binary tree.
// Special algoritm for encoding relative positioning of the cells
// allow to save total memory allocaction needed for the system of cells.
//
//_________________________________________________________________________________

#include "Riostream.h"
#include "TFoamCell.h"
#include "TFoamVect.h"


ClassImp(TFoamCell);

//________________________________________________________________________________
TFoamCell::TFoamCell()
{
// Default constructor for streamer

  fParent  = 0;
  fDaught0 = 0;
  fDaught1 = 0;
}

//_________________________________________________________________________________
TFoamCell::TFoamCell(Int_t kDim)
{
// User constructor allocating single empty Cell
  if (  kDim >0){
    //---------=========----------
    fkDim     = kDim;
    fStatus   = 1;
    fParent   = 0;
    fDaught0  = 0;
    fDaught1  = 0;
    fXdiv     = 0.0;
    fBest     = 0;
    fVolume   = 0.0;
    fIntegral = 0.0;
    fDrive    = 0.0;
    fPrimary  = 0.0;
  } else
    Error("TFoamCell","Dimension has to be >0 \n ");
}

//_________________________________________________________________________________
TFoamCell::TFoamCell(TFoamCell &From): TObject(From)
{
// Copy constructor (not tested!)

  Error("TFoamCell", "+++++ NEVER USE Copy constructor for TFoamCell \n");
  fStatus      = From.fStatus;
  fParent      = From.fParent;
  fDaught0     = From.fDaught0;
  fDaught1     = From.fDaught1;
  fXdiv        = From.fXdiv;
  fBest        = From.fBest;
  fVolume      = From.fVolume;
  fIntegral    = From.fIntegral;
  fDrive       = From.fDrive;
  fPrimary     = From.fPrimary;
}

//___________________________________________________________________________________
TFoamCell::~TFoamCell()
{
// Destructor
}

//___________________________________________________________________________________
TFoamCell& TFoamCell::operator=(TFoamCell &From)
{
// Substitution operator = (never used)

  Info("TFoamCell", "operator=\n ");
  if (&From == this) return *this;
  fStatus      = From.fStatus;
  fParent      = From.fParent;
  fDaught0     = From.fDaught0;
  fDaught1     = From.fDaught1;
  fXdiv        = From.fXdiv;
  fBest        = From.fBest;
  fVolume      = From.fVolume;
  fIntegral    = From.fIntegral;
  fDrive       = From.fDrive;
  fPrimary     = From.fPrimary;
  return *this;
}


//___________________________________________________________________________________
void TFoamCell::Fill(Int_t Status, TFoamCell *Parent, TFoamCell *Daugh1, TFoamCell *Daugh2)
{
// Fills in certain data into newly allocated cell

  fStatus  = Status;
  fParent  = Parent;
  fDaught0 = Daugh1;
  fDaught1 = Daugh2;
}

////////////////////////////////////////////////////////////////////////////////
//              GETTERS/SETTERS
////////////////////////////////////////////////////////////////////////////////

//_____________________________________________________________________________________
void    TFoamCell::GetHcub( TFoamVect &Posi, TFoamVect &Size)
{
// Provides size and position of the cell
// These parameter are calculated by analysing information in all parents
// cells up to the root cell. It takes time but saves memory.
  if(fkDim<1) return;
    TFoamCell *pCell,*dCell;
    Posi = 0.0; Size=1.0; // load all components
    dCell = this;
    while(dCell != 0){
      pCell = dCell->GetPare();
      if( pCell== 0) break;
      Int_t    kDiv = pCell->fBest;
      Double_t xDivi = pCell->fXdiv;
        if(         dCell == pCell->GetDau0()  ){
          Size[kDiv]=Size[kDiv]*xDivi;
          Posi[kDiv]=Posi[kDiv]*xDivi;
        }else if(   dCell == pCell->GetDau1()  ){
          Size[kDiv]=Size[kDiv]*(1.0-xDivi);
          Posi[kDiv]=Posi[kDiv]*(1.0-xDivi)+xDivi;
        }else{
          Error("GetHcub ","Something wrong with linked tree \n");
        }
      dCell=pCell;
    }//while
}//GetHcub

//______________________________________________________________________________________
void    TFoamCell::GetHSize( TFoamVect &Size)
{
// Provides size of the cell
// Size parameters are calculated by analysing information in all parents
// cells up to the root cell. It takes time but saves memory.
  if(fkDim<1) return;
    TFoamCell *pCell,*dCell;
    Size=1.0; // load all components
    dCell = this;
    while(dCell != 0){
      pCell = dCell->GetPare();
      if( pCell== 0) break;
      Int_t    kDiv = pCell->fBest;
      Double_t xDivi = pCell->fXdiv;
        if(        dCell == pCell->GetDau0() ){
          Size[kDiv]=Size[kDiv]*xDivi;
        }else if(  dCell == pCell->GetDau1()  ){
          Size[kDiv]=Size[kDiv]*(1.0-xDivi);
        }else{
          Error("GetHSize ","Something wrong with linked tree \n");
        }
      dCell=pCell;
    }//while
}//GetHSize

//_________________________________________________________________________________________
void TFoamCell::CalcVolume(void)
{
// Calculates volume of the cell using size params which are calculated

  Int_t k;
  Double_t volu=1.0;
  if(fkDim>0){         // h-cubical subspace
      TFoamVect Size(fkDim);
      GetHSize(Size);
      for(k=0; k<fkDim; k++) volu *= Size[k];
  }
  fVolume =volu;
}

//__________________________________________________________________________________________
void TFoamCell::PrintContent()
{
// Printout of the cell geometry parameters for the debug purpose

  cout <<  " Status= "<<     fStatus   <<",";
  cout <<  " Volume= "<<     fVolume   <<",";
  cout <<  " TrueInteg= " << fIntegral <<",";
  cout <<  " DriveInteg= "<< fDrive    <<",";
  cout <<  " PrimInteg= " << fPrimary  <<",";
  cout<< endl;
  cout <<  " Xdiv= "<<fXdiv<<",";
  cout <<  " Best= "<<fBest<<",";
  cout <<  " Parent=  {"<< (GetPare() ? GetPare()->GetSerial() : -1) <<"} "; // extra DEBUG
  cout <<  " Daught0= {"<< (GetDau0() ? GetDau0()->GetSerial() : -1 )<<"} "; // extra DEBUG
  cout <<  " Daught1= {"<< (GetDau1() ? GetDau1()->GetSerial()  : -1 )<<"} "; // extra DEBUG
  cout<< endl;
  //
  //
  if(fkDim>0 ){
    TFoamVect Posi(fkDim); TFoamVect Size(fkDim);
    (this)->GetHcub(Posi,Size);
    cout <<"   Posi= "; Posi.PrintCoord(); cout<<","<< endl;
    cout <<"   Size= "; Size.PrintCoord(); cout<<","<< endl;
  }
}
///////////////////////////////////////////////////////////////////
//        End of  class  TFoamCell                                  //
///////////////////////////////////////////////////////////////////
