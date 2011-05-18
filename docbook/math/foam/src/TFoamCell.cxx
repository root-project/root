// @(#)root/foam:$Id$
// Author: S. Jadach <mailto:Stanislaw.jadach@ifj.edu.pl>, P.Sawicki <mailto:Pawel.Sawicki@ifj.edu.pl>

//_________________________________________________________________________________
//
// Class TFoamCell  used in TFoam
// ==============================
// Objects of this class are hyper-rectangular cells organized in the binary tree.
// Special algorithm for encoding relative positioning of the cells
// allow to save total memory allocation needed for the system of cells.
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
   if (  kDim >0) {
      //---------=========----------
      fDim     = kDim;
      fSerial   = 0;
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
TFoamCell& TFoamCell::operator=(const TFoamCell &From)
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
void    TFoamCell::GetHcub( TFoamVect &cellPosi, TFoamVect &cellSize)  const
{
// Provides size and position of the cell
// These parameter are calculated by analyzing information in all parents
// cells up to the root cell. It takes time but saves memory.
   if(fDim<1) return;
   const TFoamCell *pCell,*dCell;
   cellPosi = 0.0; cellSize=1.0; // load all components
   dCell = this;
   while(dCell != 0) {
      pCell = dCell->GetPare();
      if( pCell== 0) break;
      Int_t    kDiv = pCell->fBest;
      Double_t xDivi = pCell->fXdiv;
      if(dCell == pCell->GetDau0()  ) {
         cellSize[kDiv] *=xDivi;
         cellPosi[kDiv] *=xDivi;
      } else if(   dCell == pCell->GetDau1()  ) {
         cellSize[kDiv] *=(1.0-xDivi);
         cellPosi[kDiv]  =cellPosi[kDiv]*(1.0-xDivi)+xDivi;
      } else {
         Error("GetHcub ","Something wrong with linked tree \n");
      }
      dCell=pCell;
   }//while
}//GetHcub

//______________________________________________________________________________________
void    TFoamCell::GetHSize( TFoamVect &cellSize)  const
{
// Provides size of the cell
// Size parameters are calculated by analyzing information in all parents
// cells up to the root cell. It takes time but saves memory.
   if(fDim<1) return;
   const TFoamCell *pCell,*dCell;
   cellSize=1.0; // load all components
   dCell = this;
   while(dCell != 0) {
      pCell = dCell->GetPare();
      if( pCell== 0) break;
      Int_t    kDiv = pCell->fBest;
      Double_t xDivi = pCell->fXdiv;
      if(dCell == pCell->GetDau0() ) {
         cellSize[kDiv]=cellSize[kDiv]*xDivi;
      } else if(dCell == pCell->GetDau1()  ) {
         cellSize[kDiv]=cellSize[kDiv]*(1.0-xDivi);
      } else {
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
   if(fDim>0) {         // h-cubical subspace
      TFoamVect cellSize(fDim);
      GetHSize(cellSize);
      for(k=0; k<fDim; k++) volu *= cellSize[k];
   }
   fVolume =volu;
}

//__________________________________________________________________________________________
void TFoamCell::Print(Option_t *option) const
{
// Printout of the cell geometry parameters for the debug purpose

   if(!option) Error("Print", "No option set\n");

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
   if(fDim>0 ) {
      TFoamVect cellPosi(fDim); TFoamVect cellSize(fDim);
      GetHcub(cellPosi,cellSize);
      cout <<"   Posi= "; cellPosi.Print("1"); cout<<","<< endl;
      cout <<"   Size= "; cellSize.Print("1"); cout<<","<< endl;
   }
}
///////////////////////////////////////////////////////////////////
//        End of  class  TFoamCell                               //
///////////////////////////////////////////////////////////////////
