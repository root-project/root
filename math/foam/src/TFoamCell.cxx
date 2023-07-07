// @(#)root/foam:$Id$
// Author: S. Jadach <mailto:Stanislaw.jadach@ifj.edu.pl>, P.Sawicki <mailto:Pawel.Sawicki@ifj.edu.pl>

/** \class TFoamCell

Used by TFoam

Objects of this class are hyper-rectangular cells organized in the binary tree.
Special algorithm for encoding relative positioning of the cells
allow to save total memory allocation needed for the system of cells.
*/


#include <iostream>
#include "TFoamCell.h"
#include "TFoamVect.h"


ClassImp(TFoamCell);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor for streamer

TFoamCell::TFoamCell()
{
}

////////////////////////////////////////////////////////////////////////////////
/// User constructor allocating single empty Cell

TFoamCell::TFoamCell(Int_t kDim)
{
   if (  kDim >0) {
      //---------=========----------
      fDim     = kDim;
      fSerial   = 0;
      fStatus   = 1;
      fXdiv     = 0.0;
      fBest     = 0;
      fVolume   = 0.0;
      fIntegral = 0.0;
      fDrive    = 0.0;
      fPrimary  = 0.0;
   } else
      Error("TFoamCell","Dimension has to be >0 \n ");
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TFoamCell::~TFoamCell()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Fills in certain data into newly allocated cell

void TFoamCell::Fill(Int_t Status, TFoamCell *Parent, TFoamCell *Daugh1, TFoamCell *Daugh2)
{
   fStatus  = Status;
   fParentIdx  = Parent ? Parent->fSerial : -1;
   fDaught0Idx = Daugh1 ? Daugh1->fSerial : -1;
   fDaught1Idx = Daugh2 ? Daugh2->fSerial : -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Provides size and position of the cell
/// These parameter are calculated by analyzing information in all parents
/// cells up to the root cell. It takes time but saves memory.

void    TFoamCell::GetHcub( TFoamVect &cellPosi, TFoamVect &cellSize)  const
{
   if(fDim<1) return;
   const TFoamCell *pCell,*dCell;
   cellPosi = 0.0; cellSize=1.0; // load all components
   dCell = this;
   while(dCell != nullptr) {
      pCell = dCell->GetPare();
      if( pCell== nullptr) break;
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
}

////////////////////////////////////////////////////////////////////////////////
/// Provides size of the cell
/// Size parameters are calculated by analyzing information in all parents
/// cells up to the root cell. It takes time but saves memory.

void    TFoamCell::GetHSize( TFoamVect &cellSize)  const
{
   if(fDim<1) return;
   const TFoamCell *pCell,*dCell;
   cellSize=1.0; // load all components
   dCell = this;
   while(dCell != nullptr) {
      pCell = dCell->GetPare();
      if( pCell== nullptr) break;
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
}

////////////////////////////////////////////////////////////////////////////////
/// Calculates volume of the cell using size params which are calculated

void TFoamCell::CalcVolume(void)
{
   Int_t k;
   Double_t volu=1.0;
   if(fDim>0) {         // h-cubical subspace
      TFoamVect cellSize(fDim);
      GetHSize(cellSize);
      for(k=0; k<fDim; k++) volu *= cellSize[k];
   }
   fVolume =volu;
}

////////////////////////////////////////////////////////////////////////////////
/// Printout of the cell geometry parameters for the debug purpose

void TFoamCell::Print(Option_t *option) const
{
   if(!option) Error("Print", "No option set\n");

   std::cout <<  " Status= "<<     fStatus   <<",";
   std::cout <<  " Volume= "<<     fVolume   <<",";
   std::cout <<  " TrueInteg= " << fIntegral <<",";
   std::cout <<  " DriveInteg= "<< fDrive    <<",";
   std::cout <<  " PrimInteg= " << fPrimary  <<",";
   std::cout<< std::endl;
   std::cout <<  " Xdiv= "<<fXdiv<<",";
   std::cout <<  " Best= "<<fBest<<",";
   std::cout <<  " Parent=  {"<< (GetPare() ? GetPare()->GetSerial() : -1) <<"} "; // extra DEBUG
   std::cout <<  " Daught0= {"<< (GetDau0() ? GetDau0()->GetSerial() : -1 )<<"} "; // extra DEBUG
   std::cout <<  " Daught1= {"<< (GetDau1() ? GetDau1()->GetSerial()  : -1 )<<"} "; // extra DEBUG
   std::cout<< std::endl;
   //
   //
   if(fDim>0 ) {
      TFoamVect cellPosi(fDim); TFoamVect cellSize(fDim);
      GetHcub(cellPosi,cellSize);
      std::cout <<"   Posi= "; cellPosi.Print("1"); std::cout<<","<< std::endl;
      std::cout <<"   Size= "; cellSize.Print("1"); std::cout<<","<< std::endl;
   }
}
