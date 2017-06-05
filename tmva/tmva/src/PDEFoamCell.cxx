// @(#)root/tmva $Id$
// Author: S.Jadach, Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamCell                                                           *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Objects of this class are hyperrectangular cells organized in             *
 *      the binary tree. Special algoritm for encoding relative                   *
 *      positioning of the cells saves total memory allocation needed             *
 *      for the system of cells.                                                  *
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

/*! \class TMVA::PDEFoamCell
\ingroup TMVA

*/
#include "TMVA/PDEFoamCell.h"

#include "TMVA/PDEFoamVect.h"

#include <iostream>
#include <ostream>

#include "Rtypes.h"
#include "TObject.h"
#include "TRef.h"

using namespace std;

ClassImp(TMVA::PDEFoamCell);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor for streamer

TMVA::PDEFoamCell::PDEFoamCell()
: TObject(),
   fDim(0),
   fSerial(0),
   fStatus(1),
   fParent(0),
   fDaught0(0),
   fDaught1(0),
   fXdiv(0.0),
   fBest(0),
   fVolume(0.0),
   fIntegral(0.0),
   fDrive(0.0),
   fElement(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// User constructor allocating single empty Cell

TMVA::PDEFoamCell::PDEFoamCell(Int_t kDim)
   : TObject(),
     fDim(kDim),
     fSerial(0),
     fStatus(1),
     fParent(0),
     fDaught0(0),
     fDaught1(0),
     fXdiv(0.0),
     fBest(0),
     fVolume(0.0),
     fIntegral(0.0),
     fDrive(0.0),
     fElement(0)
{
   if ( kDim <= 0 )
      Error( "PDEFoamCell", "Dimension has to be >0" );
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TMVA::PDEFoamCell::PDEFoamCell(const PDEFoamCell &cell)
   : TObject(),
     fDim     (cell.fDim),
     fSerial  (cell.fSerial),
     fStatus  (cell.fStatus),
     fParent  (cell.fParent),
     fDaught0 (cell.fDaught0),
     fDaught1 (cell.fDaught1),
     fXdiv    (cell.fXdiv),
     fBest    (cell.fBest),
     fVolume  (cell.fVolume),
     fIntegral(cell.fIntegral),
     fDrive   (cell.fDrive),
     fElement (cell.fElement)
{
   Error( "PDEFoamCell", "COPY CONSTRUCTOR NOT IMPLEMENTED" );
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TMVA::PDEFoamCell::~PDEFoamCell()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Fills in certain data into newly allocated cell

void TMVA::PDEFoamCell::Fill(Int_t status, PDEFoamCell *parent, PDEFoamCell *daugh1, PDEFoamCell *daugh2)
{
   fStatus  = status;
   fParent  = parent;
   fDaught0 = daugh1;
   fDaught1 = daugh2;
}

////////////////////////////////////////////////////////////////////////////////
//              GETTERS/SETTERS
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Provides size and position of the cell
/// These parameter are calculated by analyzing information in all parents
/// cells up to the root cell. It takes time but saves memory.

void    TMVA::PDEFoamCell::GetHcub( PDEFoamVect &cellPosi, PDEFoamVect &cellSize)  const
{
   if(fDim<1) return;
   const PDEFoamCell *pCell,*dCell;
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
         Error( "GetHcub ","Something wrong with linked tree \n");
      }
      dCell=pCell;
   }//while
}//GetHcub

////////////////////////////////////////////////////////////////////////////////
/// Provides size of the cell
/// Size parameters are calculated by analyzing information in all parents
/// cells up to the root cell. It takes time but saves memory.

void    TMVA::PDEFoamCell::GetHSize( PDEFoamVect &cellSize)  const
{
   if(fDim<1) return;
   const PDEFoamCell *pCell,*dCell;
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
         Error( "GetHSize ","Something wrong with linked tree \n");
      }
      dCell=pCell;
   }//while
}//GetHSize

////////////////////////////////////////////////////////////////////////////////
/// Calculates volume of the cell using size params which are calculated

void TMVA::PDEFoamCell::CalcVolume(void)
{
   Int_t k;
   Double_t volu=1.0;
   if(fDim>0) {         // h-cubical subspace
      PDEFoamVect cellSize(fDim);
      GetHSize(cellSize);
      for(k=0; k<fDim; k++) volu *= cellSize[k];
   }
   fVolume =volu;
}

////////////////////////////////////////////////////////////////////////////////
/// Get depth of cell in binary tree, where the root cell has depth
/// 1

UInt_t TMVA::PDEFoamCell::GetDepth()
{
   // check whether we are in the root cell
   if (fParent == 0)
      return 1;

   UInt_t depth = 1;
   PDEFoamCell *cell = this;
   while ((cell=cell->GetPare()) != 0){
      ++depth;
   }
   return depth;
}

////////////////////////////////////////////////////////////////////////////////
/// Get depth of cell tree, starting at this cell.

UInt_t TMVA::PDEFoamCell::GetTreeDepth(UInt_t depth)
{
   if (GetStat() == 1)    // this is an active cell
      return depth + 1;

   UInt_t depth0 = 0, depth1 = 0;
   if (GetDau0() != NULL)
      depth0 = GetDau0()->GetTreeDepth(depth+1);
   if (GetDau1() != NULL)
      depth1 = GetDau1()->GetTreeDepth(depth+1);

   return (depth0 > depth1 ? depth0 : depth1);
}

////////////////////////////////////////////////////////////////////////////////
/// Printout of the cell geometry parameters for the debug purpose

void TMVA::PDEFoamCell::Print(Option_t *option) const
{
   if (!option) Error( "Print", "No option set\n");

   std::cout <<  " Status= "<<     fStatus   <<",";
   std::cout <<  " Volume= "<<     fVolume   <<",";
   std::cout <<  " TrueInteg= " << fIntegral <<",";
   std::cout <<  " DriveInteg= "<< fDrive    <<",";
   std::cout << std::endl;;
   std::cout <<  " Xdiv= "<<fXdiv<<",";
   std::cout <<  " Best= "<<fBest<<",";
   std::cout <<  " Parent=  {"<< (GetPare() ? GetPare()->GetSerial() : -1) <<"} "; // extra DEBUG
   std::cout <<  " Daught0= {"<< (GetDau0() ? GetDau0()->GetSerial() : -1 )<<"} "; // extra DEBUG
   std::cout <<  " Daught1= {"<< (GetDau1() ? GetDau1()->GetSerial()  : -1 )<<"} "; // extra DEBUG
   std::cout << std::endl;;
   //
   //
   if (fDim>0 ) {
      PDEFoamVect cellPosi(fDim); PDEFoamVect cellSize(fDim);
      GetHcub(cellPosi,cellSize);
      std::cout <<"   Posi= "; cellPosi.Print("1"); std::cout<<","<< std::endl;;
      std::cout <<"   Size= "; cellSize.Print("1"); std::cout<<","<< std::endl;;
   }
}
