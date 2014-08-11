// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   25/01/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVolumeViewIter.h"
#include "TObjArray.h"
#include "TVolumeView.h"
#include "TDataSetIter.h"
#include "TGeometry.h"

/////////////////////////////////////////////////////////////////////////////////
//
//   TVolumeViewIter is a special class-iterator to
//   iterate over GEANT geometry dataset TVolumeView.
//   Class should provide a "standard" well-known
//   "TDataSetIter" interface to navigate and access
//   the geometry information supplied by Tgeant_Maker
//   as the TVolume object. Apart of the the base
//   TDataSetIter this special class may supply
//   not only pointer to the selected dataset but some
//   "position" information (like translate vectors and
//   rotation matrice).
//
/////////////////////////////////////////////////////////////////////////////////

ClassImp(TVolumeViewIter)
//______________________________________________________________________________
TVolumeViewIter::TVolumeViewIter(TVolumeView *view, Int_t depth, Bool_t dir):
           TDataSetIter(view,depth,dir),fPositions(0)
{
   //to be documented
}

//______________________________________________________________________________
TVolumeViewIter::~TVolumeViewIter()
{
   //to be documented
   if (fPositions) { fPositions->Delete(); delete fPositions; }
}
//______________________________________________________________________________
const TVolumePosition *TVolumeViewIter::GetPosition(Int_t level) const
{
   //to be documented
   const TVolumePosition *pos = 0;
   if (fPositions) {
      Int_t thisLevel = level;
      if (!thisLevel) thisLevel = fDepth;
      pos=(TVolumePosition *)fPositions->At(thisLevel);
   }
   return pos;
}

//______________________________________________________________________________
TVolumePosition *TVolumeViewIter::operator[](Int_t level)
{
   //to be documented
   const TVolumePosition *pos = GetPosition(level);
   if (pos) return new TVolumePosition(*pos);
   else {
      Error("operator[]"," GetPosition: %d %d 0x%lx", level,fDepth, (Long_t)fPositions);
      return 0;
   }
}

//______________________________________________________________________________
void TVolumeViewIter::Notify(TDataSet *set)
{
   //to be documented
   if (!set) return;
   TVolumeView     *view         = (TVolumeView *) set;
   TVolumePosition *position     = 0;
   position = view->GetPosition();
   UpdateTempMatrix(position);
}

//______________________________________________________________________________
TVolumePosition *TVolumeViewIter::UpdateTempMatrix(TVolumePosition *curPosition)
{
   // Pick the "old" position by pieces
   TVolumePosition *newPosition = 0;
   TVolume *curNode = 0;
   UInt_t curPositionId    = 0;
   if (curPosition) {
      curNode       = curPosition->GetNode();
      curPositionId = curPosition->GetId();
   } else {
      Error("UpdateTempMatrix","No position has been defined");
      return 0;
   }
   if (fDepth-1) {
      TVolumePosition *oldPosition = 0;
      const TRotMatrix *oldMatrix = 0;
      oldPosition = fPositions ? (TVolumePosition *)fPositions->At(fDepth-1):0;
      Double_t oldTranslation[] = { 0, 0, 0 };
      if (oldPosition) {
         oldMatrix         = oldPosition->GetMatrix();
         oldTranslation[0] = oldPosition->GetX();
         oldTranslation[1] = oldPosition->GetY();
         oldTranslation[2] = oldPosition->GetZ();
      }

      // Pick the "current" position by pieces
      const TRotMatrix *curMatrix        = curPosition->GetMatrix();

      // Create a new position
      Double_t newTranslation[3];
      Double_t newMatrix[9];

      if(oldMatrix) {
         TGeometry::UpdateTempMatrix(oldTranslation,((TRotMatrix *)oldMatrix)->GetMatrix()
                       ,curPosition->GetX(),curPosition->GetY(),curPosition->GetZ()
                       ,((TRotMatrix *)curMatrix)->GetMatrix()
                       ,newTranslation,newMatrix);
         Int_t num = gGeometry->GetListOfMatrices()->GetSize();
         Char_t anum[100];
         snprintf(anum,100,"%d",num+1);
         newPosition = SetPositionAt(curNode
                                ,newTranslation[0],newTranslation[1],newTranslation[2]
                                ,new TRotMatrix(anum,"NodeView",newMatrix));
         newPosition->SetMatrixOwner();
      } else {
         newTranslation[0] = oldTranslation[0] + curPosition->GetX();
         newTranslation[1] = oldTranslation[1] + curPosition->GetY();
         newTranslation[2] = oldTranslation[2] + curPosition->GetZ();
         newPosition = SetPositionAt(curNode,newTranslation[0],newTranslation[1],newTranslation[2]);
         if (newPosition) {;} //intentionally not used
      }
   } else {
      newPosition =  SetPositionAt(*curPosition);
      // printf(" new level %d %s\n",fDepth, curNode->GetName();
   }
   if (newPosition) newPosition->SetId(curPositionId);
   return newPosition;
}

//______________________________________________________________________________
void TVolumeViewIter::ResetPosition(Int_t level, TVolumePosition *newPosition)
{
   //to be documented

   Int_t thisLevel = level;
   if (!thisLevel) thisLevel = fDepth;
   TVolumePosition *thisPosition  =  (TVolumePosition *) GetPosition(level);
   if (newPosition)
      *thisPosition =  *newPosition;
}

//______________________________________________________________________________
void TVolumeViewIter::Reset(TDataSet *l,Int_t depth)
{
   //to be documented

   TDataSetIter::Reset(l,depth);
}

//______________________________________________________________________________
TVolumePosition *TVolumeViewIter::SetPositionAt(TVolume *node,Double_t x, Double_t y, Double_t z, TRotMatrix *matrix)
{
   //to be documented

   if (!fPositions)  fPositions = new TObjArray(100);
   TVolumePosition *position =  (TVolumePosition *) fPositions->At(fDepth);
   if (position) position->Reset(node,x,y,z,matrix);
   else {
      position = new TVolumePosition(node,x,y,z,matrix);
      fPositions->AddAtAndExpand(position,fDepth);
   }
   return position;
}

//______________________________________________________________________________
TVolumePosition *TVolumeViewIter::SetPositionAt(TVolumePosition &curPosition)
{
   //to be documented

   if (!fPositions)  fPositions = new TObjArray(100);
   TVolumePosition *position =  (TVolumePosition *) fPositions->At(fDepth);
   if (position) *position = curPosition;
   else {
      position = new TVolumePosition(curPosition);
      fPositions->AddAtAndExpand(position,fDepth);
   }
   return position;
}


