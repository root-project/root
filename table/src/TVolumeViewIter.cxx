// @(#)root/star:$Name:  $:$Id: TVolumeViewIter.cxx,v 1.2 2001/05/30 06:03:43 brun Exp $
// Author: Valery Fine(fine@bnl.gov)   25/01/99

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
}

//______________________________________________________________________________
TVolumeViewIter::~TVolumeViewIter()
{
  if (fPositions) { fPositions->Delete(); delete fPositions; }
}
//______________________________________________________________________________
const TVolumePosition *TVolumeViewIter::GetPosition(Int_t level) const
{
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
  const TVolumePosition *pos = GetPosition(level);
  if (pos) return new TVolumePosition(*pos);
  else {
     Error("operator[]"," GetPosition: %d %d %x", level,fDepth, fPositions);
     return 0;
  }
}

//______________________________________________________________________________
void TVolumeViewIter::Notify(TDataSet *set)
{
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
  }
  if (fDepth-1) {
    TVolumePosition *oldPosition = 0;
    TRotMatrix *oldMatrix = 0;
    oldPosition = fPositions ? (TVolumePosition *)fPositions->At(fDepth-1):0;
    Double_t oldTranslation[] = { 0, 0, 0 };
    if (oldPosition)
    {
      oldMatrix         = oldPosition->GetMatrix();
      oldTranslation[0] = oldPosition->GetX();
      oldTranslation[1] = oldPosition->GetY();
      oldTranslation[2] = oldPosition->GetZ();
    }

    // Pick the "current" position by pieces
    TRotMatrix *curMatrix        = curPosition->GetMatrix();

    // Create a new position
    Double_t newTranslation[3];
    Double_t newMatrix[9];

    if(oldMatrix)
    {
      TGeometry::UpdateTempMatrix(oldTranslation,oldMatrix->GetMatrix()
                       ,curPosition->GetX(),curPosition->GetY(),curPosition->GetZ(),curMatrix->GetMatrix()
                       ,newTranslation,newMatrix);
      Int_t num = gGeometry->GetListOfMatrices()->GetSize();
      Char_t anum[100];
      sprintf(anum,"%d",num+1);
      newPosition = SetPositionAt(curNode
                                ,newTranslation[0],newTranslation[1],newTranslation[2]
                                ,new TRotMatrix(anum,"NodeView",newMatrix));
    }
    else {
       newTranslation[0] = oldTranslation[0] + curPosition->GetX();
       newTranslation[1] = oldTranslation[1] + curPosition->GetY();
       newTranslation[2] = oldTranslation[2] + curPosition->GetZ();
       newPosition = SetPositionAt(curNode,newTranslation[0],newTranslation[1],newTranslation[2]);
    }
  }
  else if (curPosition)  {
         newPosition =  SetPositionAt(*curPosition);
//         printf(" new level %d %s\n",fDepth, curNode->GetName());
       }
       else
         Error("UpdateTempMatrix","No position has been defined");
  if (newPosition) newPosition->SetId(curPositionId);
  return newPosition;
}

//______________________________________________________________________________
void TVolumeViewIter::ResetPosition(Int_t level, TVolumePosition *newPosition)
{
  Int_t thisLevel = level;
  if (!thisLevel) thisLevel = fDepth;
  TVolumePosition *thisPosition  =  (TVolumePosition *) GetPosition(level);
  if (newPosition)
     *thisPosition =  *newPosition;
}

//______________________________________________________________________________
void TVolumeViewIter::Reset(TDataSet *l,Int_t depth)
{
  TDataSetIter::Reset(l,depth);
}

//______________________________________________________________________________
TVolumePosition *TVolumeViewIter::SetPositionAt(TVolume *node,Double_t x, Double_t y, Double_t z, TRotMatrix *matrix)
{
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
   if (!fPositions)  fPositions = new TObjArray(100);
   TVolumePosition *position =  (TVolumePosition *) fPositions->At(fDepth);
   if (position) *position = curPosition;
   else {
      position = new TVolumePosition(curPosition);
      fPositions->AddAtAndExpand(position,fDepth);
    }
   return position;
}


