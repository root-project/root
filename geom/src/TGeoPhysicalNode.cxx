// @(#)root/geom:$Name:  $:$Id: $
// Author: Andrei Gheata   17/02/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// TGeoPhysicalNode
//_________

#include "TGeoManager.h"
#include "TGeoCache.h"
#include "TGeoMatrix.h"
#include "TGeoShape.h"
#include "TGeoVolume.h"
#include "TVirtualGeoPainter.h"

#include "TGeoPhysicalNode.h"

// statics and globals

ClassImp(TGeoPhysicalNode)

//_____________________________________________________________________________
TGeoPhysicalNode::TGeoPhysicalNode()
{
// Default constructor
   fLevel        = 0;
   fMatrices     = 0;
   fNodes        = 0;
   SetVisibility(kTRUE);
   SetVisibleFull(kFALSE);
   SetIsVolAtt(kTRUE);
}

//_____________________________________________________________________________
TGeoPhysicalNode::TGeoPhysicalNode(const char *path)
{
// Constructor
   if (!strlen(path)) {
      Error("ctor", "path not valid");
      return;
   }
   fLevel  = 0;
   fMatrices = new TObjArray(30);
   fNodes    = new TObjArray(30);
   SetPath(path);   
   SetVisibility(kTRUE);
   SetVisibleFull(kFALSE);
   SetIsVolAtt(kTRUE);
}

//_____________________________________________________________________________
TGeoPhysicalNode::~TGeoPhysicalNode()
{
// Destructor
   if (fMatrices) {
      fMatrices->Delete();
      delete fMatrices;
   }   
   if (fNodes) delete fNodes;
}

//_____________________________________________________________________________
void TGeoPhysicalNode::Align(TGeoMatrix * /*newmat*/, TGeoShape * /*newshape*/)
{
   Warning("Align","Not yet implemenetd");
}   

//_____________________________________________________________________________
void TGeoPhysicalNode::cd() const
{

}

//_____________________________________________________________________________
void TGeoPhysicalNode::Draw(Option_t * /*option*/)
{

}

//_____________________________________________________________________________
TGeoNode *TGeoPhysicalNode::GetMother(Int_t levup) const
{
// Return parent at LEVUP generation
   Int_t ind = fLevel-levup;
   if (ind<0) return 0;
   return (TGeoNode*)fNodes->UncheckedAt(ind);
}   

//_____________________________________________________________________________
TGeoHMatrix *TGeoPhysicalNode::GetMatrix(Int_t level) const
{
// Return global matrix for node at LEVEL.
   if (level<0) return (TGeoHMatrix*)fMatrices->UncheckedAt(fLevel);
   if (level>fLevel) return 0;
   return (TGeoHMatrix*)fMatrices->UncheckedAt(level);
}

//_____________________________________________________________________________
TGeoNode *TGeoPhysicalNode::GetNode(Int_t level) const
{
// Return node in branch at LEVEL. If not specified, return last leaf.
   if (level<0) return (TGeoNode*)fNodes->UncheckedAt(fLevel);
   if (level>fLevel) return 0;
   return (TGeoNode*)fNodes->UncheckedAt(level);
}   

//_____________________________________________________________________________
TGeoVolume *TGeoPhysicalNode::GetVolume(Int_t level) const
{
// Return volume associated with node at LEVEL in the branch
   TGeoNode *node = GetNode(level);
   if (node) return node->GetVolume();
   return 0;
}

//_____________________________________________________________________________
TGeoShape *TGeoPhysicalNode::GetShape(Int_t level) const
{
// Return shape associated with volume.
   TGeoVolume *vol = GetVolume(level);
   if (vol) return vol->GetShape();
   return 0;
}   

//_____________________________________________________________________________
void TGeoPhysicalNode::Paint(Option_t * /*option*/)
{
// Paint this node and its content according to visualization settings.
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
//   painter->PaintNode(this, option);
}

//_____________________________________________________________________________
void TGeoPhysicalNode::SetBranchAsState()
{
// Set node branch according to current state
   TGeoNodeCache *cache = gGeoManager->GetCache();
   if (!cache) {
      Error("SetBranchAsState","no state available");
      return;
   }
   if (!cache->IsDummy()) {
      Error("SetBranchAsState", "not implemented for full cache");
      return;
   }   
   
   if (!fNodes)    fNodes = new TObjArray(30);
   if (!fMatrices) fMatrices = new TObjArray(30);
   fLevel = gGeoManager->GetLevel();   
   TGeoHMatrix **matrices = (TGeoHMatrix **) cache->GetMatrices();
   TGeoNode **branch = (TGeoNode **) cache->GetBranch();
   for (Int_t i=0; i<=fLevel; i++) {
      fNodes->AddAt(branch[i],i);
      fMatrices->AddAt(new TGeoHMatrix(*matrices[i]),i);
   }   
}

//_____________________________________________________________________________
Bool_t TGeoPhysicalNode::SetPath(const char *path)
{
// Specify the path for this node.
   if (!gGeoManager->cd(path)) {
      Error("SetPath","wrong path -> maybe RestoreMasterVolume");
      return kFALSE;
   }
   SetBranchAsState();
   return kTRUE;
}

