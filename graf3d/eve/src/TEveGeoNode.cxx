// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveGeoNode.h"
#include "TEveManager.h"
#include "TEvePolygonSetProjected.h"

#include "TEveGeoShapeExtract.h"

#include "TROOT.h"
#include "TPad.h"
#include "TBuffer3D.h"
#include "TVirtualViewer3D.h"
#include "TColor.h"
#include "TFile.h"

#include "TGeoShape.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TGeoShapeAssembly.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TVirtualGeoPainter.h"

//==============================================================================
//==============================================================================
// TEveGeoNode
//==============================================================================

//______________________________________________________________________________
//
// Wrapper for TGeoNode that allows it to be shown in GUI and controlled as a TEveElement.

ClassImp(TEveGeoNode);

//______________________________________________________________________________
TEveGeoNode::TEveGeoNode(TGeoNode* node) :
   TEveElement(),
   TObject(),
   fNode(node)
{
   // Constructor.

   // Hack!! Should use cint to retrieve TAttLine::fLineColor offset.
   char* l = (char*) dynamic_cast<TAttLine*>(node->GetVolume());
   SetMainColorPtr((Color_t*)(l + sizeof(void*)));

   fRnrSelf = fNode->TGeoAtt::IsVisible();
}

//______________________________________________________________________________
const Text_t* TEveGeoNode::GetName()  const
{
   // Return name, taken from geo-node. Used via TObject.

   return fNode->GetName();
}

const Text_t* TEveGeoNode::GetTitle() const
{
   // Return title, taken from geo-node. Used via TObject.

   return fNode->GetTitle();
}

//______________________________________________________________________________
const Text_t* TEveGeoNode::GetElementName()  const
{
   // Return name, taken from geo-node. Used via TEveElement.

   return fNode->GetName();
}

const Text_t* TEveGeoNode::GetElementTitle() const
{
   // Return title, taken from geo-node. Used via TEveElement.

   return fNode->GetTitle();
}

/******************************************************************************/

void TEveGeoNode::ExpandIntoListTree(TGListTree* ltree,
                                     TGListTreeItem* parent)
{
   // Checks if child-nodes have been imported ... imports them if not.
   // Then calls TEveElement::ExpandIntoListTree.

   if (fChildren.empty() && fNode->GetVolume()->GetNdaughters() > 0) {
      TIter next(fNode->GetVolume()->GetNodes());
      TGeoNode* dnode;
      while ((dnode = (TGeoNode*) next()) != 0) {
         TEveGeoNode* node_re = new TEveGeoNode(dnode);
         AddElement(node_re);
      }
   }
   TEveElement::ExpandIntoListTree(ltree, parent);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoNode::SetRnrSelf(Bool_t rnr)
{
   // Set render state of self, propagate also to TGeoNode.

   TEveElement::SetRnrSelf(rnr);
   fNode->SetVisibility(rnr);
}

//______________________________________________________________________________
void TEveGeoNode::SetRnrChildren(Bool_t rnr)
{
   // Set render state of children, propagate also to TGeoNode.

   TEveElement::SetRnrChildren(rnr);
   fNode->VisibleDaughters(rnr);
}

//______________________________________________________________________________
void TEveGeoNode::SetRnrState(Bool_t rnr)
{
   // Set common render state, propagate also to TGeoNode.

   TEveElement::SetRnrState(rnr);
   fNode->SetVisibility(rnr);
   fNode->VisibleDaughters(rnr);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoNode::SetMainColor(Color_t color)
{
   // Set color, propagate to volume's line color.

   fNode->GetVolume()->SetLineColor(color);
   UpdateItems();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoNode::UpdateNode(TGeoNode* node)
{
   // Updates all reve-browsers having the node in their contents.
   // All 3D-pads updated if any change found.
   //
   // Should (could?) be optimized with some assumptions about
   // volume/node structure (search for parent, know the same node can not
   // reoccur on lower level once found).

   static const TEveException eH("TEveGeoNode::UpdateNode ");

   // printf("%s node %s %p\n", eH.Data(), node->GetName(), node);

   if(fNode == node)
      UpdateItems();

   for(List_i i=fChildren.begin(); i!=fChildren.end(); ++i) {
      ((TEveGeoNode*)(*i))->UpdateNode(node);
   }

}

//______________________________________________________________________________
void TEveGeoNode::UpdateVolume(TGeoVolume* volume)
{
   // Updates all reve-browsers having the volume in their contents.
   // All 3D-pads updated if any change found.
   //
   // Should (could?) be optimized with some assumptions about
   // volume/node structure (search for parent, know the same node can not
   // reoccur on lower level once found).

   static const TEveException eH("TEveGeoNode::UpdateVolume ");

   // printf("%s volume %s %p\n", eH.Data(), volume->GetName(), volume);

   if(fNode->GetVolume() == volume)
      UpdateItems();

   for(List_i i=fChildren.begin(); i!=fChildren.end(); ++i) {
      ((TEveGeoNode*)(*i))->UpdateVolume(volume);
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoNode::Draw(Option_t* option)
{
   // Draw the object.

   TString opt("SAME");
   opt += option;
   fNode->GetVolume()->Draw(opt);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoNode::Save(const char* file, const char* name)
{
   // Save TEveGeoShapeExtract tree starting at this node.

   TEveGeoShapeExtract* gse = DumpShapeTree(this, 0, 0);

   TFile f(file, "RECREATE");
   gse->Write(name);
   f.Close();
}

/******************************************************************************/

//______________________________________________________________________________
TEveGeoShapeExtract* TEveGeoNode::DumpShapeTree(TEveGeoNode* geon, TEveGeoShapeExtract* parent, Int_t level)
{
   // Export the node hierarchy into tree of TEveGeoShapeExtract objects.

   printf("dump_shape_tree %s \n", geon->GetName());
   TGeoNode*   tnode   = 0;
   TGeoVolume* tvolume = 0;
   TGeoShape*  tshape  = 0;

   tnode = geon->GetNode();
   if(tnode == 0) {
      printf("Null node for %s; assuming it's a holder and descending.\n", geon->GetName());
      goto do_dump;
   }

   tvolume = tnode->GetVolume();
   if(tvolume == 0) {
      printf("Null volume for %s; skipping.\n", geon->GetName());
      return 0;
   }

   tshape  = tvolume->GetShape();

do_dump:
   // transformation
   TEveTrans trans;
   if (parent) if (parent) trans.SetFromArray(parent->GetTrans());
   TGeoMatrix* gm =  tnode->GetMatrix();
   const Double_t* rm = gm->GetRotationMatrix();
   const Double_t* tv = gm->GetTranslation();
   TEveTrans t;
   t(1,1) = rm[0]; t(1,2) = rm[1]; t(1,3) = rm[2];
   t(2,1) = rm[3]; t(2,2) = rm[4]; t(2,3) = rm[5];
   t(3,1) = rm[6]; t(3,2) = rm[7]; t(3,3) = rm[8];
   t(1,4) = tv[0]; t(2,4) = tv[1]; t(3,4) = tv[2];
   trans *= t;

   TEveGeoShapeExtract* gse = new TEveGeoShapeExtract(geon->GetName(), geon->GetTitle());
   gse->SetTrans(trans.Array());
   Int_t ci = 0;
   if(tvolume) ci = tvolume->GetLineColor();
   TColor* c = gROOT->GetColor(ci);
   Float_t rgba[4] = {1, 0, 0, 1};
   if (c) {
      rgba[0] = c->GetRed();
      rgba[1] = c->GetGreen();
      rgba[2] = c->GetBlue();
   }
   gse->SetRGBA(rgba);
   Bool_t rnr = geon->GetRnrSelf();
   if(level > gGeoManager->GetVisLevel())
      rnr = kFALSE;
   gse->SetRnrSelf(rnr);
   gse->SetRnrElements(geon->GetRnrChildren());

   if(dynamic_cast<TGeoShapeAssembly*>(tshape)){
      //    printf("<TGeoShapeAssembly \n");
      tshape = 0;
   }
   gse->SetShape(tshape);
   level ++;
   if ( geon->GetNChildren())
   {
      TList* ele = new TList();
      gse->SetElements(ele);
      gse->GetElements()->SetOwner(true);

      TEveElement::List_i i = geon->BeginChildren();
      while (i != geon->EndChildren()) {
         TEveGeoNode* l = dynamic_cast<TEveGeoNode*>(*i);
         DumpShapeTree(l, gse, level+1);
         i++;
      }
   }

   if(parent)
      parent->GetElements()->Add(gse);

   return gse;
}


//==============================================================================
//==============================================================================
// TEveGeoTopNode
//==============================================================================

//______________________________________________________________________________
//
// A wrapper over a TGeoNode, possibly displaced with a global
// trasformation stored in TEveElement.
//
// It holds a pointer to TGeoManager and controls for steering of
// TGeoPainter, fVisOption, fVisLevel and fMaxVisNodes. They have the
// same meaning as in TGeoManager/TGeoPainter.

ClassImp(TEveGeoTopNode);

//______________________________________________________________________________
TEveGeoTopNode::TEveGeoTopNode(TGeoManager* manager, TGeoNode* node,
                               Int_t visopt, Int_t vislvl, Int_t maxvisnds) :
   TEveGeoNode  (node),
   fManager     (manager),
   fVisOption   (visopt),
   fVisLevel    (vislvl),
   fMaxVisNodes (maxvisnds)
{
   // Constructor.

   InitMainTrans();
   fRnrSelf = kTRUE; // Override back from TEveGeoNode.
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoTopNode::UseNodeTrans()
{
   // Use transforamtion matrix from the TGeoNode.
   // Warning: this is local transformation of the node!

   RefMainTrans().SetFrom(*fNode->GetMatrix());
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoTopNode::SetRnrSelf(Bool_t rnr)
{
   // Revert from GeoNode back to standard behaviour.

   TEveElement::SetRnrSelf(rnr);
}

//______________________________________________________________________________
void TEveGeoTopNode::SetRnrChildren(Bool_t rnr)
{
   // Revert from GeoNode back to standard behaviour.

   TEveElement::SetRnrChildren(rnr);
}

//______________________________________________________________________________
void TEveGeoTopNode::SetRnrState(Bool_t rnr)
{
   // Revert from GeoNode back to standard behaviour.

   TEveElement::SetRnrState(rnr);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoTopNode::Draw(Option_t* option)
{
   // Draw the top-node.

   AppendPad(option);
}

//______________________________________________________________________________
void TEveGeoTopNode::Paint(Option_t* option)
{
   // Paint the enclosed TGeo hierarchy with visibility level and
   // option given in data-members.
   // Uses TGeoPainter internally.

   if (fRnrSelf) {
      gGeoManager = fManager;
      TVirtualPad* pad = gPad;
      gPad = 0;
      TGeoVolume* top_volume = fManager->GetTopVolume();
      fManager->SetVisOption(fVisOption);
      if (fVisLevel > 0)
         fManager->SetVisLevel(fVisLevel);
      else
         fManager->SetMaxVisNodes(fMaxVisNodes);
      fManager->SetTopVolume(fNode->GetVolume());
      gPad = pad;
      TVirtualGeoPainter* vgp = fManager->GetGeomPainter();
      if(vgp != 0) {
         TGeoHMatrix geomat;
         if (HasMainTrans()) RefMainTrans().SetGeoHMatrix(geomat);
         vgp->PaintNode(fNode, option, &geomat);
      }
      fManager->SetTopVolume(top_volume);
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoTopNode::VolumeVisChanged(TGeoVolume* volume)
{
   // Callback for propagating volume visibility changes.

   static const TEveException eH("TEveGeoTopNode::VolumeVisChanged ");
   printf("%s volume %s %p\n", eH.Data(), volume->GetName(), (void*)volume);
   UpdateVolume(volume);
}

//______________________________________________________________________________
void TEveGeoTopNode::VolumeColChanged(TGeoVolume* volume)
{
   // Callback for propagating volume parameter changes.

   static const TEveException eH("TEveGeoTopNode::VolumeColChanged ");
   printf("%s volume %s %p\n", eH.Data(), volume->GetName(), (void*)volume);
   UpdateVolume(volume);
}

//______________________________________________________________________________
void TEveGeoTopNode::NodeVisChanged(TGeoNode* node)
{
   // Callback for propagating node visibility changes.

   static const TEveException eH("TEveGeoTopNode::NodeVisChanged ");
   printf("%s node %s %p\n", eH.Data(), node->GetName(), (void*)node);
   UpdateNode(node);
}


//==============================================================================
//==============================================================================
// TEveGeoShape
//==============================================================================

//______________________________________________________________________________
//
// Wrapper for TGeoShape with absolute positioning and color
// attributes allowing display of extracted TGeoShape's (without an
// active TGeoManager) and simplified geometries (needed for NLT
// projections).

ClassImp(TEveGeoShape);

//______________________________________________________________________________
TEveGeoShape::TEveGeoShape(const Text_t* name, const Text_t* title) :
   TEveElement   (fColor),
   TNamed        (name, title),
   fColor        (0),
   fTransparency (0),
   fShape        (0)
{
   // Constructor.

   InitMainTrans();
}

//______________________________________________________________________________
TEveGeoShape::~TEveGeoShape()
{
   // Destructor.

   if (fShape) {
      fShape->SetUniqueID(fShape->GetUniqueID() - 1);
      if (fShape->GetUniqueID() == 0)
         delete fShape;
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoShape::Paint(Option_t* /*option*/)
{
   // Paint object.

   if (fShape == 0)
      return;

   TBuffer3D& buff = (TBuffer3D&) fShape->GetBuffer3D
      (TBuffer3D::kCore, kFALSE);

   buff.fID           = this;
   buff.fColor        = fColor;
   buff.fTransparency = fTransparency;
   RefMainTrans().SetBuffer3D(buff);
   buff.fLocalFrame   = kTRUE; // Always enforce local frame (no geo manager).

   fShape->GetBuffer3D(TBuffer3D::kBoundingBox | TBuffer3D::kShapeSpecific, kTRUE);

   Int_t reqSec = gPad->GetViewer3D()->AddObject(buff);

   if (reqSec != TBuffer3D::kNone) {
      fShape->GetBuffer3D(reqSec, kTRUE);
      reqSec = gPad->GetViewer3D()->AddObject(buff);
   }

   if (reqSec != TBuffer3D::kNone)
      printf("spooky reqSec=%d for %s\n", reqSec, GetName());
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoShape::Save(const char* file, const char* name)
{
   // Save the shape tree as TEveGeoShapeExtract.
   // File is always recreated.

   TEveGeoShapeExtract* gse = DumpShapeTree(this, 0);

   TFile f(file, "RECREATE");
   gse->Write(name);
   f.Close();
}

/******************************************************************************/

//______________________________________________________________________________
TEveGeoShapeExtract* TEveGeoShape::DumpShapeTree(TEveGeoShape* gsre,
                                                 TEveGeoShapeExtract* parent)
{
   // Export this shape and its descendants into a geoshape-extract.

   TEveGeoShapeExtract* she = new TEveGeoShapeExtract(gsre->GetName(), gsre->GetTitle());
   she->SetTrans(gsre->RefMainTrans().Array());
   Int_t ci = gsre->GetColor();
   TColor* c = gROOT->GetColor(ci);
   Float_t rgba[4] = {1, 0, 0, 1 - gsre->GetMainTransparency()/100.};
   if (c)
   {
      rgba[0] = c->GetRed();
      rgba[1] = c->GetGreen();
      rgba[2] = c->GetBlue();
   }
   she->SetRGBA(rgba);
   she->SetRnrSelf(gsre->GetRnrSelf());
   she->SetRnrElements(gsre->GetRnrChildren());
   she->SetShape(gsre->GetShape());
   if ( gsre->GetNChildren())
   {
      TList* ele = new TList();
      she->SetElements(ele);
      she->GetElements()->SetOwner(true);
      TEveElement::List_i i = gsre->BeginChildren();
      while (i != gsre->EndChildren()) {
         TEveGeoShape* l = dynamic_cast<TEveGeoShape*>(*i);
         DumpShapeTree(l, she);
         i++;
      }
   }
   if(parent)
      parent->GetElements()->Add(she);

   return she;
}

//______________________________________________________________________________
TEveGeoShape* TEveGeoShape::ImportShapeExtract(TEveGeoShapeExtract* gse,
                                               TEveElement*         parent)
{
   // Import a shape extract 'gse' under element 'parent'.

   gEve->DisableRedraw();
   TEveGeoShape* gsre = SubImportShapeExtract(gse, parent);
   gsre->ElementChanged();
   gEve->EnableRedraw();
   return gsre;
}


//______________________________________________________________________________
TEveGeoShape* TEveGeoShape::SubImportShapeExtract(TEveGeoShapeExtract* gse,
                                                  TEveElement*         parent)
{
   // Recursive version for importing a shape extract tree.

   TEveGeoShape* gsre = new TEveGeoShape(gse->GetName(), gse->GetTitle());
   gsre->RefMainTrans().SetFromArray(gse->GetTrans());
   const Float_t* rgba = gse->GetRGBA();
   gsre->fColor        = TColor::GetColor(rgba[0], rgba[1], rgba[2]);
   gsre->fTransparency = (UChar_t) (100.0f*(1.0f - rgba[3]));
   gsre->SetRnrSelf(gse->GetRnrSelf());
   gsre->SetRnrChildren(gse->GetRnrElements());
   gsre->fShape = gse->GetShape();
   if (gsre->fShape)
      gsre->fShape->SetUniqueID(gsre->fShape->GetUniqueID() + 1);

   if (parent)
      parent->AddElement(gsre);

   if (gse->HasElements())
   {
      TIter next(gse->GetElements());
      TEveGeoShapeExtract* chld;
      while ((chld = (TEveGeoShapeExtract*) next()) != 0)
         SubImportShapeExtract(chld, gsre);
   }

   return gsre;
}

/******************************************************************************/

//______________________________________________________________________________
TClass* TEveGeoShape::ProjectedClass() const
{
   // Return class for projected objects, TEvePolygonSetProjected.
   // Virtual from TEveProjectable.

   return TEvePolygonSetProjected::Class();
}

/******************************************************************************/

//______________________________________________________________________________
TBuffer3D* TEveGeoShape::MakeBuffer3D()
{
   // Create a TBuffer3D suitable for presentation of the shape.
   // Transformation matrix is also applied.

   if (fShape == 0) return 0;

   if (dynamic_cast<TGeoShapeAssembly*>(fShape)) {
      // !!!! TGeoShapeAssembly makes a bad TBuffer3D
      return 0;
   }

   TBuffer3D* buff  = fShape->MakeBuffer3D();
   TEveTrans& mx    = RefMainTrans();
   if (mx.GetUseTrans())
   {
      Int_t n = buff->NbPnts();
      Double_t* pnts = buff->fPnts;
      for(Int_t k = 0; k < n; ++k)
      {
         mx.MultiplyIP(&pnts[3*k]);
      }
   }
   return buff;
}
