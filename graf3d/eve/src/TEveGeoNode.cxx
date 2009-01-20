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
   SetMainTransparency((UChar_t) fNode->GetVolume()->GetTransparency());

   fRnrSelf = fNode->TGeoAtt::IsVisible();
}

//______________________________________________________________________________
const char* TEveGeoNode::GetName()  const
{
   // Return name, taken from geo-node. Used via TObject.

   return fNode->GetName();
}

//______________________________________________________________________________
const char* TEveGeoNode::GetTitle() const
{
   // Return title, taken from geo-node. Used via TObject.

   return fNode->GetTitle();
}

//______________________________________________________________________________
const char* TEveGeoNode::GetElementName()  const
{
   // Return name, taken from geo-node. Used via TEveElement.

   return fNode->GetName();
}

//______________________________________________________________________________
const char* TEveGeoNode::GetElementTitle() const
{
   // Return title, taken from geo-node. Used via TEveElement.

   return fNode->GetTitle();
}

/******************************************************************************/

//______________________________________________________________________________
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
void TEveGeoNode::AddStamp(UChar_t bits)
{
   // Override from TEveElement.
   // Process visibility changes and forward them to fNode.

   TEveElement::AddStamp(bits);
   if (bits & kCBVisibility)
   {
      fNode->SetVisibility(fRnrSelf);
      fNode->VisibleDaughters(fRnrChildren);
   }
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveGeoNode::CanEditMainColor() const
{
   // Can edit main-color -- not available for assemblies.

   return ! fNode->GetVolume()->IsAssembly();
}

//______________________________________________________________________________
void TEveGeoNode::SetMainColor(Color_t color)
{
   // Set color, propagate to volume's line color.

   TEveElement::SetMainColor(color);
   fNode->GetVolume()->SetLineColor(color);
}

//______________________________________________________________________________
Bool_t TEveGeoNode::CanEditMainTransparency() const
{
   // Can edit main transparency -- not available for assemblies.

   return ! fNode->GetVolume()->IsAssembly();
}

//______________________________________________________________________________
UChar_t TEveGeoNode::GetMainTransparency() const
{
   // Get transparency from node, if different propagate to this.

   UChar_t t = (UChar_t) fNode->GetVolume()->GetTransparency();
   if (fMainTransparency != t)
   {
      TEveGeoNode* ncthis = const_cast<TEveGeoNode*>(this);
      ncthis->SetMainTransparency(t);
   }
   return t;
}

//______________________________________________________________________________
void TEveGeoNode::SetMainTransparency(UChar_t t)
{
   // Set transparency, propagate to volume's transparency.

   TEveElement::SetMainTransparency(t);
   fNode->GetVolume()->SetTransparency((Char_t) t);
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

   if (fNode == node)
      StampColorSelection();

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i) {
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
      StampColorSelection();

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

   static const TEveException eh("TEveGeoNode::DumpShapeTree ");

   printf("dump_shape_tree %s \n", geon->GetName());
   TGeoNode*   tnode   = 0;
   TGeoVolume* tvolume = 0;
   TGeoShape*  tshape  = 0;

   tnode = geon->GetNode();
   if (tnode == 0)
   {
      Info(eh, "Null TGeoNode for TEveGeoNode '%s': assuming it's a holder and descending.", geon->GetName());
   }
   else
   {
      tvolume = tnode->GetVolume();
      if (tvolume == 0) {
         Warning(eh, "Null TGeoVolume for TEveGeoNode '%s'; skipping its sub-tree.\n", geon->GetName());
         return 0;
      }
      tshape  = tvolume->GetShape();
   }

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
   if (tvolume) ci = tvolume->GetLineColor();
   TColor* c = gROOT->GetColor(ci);
   Float_t rgba[4] = {1, 0, 0, 1};
   if (c) {
      rgba[0] = c->GetRed();
      rgba[1] = c->GetGreen();
      rgba[2] = c->GetBlue();
   }
   gse->SetRGBA(rgba);
   Bool_t rnr = geon->GetRnrSelf();
   if (level > gGeoManager->GetVisLevel())
      rnr = kFALSE;
   gse->SetRnrSelf(rnr);
   gse->SetRnrElements(geon->GetRnrChildren());

   if (dynamic_cast<TGeoShapeAssembly*>(tshape)) {
      Info(eh, "TGeoShapeAssembly name='%s' encountered in traversal. This is not supported.", tshape->GetName());
      tshape = 0;
   }
   gse->SetShape(tshape);
   ++level;
   if (geon->HasChildren())
   {
      TList* ele = new TList();
      gse->SetElements(ele);
      gse->GetElements()->SetOwner(true);

      TEveElement::List_i i = geon->BeginChildren();
      while (i != geon->EndChildren())
      {
         TEveGeoNode* l = dynamic_cast<TEveGeoNode*>(*i);
         DumpShapeTree(l, gse, level+1);
         ++i;
      }
   }

   if (parent)
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
void TEveGeoTopNode::AddStamp(UChar_t bits)
{
   // Revert from TEveGeoNode back to standard behaviour, that is,
   // do not pass visibility chanes to fNode as they are honoured
   // in Paint() method.

   TEveElement::AddStamp(bits);
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

   static const TEveException eh("TEveGeoTopNode::VolumeVisChanged ");
   printf("%s volume %s %p\n", eh.Data(), volume->GetName(), (void*)volume);
   UpdateVolume(volume);
}

//______________________________________________________________________________
void TEveGeoTopNode::VolumeColChanged(TGeoVolume* volume)
{
   // Callback for propagating volume parameter changes.

   static const TEveException eh("TEveGeoTopNode::VolumeColChanged ");
   printf("%s volume %s %p\n", eh.Data(), volume->GetName(), (void*)volume);
   UpdateVolume(volume);
}

//______________________________________________________________________________
void TEveGeoTopNode::NodeVisChanged(TGeoNode* node)
{
   // Callback for propagating node visibility changes.

   static const TEveException eh("TEveGeoTopNode::NodeVisChanged ");
   printf("%s node %s %p\n", eh.Data(), node->GetName(), (void*)node);
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
//
// TGeoCompositeShapes are currently NOT supported. This is planned
// for ROOT-5.24.

namespace
{
TGeoManager* init_geo_mangeur()
{
   // Create a phony geo manager that
   TGeoManager* old = gGeoManager;
   gGeoManager = 0;
   TGeoManager* mgr = new TGeoManager();
   mgr->SetNameTitle("TEveGeoShape::fgGeoMangeur",
                     "Static geo manager used for wrapped TGeoShapes.");
   gGeoManager = old;
   return mgr;
}
}

ClassImp(TEveGeoShape);

TGeoManager* TEveGeoShape::fgGeoMangeur = init_geo_mangeur();

//______________________________________________________________________________
TGeoManager* TEveGeoShape::GetGeoMangeur()
{
   // Return static geo-manager that is used intenally to make shapes
   // lead a happy life.
   // Set gGeoManager to this object when creating TGeoShapes to be
   // passed into TEveGeoShapes.

   return fgGeoMangeur;
}

//______________________________________________________________________________
TEveGeoShape::TEveGeoShape(const char* name, const char* title) :
   TEveElement   (fColor),
   TNamed        (name, title),
   fColor        (0),
   fNSegments    (0),
   fShape        (0)
{
   // Constructor.

   InitMainTrans();
}

//______________________________________________________________________________
TEveGeoShape::~TEveGeoShape()
{
   // Destructor.

   SetShape(0);
}

//______________________________________________________________________________
void TEveGeoShape::SetShape(TGeoShape* s)
{
   // Set TGeoShape shown by this object.

   TEveGeoManagerHolder gmgr(fgGeoMangeur);

   if (fShape) {
      fShape->SetUniqueID(fShape->GetUniqueID() - 1);
      if (fShape->GetUniqueID() == 0)
         delete fShape;
   }
   fShape = s;
   if (fShape) {
      fShape->SetUniqueID(fShape->GetUniqueID() + 1);
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoShape::Paint(Option_t* /*option*/)
{
   // Paint object.

   static const TEveException eh("TEveGeoShape::Paint ");

   if (fShape == 0)
      return;

   TEveGeoManagerHolder gmgr(fgGeoMangeur, fNSegments);

   TBuffer3D& buff = (TBuffer3D&) fShape->GetBuffer3D
      (TBuffer3D::kCore, kFALSE);

   buff.fID           = this;
   buff.fColor        = GetMainColor();
   buff.fTransparency = GetMainTransparency();
   RefMainTrans().SetBuffer3D(buff);
   buff.fLocalFrame   = kTRUE; // Always enforce local frame (no geo manager).


   Int_t sections = TBuffer3D::kBoundingBox | TBuffer3D::kShapeSpecific;
   if (fNSegments > 2)
      sections |= TBuffer3D::kRawSizes | TBuffer3D::kRaw;
   fShape->GetBuffer3D(sections, kTRUE);

   Int_t reqSec = gPad->GetViewer3D()->AddObject(buff);

   if (reqSec != TBuffer3D::kNone) {
      // This shouldn't happen, but I suspect it does sometimes.
      if (reqSec & TBuffer3D::kCore)
         Warning(eh, "Core section required again for shape='%s'. This shouldn't happen.", GetName());
      fShape->GetBuffer3D(reqSec, kTRUE);
      reqSec = gPad->GetViewer3D()->AddObject(buff);
   }

   if (reqSec != TBuffer3D::kNone)
      Warning(eh, "Extra section required: reqSec=%d, shape=%s.", reqSec, GetName());
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
   if (gsre->HasChildren())
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
   if (parent)
      parent->GetElements()->Add(she);

   return she;
}

//______________________________________________________________________________
TEveGeoShape* TEveGeoShape::ImportShapeExtract(TEveGeoShapeExtract* gse,
                                               TEveElement*         parent)
{
   // Import a shape extract 'gse' under element 'parent'.

   TEveGeoManagerHolder gmgr(fgGeoMangeur);
   TEveManager::TRedrawDisabler redrawOff(gEve);
   TEveGeoShape* gsre = SubImportShapeExtract(gse, parent);
   gsre->ElementChanged();
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
   gsre->SetMainColorRGB(rgba[0], rgba[1], rgba[2]);
   gsre->SetMainAlpha(rgba[3]);
   gsre->SetRnrSelf(gse->GetRnrSelf());
   gsre->SetRnrChildren(gse->GetRnrElements());
   gsre->SetShape(gse->GetShape());

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

   TEveGeoManagerHolder gmgr(fgGeoMangeur, fNSegments);

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
