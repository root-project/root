// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveProjectionManager.h"
#include "TEveManager.h"
#include "TEveProjectionBases.h"

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"

#include "TClass.h"

#include <list>

//______________________________________________________________________________
// TEveProjectionManager
//
// Manager class for steering of projections and managing projected
// objects.
//
// Recursively projects TEveElement's and draws axis in the projected
// scene.  It enables to interactivly set TEveProjection parameters
// and updates projected scene accordingly.

ClassImp(TEveProjectionManager)

//______________________________________________________________________________
TEveProjectionManager::TEveProjectionManager():
   TEveElementList("TEveProjectionManager",""),

   fProjection (0),

   fDrawCenter(kFALSE),
   fDrawOrigin(kFALSE),

   fSplitInfoMode(0),
   fSplitInfoLevel(1),
   fAxisColor(0),

   fCurrentDepth(0)
{
   // Constructor.

   fProjection  = new TEveCircularFishEyeProjection(fCenter);
   UpdateName();
}

//______________________________________________________________________________
TEveProjectionManager::~TEveProjectionManager()
{
   // Destructor.

   if(fProjection) delete fProjection;
}

//______________________________________________________________________________
void TEveProjectionManager::UpdateName()
{
   // Updates name to have consitent information with prjection.

   SetName(Form ("%s (%3.1f)", fProjection->GetName(), fProjection->GetDistortion()*1000));
   UpdateItems();
}

//______________________________________________________________________________
void TEveProjectionManager::SetProjection(TEveProjection::EPType_e type, Float_t distort)
{
   // Set projection type and distortion.

   static const TEveException eH("TEveProjectionManager::SetProjection ");

   delete fProjection;
   fProjection = 0;

   switch (type)
   {
      case TEveProjection::kPT_CFishEye:
      {
         fProjection  = new TEveCircularFishEyeProjection(fCenter);
         break;
      }
      case TEveProjection::kPT_RhoZ:
      {
         fProjection  = new TEveRhoZProjection(fCenter);
         break;
      }
      default:
         throw(eH + "projection type not valid.");
         break;
   }
   fProjection->SetDistortion(distort);
   UpdateName();
}
//______________________________________________________________________________
void TEveProjectionManager::SetCenter(Float_t x, Float_t y, Float_t z)
{
   // Set projection center and rebuild projected scene.

   fCenter.Set(x, y, z);
   fProjection->SetCenter(fCenter);
   ProjectChildren();
}

//______________________________________________________________________________
Bool_t TEveProjectionManager::HandleElementPaste(TEveElement* el)
{
   // React to element being pasted or dnd-ed.
   // Return true if redraw is needed (virtual method).

   List_t::size_type n_children  = fChildren.size();
   ImportElements(el);
   return n_children != fChildren.size();
}

//______________________________________________________________________________
Bool_t TEveProjectionManager::ShouldImport(TEveElement* rnr_el)
{
   // Returns true if rnr_el or any of its children is NTLProjectable.

   if (rnr_el->IsA()->InheritsFrom(TEveProjectable::Class()))
      return kTRUE;
   for (List_i i=rnr_el->BeginChildren(); i!=rnr_el->EndChildren(); ++i)
      if (ShouldImport(*i))
         return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
void TEveProjectionManager::ImportElementsRecurse(TEveElement* rnr_el, TEveElement* parent)
{
   // If rnr_el is TEveProjectable add projected instance else add
   // plain TEveElementList to parent. Call same function on rnr_el
   // children.

   static const TEveException eh("TEveProjectionManager::ImportElementsRecurse ");

   if (ShouldImport(rnr_el))
   {
      TEveElement  *new_re = 0;
      TEveProjected   *new_pr = 0;
      TEveProjectable *pble   = dynamic_cast<TEveProjectable*>(rnr_el);
      if (pble)
      {
         new_re = (TEveElement*) pble->ProjectedClass()->New();
         new_pr = dynamic_cast<TEveProjected*>(new_re);
         new_pr->SetProjection(this, pble);
         new_pr->SetDepth(fCurrentDepth);
      }
      else
      {
         new_re = new TEveElementList;
      }
      TObject *tobj   = rnr_el->GetObject(eh);
      new_re->SetRnrElNameTitle(Form("NLT %s", tobj->GetName()),
                                tobj->GetTitle());
      new_re->SetRnrSelf     (rnr_el->GetRnrSelf());
      new_re->SetRnrChildren(rnr_el->GetRnrChildren());
      gEve->AddElement(new_re, parent);

      for (List_i i=rnr_el->BeginChildren(); i!=rnr_el->EndChildren(); ++i)
         ImportElementsRecurse(*i, new_re);
   }
}

//______________________________________________________________________________
void TEveProjectionManager::ImportElements(TEveElement* rnr_el)
{
   // Recursively import elements and update projection on the projected objects.

   ImportElementsRecurse(rnr_el, this);
   ProjectChildren();
}

//______________________________________________________________________________
void TEveProjectionManager::ProjectChildrenRecurse(TEveElement* rnr_el)
{
   // Go recursively through rnr_el tree and call UpdateProjection() on TEveProjected.

   TEveProjected* pted = dynamic_cast<TEveProjected*>(rnr_el);
   if (pted)
   {
      pted->UpdateProjection();
      TAttBBox* bb = dynamic_cast<TAttBBox*>(pted);
      if(bb)
      {
         Float_t* b = bb->AssertBBox();
         BBoxCheckPoint(b[0], b[2], b[4]);
         BBoxCheckPoint(b[1], b[3], b[5]);
      }
      rnr_el->ElementChanged(kFALSE);
   }

   for (List_i i=rnr_el->BeginChildren(); i!=rnr_el->EndChildren(); ++i)
      ProjectChildrenRecurse(*i);
}

//______________________________________________________________________________
void TEveProjectionManager::ProjectChildren()
{
   // Project children recursevly, update BBox and notify ReveManger
   // the scenes have chenged.

   BBoxZero();
   ProjectChildrenRecurse(this);
   AssertBBoxExtents(0.1);
   {
      using namespace TMath;
      fBBox[0] = 10.0f * Floor(fBBox[0]/10.0f);
      fBBox[1] = 10.0f * Ceil (fBBox[1]/10.0f);
      fBBox[2] = 10.0f * Floor(fBBox[2]/10.0f);
      fBBox[3] = 10.0f * Ceil (fBBox[3]/10.0f);
   }

   List_t scenes;
   CollectSceneParentsFromChildren(scenes, 0);
   gEve->ScenesChanged(scenes);
}

//______________________________________________________________________________
void TEveProjectionManager::Paint(Option_t* /*option*/)
{
   // Paint this object. Only direct rendering is supported.

   static const TEveException eH("TEveProjectionManager::Paint ");
   TBuffer3D buff(TBuffer3DTypes::kGeneric);

   // Section kCore
   buff.fID           = this;
   buff.fColor        = fAxisColor;
   buff.fTransparency = 0;
   buff.SetSectionsValid(TBuffer3D::kCore);

   Int_t reqSections = gPad->GetViewer3D()->AddObject(buff);
   if (reqSections != TBuffer3D::kNone)
      Error(eH, "only direct GL rendering supported.");
}

//______________________________________________________________________________
void TEveProjectionManager::ComputeBBox()
{
   // Virtual from TAttBBox; fill bounding-box information.

   static const TEveException eH("TEveProjectionManager::ComputeBBox ");

   if(GetNChildren() == 0) {
      BBoxZero();
      return;
   }

   BBoxInit();
}
