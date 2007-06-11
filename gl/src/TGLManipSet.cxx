// @(#)root/gl:$Name$:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLManipSet.h"

#include "TGLTransManip.h"
#include "TGLScaleManip.h"
#include "TGLRotateManip.h"

#include "TGLPhysicalShape.h"
#include "TGLRnrCtx.h"

#include "TGLIncludes.h"

#include <KeySymbols.h>
#include <TVirtualX.h>

//______________________________________________________________________
// TGLManipSet
//
// Combine all available manipulators in a collection.
//
// At first I wanted to merge them back into TGLManip (to have a
// single class) but then it seemed somehow messy.
// Maybe next time.

ClassImp(TGLManipSet)

TGLManipSet::TGLManipSet() :
   fType     (kTrans),
   fDrawBBox (kFALSE)
{
   // Constructor.

   fManip[kTrans]  = new TGLTransManip;
   fManip[kScale]  = new TGLScaleManip;
   fManip[kRotate] = new TGLRotateManip;
}

//______________________________________________________________________
TGLManipSet::~TGLManipSet()
{
   // Destructor.

   for (Int_t i=kTrans; i<kEndType; ++i)
      delete fManip[i];
}

//______________________________________________________________________
void TGLManipSet::SetPShape(TGLPhysicalShape * shape)
{
   // Set phys-shape, override of virtual from TGLPShapeRef.
   // Forward to all managed manipulators.

   TGLPShapeRef::SetPShape(shape);
   for (Int_t i=kTrans; i<kEndType; ++i)
      fManip[i]->Attach(shape);
}

/**************************************************************************/
/**************************************************************************/

//______________________________________________________________________
Bool_t TGLManipSet::MouseEnter(UInt_t* /*record*/)
{
   // Mouse has enetered this element.
   // Always accept.

   TGLManip* manip = GetCurrentManip();
   manip->SetActive(kFALSE);
   manip->SetSelectedWidget(0);
   return kTRUE;
}

//______________________________________________________________________
Bool_t TGLManipSet::Handle(TGLRnrCtx& rnrCtx, Event_t* event, UInt_t* record)
{
   // Handle overlay event.
   // Return TRUE if event was handled.

   TGLManip* manip = GetCurrentManip();
   UInt_t    wid   = record[0] > 1 ? record[4] : 0;
   Bool_t    chg   = kFALSE;
   if (manip->GetSelectedWidget() != wid && ! manip->GetActive())
   {
      manip->SetSelectedWidget(wid);
      chg = kTRUE;
   }

   switch (event->fType)
   {
      case kButtonPress:
      {
         return manip->HandleButton(*event, rnrCtx.RefCamera());
      }
      case kButtonRelease:
      {
         manip->SetActive(kFALSE);
         return kTRUE;
      }
      case kMotionNotify:
      {
         return manip->HandleMotion(*event, rnrCtx.RefCamera()) || chg;
      }
      case kGKeyPress:
      {
         switch (rnrCtx.GetEventKeySym())
         {
            case kKey_V: case kKey_v:
               SetManipType(kTrans);
               return kTRUE;
            case kKey_C: case kKey_c:
               SetManipType(kRotate);
               return kTRUE;
            case kKey_X: case kKey_x:
               SetManipType(kScale);
               return kTRUE;
            default:
               return kFALSE;
         }
      }
      default:
      {
         return kFALSE;
      }
   }
}

//______________________________________________________________________
void TGLManipSet::MouseLeave()
{
   // Mouse has left the element.

   TGLManip* manip = GetCurrentManip();
   manip->SetActive(kFALSE);
   manip->SetSelectedWidget(0);
}

//______________________________________________________________________
void TGLManipSet::Render(TGLRnrCtx& rnrCtx)
{
   // Render the manipulator and bounding-box.

   if (fPShape == 0)
      return;

   fManip[fType]->Draw(rnrCtx.RefCamera());

   if (fDrawBBox && ! rnrCtx.Selection())
   {
      // TODO: This must be replaced by some color in rnrCtx,
      // like def-overlay-color, background-color, foreground-color
      // Or at least bkgcol ... i can then find high contrast.
      if (rnrCtx.SceneStyle() == TGLRnrCtx::kFill       ||
          rnrCtx.SceneStyle() == TGLRnrCtx::kWireFrame)
      {
         // White for wireframe and fill style,
         glColor3d(1.0, 1.0, 1.0);
      } else {
         // Red for outlines
         glColor3d(1.0, 0.0, 0.0);
      }
      glDisable(GL_LIGHTING);
      fPShape->BoundingBox().Draw();
      glEnable(GL_LIGHTING);
   }
}

//______________________________________________________________________
void TGLManipSet::SetManipType(Int_t type)
{
   // Set manipulator type, range checked.

   if (type < 0 || type >= kEndType)
      return;
   fType = (EManip) type;
}
