// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLOverlay.h"

//==============================================================================
// TGLOverlayElement
//==============================================================================

//______________________________________________________________________
//
// An overlay element. Supports drawing (Render) and event-handling
//
//

ClassImp(TGLOverlayElement);

//______________________________________________________________________
Bool_t TGLOverlayElement::MouseEnter(TGLOvlSelectRecord& /*selRec*/)
{
   // Mouse has enetered this element.
   // Return TRUE if you want additional events.

   return kFALSE;
}

Bool_t TGLOverlayElement::MouseStillInside(TGLOvlSelectRecord& /*selRec*/)
{
   // A new overlay hit is about to be processed.
   // By returning FALSE one can force mouse-leave (MouseLeave will be
   // called shortly).
   // If you return TRUE, Handle will be called soon.
   // Use this if your overlay object has some inactive parts,
   // see TGLManipSet.

   return kTRUE;
}

//______________________________________________________________________
Bool_t TGLOverlayElement::Handle(TGLRnrCtx          & /*rnrCtx*/,
                                 TGLOvlSelectRecord & /*selRec*/,
                                 Event_t            * /*event*/)
{
   // Handle overlay event.
   // Return TRUE if event was handled.

   return kFALSE;
}

//______________________________________________________________________
void TGLOverlayElement::MouseLeave()
{
   // Mouse has left the element.
}


//==============================================================================
// TGLOverlayList
//==============================================================================

//______________________________________________________________________
//
// Manage a collection of overlay elements.
//
// Not used yet.

ClassImp(TGLOverlayList);
