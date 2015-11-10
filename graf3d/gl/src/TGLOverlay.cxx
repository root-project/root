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

/** \class TGLOverlayElement
\ingroup opengl
An overlay element. Supports drawing (Render) and event-handling
*/

ClassImp(TGLOverlayElement);

////////////////////////////////////////////////////////////////////////////////
/// Mouse has entered this element.
/// Return TRUE if you want additional events.

Bool_t TGLOverlayElement::MouseEnter(TGLOvlSelectRecord& /*selRec*/)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Handle overlay event.
/// Return TRUE if event was handled.

Bool_t TGLOverlayElement::Handle(TGLRnrCtx          & /*rnrCtx*/,
                                 TGLOvlSelectRecord & /*selRec*/,
                                 Event_t            * /*event*/)
{
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Mouse has left the element.

void TGLOverlayElement::MouseLeave()
{
}


/** \class TGLOverlayList
\ingroup opengl
Manage a collection of overlay elements.
Not used yet.
*/

ClassImp(TGLOverlayList);
