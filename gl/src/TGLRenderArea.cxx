// @(#)root/gl:$Name:  $:$Id: TGLRenderArea.cxx,v 1.6 2005/12/06 17:52:04 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualGL.h"
#include "TVirtualX.h"
#include "TGClient.h"

#include "TGLRenderArea.h"

ClassImp(TGLWindow)
ClassImp(TGLRenderArea)


//______________________________________________________________________________
TGLWindow::TGLWindow(Window_t id, const TGWindow *parent)
               :TGCompositeFrame(gClient, id, parent), fCtx(0)
{
   // Constructor.

   fCtx = gVirtualGL->CreateContext(fId);
   //here add diagnostic
   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier, kButtonPressMask | kButtonReleaseMask, kNone, kNone);
   gVirtualX->SelectInput(fId, kKeyPressMask | kExposureMask | kPointerMotionMask | kStructureNotifyMask | kFocusChangeMask);
   gVirtualX->SetInputFocus(fId);
}


//______________________________________________________________________________
TGLWindow::~TGLWindow()
{
   // Destructor.

   gVirtualGL->DeleteContext(fCtx);
}


//______________________________________________________________________________
Bool_t TGLWindow::HandleConfigureNotify(Event_t *event)
{
   // Handle configure notify.

   Emit("HandleConfigureNotify(Event_t*)", (Long_t)event);
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TGLWindow::HandleButton(Event_t *event)
{
   // Ensure we take focus so keyboard events are captured.

   RequestFocus();
   Emit("HandleButton(Event_t*)", (Long_t)event);
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TGLWindow::HandleDoubleClick(Event_t *event)
{
   // Handle double click.

   Emit("HandleDoubleClick(Event_t*)", (Long_t)event);
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TGLWindow::HandleKey(Event_t *event)
{
   // Handle key.

   Emit("HandleKey(Event_t*)", (Long_t)event);
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TGLWindow::HandleMotion(Event_t *event)
{
   // Handle motion.

   Emit("HandleMotion(Event_t*)", (Long_t)event);
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TGLWindow::HandleExpose(Event_t *event)
{
   // Handle expose.
   
   Emit("HandleExpose(Event_t*)", (Long_t)event);
   return kTRUE;
}


//______________________________________________________________________________
void TGLWindow::SwapBuffers()
{
   // Swap buffers.
   
   gVirtualGL->SwapBuffers(fId);
}


//______________________________________________________________________________
void TGLWindow::MakeCurrent()
{
   // Make current.
   gVirtualGL->MakeCurrent(fId, fCtx);
}


//______________________________________________________________________________
TGLRenderArea::TGLRenderArea()
                  :fArea(0)
{
   // Constructor.
}


//______________________________________________________________________________
TGLRenderArea::TGLRenderArea(Window_t wid, const TGWindow *parent)
                  :fArea(0)
{
   // Constructor.

   Window_t glWin = gVirtualGL->CreateGLWindow(wid);
   fArea = new TGLWindow(glWin, parent);
}


//______________________________________________________________________________
TGLRenderArea::~TGLRenderArea()
{
   // Destructor.

   delete fArea;
}
