// @(#)root/gui:$Name:  $:$Id: TRootEmbeddedCanvas.cxx,v 1.3 2001/02/14 15:39:35 rdm Exp $
// Author: Fons Rademakers   15/07/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootEmbeddedCanvas                                                  //
//                                                                      //
// This class creates a TGCanvas in which a TCanvas is created. Use     //
// GetCanvas() to get a pointer to the TCanvas.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRootEmbeddedCanvas.h"
#include "TCanvas.h"
#include "TROOT.h"


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootEmbeddedContainer                                               //
//                                                                      //
// Utility class used by TRootEmbeddedCanvas. The TRootEmbeddedContainer//
// is the frame embedded in the TGCanvas widget. The ROOT graphics goes //
// into this frame. This class is used to enable input events on this   //
// graphics frame and forward the events to the TRootEmbeddedCanvas     //
// handlers.                                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TRootEmbeddedContainer : public TGCompositeFrame {
private:
   TRootEmbeddedCanvas  *fCanvas;    // pointer back to embedded canvas
public:
   TRootEmbeddedContainer(TRootEmbeddedCanvas *c, Window_t id, const TGWindow *parent);

   Bool_t  HandleButton(Event_t *ev)
                { return fCanvas->HandleContainerButton(ev); }
   Bool_t  HandleDoubleClick(Event_t *ev)
                { return fCanvas->HandleContainerDoubleClick(ev); }
   Bool_t  HandleConfigureNotify(Event_t *ev)
                { TGFrame::HandleConfigureNotify(ev);
                  return fCanvas->HandleContainerConfigure(ev); }
   Bool_t  HandleKey(Event_t *ev)
                { return fCanvas->HandleContainerKey(ev); }
   Bool_t  HandleMotion(Event_t *ev)
                { return fCanvas->HandleContainerMotion(ev); }
   Bool_t  HandleExpose(Event_t *ev)
                { return fCanvas->HandleContainerExpose(ev); }
   Bool_t  HandleCrossing(Event_t *ev)
                { return fCanvas->HandleContainerCrossing(ev); }
};

//______________________________________________________________________________
TRootEmbeddedContainer::TRootEmbeddedContainer(TRootEmbeddedCanvas *c, Window_t id,
   const TGWindow *p) : TGCompositeFrame(gClient, id, p)
{
   // Create a canvas container.

   fCanvas = c;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                    kButtonPressMask | kButtonReleaseMask,
                    kNone, kNone);

   AddInput(kKeyPressMask | kKeyReleaseMask | kPointerMotionMask |
            kExposureMask | kStructureNotifyMask | kLeaveWindowMask);
}




ClassImp(TRootEmbeddedCanvas)

//______________________________________________________________________________
TRootEmbeddedCanvas::TRootEmbeddedCanvas(const char *name, const TGWindow *p,
            UInt_t w, UInt_t h, UInt_t options, ULong_t back)
   : TGCanvas(p, w, h, options, back)
{
   // Create an TCanvas embedded in a TGFrame. A pointer to the TCanvas can
   // be obtained via the GetCanvas() member function. To embed a canvas
   // derived from a TCanvas do the following:
   // TRootEmbeddedCanvas *embedded = new TRootEmbeddedCanvas(0, p, w, h);
   //      [note name must be 0, not null string ""]
   // Int_t wid = embedded->GetCanvasWindowId();
   // TMyCanvas *myc = new TMyCanvas("myname", 10, 10, wid);
   // embedded->AdoptCanvas(myc);
   //      [ the MyCanvas is adopted by the embedded canvas and will be
   //        destroyed by it ]

   fButton  = 0;
   fAutoFit = kTRUE;

   fCWinId = gVirtualX->InitWindow((ULong_t)GetViewPort()->GetId());
   Window_t win = gVirtualX->GetWindowID(fCWinId);
   fCanvasContainer = new TRootEmbeddedContainer(this, win, GetViewPort());
   SetContainer(fCanvasContainer);

   if (name)
      fCanvas = new TCanvas(name, 10, 10, fCWinId);
}

//______________________________________________________________________________
TRootEmbeddedCanvas::~TRootEmbeddedCanvas()
{
   // Delete embedded ROOT canvas.

   delete fCanvas;
   delete fCanvasContainer;
}

//______________________________________________________________________________
Bool_t TRootEmbeddedCanvas::HandleContainerButton(Event_t *event)
{
   // Handle mouse button events in the canvas container.

   Int_t button = event->fCode;
   Int_t x = event->fX;
   Int_t y = event->fY;

   if (event->fType == kButtonPress) {
      fButton = button;
      if (button == kButton1)
         fCanvas->HandleInput(kButton1Down, x, y);
      if (button == kButton2)
         fCanvas->HandleInput(kButton2Down, x, y);
      if (button == kButton3) {
         fCanvas->HandleInput(kButton3Down, x, y);
         fButton = 0;  // button up is consumed by TContextMenu
      }

   } else if (event->fType == kButtonRelease) {
      if (button == kButton1)
         fCanvas->HandleInput(kButton1Up, x, y);
      if (button == kButton2)
         fCanvas->HandleInput(kButton2Up, x, y);
      if (button == kButton3)
         fCanvas->HandleInput(kButton3Up, x, y);

      fButton = 0;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootEmbeddedCanvas::HandleContainerDoubleClick(Event_t *event)
{
   // Handle mouse button double click events in the canvas container.

   Int_t button = event->fCode;
   Int_t x = event->fX;
   Int_t y = event->fY;

   if (button == kButton1)
      fCanvas->HandleInput(kButton1Double, x, y);
   if (button == kButton2)
      fCanvas->HandleInput(kButton2Double, x, y);
   if (button == kButton3)
      fCanvas->HandleInput(kButton3Double, x, y);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootEmbeddedCanvas::HandleContainerConfigure(Event_t *)
{
   // Handle configure (i.e. resize) event.

   if (fAutoFit) {
      fCanvas->Resize();
      fCanvas->Update();
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootEmbeddedCanvas::HandleContainerKey(Event_t *event)
{
   // Handle keyboard events in the canvas container.

   if (event->fType == kGKeyPress) {
      fButton = event->fCode;
      UInt_t keysym;
      char str[2];
      gVirtualX->LookupString(event, str, sizeof(str), keysym);
      if (str[0] == 3)   // ctrl-c sets the interrupt flag
         gROOT->SetInterrupt();
      fCanvas->HandleInput(kKeyPress, str[0], keysym);
   } else if (event->fType == kKeyRelease)
      fButton = 0;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootEmbeddedCanvas::HandleContainerMotion(Event_t *event)
{
   // Handle mouse motion event in the canvas container.

   Int_t x = event->fX;
   Int_t y = event->fY;

   if (fButton == 0)
      fCanvas->HandleInput(kMouseMotion, x, y);
   if (fButton == kButton1)
      fCanvas->HandleInput(kButton1Motion, x, y);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootEmbeddedCanvas::HandleContainerExpose(Event_t *event)
{
   // Handle expose events.

   if (event->fCount == 0)
      fCanvas->Flush();

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootEmbeddedCanvas::HandleContainerCrossing(Event_t *event)
{
   // Handle enter/leave events. Only leave is activated at the moment.

   if (event->fType == kLeaveNotify)
      fCanvas->HandleInput(kMouseLeave, 0, 0);

   return kTRUE;
}
