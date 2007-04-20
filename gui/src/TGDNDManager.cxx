// @(#)root/gui:$Name:  $:$Id: TGDNDManager.cxx,v 1.2 2007/04/20 11:43:22 rdm Exp $
// Author: Bertrand Bellenot   19/04/07

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGFrame.h"
#include "TTimer.h"
#include "TGDNDManager.h"
#include "TRootCanvas.h"

#define ROOTDND_PROTOCOL_VERSION      4
#define XA_ATOM ((Atom_t) 4)
#define XA_WINDOW ((Atom_t) 33)

Atom_t TGDNDManager::fgDNDaware         = kNone;
Atom_t TGDNDManager::fgDNDselection     = kNone;
Atom_t TGDNDManager::fgDNDproxy         = kNone;

Atom_t TGDNDManager::fgDNDenter         = kNone;
Atom_t TGDNDManager::fgDNDleave         = kNone;
Atom_t TGDNDManager::fgDNDposition      = kNone;
Atom_t TGDNDManager::fgDNDstatus        = kNone;
Atom_t TGDNDManager::fgDNDdrop          = kNone;
Atom_t TGDNDManager::fgDNDfinished      = kNone;
Atom_t TGDNDManager::fgDNDversion       = kNone;

Atom_t TGDNDManager::fgDNDactionCopy    = kNone;
Atom_t TGDNDManager::fgDNDactionMove    = kNone;
Atom_t TGDNDManager::fgDNDactionLink    = kNone;
Atom_t TGDNDManager::fgDNDactionAsk     = kNone;
Atom_t TGDNDManager::fgDNDactionPrivate = kNone;

Atom_t TGDNDManager::fgDNDtypeList      = kNone;
Atom_t TGDNDManager::fgDNDactionList    = kNone;
Atom_t TGDNDManager::fgDNDactionDescrip = kNone;

Atom_t TGDNDManager::fgXAWMState     = kNone;
Atom_t TGDNDManager::fgXCDNDData     = kNone;

Bool_t TGDNDManager::fgInit = kFALSE;

// TODO:
// - add an TGFrame::HandleDNDstatus event handler?
// - implement INCR protocol
// - cache several requests?

TGDNDManager *gDNDManager = 0;

Cursor_t TGDragWindow::fgDefaultCursor = kNone;


//______________________________________________________________________________
TGDragWindow::TGDragWindow(const TGWindow *p, Pixmap_t pic, Pixmap_t mask,
                           UInt_t options, Pixel_t back) :
   TGFrame(p, 32, 32, options, back)
{
   // TGDragWindow constructor.

   if (fgDefaultCursor == kNone) {
      fgDefaultCursor = gVirtualX->CreateCursor(kTopLeft);
   }

   fPic = pic;
   fMask = mask;

   SetWindowAttributes_t wattr;

   wattr.fMask = kWAOverrideRedirect | kWASaveUnder;
   wattr.fSaveUnder = kTRUE;
   wattr.fOverrideRedirect = kTRUE;

   gVirtualX->ChangeWindowAttributes(fId, &wattr);

   int x, y;

   gVirtualX->GetWindowSize(fPic, x, y, fPw, fPh);

   wattr.fMask = kWAOverrideRedirect;
   wattr.fOverrideRedirect = kTRUE;

   // This input window is used to make the dragging smoother when using
   // highly complicated shapped windows (like labels and semitransparent
   // icons), for some obscure reason most of the motion events get lost
   // while the pointer is over the shaped window.

   //fInput = gVirtualX->CreateWindow(fParent->GetId(), 0, 0, fWidth,
   //                                 fHeight, 0, 0, 0, 0, &wattr, 0);
   fInput = fId;

   Resize(GetDefaultSize());

   gVirtualX->ShapeCombineMask(fId, 0, 0, fMask);

   gVirtualX->SetCursor(fId, fgDefaultCursor);
}

//______________________________________________________________________________
TGDragWindow::~TGDragWindow()
{
   // TGDragWindow destructor.

   //gVirtualX->DestroyWindow(fInput);
}

//______________________________________________________________________________
void TGDragWindow::MapWindow()
{
   // Map TGDragWindow.

   TGFrame::MapWindow();
   //gVirtualX->MapWindow(fInput);
}

//______________________________________________________________________________
void TGDragWindow::UnmapWindow()
{
   // Unmap TGDragWindow.

   TGFrame::UnmapWindow();
   //gVirtualX->UnmapWindow(fInput);
}

//______________________________________________________________________________
void TGDragWindow::RaiseWindow()
{
   // Raise TGDragWindow.

   TGFrame::RaiseWindow();
   //gVirtualX->RaiseWindow(fInput);
}

//______________________________________________________________________________
void TGDragWindow::LowerWindow()
{
   // Lower TGDragWindow.

   //gVirtualX->LowerWindow(fInput);
   TGFrame::LowerWindow();
}

//______________________________________________________________________________
void TGDragWindow::MapRaised()
{
   // Map and Raise TGDragWindow.

   TGFrame::MapRaised();
   //gVirtualX->MapRaised(fInput);
}

//______________________________________________________________________________
void TGDragWindow::Layout()
{
   // Layout TGDragWindow.

   gVirtualX->ShapeCombineMask(fId, 0, 0, fMask);
}

//______________________________________________________________________________
void TGDragWindow::DoRedraw()
{
   // Redraw TGDragWindow.

   gVirtualX->CopyArea(fPic, fId, GetBckgndGC()(), 0, 0, fWidth, fHeight, 0, 0);
}

//______________________________________________________________________________
TGDNDManager::TGDNDManager(TGFrame *toplevel, Atom_t *typelist)
{
   // TGDNDManager constructor.

   if (gDNDManager)
      return;

   fMain = toplevel;
   fVersion = ROOTDND_PROTOCOL_VERSION;
   fTypelist = typelist;

   if (!fgInit) {
      InitAtoms();
      fgInit = kTRUE;
   }

   //Reset();
   fDropTimeout = 0;

   fSource = kNone;
   fTarget = kNone;
   fTargetIsDNDaware = kFALSE;
   fStatusPending = kFALSE;
   fDropAccepted = kFALSE;  // this would become obsoleted by _acceptedAction
   fAcceptedAction = kNone; // target's accepted action
   fLocalAction = kNone;    // our last specified action when we act as source
   fDragging = kFALSE;
   fDragWin = 0;
   fLocalSource = 0;
   fLocalTarget = 0;
   fPic = fMask = kNone;
   fDraggerTypes = 0;
   fDropType = kNone;

   fGrabEventMask = kButtonPressMask | kButtonReleaseMask | kButtonMotionMask;

   fDNDNoDropCursor = gVirtualX->CreateCursor(kNoDrop); // kNoDrop

   // set the aware prop

   //if (fMain) {
   //   SetAware(fTypelist);
   //   SetTypeList(fTypelist);
   //}

   fProxyOurs = kFALSE;
   gDNDManager = this;
}

//______________________________________________________________________________
TGDNDManager::~TGDNDManager()
{
   // TGDNDManager destructor.

   // remove the proxy prop if we own it
   if (fProxyOurs)
      RemoveRootProxy();

   // remove the aware prop ant the types list, if any
   if (fMain) {
      gVirtualX->DeleteProperty(fMain->GetId(), fgDNDaware);
      gVirtualX->DeleteProperty(fMain->GetId(), fgDNDtypeList);
   }
   if (fDropTimeout) delete fDropTimeout;

   // delete the drag pixmap, if any
   if (fDragWin) {
      fDragWin->DeleteWindow();
      fDragWin = 0;
   }
   if (fPic != kNone) gVirtualX->DeletePixmap(fPic);
   if (fMask != kNone) gVirtualX->DeletePixmap(fMask);

   if (fDraggerTypes) delete[] fDraggerTypes;
}

Atom_t TGDNDManager::GetDNDaware() { return fgDNDaware; }
Atom_t TGDNDManager::GetDNDselection() { return fgDNDselection; }
Atom_t TGDNDManager::GetDNDproxy() { return fgDNDproxy; }
Atom_t TGDNDManager::GetDNDenter() { return fgDNDenter; }
Atom_t TGDNDManager::GetDNDleave() { return fgDNDleave; }
Atom_t TGDNDManager::GetDNDposition() { return fgDNDposition; }
Atom_t TGDNDManager::GetDNDstatus() { return fgDNDstatus; }
Atom_t TGDNDManager::GetDNDdrop() { return fgDNDdrop; }
Atom_t TGDNDManager::GetDNDfinished() { return fgDNDfinished; }
Atom_t TGDNDManager::GetDNDversion() { return fgDNDversion; }
Atom_t TGDNDManager::GetDNDactionCopy() { return fgDNDactionCopy; }
Atom_t TGDNDManager::GetDNDactionMove() { return fgDNDactionMove; }
Atom_t TGDNDManager::GetDNDactionLink() { return fgDNDactionLink; }
Atom_t TGDNDManager::GetDNDactionAsk() { return fgDNDactionAsk; }
Atom_t TGDNDManager::GetDNDactionPrivate() { return fgDNDactionPrivate; }
Atom_t TGDNDManager::GetDNDtypeList() { return fgDNDtypeList; }
Atom_t TGDNDManager::GetDNDactionList() { return fgDNDactionList; }
Atom_t TGDNDManager::GetDNDactionDescrip() { return fgDNDactionDescrip; }
Atom_t TGDNDManager::GetXCDNDData() { return fgXCDNDData; }

//______________________________________________________________________________
void TGDNDManager::InitAtoms()
{
   // Initialize drag and drop atoms.

   // awareness
   fgDNDaware = gVirtualX->InternAtom("XdndAware", kFALSE);

   // selection
   fgDNDselection = gVirtualX->InternAtom("XdndSelection", kFALSE);

   // proxy window
   fgDNDproxy = gVirtualX->InternAtom("XdndProxy", kFALSE);

   // messages
   fgDNDenter    = gVirtualX->InternAtom("XdndEnter", kFALSE);
   fgDNDleave    = gVirtualX->InternAtom("XdndLeave", kFALSE);
   fgDNDposition = gVirtualX->InternAtom("XdndPosition", kFALSE);
   fgDNDstatus   = gVirtualX->InternAtom("XdndStatus", kFALSE);
   fgDNDdrop     = gVirtualX->InternAtom("XdndDrop", kFALSE);
   fgDNDfinished = gVirtualX->InternAtom("XdndFinished", kFALSE);

   // actions
   fgDNDactionCopy    = gVirtualX->InternAtom("XdndActionCopy", kFALSE);
   fgDNDactionMove    = gVirtualX->InternAtom("XdndActionMove", kFALSE);
   fgDNDactionLink    = gVirtualX->InternAtom("XdndActionLink", kFALSE);
   fgDNDactionAsk     = gVirtualX->InternAtom("XdndActionAsk", kFALSE);
   fgDNDactionPrivate = gVirtualX->InternAtom("XdndActionPrivate", kFALSE);

   // types list
   fgDNDtypeList      = gVirtualX->InternAtom("XdndTypeList", kFALSE);
   fgDNDactionList    = gVirtualX->InternAtom("XdndActionList", kFALSE);
   fgDNDactionDescrip = gVirtualX->InternAtom("XdndActionDescription", kFALSE);

   // misc
   fgXAWMState = gVirtualX->InternAtom("WM_STATE", kFALSE);
   fgXCDNDData = gVirtualX->InternAtom("_XC_DND_DATA", kFALSE);
}

static int ArrayLength(Atom_t *a)
{
   // Returns length of array a.

   int n;

   for (n = 0; a[n]; n++);
   return n;
}

//______________________________________________________________________________
Bool_t TGDNDManager::IsDNDAware(Window_t win, Atom_t *typelist)
{
   // Check if window win is DND aware.

   return gVirtualX->IsDNDAware(win, typelist);
}

//______________________________________________________________________________
Window_t TGDNDManager::FindWindow(Window_t root, int x, int y, int maxd)
{
   // Search for DND aware window at position x,y.

   if (maxd <= 0) return kNone;

   if (fDragWin && fDragWin->HasWindow(root)) return kNone;

   return gVirtualX->FindRWindow(root, fDragWin ? fDragWin->GetId() : 0,
                                 fDragWin ? fDragWin->GetInputId() : 0,
                                 x, y, maxd);
}


//______________________________________________________________________________
Window_t TGDNDManager::GetRootProxy()
{
   // Get root window proxy.

   Atom_t actual;
   Int_t format = 32;
   ULong_t count, remaining;
   unsigned char *data = 0;
   Window_t win, proxy = kNone;

   // search for XdndProxy property on the root window...

   // XSync(_dpy, kFALSE);      // get to known state...
   gVirtualX->UpdateWindow(0);

   //oldhandler = XSetErrorHandler(TGDNDManager::CatchXError);
   //target_error = kFALSE;

   gVirtualX->GetProperty(gVirtualX->GetDefaultRootWindow(),
                          fgDNDproxy, 0, 1, kFALSE, XA_WINDOW,
                          &actual, &format, &count, &remaining, &data);

   if ((actual == XA_WINDOW) && (format == 32) && (count > 0) && data) {

      // found the XdndProxy property, now check for the proxy window...
      win = *((Window_t *) data);
      delete[] data;
      data = 0;

      gVirtualX->GetProperty(win, fgDNDproxy, 0, 1, kFALSE, XA_WINDOW,
                             &actual, &format, &count, &remaining, &data);

      // XSync(_dpy, kFALSE);      // force the error...
      gVirtualX->UpdateWindow(0);

      if ((actual == XA_WINDOW) && (format == 32) && (count > 0) && data) {
         if (*((Window_t *) data) == win) {

            // proxy window exists and is correct
            proxy = win;
         }
      }
   }
   if (data) delete[] data;
   //oldhandler = XSetErrorHandler(oldhandler);
   return proxy;
}

//______________________________________________________________________________
Bool_t TGDNDManager::HandleClientMessage(Event_t *event)
{
   // Handle DND related client messages.

   if (event->fHandle == fgDNDenter) {
      HandleDNDenter((Window_t) event->fUser[0], event->fUser[1],
                     (Atom_t *) &event->fUser[2]);

   } else if (event->fHandle == fgDNDleave) {
      HandleDNDleave((Window_t) event->fUser[0]);

   } else if (event->fHandle == fgDNDposition) {
      HandleDNDposition((Window_t) event->fUser[0],
                       (Int_t) (event->fUser[2] >> 16) & 0xFFFF,  // x_root
                       (Int_t) (event->fUser[2] & 0xFFFF),        // y_root
                       (Atom_t) event->fUser[4],                  // action
                       (Time_t) event->fUser[3]);                 // timestamp

   } else if (event->fHandle == fgDNDstatus) {
      Rectangle_t skip;
      skip.fX      = (event->fUser[2] >> 16) & 0xFFFF;
      skip.fY      = (event->fUser[2] & 0xFFFF);
      skip.fWidth  = (event->fUser[3] >> 16) & 0xFFFF;
      skip.fHeight = (event->fUser[3] & 0xFFFF);

      HandleDNDstatus((Window_t) event->fUser[0],
                      (int) (event->fUser[1] & 0x1),
                       skip, (Atom_t) event->fUser[4]);

   } else if (event->fHandle == fgDNDdrop) {
      HandleDNDdrop((Window_t) event->fUser[0], (Time_t) event->fUser[2]);

   } else if (event->fHandle == fgDNDfinished) {
      HandleDNDfinished((Window_t) event->fUser[0]);

   } else {
      return kFALSE;  // not for us...
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGDNDManager::HandleTimer(TTimer *t)
{
   // Handle Drop timeout.

   if (t == fDropTimeout) {
      // The drop operation timed out without receiving
      // status confirmation from the target. Send a
      // leave message instead (and notify the user or widget).
      delete fDropTimeout;
      fDropTimeout = 0;

      SendDNDleave(fTarget);
      fStatusPending = kFALSE;

      if (fLocalSource) fLocalSource->HandleDNDfinished();
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TGDNDManager::SendDNDenter(Window_t target)
{
   // Send DND enter message to target window.

   Int_t i, n;
   Event_t event;

   event.fType   = kClientMessage;
   event.fWindow = target;
   event.fHandle = fgDNDenter;
   event.fFormat = 32;

   event.fUser[0] = fMain->GetId();  // from;

   n = ArrayLength(fTypelist);

   event.fUser[1] = ((n > 3) ? 1L : 0L) | (fUseVersion << 24);

   // set the first 1-3 data types

   for (i = 0; i < 3; ++i)
      event.fUser[2+i] = (i < n) ? fTypelist[i] : kNone;

   gVirtualX->SendEvent(target, &event);
}

//______________________________________________________________________________
void TGDNDManager::SendDNDleave(Window_t target)
{
   // Send DND leave message to target window.

   Event_t event;

   event.fType    = kClientMessage;
   event.fWindow  = target;
   event.fHandle  = fgDNDleave;
   event.fFormat  = 32;

   event.fUser[0] = fMain->GetId();  // from;
   event.fUser[1] = 0L;

   event.fUser[2] = 0L;
   event.fUser[3] = 0L;
   event.fUser[4] = 0L;

   gVirtualX->SendEvent(target, &event);
}

//______________________________________________________________________________
void TGDNDManager::SendDNDposition(Window_t target, int x, int y,
                                  Atom_t action, Time_t timestamp)
{
   // Send DND position message to target window.

   Event_t event;

   event.fType    = kClientMessage;
   event.fWindow  = target;
   event.fHandle  = fgDNDposition;
   event.fFormat  = 32;

   event.fUser[0] = fMain->GetId();  // from;
   event.fUser[1] = 0L;

   event.fUser[2] = (x << 16) | y;   // root coodinates
   event.fUser[3] = timestamp;       // timestamp for retrieving data
   event.fUser[4] = action;          // requested action

   gVirtualX->SendEvent(target, &event);
}

//______________________________________________________________________________
void TGDNDManager::SendDNDstatus(Window_t source, Atom_t action)
{
   // Send DND status message to source window.

   Event_t event;

   event.fType    = kClientMessage;
   event.fWindow  = source;
   event.fHandle  = fgDNDstatus;
   event.fFormat  = 32;

   event.fUser[0] = fMain->GetId();    // from;
   event.fUser[1] = (action == kNone) ? 0L : 1L;

   event.fUser[2] = 0L;                // empty rectangle
   event.fUser[3] = 0L;
   event.fUser[4] = action;            // accepted action

   gVirtualX->SendEvent(source, &event);
}

//______________________________________________________________________________
void TGDNDManager::SendDNDdrop(Window_t target)
{
   // Send DND drop message to target window.

   Event_t event;

   event.fType    = kClientMessage;
   event.fWindow  = target;
   event.fHandle  = fgDNDdrop;
   event.fFormat  = 32;

   event.fUser[0] = fMain->GetId();    // from;
   event.fUser[1] = 0L;                // reserved
   event.fUser[2] = 0L; //CurrentTime;       // timestamp
   event.fUser[3] = 0L;
   event.fUser[4] = 0L;

   gVirtualX->SendEvent(target, &event);
}

//______________________________________________________________________________
void TGDNDManager::SendDNDfinished(Window_t source)
{
   // Send DND finished message to source window.

   Event_t event;

   event.fType    = kClientMessage;
   event.fWindow  = source;
   event.fHandle  = fgDNDfinished;
   event.fFormat  = 32;

   event.fUser[0] = fMain->GetId();    // from;
   event.fUser[1] = 0L;                // reserved
   event.fUser[2] = 0L;
   event.fUser[3] = 0L;
   event.fUser[4] = 0L;

   gVirtualX->SendEvent(source, &event);
}

//______________________________________________________________________________
Bool_t TGDNDManager::HandleDNDenter(Window_t src, Long_t vers, Atom_t dataTypes[3])
{
   // Handle DND enter event.

   fSource = src;

   if (fDraggerTypes) delete[] fDraggerTypes;

   if (vers & 1) {  // more than 3 data types?
      Atom_t type, *a;
      Int_t format = 32;
      ULong_t i, count, remaining;
      unsigned char *data = 0;

      gVirtualX->GetProperty(src, fgDNDtypeList,
                             0, 0x8000000L, kFALSE, XA_ATOM,
                             &type, &format, &count, &remaining, &data);

      if (type != XA_ATOM || format != 32 || !data) {
         count = 0;
      }

      fDraggerTypes = new Atom_t[count+4];

      a = (Atom_t *) data;
      for (i = 0; i < count; i++)
         fDraggerTypes[i] = a[i];

      fDraggerTypes[i] = kNone;

      if (data) delete[] data;

   } else {
      fDraggerTypes = new Atom_t[4];

      fDraggerTypes[0] = dataTypes[0];
      fDraggerTypes[1] = dataTypes[1];
      fDraggerTypes[2] = dataTypes[2];

      fDraggerTypes[3] = kNone;
   }

   // the following is not strictly neccessary, unless the previous
   // dragging application crashed without sending XdndLeave
   if (fLocalTarget) fLocalTarget->HandleDNDleave();
   fLocalTarget = 0;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGDNDManager::HandleDNDleave(Window_t /*src*/)
{
   // Handle DND leave event.

   fSource = kNone;
   if (fLocalTarget) fLocalTarget->HandleDNDleave();
   fLocalTarget = 0;

   if (fDraggerTypes) delete[] fDraggerTypes;
   fDraggerTypes = 0;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGDNDManager::HandleDNDposition(Window_t source, Int_t x_root, Int_t y_root,
                                      Atom_t action, Time_t /*timestamp*/)
{
   // Handle DND position event.

   Int_t x, y;
   Window_t child;
   TGFrame *f = 0, *main = 0;
   TGWindow *w = 0;
   Window_t wtarget = 0;

   wtarget = FindWindow(gVirtualX->GetDefaultRootWindow(), x_root, y_root, 10);

   if (wtarget) {
      gVirtualX->TranslateCoordinates(gVirtualX->GetDefaultRootWindow(),
                                      wtarget, x_root, y_root, x, y, child);
      w = gClient->GetWindowById(wtarget);
      if (w)
         f = dynamic_cast<TGFrame *>(w);
   }

   if (f != fLocalTarget) {
      if (fLocalTarget) fLocalTarget->HandleDNDleave();
      fLocalTarget = f;
      if (fLocalTarget) {
         main = (TGFrame *)fLocalTarget->GetMainFrame();
         main->RaiseWindow();
         if (fMain == 0)
            fMain = main;
         fDropType = fLocalTarget->HandleDNDenter(fDraggerTypes);
      }
   }
   // query the target widget to determine whether it accepts the
   // required action
   if (fLocalTarget) {
      action = (fDropType == kNone) ? kNone :
              fLocalTarget->HandleDNDposition(x, y, action, x_root, y_root);
   } else if (fProxyOurs) {
      action = fMain->HandleDNDposition(x, y, action, x_root, y_root);
   } else {
      action = kNone;
   }
   SendDNDstatus(source, fLocalAction = action);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGDNDManager::HandleDNDstatus(Window_t target, Int_t accepted,
                                    Rectangle_t /*area*/, Atom_t action)
{
   // Handle DND status event.

   if (target) {
      fStatusPending = kFALSE;
      if (accepted) {
         fDropAccepted = kTRUE;
         fAcceptedAction = action;
         if (fDragWin)
            gVirtualX->ChangeActivePointerGrab(fDragWin->GetId(),
                                               fGrabEventMask, kNone);
      } else {
         fDropAccepted = kFALSE;
         fAcceptedAction = kNone;
         if (fDragWin)
            gVirtualX->ChangeActivePointerGrab(fDragWin->GetId(),
                                               fGrabEventMask,
                                               fDNDNoDropCursor);
      }
      if (fDropTimeout) {   // were we waiting for this to do the drop?
         delete fDropTimeout;
         fDropTimeout = 0;
         SendDNDdrop(fTarget);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGDNDManager::HandleDNDdrop(Window_t source, Time_t timestamp)
{
   // Handle DND drop event.

   // to get the data, we must call XConvertSelection with
   // the timestamp in XdndDrop, wait for SelectionNotify
   // to arrive to retrieve the data, and when we are finished,
   // send a XdndFinished message to the source.

   if (fMain && fDropType != kNone) {
      gVirtualX->ChangeProperties(fMain->GetId(), fgXCDNDData, fDropType,
                                  8, (unsigned char *) 0, 0);

      gVirtualX->ConvertSelection(fMain->GetId(), fgDNDselection, fDropType,
                                  fgXCDNDData, timestamp);
   }

   fSource = source;
   SendDNDfinished(source);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGDNDManager::HandleDNDfinished(Window_t /*target*/)
{
   // Handle DND finished event.

   if (fLocalSource) fLocalSource->HandleDNDfinished();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGDNDManager::HandleSelectionRequest(Event_t *event)
{
   // Handle selection request event.

   if ((Atom_t)event->fUser[1] == fgDNDselection) {
      Event_t xevent;
      TDNDdata *dnddata = 0;
      char *data;
      int len;

      // get the data from the drag source widget
      if (fLocalSource)
         dnddata = fLocalSource->GetDNDdata(event->fUser[2]);

      data = dnddata ? (char *) dnddata->fData : (char *) "";
      len  = dnddata ? dnddata->fDataLength : 0;

      if ((Atom_t)event->fUser[3] == kNone) {
         //printf("warning: kNone property specified in SelectionRequest\n");
         event->fUser[3] = fgXCDNDData;
      }

      gVirtualX->ChangeProperties(event->fUser[0], event->fUser[3],
                                  event->fUser[2], 8,
                                  (unsigned char *) data, len);

      xevent.fType    = kSelectionNotify;
      xevent.fTime    = event->fTime;
      xevent.fUser[0] = event->fUser[0]; // requestor
      xevent.fUser[1] = event->fUser[1]; // selection
      xevent.fUser[2] = event->fUser[2]; // target;
      xevent.fUser[3] = event->fUser[3]; // property;
      gVirtualX->SendEvent(event->fUser[0], &xevent);

      return kTRUE;

   } else {
      return kFALSE;  // not for us...
   }
}

//______________________________________________________________________________
Bool_t TGDNDManager::HandleSelection(Event_t *event)
{
   // Handle selection event.

   if ((Atom_t)event->fUser[1] == fgDNDselection) {
      Atom_t actual = fDropType;
      Int_t format = 8;
      ULong_t count, remaining;
      unsigned char *data = 0;

      gVirtualX->GetProperty(event->fUser[0], event->fUser[3],
                             0, 0x8000000L, kTRUE, event->fUser[2],
                             &actual, &format, &count, &remaining, &data);

      if ((actual != fDropType) || (format != 8) || (count == 0) || !data) {
         if (data) delete[] data;
         return kFALSE;
      }

      if (fSource != kNone) SendDNDfinished(fSource);

      // send the data to the target widget

      if (fLocalTarget) {
         TDNDdata dndData(actual, data, count, fLocalAction);
         fLocalTarget->HandleDNDdrop(&dndData);
         if (fDraggerTypes) delete[] fDraggerTypes;
         fDraggerTypes = 0;
      }

      fSource = kNone;
      fLocalAction = kNone;

//      delete[] data;

      return kTRUE;

   } else {
      return kFALSE;  // not for us...
   }
}

//______________________________________________________________________________
void TGDNDManager::SetDragPixmap(Pixmap_t pic, Pixmap_t mask,
                                int hot_x, int hot_y)
{
   // Set drag window pixmaps and hotpoint.

   fPic  = pic;
   fMask = mask;
   fHotx = hot_x;
   fHoty = hot_y;
}

//______________________________________________________________________________
Bool_t TGDNDManager::StartDrag(TGFrame *src, int x_root, int y_root,
                              Window_t grabWin)
{
   // Start dragging.

   if (fDragging) return kTRUE;

   fLocalSource = src;

   if ((TGWindow *)fMain != src->GetMainFrame()) {
      fMain = (TGFrame *)src->GetMainFrame();
   }

   if (!gVirtualX->SetSelectionOwner(fMain->GetId(), fgDNDselection)) {
      // hmmm... failed to acquire ownership of XdndSelection!
      return kFALSE;
   }

   if (grabWin == kNone) grabWin = fMain->GetId();

   gVirtualX->GrabPointer(grabWin, fGrabEventMask, kNone, fDNDNoDropCursor, kTRUE, kFALSE);

   fLocalTarget = 0;
   fDragging = kTRUE;
   fTarget = kNone;
   fTargetIsDNDaware = kFALSE;
   fStatusPending = kFALSE;
   if (fDropTimeout) delete fDropTimeout;
   fDropTimeout = 0;
   fDropAccepted = kFALSE;
   fAcceptedAction = kNone;
   fLocalAction = kNone;

   if (!fDragWin && fPic != kNone && fMask != kNone) {
      fDragWin = new TGDragWindow(gClient->GetDefaultRoot(), fPic, fMask);
      fDragWin->Move((x_root-fHotx)|1, (y_root-fHoty)|1);
      fDragWin->MapSubwindows();
      fDragWin->MapRaised();
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGDNDManager::Drop()
{
   // Drop.

   if (!fDragging) return kFALSE;

   if (fTargetIsDNDaware) {
      if (fDropAccepted) {
         if (fStatusPending) {
            if (fDropTimeout) delete fDropTimeout;
            fDropTimeout = new TTimer(this, 5000);
         } else {
            SendDNDdrop(fTarget);
         }
      } else {
         SendDNDleave(fTarget);
         fStatusPending = kFALSE;
      }
   }
   EndDrag();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGDNDManager::EndDrag()
{
   // End dragging.

   if (!fDragging) return kFALSE;

   gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);

   if (fSource)
      SendDNDfinished(fSource);
   if (fLocalSource)
      fLocalSource->HandleDNDfinished();

   fDragging = kFALSE;
   if (fDragWin) {
      fDragWin->DeleteWindow();
      fDragWin = 0;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGDNDManager::Drag(int x_root, int y_root, Atom_t action, Time_t timestamp)
{
   // Process drag event.

   if (!fDragging) return kFALSE;

   Window_t newTarget = FindWindow(gVirtualX->GetDefaultRootWindow(),
                                   x_root, y_root, 10);

   if (newTarget == kNone) {
      Window_t t = GetRootProxy();
      if (t != kNone) newTarget = t;
   }

   if (fTarget != newTarget) {

      if (fTargetIsDNDaware) SendDNDleave(fTarget);

      fTarget = newTarget;
      fTargetIsDNDaware = IsDNDAware(fTarget);
      fStatusPending = kFALSE;
      fDropAccepted = kFALSE;
      fAcceptedAction = kNone;

      if (fTargetIsDNDaware) SendDNDenter(fTarget);

      if (fDragWin)
         gVirtualX->ChangeActivePointerGrab(fDragWin->GetId(), fGrabEventMask,
                                            fDNDNoDropCursor);
   }

   if (fTargetIsDNDaware && !fStatusPending) {
      SendDNDposition(fTarget, x_root, y_root, action, timestamp);

      // this is to avoid sending XdndPosition messages over and over
      // if the target is not responding
      fStatusPending = kTRUE;
   }

   if (fDragWin) {
      fDragWin->RaiseWindow();
      fDragWin->Move((x_root-fHotx)|1, (y_root-fHoty)|1);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGDNDManager::SetRootProxy()
{
   // Set root window proxy.

   Window_t mainw = fMain->GetId();
   int result = kFALSE;

   if (GetRootProxy() == kNone) {
      gVirtualX->ChangeProperties(gVirtualX->GetDefaultRootWindow(),
                                  fgDNDproxy, XA_WINDOW, 32,
                                  (unsigned char *) &mainw, 1);
      gVirtualX->ChangeProperties(mainw, fgDNDproxy, XA_WINDOW, 32,
                                  (unsigned char *) &mainw, 1);

      fProxyOurs = kTRUE;
      result = kTRUE;
   }
   // XSync(_dpy, kFALSE);
   gVirtualX->UpdateWindow(0);
   return result;
}

//______________________________________________________________________________
Bool_t TGDNDManager::RemoveRootProxy()
{
   // Remove root window proxy.

   if (!fProxyOurs) return kFALSE;

   gVirtualX->DeleteProperty(fMain->GetId(), fgDNDproxy);
   gVirtualX->DeleteProperty(gVirtualX->GetDefaultRootWindow(), fgDNDproxy);
   // the following is to ensure that the properties
   // (specially the one on the root window) are deleted
   // in case the application is exiting...

   // XSync(_dpy, kFALSE);
   gVirtualX->UpdateWindow(0);

   fProxyOurs = kFALSE;

   return kTRUE;
}
