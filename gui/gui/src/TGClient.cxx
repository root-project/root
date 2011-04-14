// @(#)root/gui:$Id$
// Author: Fons Rademakers   27/12/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGClient                                                             //
//                                                                      //
// Window client. In client server windowing systems, like X11 this     //
// class is used to make the initial connection to the window server.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfigure.h"

#include "TGClient.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TSystem.h"
#include "TEnv.h"
#include "THashList.h"
#include "TSysEvtHandler.h"
#include "TVirtualX.h"
#include "TGWindow.h"
#include "TGResourcePool.h"
#include "TGGC.h"
#include "TGFont.h"
#include "TGMimeTypes.h"
#include "TGFrame.h"
#include "TGIdleHandler.h"


// Global pointer to the TGClient object
TGClient *gClient = 0;

// Initialize gClient in case libGui is loaded in batch mode
extern "C" void G__cpp_setup_tagtableG__Gui1();
class TGClientInit {
public:
   TGClientInit() { 
      if (gROOT && gROOT->IsBatch()) {
         // Insure that the CINT dictionary is initialized __before__ the TGClient creation which
         // will induce the creation of a TClass object which will need the dictionary for TGClient!
         G__cpp_setup_tagtableG__Gui1(); 
         new TGClient();
      }
      TApplication::NeedGraphicsLibs(); 
   }
};
static TGClientInit gClientInit;


//----- Graphics Input handler -------------------------------------------------
//______________________________________________________________________________
class TGInputHandler : public TFileHandler {
private:
   TGClient  *fClient;   // connection to display server
public:
   TGInputHandler(TGClient *c, Int_t fd) : TFileHandler(fd, 1) { fClient = c; }
   Bool_t Notify();
   // Important: don't override ReadNotify()
};

//______________________________________________________________________________
Bool_t TGInputHandler::Notify()
{
   // Notify input from the display server.

   return fClient->HandleInput();
}


ClassImp(TGClient)

//______________________________________________________________________________
TGClient::TGClient(const char *dpyName)
{
   // Create a connection with the display sever on host dpyName and setup
   // the complete GUI system, i.e., graphics contexts, fonts, etc. for all
   // widgets.

   fRoot         = 0;
   fPicturePool  = 0;
   fMimeTypeList = 0;
   fWlist        = 0;
   fPlist        = 0;
   fUWHandlers   = 0;
   fIdleHandlers = 0;

   if (gClient) {
      Error("TGClient", "only one instance of TGClient allowed");
      MakeZombie();
      return;
   }

   // Set DISPLAY based on utmp (only if DISPLAY is not yet set).
   gSystem->SetDisplay();

   // Open the connection to the display
   if ((fXfd = gVirtualX->OpenDisplay(dpyName)) < 0) {
      Error("TGClient", "can't open display \"%s\", switching to batch mode...\n In case you run from a remote ssh session, reconnect with ssh -Y",
            gVirtualX->DisplayName(dpyName));
      MakeZombie();
      return;
   }

   if (fXfd >= 0 && !gROOT->IsBatch()) {
      TGInputHandler *xi = new TGInputHandler(this, fXfd);
      if (fXfd) gSystem->AddFileHandler(xi);
      // X11 events are handled via gXDisplay->Notify() in
      // TUnixSystem::DispatchOneEvent(). When no events available we wait for
      // events on all TFileHandlers including this one via a select() call.
      // However, X11 events are always handled via gXDisplay->Notify() and not
      // via the ReadNotify() (therefore TGInputHandler should not override
      // TFileHandler::ReadNotify()).
      gXDisplay = xi;
   }

   // Initialize internal window list. Use a THashList for fast
   // finding of windows based on window id (see GetWindowById()).

   fWlist = new THashList(200);
   fPlist = new TList;

   // Create root window

   fDefaultRoot = fRoot = new TGFrame(this, gVirtualX->GetDefaultRootWindow());

   // Setup some atoms (defined in TVirtualX)...

   gWM_DELETE_WINDOW = gVirtualX->InternAtom("WM_DELETE_WINDOW", kFALSE);
   gMOTIF_WM_HINTS   = gVirtualX->InternAtom("_MOTIF_WM_HINTS", kFALSE);
   gROOT_MESSAGE     = gVirtualX->InternAtom("_ROOT_MESSAGE", kFALSE);

   // Create the graphics event handler, an object for the root window,
   // a picture pool, mimetype list, etc...

   fGlobalNeedRedraw = kFALSE;
   fForceRedraw      = kFALSE;
   fWaitForWindow    = kNone;
   fWaitForEvent     = kOtherEvent;

   fResourcePool    = new TGResourcePool(this);

   fPicturePool     = fResourcePool->GetPicturePool();
   fGCPool          = fResourcePool->GetGCPool();
   fFontPool        = fResourcePool->GetFontPool();

   fMimeTypeList    = fResourcePool->GetMimeTypes();
   fDefaultColormap = fResourcePool->GetDefaultColormap();

   // Set some color defaults...

   fWhite        = fResourcePool->GetWhiteColor();
   fBlack        = fResourcePool->GetBlackColor();
   fBackColor    = fResourcePool->GetFrameBgndColor();
   fForeColor    = fResourcePool->GetFrameFgndColor();
   fHilite       = GetHilite(fBackColor);
   fShadow       = GetShadow(fBackColor);
   fSelForeColor = fResourcePool->GetSelectedFgndColor();
   fSelBackColor = fResourcePool->GetSelectedBgndColor();

   fStyle        = 0;
   TString style = gEnv->GetValue("Gui.Style", "modern");
   if (style.Contains("modern", TString::kIgnoreCase))
      fStyle = 1;

   gClient = this;
}

//______________________________________________________________________________
const TGWindow *TGClient::GetRoot() const
{
   // Returns current root (i.e. base) window. By changing the root
   // window one can change the window hierarchy, e.g. a top level
   // frame (TGMainFrame) can be embedded in another window.

   return fRoot;
}

//______________________________________________________________________________
const TGWindow *TGClient::GetDefaultRoot() const
{
   // Returns the root (i.e. desktop) window. Should only be used as parent
   // for frames that will never be embedded, like popups, message boxes,
   // etc. (like TGToolTips, TGMessageBox, etc.).

   return fDefaultRoot;
}

//______________________________________________________________________________
void TGClient::SetRoot(TGWindow *root)
{
   // Sets the current root (i.e. base) window. By changing the root
   // window one can change the window hierarchy, e.g. a top level
   // frame (TGMainFrame) can be embedded in another window.

   fRoot = root ? root : fDefaultRoot;
}

//______________________________________________________________________________
void TGClient::SetStyle(const char *style)
{
   // Set the button style (modern or classic).

   fStyle = 0;
   if (style && strstr(style, "modern"))
      fStyle = 1;
}

//______________________________________________________________________________
UInt_t TGClient::GetDisplayWidth() const
{
   // Get display width.

   Int_t  x, y;
   UInt_t w, h;

   gVirtualX->GetGeometry(-1, x, y, w, h);

   return w;
}

//______________________________________________________________________________
UInt_t TGClient::GetDisplayHeight() const
{
   // Get display height.

   Int_t  x, y;
   UInt_t w, h;

   gVirtualX->GetGeometry(-1, x, y, w, h);

   return h;
}

//______________________________________________________________________________
const TGPicture *TGClient::GetPicture(const char *name)
{
   // Get picture from the picture pool. Picture must be freed using
   // TGClient::FreePicture(). If picture is not found 0 is returned.

   return fPicturePool->GetPicture(name);
}

//______________________________________________________________________________
const TGPicture *TGClient::GetPicture(const char *name,
                                      UInt_t new_width, UInt_t new_height)
{
   // Get picture with specified size from pool (picture will be scaled if
   // necessary). Picture must be freed using TGClient::FreePicture(). If
   // picture is not found 0 is returned.

   return fPicturePool->GetPicture(name, new_width, new_height);
}

//______________________________________________________________________________
void TGClient::FreePicture(const TGPicture *pic)
{
   // Free picture resource.

   if (pic) fPicturePool->FreePicture(pic);
}

//______________________________________________________________________________
TGGC *TGClient::GetGC(GCValues_t *values, Bool_t rw)
{
   // Get graphics context from the gc pool. Context must be freed via
   // TGClient::FreeGC(). If rw is true a new read/write-able GC
   // is returned, otherwise a shared read-only context is returned.
   // For historical reasons it is also possible to create directly a
   // TGGC object, but it is advised to use this new interface only.

   return fGCPool->GetGC(values, rw);
}

//______________________________________________________________________________
void TGClient::FreeGC(const TGGC *gc)
{
   // Free a graphics context.

   fGCPool->FreeGC(gc);
}

//______________________________________________________________________________
void TGClient::FreeGC(GContext_t gc)
{
   // Free a graphics context.

   fGCPool->FreeGC(gc);
}

//______________________________________________________________________________
TGFont *TGClient::GetFont(const char *font, Bool_t fixedDefault)
{
   // Get a font from the font pool. Fonts must be freed via
   // TGClient::FreeFont(). Returns 0 in case of error or if font
   // does not exist. If fixedDefault is false the "fixed" font
   // will not be substituted as fallback when the asked for font
   // does not exist.

   return fFontPool->GetFont(font, fixedDefault);
}

//______________________________________________________________________________
TGFont *TGClient::GetFont(const TGFont *font)
{
   // Get again specified font. Will increase its usage count.

   return fFontPool->GetFont(font);
}

//______________________________________________________________________________
void TGClient::FreeFont(const TGFont *font)
{
   // Free a font.

   fFontPool->FreeFont(font);
}

//______________________________________________________________________________
void TGClient::NeedRedraw(TGWindow *w, Bool_t force)
{
   // Set redraw flags.
   if (gVirtualX->NeedRedraw((ULong_t)w,force)) return;
   if (force) {
      w->DoRedraw();
      return;
   }
   w->fNeedRedraw = kTRUE;
   fGlobalNeedRedraw = kTRUE;
}

//______________________________________________________________________________
Bool_t TGClient::GetColorByName(const char *name, Pixel_t &pixel) const
{
   // Get a color by name. If color is found return kTRUE and pixel is
   // set to the color's pixel value, kFALSE otherwise.

   ColorStruct_t      color;
   WindowAttributes_t attributes = WindowAttributes_t();
   Bool_t             status = kTRUE;

   gVirtualX->GetWindowAttributes(fRoot->GetId(), attributes);
   color.fPixel = 0;
   if (!gVirtualX->ParseColor(attributes.fColormap, name, color)) {
      Error("GetColorByName", "couldn't parse color %s", name);
      status = kFALSE;
   } else if (!gVirtualX->AllocColor(attributes.fColormap, color)) {
      Warning("GetColorByName", "couldn't retrieve color %s.\n"
              "Please close any other application, like netscape, "
              "that might exhaust\nthe colormap and start ROOT again", name);
      status = kFALSE;
   }

   pixel = color.fPixel;

   return status;
}

//______________________________________________________________________________
FontStruct_t TGClient::GetFontByName(const char *name, Bool_t fixedDefault) const
{
   // Get a font by name. If font is not found, fixed font is returned,
   // if fixed font also does not exist return 0 and print error.
   // The loaded font needs to be freed using TVirtualX::DeleteFont().
   // If fixedDefault is false the "fixed" font will not be substituted
   // as fallback when the asked for font does not exist.

   if (gROOT->IsBatch())
      return (FontStruct_t) -1;

   FontStruct_t font = gVirtualX->LoadQueryFont(name);

   if (!font && fixedDefault) {
      font = gVirtualX->LoadQueryFont("fixed");
      if (font)
         Warning("GetFontByName", "couldn't retrieve font %s, using \"fixed\"", name);
   }
   if (!font) {
      if (fixedDefault)
         Error("GetFontByName", "couldn't retrieve font %s nor backup font \"fixed\"", name);
      else
         Warning("GetFontByName", "couldn't retrieve font %s", name);
   }

   return font;
}

//______________________________________________________________________________
Pixel_t TGClient::GetHilite(Pixel_t base_color) const
{
   // Return pixel value of hilite color based on base_color.

   ColorStruct_t      color, white_p;
   WindowAttributes_t attributes = WindowAttributes_t();

   gVirtualX->GetWindowAttributes(fRoot->GetId(), attributes);

   color.fPixel = base_color;
   gVirtualX->QueryColor(attributes.fColormap, color);

   GetColorByName("white", white_p.fPixel);
   gVirtualX->QueryColor(attributes.fColormap, white_p);

   color.fRed   = TMath::Max((UShort_t)(white_p.fRed/5),   color.fRed);
   color.fGreen = TMath::Max((UShort_t)(white_p.fGreen/5), color.fGreen);
   color.fBlue  = TMath::Max((UShort_t)(white_p.fBlue/5),  color.fBlue);

   color.fRed   = (UShort_t)TMath::Min((Int_t)white_p.fRed,   (Int_t)(color.fRed*140)/100);
   color.fGreen = (UShort_t)TMath::Min((Int_t)white_p.fGreen, (Int_t)(color.fGreen*140)/100);
   color.fBlue  = (UShort_t)TMath::Min((Int_t)white_p.fBlue,  (Int_t)(color.fBlue*140)/100);

   if (!gVirtualX->AllocColor(attributes.fColormap, color))
      Error("GetHilite", "couldn't allocate hilight color");

   return color.fPixel;
}

//______________________________________________________________________________
Pixel_t TGClient::GetShadow(Pixel_t base_color) const
{
   // Return pixel value of shadow color based on base_color.
   // Shadow is 60% of base_color intensity.

   ColorStruct_t      color;
   WindowAttributes_t attributes = WindowAttributes_t();

   gVirtualX->GetWindowAttributes(fRoot->GetId(), attributes);

   color.fPixel = base_color;
   gVirtualX->QueryColor(attributes.fColormap, color);

   color.fRed   = (UShort_t)((color.fRed*60)/100);
   color.fGreen = (UShort_t)((color.fGreen*60)/100);
   color.fBlue  = (UShort_t)((color.fBlue*60)/100);

   if (!gVirtualX->AllocColor(attributes.fColormap, color))
      Error("GetShadow", "couldn't allocate shadow color");

   return color.fPixel;
}

//______________________________________________________________________________
void TGClient::FreeColor(Pixel_t color) const
{
   // Free color.

   gVirtualX->FreeColor(fDefaultColormap, color);
}

//______________________________________________________________________________
void TGClient::RegisterWindow(TGWindow *w)
{
   // Add a TGWindow to the clients list of windows.

   fWlist->Add(w);

   // Emits signal
   RegisteredWindow(w->GetId());
}

//______________________________________________________________________________
void TGClient::UnregisterWindow(TGWindow *w)
{
   // Remove a TGWindow from the list of windows.

   fWlist->Remove(w);
}

//______________________________________________________________________________
void TGClient::RegisterPopup(TGWindow *w)
{
   // Add a popup menu to the list of popups. This list is used to pass
   // events to popup menus that are popped up over a transient window which
   // is waited for (see WaitFor()).

   fPlist->Add(w);

   // Emits signal
   RegisteredWindow(w->GetId());
}

//______________________________________________________________________________
void TGClient::UnregisterPopup(TGWindow *w)
{
   // Remove a popup menu from the list of popups.

   fPlist->Remove(w);
}

//______________________________________________________________________________
void TGClient::AddUnknownWindowHandler(TGUnknownWindowHandler *h)
{
   // Add handler for unknown (i.e. unregistered) windows.

   if (!fUWHandlers) {
      fUWHandlers = new TList;
      fUWHandlers->SetOwner();
   }

   fUWHandlers->Add(h);
}

//______________________________________________________________________________
void TGClient::RemoveUnknownWindowHandler(TGUnknownWindowHandler *h)
{
   // Remove handler for unknown (i.e. unregistered) windows.

   fUWHandlers->Remove(h);
}

//______________________________________________________________________________
void TGClient::AddIdleHandler(TGIdleHandler *h)
{
   // Add handler for idle events.

   if (!fIdleHandlers) {
      fIdleHandlers = new TList;
      fIdleHandlers->SetOwner();
   }

   fIdleHandlers->Add(h);
}

//______________________________________________________________________________
void TGClient::RemoveIdleHandler(TGIdleHandler *h)
{
   // Remove handler for idle events.

   fIdleHandlers->Remove(h);
}

//______________________________________________________________________________
TGWindow *TGClient::GetWindowById(Window_t wid) const
{
   // Find a TGWindow via its handle. If window is not found return 0.

   TGWindow  wt(wid);

   return (TGWindow *) fWlist->FindObject(&wt);
}

//______________________________________________________________________________
TGWindow *TGClient::GetWindowByName(const char *name) const
{
   // Find a TGWindow via its name (unique name used in TGWindow::SavePrimitive).
   // If window is not found return 0.

   TIter next(fWlist);

   TObject *obj;
   while ((obj = next())) {
      TString n = obj->GetName();
      if (n == name) {
         return (TGWindow*)obj;
      }
   }
   return 0;
}

//______________________________________________________________________________
TGClient::~TGClient()
{
   // Closing down client: cleanup and close X connection.

   if (IsZombie())
      return;

   if (fWlist)
      fWlist->Delete("slow");
   delete fWlist;
   delete fPlist;
   delete fUWHandlers;
   delete fIdleHandlers;
   delete fResourcePool;

   gVirtualX->CloseDisplay(); // this should do a cleanup of the remaining
                              // X allocated objects...
}

//______________________________________________________________________________
Bool_t TGClient::ProcessOneEvent()
{
   // Process one event. This method should only be called when there is
   // a GUI event ready to be processed. If event has been processed
   // kTRUE is returned. If processing of a specific event type for a specific
   // window was requested kFALSE is returned when specific event has been
   // processed, kTRUE otherwise. If no more pending events return kFALSE.

   Event_t event;

   if (!fRoot) return kFALSE;
   if (gVirtualX->EventsPending()) {
      gVirtualX->NextEvent(event);
      if (fWaitForWindow == kNone) {
         HandleEvent(&event);
         if (fForceRedraw)
            DoRedraw();
         return kTRUE;
      } else {
         HandleMaskEvent(&event, fWaitForWindow);
         if ((event.fType == fWaitForEvent) && (event.fWindow == fWaitForWindow))
            fWaitForWindow = kNone;
         if (fForceRedraw)
            DoRedraw();
         return kTRUE;
      }
   }

   // if nothing else to do redraw windows that need redrawing
   if (DoRedraw()) return kTRUE;

   // process one idle event if there is nothing else to do
   if (ProcessIdleEvent()) return kTRUE;

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGClient::ProcessIdleEvent()
{
   // Process one idle event.

   if (fIdleHandlers) {
      TGIdleHandler *ih = (TGIdleHandler *) fIdleHandlers->First();
      if (ih) {
         RemoveIdleHandler(ih);
         ih->HandleEvent();
         return kTRUE;
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGClient::HandleInput()
{
   // Handles input from the display server. Returns kTRUE if one or more
   // events have been processed, kFALSE otherwise.

   Bool_t handledevent = kFALSE;

   while (ProcessOneEvent())
      handledevent = kTRUE;
   return handledevent;
}

//______________________________________________________________________________
void TGClient::WaitFor(TGWindow *w)
{
   // Wait for window to be destroyed.

   Window_t wsave    = fWaitForWindow;
   EGEventType esave = fWaitForEvent;

   fWaitForWindow = w->GetId();
   fWaitForEvent  = kDestroyNotify;

   while (fWaitForWindow != kNone) {
      if (esave == kUnmapNotify)
         wsave = kNone;
      gSystem->ProcessEvents();//gSystem->InnerLoop();
      gSystem->Sleep(5);
   }

   fWaitForWindow = wsave;
   fWaitForEvent  = esave;
}

//______________________________________________________________________________
void TGClient::WaitForUnmap(TGWindow *w)
{
   // Wait for window to be unmapped.

   Window_t wsave    = fWaitForWindow;
   EGEventType esave = fWaitForEvent;

   fWaitForWindow = w->GetId();
   fWaitForEvent  = kUnmapNotify;

   while (fWaitForWindow != kNone) {
      gSystem->ProcessEvents();//gSystem->InnerLoop();
      gSystem->Sleep(5);
   }

   fWaitForWindow = wsave;
   fWaitForEvent  = esave;
}

//______________________________________________________________________________
void TGClient::ResetWaitFor(TGWindow *w)
{
   // reset waiting

   if (fWaitForWindow == w->GetId()) fWaitForWindow = kNone;
}

//______________________________________________________________________________
Bool_t TGClient::ProcessEventsFor(TGWindow *w)
{
   // Like gSystem->ProcessEvents() but then only allow events for w to
   // be processed. For example to interrupt the processing and destroy
   // the window, call gROOT->SetInterrupt() before destroying the window.

   Window_t wsave    = fWaitForWindow;
   EGEventType esave = fWaitForEvent;

   fWaitForWindow = w->GetId();
   fWaitForEvent  = kDestroyNotify;

   Bool_t intr = gSystem->ProcessEvents();

   fWaitForWindow = wsave;
   fWaitForEvent  = esave;

   return intr;
}

//______________________________________________________________________________
Bool_t TGClient::DoRedraw()
{
   // Redraw all windows that need redrawing. Returns kFALSE if no redraw
   // was needed, kTRUE otherwise.
   // Only redraw the application's windows when the event queue
   // does not contain expose event anymore.

   if (!fGlobalNeedRedraw) return kFALSE;

   TGWindow *w;
   TObjLink *lnk = fWlist->FirstLink();
   while (lnk) {
      w = (TGWindow *) lnk->GetObject();
      if (w->fNeedRedraw) {
         w->DoRedraw();
         w->fNeedRedraw = kFALSE;
      }
      lnk = lnk->Next();
   }

   fGlobalNeedRedraw = kFALSE;
   fForceRedraw      = kFALSE;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGClient::HandleEvent(Event_t *event)
{
   // Handle a GUI event.

   TGWindow *w;

   // Emit signal for event recorder(s)
   if (event->fType != kConfigureNotify) {
      ProcessedEvent(event, 0);
   }

   // Find window where event happened
   if ((w = GetWindowById(event->fWindow)) == 0) {
      if (fUWHandlers && fUWHandlers->GetSize() > 0) {
         TGUnknownWindowHandler *unkwh;
         TListIter it(fUWHandlers);
         while ((unkwh = (TGUnknownWindowHandler*)it.Next())) {
            if (unkwh->HandleEvent(event))
               return kTRUE;
         }
      }
      //Warning("HandleEvent", "unknown window %ld not handled\n",
      //        event->fWindow);
      return kFALSE;
   }

   // and let it handle the event
   w->HandleEvent(event);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGClient::HandleMaskEvent(Event_t *event, Window_t wid)
{
   // Handle masked events only if window wid is the window for which the
   // event was reported or if wid is a parent of the event window. The not
   // masked event are handled directly. The masked events are:
   // kButtonPress, kButtonRelease, kKeyPress, kKeyRelease, kEnterNotify,
   // kLeaveNotify, kMotionNotify.

   TGWindow *w, *ptr, *pop;

   if ((w = GetWindowById(event->fWindow)) == 0) return kFALSE;

   // Emit signal for event recorder(s)
   if (event->fType != kConfigureNotify) {
      ProcessedEvent(event, wid);
   }

   // This breaks class member protection, but TGClient is a friend of
   // TGWindow and _should_ know what to do and what *not* to do...

   for (ptr = w; ptr->fParent != 0; ptr = (TGWindow *) ptr->fParent) {
      if ((ptr->fId == wid) ||
          ((event->fType != kButtonPress) &&
           (event->fType != kButtonRelease) &&
           (event->fType != kGKeyPress) &&
           (event->fType != kKeyRelease) &&
           (event->fType != kEnterNotify) &&
           (event->fType != kLeaveNotify) &&
           (event->fType != kMotionNotify))) {
         w->HandleEvent(event);
         return kTRUE;
      }
   }

   // check if this is a popup menu
   TIter next(fPlist);
   while ((pop = (TGWindow *) next())) {
      for (ptr = w; ptr->fParent != 0; ptr = (TGWindow *) ptr->fParent) {
         if ((ptr->fId == pop->fId) &&
             ((event->fType == kButtonPress) ||
              (event->fType == kButtonRelease) ||
              (event->fType == kGKeyPress) ||
              (event->fType == kKeyRelease) ||
              (event->fType == kEnterNotify) ||
              (event->fType == kLeaveNotify) ||
              (event->fType == kMotionNotify))) {
            w->HandleEvent(event);
            return kTRUE;
         }
      }
   }

   if (event->fType == kButtonPress || event->fType == kGKeyPress)
      gVirtualX->Bell(0);

   return kFALSE;
}

//______________________________________________________________________________
void TGClient::ProcessLine(TString cmd, Long_t msg, Long_t parm1, Long_t parm2)
{
   // Execute string "cmd" via the interpreter. Before executing replace
   // in the command string the token $MSG, $PARM1 and $PARM2 by msg,
   // parm1 and parm2, respectively. The function in cmd string must accept
   // these as longs.

   if (cmd.IsNull()) return;

   char s[32];

   snprintf(s, sizeof(s), "%ld", msg);
   cmd.ReplaceAll("$MSG", s);

   snprintf(s, sizeof(s), "%ld", parm1);
   cmd.ReplaceAll("$PARM1", s);

   snprintf(s, sizeof(s), "%ld", parm2);
   cmd.ReplaceAll("$PARM2", s);

   gROOT->ProcessLine(cmd.Data());
}

//______________________________________________________________________________
Bool_t TGClient::IsEditDisabled() const
{
   // Returns kTRUE if edit/guibuilding is forbidden.

   return (fDefaultRoot->GetEditDisabled() == 1);
}

//______________________________________________________________________________
void TGClient::SetEditDisabled(Bool_t on)
{
   // If on is kTRUE editting/guibuilding is forbidden.

   fDefaultRoot->SetEditDisabled(on);
}

//______________________________________________________________________________
void TGClient::ProcessedEvent(Event_t *event, Window_t wid)
{
   // Emits a signal when an event has been processed.
   // Used in TRecorder.

   Long_t args[2];
   args[0] = (Long_t) event;
   args[1] = (Long_t) wid;

   Emit("ProcessedEvent(Event_t*, Window_t)", args);
}

//______________________________________________________________________________
void TGClient::RegisteredWindow(Window_t w)
{
   // Emits a signal when a Window has been registered in TGClient.
   // Used in TRecorder.

   Emit("RegisteredWindow(Window_t)", w);
}
