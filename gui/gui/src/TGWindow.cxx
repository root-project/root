// @(#)root/gui:$Id$
// Author: Fons Rademakers   28/12/97

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
// TGWindow                                                             //
//                                                                      //
// ROOT GUI Window base class.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGWindow.h"
#include "Riostream.h"
#include "TApplication.h"

ClassImp(TGWindow)
ClassImp(TGUnknownWindowHandler)

Int_t TGWindow::fgCounter = 0;

//______________________________________________________________________________
TGWindow::TGWindow(const TGWindow *p, Int_t x, Int_t y, UInt_t w, UInt_t h,
                   UInt_t border, Int_t depth, UInt_t clss, void *visual,
                   SetWindowAttributes_t *attr, UInt_t wtype)
{
   // Create a new window. Parent p must exist otherwise the root window
   // is taken as parent. No arguments specified results in values from
   // parent to be taken (or defaults).

   UInt_t type = wtype;
   fId = 0;
   fParent = 0;

   if (!p && gClient) {
      p = gClient->GetRoot();
   }

   if (p) {
      fClient = p->fClient;
      if (fClient->IsEditable()) type = wtype & ~1;

      fParent = p;
      if (fParent && fParent->IsMapSubwindows()) {
         fId = gVirtualX->CreateWindow(fParent->fId, x, y,
                                     TMath::Max(w, (UInt_t) 1),
                                     TMath::Max(h, (UInt_t) 1), border,
                                     depth, clss, visual, attr, type);
         fClient->RegisterWindow(this);
      }
      fNeedRedraw = kFALSE;

      // name will be used in SavePrimitive methods
      fgCounter++;
      fName = "frame";
      fName += fgCounter;
   }
   fEditDisabled = (fId != gVirtualX->GetDefaultRootWindow()) && fParent ?
                    (fParent->fEditDisabled == kEditDisable) : 0;

   // add protection for the root window on Cocoa (MacOS X)
   if (fClient && fClient->GetDefaultRoot())
      SetWindowName();
}

//______________________________________________________________________________
TGWindow::TGWindow(TGClient *c, Window_t id, const TGWindow *parent)
{
   // Create a copy of a window.

   fClient = c;
   fId     = id;
   fParent = parent;
   fClient->RegisterWindow(this);
   fNeedRedraw = kFALSE;
   fEditDisabled = (fId != gVirtualX->GetDefaultRootWindow()) && fParent ?
                    fParent->fEditDisabled : kFALSE;

   // name used in SavePrimitive methods
   fgCounter++;
   fName = "frame";
   fName += fgCounter;
}

//______________________________________________________________________________
TGWindow::~TGWindow()
{
   // Window destructor. Unregisters the window.

   if (fClient) {
      if (fParent == fClient->GetDefaultRoot())
         DestroyWindow();
      fClient->UnregisterWindow(this);
   }
}

//______________________________________________________________________________
void TGWindow::SetWindowName(const char *name)
{
   // Set window name.

   if (!name && gDebug > 0) {
      // set default frame names only when in debug mode
      TString wname = ClassName();
      wname += "::" + fName;
      gVirtualX->SetWindowName(fId, (char *)wname.Data());
   } else {
      gVirtualX->SetWindowName(fId, (char *)name);
   }
}

//______________________________________________________________________________
const TGWindow *TGWindow::GetMainFrame() const
{
   // Returns top level main frame.

   return ((fParent == 0) || (fParent == fClient->GetDefaultRoot())) ? this : fParent->GetMainFrame();
}

//______________________________________________________________________________
void TGWindow::ReparentWindow(const TGWindow *p, Int_t x, Int_t y)
{
   // Reparent window, make p the new parent and position the window at
   // position (x,y) in new parent.

   if (p == fParent) return;

   if (p) {
      gVirtualX->ReparentWindow(fId, p->GetId(), x, y);
      gVirtualX->Update(1);
   }
   fParent = p;
}

//______________________________________________________________________________
void TGWindow::Move(Int_t x, Int_t y)
{
   // Move the window.

   gVirtualX->MoveWindow(fId, x, y);
}

//______________________________________________________________________________
void TGWindow::Resize(UInt_t w, UInt_t h)
{
   // Resize the window.

   gVirtualX->ResizeWindow(fId, TMath::Max(w, (UInt_t)1), TMath::Max(h, (UInt_t)1));
}

//______________________________________________________________________________
void TGWindow::MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Move and resize the window.

   gVirtualX->MoveResizeWindow(fId, x, y, TMath::Max(w, (UInt_t)1), TMath::Max(h, (UInt_t)1));
}

//______________________________________________________________________________
Bool_t TGWindow::IsMapped()
{
   // Returns kTRUE if window is mapped on screen, kFALSE otherwise.

   WindowAttributes_t attr;

   gVirtualX->GetWindowAttributes(fId, attr);
   return (attr.fMapState != kIsUnmapped);
}

//______________________________________________________________________________
void TGWindow::Print(Option_t *option) const
{
   // Print window id.
   // If option is "tree" - print all parent windows tree

   TString opt = option;

   if (opt.Contains("tree")) {

      const TGWindow *parent = fParent;
      std::cout << ClassName() << ":\t" << fId << std::endl;

      while (parent && (parent != fClient->GetDefaultRoot())) {
         std::cout << "\t" << parent->ClassName() << ":\t" << parent->GetId() << std::endl;
         parent = parent->GetParent();
      }
   } else {
      std::cout << ClassName() << ":\t" << fId << std::endl;
   }
}

//______________________________________________________________________________
Int_t TGWindow::GetCounter()
{
   // Return global window counter (total number of created windows).

   return fgCounter;
}

//______________________________________________________________________________
const char *TGWindow::GetName()const
{
   // Return unique name, used in SavePrimitive methods.

   TGWindow *w = (TGWindow*)this;

   if (fName.BeginsWith("frame")) {
      TString cname = ClassName();
      if (cname.BeginsWith("TGed"))
         cname.Replace(0, 1, 'f');
      else if (cname.BeginsWith("TG"))
         cname.Replace(0,2,'f');
      else
         cname.Replace(0, 1, 'f');
      w->fName.Remove(0,5);
      w->fName = cname + w->fName;
   }

   if (w->fName.Contains(" "))
      w->fName.ReplaceAll(" ", "");
   if (w->fName.Contains(":"))
      w->fName.ReplaceAll(":", "");

   return fName.Data();
}
