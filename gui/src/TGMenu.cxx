// @(#)root/gui:$Name:  $:$Id: TGMenu.cxx,v 1.3 2000/10/04 23:40:07 rdm Exp $
// Author: Fons Rademakers   09/01/98

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
// TGMenuBar, TGPopupMenu, TGMenuTitle and TGMenuEntry                  //
//                                                                      //
// This header contains all different menu classes.                     //
//                                                                      //
// Selecting a menu item will generate the event:                       //
// kC_COMMAND, kCM_MENU, menu id, user data.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGMenu.h"
#include "TTimer.h"
#include "TMath.h"
#include "TSystem.h"


ClassImp(TGMenuBar)
ClassImp(TGMenuTitle)
ClassImpQ(TGPopupMenu)


//______________________________________________________________________________
class TPopupDelayTimer : public TTimer {
private:
   TGPopupMenu   *fPopup;
public:
   TPopupDelayTimer(TGPopupMenu *p, Long_t ms) : TTimer(ms, kTRUE) { fPopup = p; }
   Bool_t Notify();
};

//______________________________________________________________________________
Bool_t TPopupDelayTimer::Notify()
{
   fPopup->HandleTimer(0);
   Reset();
   return kFALSE;
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGMenuBar member functions.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TGMenuBar::TGMenuBar(const TGWindow *p, UInt_t w, UInt_t h, UInt_t options)
   : TGHorizontalFrame(p, w, h, options | kHorizontalFrame)
{
   // Create a menu bar object.

   fCurrent = 0;
   fTitles  = new TList;
   fStick   = kTRUE;

   gVirtualX->GrabButton(fId, kButton1, kAnyModifier,
                    kButtonPressMask | kButtonReleaseMask | kEnterWindowMask,
                    kNone, kNone);
}

//______________________________________________________________________________
TGMenuBar::~TGMenuBar()
{
   // Delete menu bar object. Removes also the hot keys from the main frame,
   // so hitting them will not cause the menus to popup.

   TGFrameElement *el;
   TGMenuTitle    *t;
   Int_t           keycode;

   const TGMainFrame *main = (TGMainFrame *) GetMainFrame();

   TIter next(fList);
   while ((el = (TGFrameElement *) next())) {
      t = (TGMenuTitle *) el->fFrame;
      if ((keycode = t->GetHotKeyCode()) != 0)
         main->RemoveBind(this, keycode, kKeyMod1Mask);
   }

   // delete TGMenuTitles
   if (fTitles) fTitles->Delete();
   delete fTitles;
}

//______________________________________________________________________________
void TGMenuBar::AddPopup(TGHotString *s, TGPopupMenu *menu, TGLayoutHints *l)
{
   // Add popup menu to menu bar. The hot string will be adopted by the
   // menubar (actually the menu title) and deleted when possible.

   TGMenuTitle *t;
   Int_t keycode;

   AddFrame(t = new TGMenuTitle(this, s, menu), l);
   fTitles->Add(t);  // keep track of menu titles for later cleanup in dtor

   if ((keycode = t->GetHotKeyCode()) != 0) {
      const TGMainFrame *main = (TGMainFrame *) GetMainFrame();
      main->BindKey(this, keycode, kKeyMod1Mask);
   }
}

//______________________________________________________________________________
void TGMenuBar::AddPopup(const char *s, TGPopupMenu *menu, TGLayoutHints *l)
{
   // Add popup menu to menu bar.

   AddPopup(new TGHotString(s), menu, l);
}

//______________________________________________________________________________
void TGMenuBar::AddFrame(TGFrame *f, TGLayoutHints *l)
{
   // Private version of AddFrame for menubar, to make sure that we
   // indeed only add TGMenuTitle objects to it.

   if (f->IsA() == TGMenuTitle::Class())
      TGCompositeFrame::AddFrame(f, l);
   else
      Error("AddFrame", "may only add TGMenuTitle objects to a menu bar");
}

//______________________________________________________________________________
Bool_t TGMenuBar::HandleMotion(Event_t *event)
{
   // Handle a mouse motion event in a menu bar.

   Int_t        dummy;
   Window_t     wtarget;
   TGMenuTitle *target;

   fStick = kFALSE; // use some threshold!

   gVirtualX->TranslateCoordinates(fId, fId, event->fX, event->fY,
                              dummy, dummy, wtarget);
   target = (TGMenuTitle*) fClient->GetWindowById(wtarget);

   if (target != 0 && target != fCurrent) {
      // deactivate all others
      TGFrameElement *el;
      TIter next(fList);
      while ((el = (TGFrameElement *) next()))
         ((TGMenuTitle*)el->fFrame)->SetState(kFALSE);

      target->SetState(kTRUE);
      fStick   = kTRUE;
      fCurrent = target;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGMenuBar::HandleButton(Event_t *event)
{
   // Handle a mouse button event in a menubar.

   Int_t        dummy;
   Window_t     wtarget;
   TGMenuTitle *target;

   // We don't need to check the button number as GrabButton will
   // only allow button1 events

   if (event->fType == kButtonPress) {

      gVirtualX->TranslateCoordinates(fId, fId, event->fX, event->fY,
                                 dummy, dummy, wtarget);
      target = (TGMenuTitle*) fClient->GetWindowById(wtarget);

      if (target != 0) {
         fStick = kTRUE;

         if (target != fCurrent) {
            // deactivate all others
            TGFrameElement *el;
            TIter next(fList);
            while ((el = (TGFrameElement *) next()))
               ((TGMenuTitle*)el->fFrame)->SetState(kFALSE);

            target->SetState(kTRUE);
            fStick   = kTRUE;
            fCurrent = target;

            gVirtualX->GrabPointer(fId, kButtonPressMask | kButtonReleaseMask |
                              kPointerMotionMask, kNone, fgDefaultCursor);
         }
      }
   }

   if (event->fType == kButtonRelease) {
      if (fStick) {
         fStick = kFALSE;
         return kTRUE;
      }
      gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);  // ungrab pointer

      TGFrameElement *el;
      TIter next(fList);
      while ((el = (TGFrameElement *) next()))
         ((TGMenuTitle*)el->fFrame)->SetState(kFALSE);

      if (fCurrent != 0) {
         target   = fCurrent; // tricky, because WaitFor
         fCurrent = 0;
         target->DoSendMessage();
      }
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGMenuBar::HandleKey(Event_t *event)
{
   // Handle keyboard events in a menu bar.

   TGMenuTitle *target = 0;

   if (event->fType == kGKeyPress) {

      TGFrameElement *el;
      TIter next(fList);
      while ((el = (TGFrameElement *) next())) {
         target = (TGMenuTitle *) el->fFrame;
         if ((Int_t)event->fCode == target->GetHotKeyCode()) break;
      }
      if (el == 0) target = 0;

      if (target != 0) {
         fStick = kTRUE;

         if (target != fCurrent) {
            // deactivate all others
            next.Reset();
            while ((el = (TGFrameElement *) next()))
               ((TGMenuTitle*)el->fFrame)->SetState(kFALSE);

            target->SetState(kTRUE);
            fStick   = kTRUE;
            fCurrent = target;

            gVirtualX->GrabPointer(fId, kButtonPressMask | kButtonReleaseMask |
                              kPointerMotionMask, kNone, fgDefaultCursor);
         }
      }
   }

   if (event->fType == kKeyRelease) {
      if (fStick) {
         fStick = kFALSE;
         return kTRUE;
      }
      gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);  // ungrab pointer

      TGFrameElement *el;
      TIter next(fList);
      while ((el = (TGFrameElement *) next()))
         ((TGMenuTitle*)el->fFrame)->SetState(kFALSE);

      if (fCurrent != 0) {
         target   = fCurrent; // tricky, because WaitFor
         fCurrent = 0;
         target->DoSendMessage();
      }
   }

   return kTRUE;
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGPopupMenu member functions.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TGPopupMenu::TGPopupMenu(const TGWindow *p, UInt_t w, UInt_t h, UInt_t options)
    : TGFrame(p, w, h, options | kOwnBackground)
{
   // Create a popup menu.

   fNormGC       = fgDefaultGC();
   fSelGC        = fgDefaultSelectedGC();
   fSelbackGC    = fgDefaultSelectedBackgroundGC();
   fFontStruct   = fgDefaultFontStruct;
   fHifontStruct = fgHilightFontStruct;

   fDelay     = 0;
   fEntryList = new TList;

   fBorderWidth = 3;
   fHeight      = 6;
   fWidth       = 8;
   fXl          = 16;
   fMsgWindow   = p;
   fStick       = kTRUE;
   fCurrent     = 0;
   fHasGrab     = kFALSE;

   SetWindowAttributes_t wattr;
   wattr.fMask             = kWAOverrideRedirect | kWASaveUnder;
   wattr.fOverrideRedirect = kTRUE;
   wattr.fSaveUnder        = kTRUE;

   gVirtualX->ChangeWindowAttributes(fId, &wattr);

   AddInput(kPointerMotionMask | kEnterWindowMask | kLeaveWindowMask);
}

//______________________________________________________________________________
TGPopupMenu::~TGPopupMenu()
{
   // Delete a popup menu.

   if (fEntryList) fEntryList->Delete();
   delete fEntryList;
   delete fDelay;
}

//______________________________________________________________________________
void TGPopupMenu::AddEntry(TGHotString *s, Int_t id, void *ud, const TGPicture *p)
{
   // Add a menu entry. The hotstring is adopted by the menu (actually by
   // the TGMenuEntry) and deleted when possible. A possible picture is
   // borrowed from the picture pool and therefore not adopted.

   TGMenuEntry *nw = new TGMenuEntry;

   nw->fLabel    = s;
   nw->fPic      = p;
   nw->fType     = kMenuEntry;
   nw->fEntryId  = id;
   nw->fUserData = ud;
   nw->fPopup    = 0;
   nw->fStatus   = kMenuEnableMask;
   nw->fEx       = 2;
   nw->fEy       = fHeight-2;

   fEntryList->Add(nw);

   UInt_t tw, pw = 0;
   tw = gVirtualX->TextWidth(fHifontStruct, s->GetString(), s->GetLength());
   if (p) {
      pw = p->GetWidth();
      if (pw+12 > fXl) { fWidth += pw+12-fXl; fXl = pw+12; }
   }

   Int_t max_ascent, max_descent;
   fWidth = TMath::Max(fWidth, tw + pw /*+8*/+18+12);
   gVirtualX->GetFontProperties(fHifontStruct, max_ascent, max_descent);
   fHeight += max_ascent + max_descent + 3;

   Resize(fWidth, fHeight);
}

//______________________________________________________________________________
void TGPopupMenu::AddEntry(const char *s, Int_t id, void *ud, const TGPicture *p)
{
   // Add a menu entry. The string s in not adopted.

   AddEntry(new TGHotString(s), id, ud, p);
}

//______________________________________________________________________________
void TGPopupMenu::AddSeparator()
{
   // Add a menu separator to the menu.

   TGMenuEntry *nw = new TGMenuEntry;

   nw->fLabel    = 0;
   nw->fPic      = 0;
   nw->fType     = kMenuSeparator;
   nw->fEntryId  = -1;
   nw->fUserData = 0;
   nw->fPopup    = 0;
   nw->fStatus   = kMenuEnableMask;
   nw->fEx       = 2;
   nw->fEy       = fHeight-2;

   fEntryList->Add(nw);

   fHeight += 4;

   Resize(fWidth, fHeight);
}

//______________________________________________________________________________
void TGPopupMenu::AddLabel(TGHotString *s, const TGPicture *p)
{
   // Add a menu label to the menu. The hotstring is adopted by the menu
   // (actually by the TGMenuEntry) and deleted when possible. A possible
   // picture is borrowed from the picture pool and therefore not adopted.

   TGMenuEntry *nw = new TGMenuEntry;

   nw->fLabel    = s;
   nw->fPic      = p;
   nw->fType     = kMenuLabel;
   nw->fEntryId  = -1;
   nw->fUserData = 0;
   nw->fPopup    = 0;
   nw->fStatus   = kMenuEnableMask | kMenuDefaultMask;
   nw->fEx       = 2;
   nw->fEy       = fHeight-2;

   fEntryList->Add(nw);

   UInt_t tw, pw = 0;
   tw = gVirtualX->TextWidth(fHifontStruct, s->GetString(), s->GetLength());
   if (p) {
      pw = p->GetWidth();
      if (pw+12 > fXl) { fWidth += pw+12-fXl; fXl = pw+12; }
   }

   Int_t max_ascent, max_descent;
   fWidth = TMath::Max(fWidth, tw + pw /*+8*/+18+12);
   gVirtualX->GetFontProperties(fHifontStruct, max_ascent, max_descent);
   fHeight += max_ascent + max_descent + 3;

   Resize(fWidth, fHeight);
}

//______________________________________________________________________________
void TGPopupMenu::AddLabel(const char *s, const TGPicture *p)
{
   // Add a menu label to the menu. The string s in not adopted.

   AddLabel(new TGHotString(s), p);
}

//______________________________________________________________________________
void TGPopupMenu::AddPopup(TGHotString *s, TGPopupMenu *popup)
{
   // Add a (cascading) popup menu to a popup menu. The hotstring is adopted
   // by the menu (actually by the TGMenuEntry) and deleted when possible.

   TGMenuEntry *nw = new TGMenuEntry;

   nw->fLabel    = s;
   nw->fPic      = 0;
   nw->fType     = kMenuPopup;
   nw->fEntryId  = -2;
   nw->fUserData = 0;
   nw->fPopup    = popup;
   nw->fStatus   = kMenuEnableMask;
   nw->fEx       = 2;
   nw->fEy       = fHeight-2;

   fEntryList->Add(nw);

   UInt_t tw = gVirtualX->TextWidth(fHifontStruct, s->GetString(), s->GetLength());

   Int_t max_ascent, max_descent;
   fWidth = TMath::Max(fWidth, tw +8+18+12);
   gVirtualX->GetFontProperties(fHifontStruct, max_ascent, max_descent);
   fHeight += max_ascent + max_descent + 3;

   Resize(fWidth, fHeight);
}

//______________________________________________________________________________
void TGPopupMenu::AddPopup(const char *s, TGPopupMenu *popup)
{
   // Add a (cascading) popup menu to a popup menu. The string s is not
   // adopted.

   AddPopup(new TGHotString(s), popup);
}

//______________________________________________________________________________
void TGPopupMenu::PlaceMenu(Int_t x, Int_t y, Bool_t stick_mode, Bool_t grab_pointer)
{
   // Popup a popup menu. If stick mode is true keep the menu up. If
   // grab_pointer is true the pointer will be grabbed, which means that
   // all pointer events will go to the popup menu, idependent of in
   // which window the pointer is.

   Int_t  rx, ry;
   UInt_t rw, rh;

   fStick = stick_mode;
   fCurrent = 0;

   // Parent is root window for a popup menu
   gVirtualX->GetWindowSize(fParent->GetId(), rx, ry, rw, rh);

   if (x < 0) x = 0;
   if (x + fWidth > rw) x = rw - fWidth;
   if (y < 0) y = 0;
   if (y + fHeight > rh) y = rh - fHeight;

   Move(x, y);
   MapRaised();

   if (grab_pointer) {
      gVirtualX->GrabPointer(fId, kButtonPressMask | kButtonReleaseMask |
                        kPointerMotionMask, kNone, fgDefaultCursor);
      fHasGrab = kTRUE;
   } else {
      fHasGrab = kFALSE;
   }
}

//______________________________________________________________________________
Int_t TGPopupMenu::EndMenu(void *&userData)
{
   // Close menu and return ID of selected menu item.
   // In case of cascading menus, recursively close all menus.

   Int_t id;

   if (fDelay) fDelay->Remove();

   // destroy any cascaded childs and get any ID

   if (fCurrent != 0) {

      // deactivate the entry
      fCurrent->fStatus &= ~kMenuActiveMask;

      if (fCurrent->fType == kMenuPopup) {
         id = fCurrent->fPopup->EndMenu(userData);
      } else {
         // return the ID if the entry is enabled, otherwise -1
         if (fCurrent->fStatus & kMenuEnableMask) {
            id       = fCurrent->fEntryId;
            userData = fCurrent->fUserData;
         } else {
            id       = -1;
            userData = 0;
         }
      }

   } else {
      // if no entry selected...
      id       = -1;
      userData = 0;
   }

   // then unmap itself
   UnmapWindow();

   return id;
}

//______________________________________________________________________________
Bool_t TGPopupMenu::HandleButton(Event_t *event)
{
   // Handle button event in the popup menu.

   int   id;
   void *ud;

   if (event->fType == kButtonRelease) {
      if (fStick) {
         fStick = kFALSE;
         return kTRUE;
      }
//    if (fCurrent != NULL)
//      if (fCurrent->fType == kMenuPopup) return kTRUE;
      id = EndMenu(ud);
      if (fHasGrab) gVirtualX->GrabPointer(0, 0, 0, 0, kFALSE);  // ungrab
      if (fCurrent != 0) {
         fCurrent->fStatus &= ~kMenuActiveMask;
         if (fCurrent->fStatus & kMenuEnableMask) {
            SendMessage(fMsgWindow, MK_MSG(kC_COMMAND, kCM_MENU), id,
                        (Long_t)fCurrent->fUserData);
            Activated(id);
         }
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGPopupMenu::HandleCrossing(Event_t *event)
{
   // Handle pointer crossing event in popup menu.

   if (event->fType == kEnterNotify) {
      Int_t h, max_ascent, max_descent;
      gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);

      TGMenuEntry *ptr;
      TIter next(fEntryList);

      while ((ptr = (TGMenuEntry *) next())) {
         if (ptr->fType == kMenuSeparator) {
            h = 4;
         } else {
            h = max_ascent + max_descent + 3;
         }
         if ((event->fX >= ptr->fEx) && (event->fX <= ptr->fEx+(Int_t)fWidth-10) &&
             (event->fY >= ptr->fEy) && (event->fY <= ptr->fEy+h)) break;
      }
      Activate(ptr);
   } else {
      Activate(0);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGPopupMenu::HandleMotion(Event_t *event)
{
   // Handle pointer motion event in popup menu.

   TGFrame::HandleMotion(event);

   Int_t h, max_ascent, max_descent;
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);

   TGMenuEntry *ptr;
   TIter next(fEntryList);

   fStick = kFALSE;   // be careful with this, use some threshold
   while ((ptr = (TGMenuEntry *) next())) {
      if (ptr->fType == kMenuSeparator) {
         h = 4;
      } else {
         h = max_ascent + max_descent + 3;
      }
      if ((event->fX >= ptr->fEx) && (event->fX <= ptr->fEx+(Int_t)fWidth-4) &&  //fWidth-10??
          (event->fY >= ptr->fEy) && (event->fY <= ptr->fEy+h)) break;
   }
   Activate(ptr);

   return kTRUE;
}

//______________________________________________________________________________
void TGPopupMenu::Activate(TGMenuEntry *entry)
{
   // Activate a menu entry in a popup menu.

   if (entry == fCurrent) return;

   //-- Deactivate the current entry

   if (fCurrent != 0) {
      void *ud;
      if (entry == 0 && fCurrent->fType == kMenuPopup) return;
      if (fCurrent->fType == kMenuPopup) fCurrent->fPopup->EndMenu(ud);
      fCurrent->fStatus &= ~kMenuActiveMask;
      DrawEntry(fCurrent);
   }

   if (fDelay) fDelay->Remove();

   //-- Activate the new one

   if (entry) {
      entry->fStatus |= kMenuActiveMask;
      DrawEntry(entry);
      if (entry->fType == kMenuPopup) {
         if (!fDelay) fDelay = new TPopupDelayTimer(this, 350);
         fDelay->Reset();
         gSystem->AddTimer(fDelay);
         // after delay expires it will popup the cascading popup menu
         // iff it is still the current entry
      } else if (entry->fType == kMenuEntry) {
         // test...
         SendMessage(fMsgWindow, MK_MSG(kC_COMMAND, kCM_MENUSELECT),
                     entry->fEntryId, (Long_t)entry->fUserData);
         Highlighted(entry->fEntryId);
      }
   }
   fCurrent = entry;
}

//______________________________________________________________________________
Bool_t TGPopupMenu::HandleTimer(TTimer *)
{
   // If TPopupDelayTimer times out popup cascading popup menu (if it is
   // still the current entry).

   if (fCurrent != 0)
      if (fCurrent->fType == kMenuPopup) {
         Int_t    ax, ay;
         Window_t wdummy;

         gVirtualX->TranslateCoordinates(fId,
                                    (fCurrent->fPopup->GetParent())->GetId(),
                                    fCurrent->fEx+fWidth, fCurrent->fEy,
                                    ax, ay, wdummy);

         fCurrent->fPopup->PlaceMenu(ax-5, ay-1, kFALSE, kFALSE);
      }

   fDelay->Remove();

   return kTRUE;
}

//______________________________________________________________________________
void TGPopupMenu::DoRedraw()
{
   // Draw popup menu.

   TGFrame::DoRedraw();

   TGMenuEntry *ptr;
   TIter next(fEntryList);

   while ((ptr = (TGMenuEntry *) next()))
      DrawEntry(ptr);
}

//______________________________________________________________________________
void TGPopupMenu::DrawEntry(TGMenuEntry *entry)
{
   // Draw popup menu entry.

   FontStruct_t  font;
   GCValues_t    gcval;

   if (entry->fStatus & kMenuDefaultMask) {
      font = fHifontStruct;
      gcval.fMask = kGCFont;
      gcval.fFont = gVirtualX->GetFontHandle(font);
      gVirtualX->ChangeGC(fNormGC, &gcval);
      gVirtualX->ChangeGC(fSelGC, &gcval);
   } else {
      font = fFontStruct;
   }

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(font, max_ascent, max_descent);
   int tx = entry->fEx + fXl;
   int ty = entry->fEy + max_ascent;

   switch (entry->fType) {
      case kMenuPopup:
      case kMenuLabel:
      case kMenuEntry:
         if ((entry->fStatus & kMenuActiveMask) && entry->fType != kMenuLabel) {
            gVirtualX->FillRectangle(fId, fSelbackGC, entry->fEx+1, entry->fEy-1,
                                fWidth-6, max_ascent + max_descent + 3);
            if (entry->fType == kMenuPopup)
               DrawTrianglePattern(fSelGC, fWidth-10, entry->fEy+3, fWidth-6, entry->fEy+11);
            if (entry->fStatus & kMenuCheckedMask)
               DrawCheckMark(fSelGC, 6, entry->fEy+3, 14, entry->fEy+11);
            if (entry->fStatus & kMenuRadioMask)
               DrawRCheckMark(fSelGC, 6, entry->fEy+3, 14, entry->fEy+11);
            if (entry->fPic != 0)
               entry->fPic->Draw(fId, fSelGC, 8, entry->fEy+1);
            entry->fLabel->Draw(fId,
                           (entry->fStatus & kMenuEnableMask) ? fSelGC : fgShadowGC(),
                           tx, ty);
         } else {
            gVirtualX->FillRectangle(fId, fgBckgndGC(), entry->fEx+1, entry->fEy-1,
                                fWidth-6, max_ascent + max_descent + 3);
            if (entry->fType == kMenuPopup)
               DrawTrianglePattern(fNormGC, fWidth-10, entry->fEy+3, fWidth-6, entry->fEy+11);
            if (entry->fStatus & kMenuCheckedMask)
               DrawCheckMark(fNormGC, 6, entry->fEy+3, 14, entry->fEy+11);
            if (entry->fStatus & kMenuRadioMask)
               DrawRCheckMark(fNormGC, 6, entry->fEy+3, 14, entry->fEy+11);
            if (entry->fPic != 0)
               entry->fPic->Draw(fId, fNormGC, 8, entry->fEy+1);
            if (entry->fStatus & kMenuEnableMask) {
               entry->fLabel->Draw(fId, fNormGC, tx, ty);
            } else {
               entry->fLabel->Draw(fId, fgHilightGC(), tx+1, ty+1);
               entry->fLabel->Draw(fId, fgShadowGC(), tx, ty);
            }
         }
         break;

      case kMenuSeparator:
         gVirtualX->DrawLine(fId, fgShadowGC(),  2, entry->fEy, fWidth-3, entry->fEy);
         gVirtualX->DrawLine(fId, fgHilightGC(), 2, entry->fEy+1, fWidth-3, entry->fEy+1);
         break;
   }

   // restore font
   if (entry->fStatus & kMenuDefaultMask) {
      gcval.fFont = gVirtualX->GetFontHandle(fFontStruct);
      gVirtualX->ChangeGC(fNormGC, &gcval);
      gVirtualX->ChangeGC(fSelGC, &gcval);
   }
}

//______________________________________________________________________________
void TGPopupMenu::DrawBorder()
{
   // Draw border round popup menu.

   gVirtualX->DrawLine(fId, fgBckgndGC(), 0, 0, fWidth-2, 0);
   gVirtualX->DrawLine(fId, fgBckgndGC(), 0, 0, 0, fHeight-2);
   gVirtualX->DrawLine(fId, fgHilightGC(), 1, 1, fWidth-3, 1);
   gVirtualX->DrawLine(fId, fgHilightGC(), 1, 1, 1, fHeight-3);

   gVirtualX->DrawLine(fId, fgShadowGC(),  1, fHeight-2, fWidth-2, fHeight-2);
   gVirtualX->DrawLine(fId, fgShadowGC(),  fWidth-2, fHeight-2, fWidth-2, 1);
   gVirtualX->DrawLine(fId, fgBlackGC(), 0, fHeight-1, fWidth-1, fHeight-1);
   gVirtualX->DrawLine(fId, fgBlackGC(), fWidth-1, fHeight-1, fWidth-1, 0);
}

//______________________________________________________________________________
void TGPopupMenu::DrawTrianglePattern(GContext_t gc, Int_t l, Int_t t,
                                      Int_t r, Int_t b)
{
   // Draw triangle pattern. Used for menu entries that are of type
   // kMenuPopup (i.e. cascading menus).

   Point_t  points[3];

   int m = (t + b) >> 1;

   points[0].fX = l;
   points[0].fY = t;
   points[1].fX = l;
   points[1].fY = b;
   points[2].fX = r;
   points[2].fY = m;

   gVirtualX->FillPolygon(fId, gc, points, 3);
}

//______________________________________________________________________________
void TGPopupMenu::DrawCheckMark(GContext_t gc, Int_t l, Int_t t, Int_t, Int_t b)
{
   // Draw check mark. Used for checked button type menu entries.

   Segment_t seg[6];

   t = (t + b - 8) >> 1; ++t;

   seg[0].fX1 = 1+l; seg[0].fY1 = 3+t; seg[0].fX2 = 3+l; seg[0].fY2 = 5+t;
   seg[1].fX1 = 1+l; seg[1].fY1 = 4+t; seg[1].fX2 = 3+l; seg[1].fY2 = 6+t;
   seg[2].fX1 = 1+l; seg[2].fY1 = 5+t; seg[2].fX2 = 3+l; seg[2].fY2 = 7+t;
   seg[3].fX1 = 3+l; seg[3].fY1 = 5+t; seg[3].fX2 = 7+l; seg[3].fY2 = 1+t;
   seg[4].fX1 = 3+l; seg[4].fY1 = 6+t; seg[4].fX2 = 7+l; seg[4].fY2 = 2+t;
   seg[5].fX1 = 3+l; seg[5].fY1 = 7+t; seg[5].fX2 = 7+l; seg[5].fY2 = 3+t;

   gVirtualX->DrawSegments(fId, gc, seg, 6);
}

//______________________________________________________________________________
void TGPopupMenu::DrawRCheckMark(GContext_t gc, Int_t l, Int_t t, Int_t r, Int_t b)
{
   // Draw radio check mark. Used for radio button type menu entries.

   Segment_t seg[5];

   t = (t + b - 5) >> 1; ++t;
   l = (l + r - 5) >> 1; ++l;

   seg[0].fX1 = 1+l; seg[0].fY1 = 0+t; seg[0].fX2 = 3+l; seg[0].fY2 = 0+t;
   seg[1].fX1 = 0+l; seg[1].fY1 = 1+t; seg[1].fX2 = 4+l; seg[1].fY2 = 1+t;
   seg[2].fX1 = 0+l; seg[2].fY1 = 2+t; seg[2].fX2 = 4+l; seg[2].fY2 = 2+t;
   seg[3].fX1 = 0+l; seg[3].fY1 = 3+t; seg[3].fX2 = 4+l; seg[3].fY2 = 3+t;
   seg[4].fX1 = 1+l; seg[4].fY1 = 4+t; seg[4].fX2 = 3+l; seg[4].fY2 = 4+t;

   gVirtualX->DrawSegments(fId, gc, seg, 5);
}

//______________________________________________________________________________
void TGPopupMenu::DefaultEntry(Int_t id)
{
   // Set default entry (default entries are drawn with bold text).

   TGMenuEntry *ptr;
   TIter next(fEntryList);

   while ((ptr = (TGMenuEntry *) next())) {
      if (ptr->fEntryId == id)
         ptr->fStatus |= kMenuDefaultMask;
      else
         ptr->fStatus &= ~kMenuDefaultMask;
   }
}

//______________________________________________________________________________
void TGPopupMenu::EnableEntry(Int_t id)
{
   // Enable entry.

   TGMenuEntry *ptr;
   TIter next(fEntryList);

   while ((ptr = (TGMenuEntry *) next()))
      if (ptr->fEntryId == id) { ptr->fStatus |= kMenuEnableMask; break; }
}

//______________________________________________________________________________
void TGPopupMenu::DisableEntry(Int_t id)
{
   // Disable entry (disabled entries appear in a sunken relieve).

   TGMenuEntry *ptr;
   TIter next(fEntryList);

   while ((ptr = (TGMenuEntry *) next()))
      if (ptr->fEntryId == id) { ptr->fStatus &= ~kMenuEnableMask; break; }
}

//______________________________________________________________________________
Bool_t TGPopupMenu::IsEntryEnabled(Int_t id)
{
   // Return true if menu entry is enabled.

   TGMenuEntry *ptr;
   TIter next(fEntryList);

   while ((ptr = (TGMenuEntry *) next()))
      if (ptr->fEntryId == id)
         return (ptr->fStatus & kMenuEnableMask) ? kTRUE : kFALSE;
   return kFALSE;
}

//______________________________________________________________________________
void TGPopupMenu::CheckEntry(Int_t id)
{
   // Check a menu entry (i.e. add a check mark in front of it).

   TGMenuEntry *ptr;
   TIter next(fEntryList);

   while ((ptr = (TGMenuEntry *) next()))
      if (ptr->fEntryId == id) { ptr->fStatus |= kMenuCheckedMask; break; }
}

//______________________________________________________________________________
void TGPopupMenu::UnCheckEntry(Int_t id)
{
   // Uncheck menu entry (i.e. remove check mark).

   TGMenuEntry *ptr;
   TIter next(fEntryList);

   while ((ptr = (TGMenuEntry *) next()))
      if (ptr->fEntryId == id) { ptr->fStatus &= ~kMenuCheckedMask; break; }
}

//______________________________________________________________________________
Bool_t TGPopupMenu::IsEntryChecked(Int_t id)
{
   // Return true if menu item is checked.

   TGMenuEntry *ptr;
   TIter next(fEntryList);

   while ((ptr = (TGMenuEntry *) next()))
      if (ptr->fEntryId == id)
         return (ptr->fStatus & kMenuCheckedMask) ? kTRUE : kFALSE;
   return kFALSE;
}

//______________________________________________________________________________
void TGPopupMenu::RCheckEntry(Int_t id, Int_t IDfirst, Int_t IDlast)
{
   // Radio-select entry (note that they cannot be unselected,
   // the selection must be moved to another entry instead).

   TGMenuEntry *ptr;
   TIter next(fEntryList);

   while ((ptr = (TGMenuEntry *) next()))
      if (ptr->fEntryId == id)
         ptr->fStatus |= kMenuRadioMask;
      else
         if (ptr->fEntryId >= IDfirst && ptr->fEntryId <= IDlast)
            ptr->fStatus &= ~kMenuRadioMask;
}

//______________________________________________________________________________
Bool_t TGPopupMenu::IsEntryRChecked(Int_t id)
{
   // Return true if menu item has radio check mark.

   TGMenuEntry *ptr;
   TIter next(fEntryList);

   while ((ptr = (TGMenuEntry *) next()))
      if (ptr->fEntryId == id)
         return (ptr->fStatus & kMenuRadioMask) ? kTRUE : kFALSE;
   return kFALSE;
}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGMenuTitle member functions.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TGMenuTitle::TGMenuTitle(const TGWindow *p, TGHotString *s, TGPopupMenu *menu,
                         GContext_t norm, FontStruct_t font, UInt_t options)
    : TGFrame(p, 1, 1, options)
{
   // Create a menu title. This object is created by a menu bar when adding
   // a popup menu. The menu title adopts the hotstring.

   fLabel      = s;
   fMenu       = menu;
   fFontStruct = font;
   fSelGC      = fgDefaultSelectedGC();
   fNormGC     = norm;
   fState      = kFALSE;
   fTitleId    = -1;

   Int_t hotchar;
   if ((hotchar = s->GetHotChar()) != 0)
      fHkeycode = gVirtualX->KeysymToKeycode(hotchar);
   else
      fHkeycode = 0;

   UInt_t tw;
   Int_t  max_ascent, max_descent;
   tw = gVirtualX->TextWidth(fFontStruct, fLabel->GetString(), fLabel->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);

   Resize(tw + 8, max_ascent + max_descent + 7);
}

//______________________________________________________________________________
void TGMenuTitle::SetState(Bool_t state)
{
   // Set state of menu title.

   fState = state;
   if (state) {
      if (fMenu != 0) {
         Int_t    ax, ay;
         Window_t wdummy;

         gVirtualX->TranslateCoordinates(fId, (fMenu->GetParent())->GetId(),
                                    0, 0, ax, ay, wdummy);

         // Place the menu just under the window :
         fMenu->PlaceMenu(ax-1, ay+fHeight, kTRUE, kFALSE); //kTRUE);
      }
   } else {
      if (fMenu != 0) {
         fTitleId = fMenu->EndMenu(fTitleData);
      }
   }
   fOptions &= ~(kSunkenFrame | kRaisedFrame);
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGMenuTitle::DoRedraw()
{
   // Draw a menu title.

   TGFrame::DoRedraw();

   int x, y, max_ascent, max_descent;
   x = y = 4;

   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);

   if (fState) {
      SetBackgroundColor(fgDefaultSelectedBackground);
      gVirtualX->ClearWindow(fId);
      fLabel->Draw(fId, fSelGC, x, y + max_ascent);
   } else {
      SetBackgroundColor(fgDefaultFrameBackground);
      gVirtualX->ClearWindow(fId);
      fLabel->Draw(fId, fNormGC, x, y + max_ascent);
   }
}

//______________________________________________________________________________
void TGMenuTitle::DoSendMessage()
{
   // Send final selected menu item to be processed.

   if (fMenu)
      if (fTitleId != -1) {
         SendMessage(fMenu->fMsgWindow, MK_MSG(kC_COMMAND, kCM_MENU), fTitleId,
                     (Long_t)fTitleData);
         fMenu->Activated(fTitleId);
      }
}

//______________________________________________________________________________
FontStruct_t TGMenuTitle::GetDefaultFontStruct()
{ return fgDefaultFontStruct; }

//______________________________________________________________________________
const TGGC &TGMenuTitle::GetDefaultGC()
{ return fgDefaultGC; }
