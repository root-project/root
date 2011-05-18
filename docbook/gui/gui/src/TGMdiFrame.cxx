// @(#)root/gui:$Id$
// Author: Bertrand Bellenot   20/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**************************************************************************

    This file is part of TGMdi an extension to the xclass toolkit.
    Copyright (C) 1998-2002 by Harald Radke, Hector Peraza.

    This application is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This application is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGMdiFrame.                                                          //
//                                                                      //
// This file contains the TGMdiFrame class.                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGFrame.h"
#include "TGMdiFrame.h"
#include "TGMdiMainFrame.h"
#include "TGMdiDecorFrame.h"
#include "Riostream.h"

ClassImp(TGMdiFrame)

//______________________________________________________________________________
TGMdiFrame::TGMdiFrame(TGMdiMainFrame *main, Int_t w, Int_t h, UInt_t options,
                       Pixel_t back) :
   TGCompositeFrame(main->GetContainer(), w, h,
                    options | kOwnBackground | kMdiFrame, back)
{
   // TGMdiFrame constructor.

   fMain = main;
   fMain->AddMdiFrame(this);  // this reparents the window
   fMdiHints = kMdiDefaultHints;
}

//______________________________________________________________________________
TGMdiFrame::~TGMdiFrame()
{
   // TGMdiFrame destructor.

   Cleanup();
   fMain->RemoveMdiFrame(this);
}

//______________________________________________________________________________
Bool_t TGMdiFrame::CloseWindow()
{
   // Close MDI frame window.

   DeleteWindow();
   return kTRUE;
}

//______________________________________________________________________________
void TGMdiFrame::DontCallClose()
{
   // Typically call this method in the slot connected to the CloseWindow()
   // signal to prevent the calling of the default or any derived CloseWindow()
   // methods to prevent premature or double deletion of this window.

   SetBit(kDontCallClose);
}

//______________________________________________________________________________
void TGMdiFrame::SetMdiHints(ULong_t mdihints)
{
   // Set MDI hints, also used to identify titlebar buttons.

   fMdiHints = mdihints;
   ((TGMdiDecorFrame *)fParent)->SetMdiButtons(mdihints);
}

//______________________________________________________________________________
void TGMdiFrame::SetWindowName(const char *name)
{
   // Set MDI window name (set titlebar title).

   ((TGMdiDecorFrame *)fParent)->SetWindowName(name);
   fMain->UpdateWinListMenu();
}

//______________________________________________________________________________
void TGMdiFrame::SetWindowIcon(const TGPicture *pic)
{
   // Set MDI window icon (titlebar icon).

   ((TGMdiDecorFrame *)fParent)->SetWindowIcon(pic);
   fMain->UpdateWinListMenu();
}

//______________________________________________________________________________
const char *TGMdiFrame::GetWindowName()
{
   // Return MDI window name.

   return ((TGMdiDecorFrame *)fParent)->GetWindowName();
}

//______________________________________________________________________________
const TGPicture *TGMdiFrame::GetWindowIcon()
{
   // Return pointer to picture used as MDI window icon (on titlebar).

   return ((TGMdiDecorFrame *)fParent)->GetWindowIcon();
}

//______________________________________________________________________________
void TGMdiFrame::Move(Int_t x, Int_t y)
{
   // Move MDI window at position x, y.

   ((TGMdiDecorFrame *)fParent)->Move(x, y);
   fX = x; fY = y;
}

//______________________________________________________________________________
TString TGMdiFrame::GetMdiHintsString() const
{
   // Returns a MDI option string - used in SavePrimitive().

   TString hints;
   if (fMdiHints == kMdiDefaultHints)
      hints = "kMdiDefaultHints";
   else {
      if (fMdiHints & kMdiClose) {
         if (hints.Length() == 0) hints = "kMdiClose";
         else                     hints += " | kMdiClose";
      }
      if (fMdiHints & kMdiRestore) {
         if (hints.Length() == 0) hints = "kMdiRestore";
         else                     hints += " | kMdiRestore";
      }
      if (fMdiHints & kMdiMove) {
         if (hints.Length() == 0) hints = "kMdiMove";
         else                     hints += " | kMdiMove";
      }
      if (fMdiHints & kMdiSize) {
         if (hints.Length() == 0) hints = "kMdiSize";
         else                     hints += " | kMdiSize";
      }
      if (fMdiHints & kMdiMinimize) {
         if (hints.Length() == 0) hints = "kMdiMinimize";
         else                     hints += " | kMdiMinimize";
      }
      if (fMdiHints & kMdiMaximize) {
         if (hints.Length() == 0) hints = "kMdiMaximize";
         else                     hints += " | kMdiMaximize";
      }
      if (fMdiHints & kMdiHelp) {
         if (hints.Length() == 0) hints = "kMdiHelp";
         else                     hints += " | kMdiHelp";
      }
      if (fMdiHints & kMdiMenu) {
         if (hints.Length() == 0) hints = "kMdiMenu";
         else                     hints += " | kMdiMenu";
      }
   }
   return hints;
}

//______________________________________________________________________________
void TGMdiFrame::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a MDIframe as a C++ statement(s) on output stream out

   char quote = '"';
   
   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);
   
   TGMdiTitleBar *tb = fMain->GetWindowList()->GetDecorFrame()->GetTitleBar();
   
   out << endl <<"   // MDI frame "<< quote << GetWindowName() << quote << endl;
   out << "   TGMdiFrame *";
   out << GetName() << " = new TGMdiFrame(" << fMain->GetName()
       << "," << GetWidth() + GetBorderWidth()*2 
       << "," << GetHeight() + tb->GetHeight() + GetBorderWidth()*2;

   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         out << ");" << endl;
      } else {
         out << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }
   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;

   SavePrimitiveSubframes(out, option);
   
   out << "   " << GetName() << "->SetWindowName(" << quote << GetWindowName()
       << quote << ");" << endl;
   out << "   " << GetName() << "->SetMdiHints(" << GetMdiHintsString()
       << ");" << endl;
   if ((GetX() != 5) && (GetY() != 23))
      out << "   " << GetName() << "->Move(" << GetX() << "," << GetY() 
          << ");" << endl;
          
   out << "   " << GetName() << "->MapSubwindows();" << endl;
   out << "   " << GetName() << "->Layout();" << endl;
}
