// @(#)root/gui:$Name:  $:$Id: TGMdiMenu.cxx,v 1.3 2004/09/10 14:00:40 brun Exp $
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
// TGMdiMenu.                                                           //
//                                                                      //
// This file contains the TGMdiMenuBar class.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGMdi.h"
#include "TGMdiMenu.h"
#include "Riostream.h"


ClassImp(TGMdiMenuBar)

//______________________________________________________________________________
TGMdiMenuBar::TGMdiMenuBar(const TGWindow *p, int w, int h) :
   TGCompositeFrame(p, w, h, kHorizontalFrame)
{
   fLHint = new TGLayoutHints(kLHintsNormal);
   fLeftHint = new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 1, 1, 1, 1);
   fBarHint = new TGLayoutHints(kLHintsExpandX | kLHintsCenterY, 1, 1, 1, 1);
   fRightHint = new TGLayoutHints(kLHintsRight | kLHintsCenterY, 1, 2, 1, 1);

   fLeft = new TGCompositeFrame(this, 10, 10, kHorizontalFrame);
   fBar = new TGMenuBar(this, 1, 20, kHorizontalFrame);
   fRight = new TGCompositeFrame(this, 10, 10, kHorizontalFrame);

   AddFrame(fLeft,  fLeftHint);
   AddFrame(fBar,   fBarHint);
   AddFrame(fRight, fRightHint);
}

//______________________________________________________________________________
TGMdiMenuBar::~TGMdiMenuBar()
{
   //

   if (!MustCleanup()) {
      delete fLHint;
      delete fLeftHint;
      delete fRightHint;
      delete fBarHint;
   }
}

//______________________________________________________________________________
void TGMdiMenuBar::AddPopup(TGHotString *s, TGPopupMenu *menu, TGLayoutHints *l)
{
   fBar->AddPopup(s, menu, l);
   // Layout();
}

//______________________________________________________________________________
void TGMdiMenuBar::AddFrames(TGMdiTitleIcon *icon, TGMdiButtons *buttons)
{
   // This is called from TGMdiMainFrame on Maximize().

   icon->ReparentWindow(fLeft);
   buttons->ReparentWindow(fRight);
   fLeft->AddFrame(icon, fLHint);
   fRight->AddFrame(buttons, fLHint);
}

//______________________________________________________________________________
void TGMdiMenuBar::RemoveFrames(TGMdiTitleIcon *icon, TGMdiButtons *buttons)
{
   // This is called from TGMdiMainFrame on Restore()

   fLeft->RemoveFrame(icon);
   fRight->RemoveFrame(buttons);
}

//______________________________________________________________________________
void TGMdiMenuBar::SavePrimitive(ofstream &out, Option_t *option)
{
   // Save a MDI menu as a C++ statement(s) on output stream out

   out << endl;
   out << "   // MDI menu bar" << endl;

   out << "   TGMdiMenuBar *";
   out << GetName() << " = new TGMdiMenuBar(" << fParent->GetName()
       << "," << GetWidth() << "," << GetHeight() << ");" << endl;

   if (!fList) return;

   out << "   TGMenuBar *" << fBar->GetName() << " = " << GetName()
       << "->GetMenuBar();" << endl;

   TGFrameElement *el;
   TIter next(fBar->GetList());

   while ((el = (TGFrameElement *)next())) {
	     el->fFrame->SavePrimitive(out, option);
      el->fLayout->SavePrimitive(out, option);
      out << ");" << endl;
   }
}
