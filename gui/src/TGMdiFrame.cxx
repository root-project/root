// @(#)root/gui:$Name:  $:$Id: TGMdiFrame.cxx,v 1.1 2004/09/03 00:25:47 rdm Exp $
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

#include <stdio.h>
#include <stdlib.h>
#include <TGFrame.h>

#include "TGMdiFrame.h"
#include "TGMdiMainFrame.h"
#include "TGMdiDecorFrame.h"


ClassImp(TGMdiFrame)

//______________________________________________________________________________
TGMdiFrame::TGMdiFrame(TGMdiMainFrame *main, Int_t w, Int_t h, UInt_t options,
                       Pixel_t back) :
   TGCompositeFrame(main->GetContainer(), w, h,
                    options | kOwnBackground | kMdiFrame, back)
{
   fMain = main;
   fMain->AddMdiFrame(this);  // this reparents the window
}

//______________________________________________________________________________
TGMdiFrame::~TGMdiFrame()
{
   fMain->RemoveMdiFrame(this);
}

//______________________________________________________________________________
Bool_t TGMdiFrame::CloseWindow()
{
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
   fMdiHints = mdihints;
   ((TGMdiDecorFrame *)fParent)->SetMdiButtons(mdihints);
}

//______________________________________________________________________________
void TGMdiFrame::SetWindowName(const char *name)
{
   ((TGMdiDecorFrame *)fParent)->SetWindowName(name);
   fMain->UpdateWinListMenu();
}

//______________________________________________________________________________
void TGMdiFrame::SetWindowIcon(const TGPicture *pic)
{
   ((TGMdiDecorFrame *)fParent)->SetWindowIcon(pic);
   fMain->UpdateWinListMenu();
}

//______________________________________________________________________________
const char *TGMdiFrame::GetWindowName()
{
   return ((TGMdiDecorFrame *)fParent)->GetWindowName();
}

//______________________________________________________________________________
const TGPicture *TGMdiFrame::GetWindowIcon()
{
   return ((TGMdiDecorFrame *)fParent)->GetWindowIcon();
}

//______________________________________________________________________________
void TGMdiFrame::Move(Int_t x, Int_t y)
{
   ((TGMdiDecorFrame *)fParent)->Move(x, y);
}
