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

    This file is part of TGMdi, an extension to the xclass toolkit.
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

#ifndef ROOT_TGMdiFrame
#define ROOT_TGMdiFrame

#include "TGFrame.h"

class TGPicture;
class TGMdiMainFrame;
class TGMdiDecorFrame;

class TGMdiFrame : public TGCompositeFrame {

friend class TGMdiMainFrame;
friend class TGMdiDecorFrame;

protected:
   enum { kDontCallClose = BIT(14) };

   TGMdiMainFrame  *fMain;       ///< pointer to the MDI main frame
   ULong_t          fMdiHints;   ///< MDI hints, also used to identify titlebar buttons

   TString GetMdiHintsString() const;

public:
   TGMdiFrame(TGMdiMainFrame *main, Int_t w, Int_t h,
              UInt_t options = 0,
              Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGMdiFrame();

   void              Move(Int_t x, Int_t y) override;
   virtual Bool_t    CloseWindow();     //*SIGNAL*
   virtual Bool_t    Help() { return kFALSE; }

   virtual void      SetMdiHints(ULong_t mdihints);
   ULong_t           GetMdiHints() const { return fMdiHints; }

   void              DontCallClose();
   void              SetWindowName(const char *name) override;
   void              SetWindowIcon(const TGPicture *pic);
   const char       *GetWindowName();
   const TGPicture  *GetWindowIcon();

   void              SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGMdiFrame, 0) // MDI Frame
};

#endif
