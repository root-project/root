// @(#)root/ged:$Id$
// Author: Ilka  Antcheva 08/03/05

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFrameEditor
#define ROOT_TFrameEditor


#include "TGedFrame.h"

class TGRadioButton;
class TGLineWidthComboBox;
class TGLayoutHints;
class TFrame;

class TFrameEditor : public TGedFrame {

protected:
   TFrame              *fFrame;
   TGRadioButton       *fBmode;            ///< set sinken frame border mode
   TGRadioButton       *fBmode0;           ///< set no border
   TGRadioButton       *fBmode1;           ///< set raised frame border mode
   TGLayoutHints       *fBmodelh;          ///< layout hints for border mode buttons
   TGLineWidthComboBox *fBsize;            ///< set frame border size

   virtual void ConnectSignals2Slots();

public:
   TFrameEditor(const TGWindow *p =0,
                Int_t width = 140, Int_t height = 30,
                UInt_t options = kChildFrame,
                Pixel_t back = GetDefaultFrameBackground());
   virtual ~TFrameEditor();

   virtual void   SetModel(TObject* obj);
   virtual void   DoBorderMode();
   virtual void   DoBorderSize(Int_t size);

   ClassDef(TFrameEditor,0)  //editor of TFrame objects
};

#endif
