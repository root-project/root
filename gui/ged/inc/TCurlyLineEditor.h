// @(#)root/ged:$Id$
// Author: Ilka  Antcheva, Otto Schaile 15/12/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCurlyLineEditor
#define ROOT_TCurlyLineEditor


#include "TGedFrame.h"

class TGNumberEntry;
class TGCheckButton;
class TCurlyLine;

class TCurlyLineEditor : public TGedFrame {

protected:
   TCurlyLine           *fCurlyLine;            ///< CurlyLineobject
   TGNumberEntry        *fStartXEntry;          ///< start  x entry
   TGNumberEntry        *fEndXEntry;            ///< end  x entry
   TGNumberEntry        *fStartYEntry;          ///< start  y entry
   TGNumberEntry        *fEndYEntry;            ///< end  y entry
   TGNumberEntry        *fAmplitudeEntry;       ///< ampl entry
   TGNumberEntry        *fWaveLengthEntry;      ///< wavelength entry
   TGCheckButton        *fIsWavy;               ///< toggle wavy / curly
   TGCompositeFrame     *fStartXFrame;

   virtual void   ConnectSignals2Slots();

public:
   TCurlyLineEditor(const TGWindow *p = 0 ,
                Int_t width = 140, Int_t height = 30,
                UInt_t options = kChildFrame,
                Pixel_t back = GetDefaultFrameBackground());
   virtual ~TCurlyLineEditor();

   virtual void   SetModel(TObject* obj);
   virtual void   ActivateBaseClassEditors(TClass* cl);
   virtual void   DoStartXY();
   virtual void   DoEndXY();
   virtual void   DoAmplitude();
   virtual void   DoWaveLength();
   virtual void   DoWavy();

   ClassDef(TCurlyLineEditor,0)  // GUI for editing arrow attributes
};

#endif
