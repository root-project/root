// @(#)root/ged:$Id$
// Author: Ilka  Antcheva, Otto Schaile 15/12/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCurlyArcEditor
#define ROOT_TCurlyArcEditor


#include "TGedFrame.h"

class TGNumberEntry;
class TCurlyArc;

class TCurlyArcEditor : public TGedFrame {

protected:
   TCurlyArc            *fCurlyArc;         ///< CurlyArc object
   TGNumberEntry        *fRadiusEntry;      ///< radius entry
   TGNumberEntry        *fPhiminEntry;      ///< Phimin entry
   TGNumberEntry        *fPhimaxEntry;      ///< Phimax entry
   TGNumberEntry        *fCenterXEntry;     ///< center x entry
   TGNumberEntry        *fCenterYEntry;     ///< center y entry

   virtual void   ConnectSignals2Slots();

public:
   TCurlyArcEditor(const TGWindow *p = 0,
                Int_t width = 140, Int_t height = 30,
                UInt_t options = kChildFrame,
                Pixel_t back = GetDefaultFrameBackground());
   virtual ~TCurlyArcEditor();

   virtual void   SetModel(TObject* obj);
   virtual void   DoRadius();
   virtual void   DoPhimin();
   virtual void   DoPhimax();
   virtual void   DoCenterXY();

   ClassDef(TCurlyArcEditor,0)  // GUI for editing arrow attributes
};

#endif
