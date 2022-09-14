// @(#)root/ged:$Id$
// Author: Ilka  Antcheva 24/04/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLineEditor
#define ROOT_TLineEditor


#include "TGedFrame.h"

class TGNumberEntry;
class TGCheckButton;
class TLine;

class TLineEditor : public TGedFrame {

protected:
   TLine           *fLine;         ///< line object
   TGNumberEntry   *fStartPointX;  ///< start point x coordinate
   TGNumberEntry   *fStartPointY;  ///< start point y coordinate
   TGNumberEntry   *fEndPointX;    ///< end point x coordinate
   TGNumberEntry   *fEndPointY;    ///< end point y coordinate
   TGCheckButton   *fVertical;     ///< set the line vertical
   TGCheckButton   *fHorizontal;   ///< set the line horizontal

   virtual void   ConnectSignals2Slots();

public:
   TLineEditor(const TGWindow *p = nullptr,
               Int_t width = 140, Int_t height = 30,
               UInt_t options = kChildFrame,
               Pixel_t back = GetDefaultFrameBackground());
   virtual ~TLineEditor();

   virtual void   SetModel(TObject* obj);
   virtual void   DoStartPoint();
   virtual void   DoEndPoint();
   virtual void   DoLineVertical();
   virtual void   DoLineHorizontal();

   ClassDef(TLineEditor,0)  // GUI for editing Line attributes
};

#endif
