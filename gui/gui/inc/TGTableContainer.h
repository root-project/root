// Author: Roel Aaij   14/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTableFrame
#define ROOT_TGTableFrame

#ifndef ROOT_TGCanvas
#include "TGCanvas.h"
#endif
#ifndef ROOT_TGTableHeader
#include "TGTableHeader.h"
#endif

class TGTableFrame : public TQObject {

protected:
   TGCompositeFrame *fFrame;  // Composite frame used as a container
   TGCanvas         *fCanvas; // Pointer to the canvas that used this frame.

public:
   TGTableFrame(const TGWindow *p, UInt_t nrows, UInt_t ncolumns);
   virtual ~TGTableFrame() { delete fFrame; }

   TGFrame *GetFrame() const { return fFrame; }

   void SetCanvas(TGCanvas *canvas) { fCanvas = canvas; }
   void HandleMouseWheel(Event_t *event);
   virtual void DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h);

   ClassDef(TGTableFrame, 0) // A frame used internally by TGTable.
};

class TGTableHeaderFrame: public TGCompositeFrame {

protected:
   Int_t    fX0;     // X coordinate of the header frame
   Int_t    fY0;     // Y coordinate of the header frame
   TGTable *fTable;  // Table that this frame belongs to

public:
   TGTableHeaderFrame(const TGWindow *p, TGTable *table = 0, UInt_t w = 1,
                      UInt_t h = 1, EHeaderType type = kColumnHeader,
                      UInt_t option = 0);
   ~TGTableHeaderFrame() {}

   virtual void DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h);

   ClassDef(TGTableHeaderFrame, 0) // A frame used internally by TGTable.
};

#endif // ROOT_TGTableFrame


