// @(#)root/gui:$Name:  $:$Id: TRootEmbeddedCanvas.h,v 1.1.1.1 2000/05/16 17:00:42 rdm Exp $
// Author: Fons Rademakers   15/07/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TRootEmbeddedCanvas
#define ROOT_TRootEmbeddedCanvas

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootEmbeddedCanvas                                                  //
//                                                                      //
// This class creates a TGCanvas in which a TCanvas is created. Use     //
// GetCanvas() to get a pointer to the TCanvas.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGCanvas
#include "TGCanvas.h"
#endif


class TCanvas;
class TRootEmbeddedContainer;


class TRootEmbeddedCanvas : public TGCanvas {

friend class TRootEmbeddedContainer;

private:
   Int_t                   fCWinId;           // window id used by embedded TCanvas
   TRootEmbeddedContainer *fCanvasContainer;  // container in canvas widget
   TCanvas                *fCanvas;           // pointer to TCanvas

   Bool_t   fAutoFit;    // when true canvas container keeps same size as canvas
   Int_t    fButton;     // currently pressed button

   Bool_t   HandleContainerButton(Event_t *ev);
   Bool_t   HandleContainerDoubleClick(Event_t *ev);
   Bool_t   HandleContainerConfigure(Event_t *ev);
   Bool_t   HandleContainerKey(Event_t *ev);
   Bool_t   HandleContainerMotion(Event_t *ev);
   Bool_t   HandleContainerExpose(Event_t *ev);
   Bool_t   HandleContainerCrossing(Event_t *ev);

public:
   TRootEmbeddedCanvas(const char *name, const TGWindow *p, UInt_t w,
            UInt_t h, UInt_t options = kSunkenFrame | kDoubleBorder,
            ULong_t back = fgDefaultFrameBackground);
   virtual ~TRootEmbeddedCanvas();

   TCanvas   *GetCanvas() const { return fCanvas; }
   void       AdoptCanvas(TCanvas *c) { fCanvas = c; }
   Int_t      GetCanvasWindowId() const { return fCWinId; }

   ClassDef(TRootEmbeddedCanvas,0)  //A ROOT TCanvas that can be embedded in a TGFrame
};

#endif
