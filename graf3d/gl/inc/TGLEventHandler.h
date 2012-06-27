// @(#)root/gl:$Id$
// Author: Bertrand Bellenot   29/01/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLEventHandler
#define ROOT_TGLEventHandler

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLEventHandler                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGEventHandler
#include "TGEventHandler.h"
#endif
#ifndef ROOT_TPoint
#include "TPoint.h"
#endif

class TGToolTip;

class TGLViewer;
class TGLPhysicalShape;

class TGLEventHandler : public TGEventHandler
{
protected:
   TGLViewer        *fGLViewer;
   TTimer           *fMouseTimer;     // mouse delay timer
   TPoint            fButtonPushPos;
   TPoint            fLastPos;
   TPoint            fLastMouseOverPos;
   TGLPhysicalShape *fLastMouseOverShape;
   TGToolTip        *fTooltip;        // tooltip for highlight
   TPoint            fLastGlobalPos;
   TPoint            fTooltipPos;
   UInt_t            fActiveButtonID;
   UInt_t            fLastEventState;
   Bool_t            fIgnoreButtonUp;
   Bool_t            fInPointerGrab;
   Bool_t            fMouseTimerRunning;
   Bool_t            fTooltipShown;
   Bool_t            fArcBall;
   Int_t             fTooltipPixelTolerance;
   Int_t             fSecSelType; // secondary selection type
   Bool_t            fDoInternalSelection;
   Bool_t            fViewerCentricControls;
   Float_t           fArrowKeyFactor;
   Float_t           fMouseDragFactor;
   Float_t           fMouseWheelFactor;

   virtual Bool_t Rotate(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2);

   virtual void   GrabMouse();
   virtual void   UnGrabMouse();

   virtual void   SelectForClicked(Event_t *event);
   virtual void   SelectForMouseOver();

   Int_t ControlValue(Int_t v) { return fViewerCentricControls ? -v : v; }

public:
   TGLEventHandler(TGWindow *w, TObject *obj);
   virtual ~TGLEventHandler();

   virtual void   ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual Bool_t HandleEvent(Event_t *event);
   virtual Bool_t HandleExpose(Event_t * event);
   virtual Bool_t HandleFocusChange(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual Bool_t HandleButton(Event_t * event);
   virtual Bool_t HandleDoubleClick(Event_t *event);
   virtual Bool_t HandleConfigureNotify(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleMotion(Event_t * event);
   virtual Bool_t HandleTimer(TTimer *t);
   virtual void   StartMouseTimer();
   virtual void   StopMouseTimer();
   virtual void   ClearMouseOver();
   virtual void   Repaint();

   virtual void   PopupContextMenu(TGLPhysicalShape* pshp, Event_t *event, Int_t gx, Int_t gy);

   virtual void   TriggerTooltip(const char* text);
   virtual void   RemoveTooltip();

   void  SetMouseOverSelectDelay(Int_t ms);
   void  SetMouseOverTooltipDelay(Int_t ms);

   Int_t GetTooltipPixelTolerance()  const { return fTooltipPixelTolerance; }
   void  SetTooltipPixelTolerance(Int_t t) { fTooltipPixelTolerance = t; }

   Int_t GetSecSelType()  const { return fSecSelType; }
   void  SetSecSelType(Int_t t) { fSecSelType = t; }

   Bool_t GetDoInternalSelection() const { return fDoInternalSelection; }
   void   SetDoInternalSelection(Bool_t x) { fDoInternalSelection = x; }

   Bool_t GetArcBall() const   { return fArcBall; }
   void   SetArcBall(Bool_t a) { fArcBall = a;    }

   ClassDef(TGLEventHandler, 0); // Base-class and default implementation of event-handler for TGLViewer.
};

#endif

