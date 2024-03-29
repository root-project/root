// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLWidget
#define ROOT_TGLWidget

#include <utility>
#include <memory>
#include <set>

#include "TGLContext.h"
#include "TVirtualGL.h"
#include "TGLFormat.h"
#include "TGFrame.h"

class TGLWidget;
class TGEventHandler;

class TGLWidget : public TGFrame, public TGLPaintDevice
{
   friend class TGLContext;

private:
   TGLContext                       *fGLContext;
   //fInnerData is for X11 - <dpy, visualInfo> pair.
   std::pair<void *, void *>         fInnerData;
   Int_t                             fWindowIndex;

   TGLFormat                         fGLFormat;
   //fFromCtor checks that SetFormat was called only from ctor.
   Bool_t                            fFromInit;

   std::set<TGLContext *>            fValidContexts;

   TGEventHandler                   *fEventHandler;

public:
   static TGLWidget* CreateDummy();

   static TGLWidget* Create(const TGWindow* parent, Bool_t selectInput,
             Bool_t shareDefault, const TGLPaintDevice *shareDevice,
             UInt_t width, UInt_t height);

   static TGLWidget* Create(const TGLFormat &format,
             const TGWindow* parent, Bool_t selectInput,
             Bool_t shareDefault, const TGLPaintDevice *shareDevice,
             UInt_t width, UInt_t height);

   ~TGLWidget() override;

   virtual void      InitGL();
   virtual void      PaintGL();

   Bool_t            MakeCurrent() override;
   Bool_t            ClearCurrent();
   void              SwapBuffers() override;
   const TGLContext *GetContext()const override;

   const  TGLFormat *GetPixelFormat()const override;

   //This function is public _ONLY_ for calls
   //via gInterpreter. Do not call it directly.
   void              SetFormat();
   //To repaint gl-widget without GUI events.
   void              ExtractViewport(Int_t *vp)const override;

   TGEventHandler   *GetEventHandler() const { return fEventHandler; }
   void              SetEventHandler(TGEventHandler *eh);

   Bool_t HandleButton(Event_t *ev) override;
   Bool_t HandleDoubleClick(Event_t *ev) override;
   Bool_t HandleConfigureNotify(Event_t *ev) override;
   Bool_t HandleKey(Event_t *ev) override;
   Bool_t HandleMotion(Event_t *ev) override;
   Bool_t HandleFocusChange(Event_t *) override;
   Bool_t HandleCrossing(Event_t *) override;

   void   DoRedraw() override;

private:
   TGLWidget(const TGLWidget &) = delete;
   TGLWidget &operator = (const TGLWidget &) = delete;

protected:
   TGLWidget(Window_t glw, const TGWindow* parent, Bool_t selectInput);

   static Window_t CreateWindow(const TGWindow* parent, const TGLFormat &format,
                                UInt_t width, UInt_t height,
                                std::pair<void *, void *>& innerData);

   void AddContext(TGLContext *ctx) override;
   void RemoveContext(TGLContext *ctx) override;

   std::pair<void *, void *> GetInnerData()const;

   ClassDefOverride(TGLWidget, 0); //Window (widget) version of TGLPaintDevice
};

#endif
