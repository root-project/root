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

#ifndef ROOT_TGLContext
#include "TGLContext.h"
#endif
#ifndef ROOT_TVirtualGL
#include "TVirtualGL.h"
#endif
#ifndef ROOT_TGLFormat
#include "TGLFormat.h"
#endif
#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGLWidget;
class TGEventHandler;

class TGLWidget : public TGFrame, public TGLPaintDevice {
   friend class TGLContext;
private:
   std::auto_ptr<TGLContext>         fGLContext;
   //fInnerData is for X11 - <dpy, visualInfo> pair.
   std::pair<void *, void *>         fInnerData;

   TGLFormat                         fGLFormat;
   //fFromCtor checks that SetFormat was called only from ctor.
   Bool_t                            fFromCtor;

   std::set<TGLContext *>            fValidContexts;

   TGEventHandler                   *fEventHandler;

public:
   TGLWidget(const TGWindow &parent, Bool_t selectInput,
             const TGLPaintDevice *shareDevice,
             UInt_t width, UInt_t height,
             UInt_t options = kSunkenFrame | kDoubleBorder,
             Pixel_t back = 0);
   TGLWidget(const TGLFormat &format, const TGWindow &parent, Bool_t selectInput,
             const TGLPaintDevice *shareDevice,
             UInt_t width, UInt_t height,
             UInt_t options = kSunkenFrame | kDoubleBorder,
             Pixel_t back = 0);
   TGLWidget(const TGWindow &parent, Bool_t selectInput,
             UInt_t width, UInt_t height,
             UInt_t options = kSunkenFrame | kDoubleBorder,
             Pixel_t back = 0);
   TGLWidget(const TGLFormat &format, const TGWindow &parent, Bool_t selectInput,
             UInt_t width, UInt_t height,
             UInt_t options = kSunkenFrame | kDoubleBorder,
             Pixel_t back = 0);

   ~TGLWidget();

   virtual void      InitGL();
   virtual void      PaintGL();

   Bool_t            MakeCurrent();
   void              SwapBuffers();
   const TGLContext *GetContext()const;

   const  TGLFormat *GetPixelFormat()const;

   //This function is public _ONLY_ for calls
   //via gInterpreter. Do not call it directly.
   void              SetFormat();
   //To repaint gl-widget without GUI events.
   void              ExtractViewport(Int_t *vp)const;

   TGEventHandler   *GetEventHandler() const { return fEventHandler; }
   void              SetEventHandler(TGEventHandler *eh);

   Bool_t HandleButton(Event_t *ev);
   Bool_t HandleDoubleClick(Event_t *ev);
   Bool_t HandleConfigureNotify(Event_t *ev);
   Bool_t HandleKey(Event_t *ev);
   Bool_t HandleMotion(Event_t *ev);
   Bool_t HandleFocusChange(Event_t *);
   Bool_t HandleCrossing(Event_t *);

   void   DoRedraw();

private:
   TGLWidget(const TGLWidget &);
   TGLWidget &operator = (const TGLWidget &);

   void CreateWidget(const TGLPaintDevice *shareDevice);
   void CreateWidget();

   void AddContext(TGLContext *ctx);
   void RemoveContext(TGLContext *ctx);

   std::pair<void *, void *> GetInnerData()const;

   ClassDef(TGLWidget, 0)//Window (widget) version of TGLPaintDevice
};

#endif
