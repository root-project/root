// @(#)root/gl:$Name:  $:$Id: TGLWidget.h,v 1.5 2007/06/18 07:02:16 brun Exp $
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
#ifndef ROOT_TGCanvas
#include "TGCanvas.h"
#endif

class TGLWidget;

/*
   Auxiliary "widget container" class.
   Does not throw (base classe can throw?).
   Immutable - after constructed, fOwner is
   invariant, cannot change.
   Non-copyable.
*/

class TGLWidgetContainer : public TGCompositeFrame {
private:
   TGLWidget *fOwner;

public:
   TGLWidgetContainer(TGLWidget *owner, Window_t id, const TGWindow *parent);

   Bool_t HandleButton(Event_t *ev);
   Bool_t HandleDoubleClick(Event_t *ev);
   Bool_t HandleConfigureNotify(Event_t *ev);
   Bool_t HandleKey(Event_t *ev);
   Bool_t HandleMotion(Event_t *ev);
   //Bool_t HandleExpose(Event_t *ev);
   
   void   DoRedraw();

private:
   TGLWidgetContainer(const TGLWidgetContainer &);
   TGLWidgetContainer &operator = (const TGLWidgetContainer &);

   ClassDef(TGLWidgetContainer, 0)//Auxilary widget container class.
};

/*
   TGLWidget. GL window with context. _Must_ _have_ a parent window
   (the 'parent' parameter of ctors). The current version inherits
   TGCanvas (I'm not sure about future versions), probably, in future
   multiple inheritance will be added - the second
   base class will be TGLPaintDevice or something like this.

   Usage:
   - Simply create TGLWidget as an embedded widget, and
     connect your slots to signals you need: HandleExpose, HandleConfigureNotify, etc.
     In your slots you can use gl API directly - under Win32 TGLWidget switches
     between threads internally (look TGLPShapeObjEditor for such usage).
   - You can write your own class, derived from TGLWidget, with PaintGL and InitGL
     overriden.

   Resources (and invariants):
   -fContainer (TGLWidgetContainer) - controlled by std::auto_ptr
   -fWindowIndex - controlled manually (see CreateWidget and dtor)
   -fGLContext - controlled by std::auto_ptr
   -visual info for X11 version, controlled manually (see CreateGLContainer and dtor)

   Exceptions:
   -can be thrown only during construction.
   -under win32 class does not throw itself (but some internal operations can throw)
   -under X11 can throw std::runtime_error (from CreateGLContext).
   -In case of exceptions resources will be freed.

   TGLWidget object is immutable as far as it was created.

   Boolean parameter defines, if you want to grab user's input or not.
   By default you want, but for example when not - see TGLPShapeObjEditor.

   Non-copyable.
*/

class TGLWidget : public TGCanvas, public TGLPaintDevice {
   friend class TGLContext;
private:
   //Widget container.
   std::auto_ptr<TGLWidgetContainer> fContainer;
   //Index, returned from gVirtualX
   Int_t                             fWindowIndex;
   std::auto_ptr<TGLContext>         fGLContext;
   //fInnerData is for X11 - <dpy, visualInfo> pair.
   std::pair<void *, void *>         fInnerData;

   TGLFormat                         fGLFormat;
   //fFromCtor checks that SetFormat was called only from ctor.
   Bool_t                            fFromCtor;

   std::set<TGLContext *>            fValidContexts;

public:
   TGLWidget(const TGWindow &parent, Bool_t selectInput, UInt_t width, UInt_t height,
             const TGLPaintDevice *shareDevice = 0, UInt_t options = kSunkenFrame | kDoubleBorder,
             Pixel_t back = GetDefaultFrameBackground());
   TGLWidget(const TGLFormat &format, const TGWindow &parent, Bool_t selectInput,
             UInt_t width, UInt_t height, const TGLPaintDevice *shareDevice = 0,
             UInt_t options = kSunkenFrame | kDoubleBorder,
             Pixel_t back = GetDefaultFrameBackground());

   ~TGLWidget();

   virtual void      InitGL();
   virtual void      PaintGL();

   Bool_t            MakeCurrent();
   void              SwapBuffers();
   const TGLContext *GetContext()const;

   //Signals. Names can be changed (for example PaintSignal, DoubleClickSignal etc.)
   Bool_t            HandleButton(Event_t *event);         //*SIGNAL*
   Bool_t            HandleDoubleClick(Event_t *event);    //*SIGNAL*
   Bool_t            HandleConfigureNotify(Event_t *event);//*SIGNAL*
   Bool_t            HandleKey(Event_t *event);            //*SIGNAL*
   Bool_t            HandleMotion(Event_t *event);         //*SIGNAL*
//   Bool_t            HandleExpose(Event_t *event);         //*SIGNAL*
   void              Repaint();                           //*SIGNAL*

   Int_t             GetWindowIndex()const;
   const  TGLFormat *GetPixelFormat()const;
   Int_t             GetContId()const;

   //This function is public _ONLY_ for calls
   //via gInterpreter. Do not call it directly.
   void              SetFormat();
   //To repaint gl-widget without GUI events.


private:
   TGLWidget(const TGLWidget &);
   TGLWidget &operator = (const TGLWidget &);

   void CreateWidget(const TGLPaintDevice *shareDevice);

   void AddContext(TGLContext *ctx);
   void RemoveContext(TGLContext *ctx);

   std::pair<void *, void *> GetInnerData()const;

   ClassDef(TGLWidget, 0)//Window (widget) version of TGLPaintDevice
};

#endif
