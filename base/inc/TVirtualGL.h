// @(#)root/base:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   05/03/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualGL
#define ROOT_TVirtualGL


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualGL                                                           //
//                                                                      //
// The TVirtualGL class is an abstract base class defining the          //
// OpenGL interface protocol. All interactions with OpenGL should be    //
// done via the global pointer gVirtualGL. If the OpenGL library is     //
// available this pointer is pointing to an instance of the TGLKernel   //
// class which provides the actual interface to OpenGL. Using this      //
// scheme of ABC we can use OpenGL in other parts of the framework      //
// without having to link with the OpenGL library in case we don't      //
// use the classes using OpenGL.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif
#ifndef ROOT_GLConstants
#include "GLConstants.h"
#endif

class TVirtualViewer3D;
class TPoints3DABC;
class TGLViewer;
class TGLCamera;
class TGLManip;
class TGLBoundingBox;
class TGLRect;

//TVirtualGLPainter is the base for histogramm painters.

class TVirtualGLPainter {
public:
   virtual ~TVirtualGLPainter(){}

   virtual void     Paint() = 0;
   virtual void     Pan(Int_t px, Int_t py) = 0;
   virtual Bool_t   PlotSelected(Int_t px, Int_t py) = 0;
   //Used by status bar in a canvas.
   virtual char    *GetPlotInfo(Int_t px, Int_t py) = 0;

   ClassDef(TVirtualGLPainter, 0); // Interface for OpenGL painter
};

//We need this class to implement TGWin32GLManager's SelectManip
class TVirtualGLManip {
public:
   virtual ~TVirtualGLManip(){}
   virtual Bool_t Select(const TGLCamera & camera, const TGLRect & rect, const TGLBoundingBox & sceneBox) = 0;

   ClassDef(TVirtualGLManip, 0); //Interface for GL manipulator
};

//This class (and its descendants) in future will replace (?)
//TVirtualGL/TGLKernel/TGWin32GL/TGX11GL

class TGLManager : public TNamed {
public:
   TGLManager();

   //index returned can be used as a result of gVirtualX->InitWindow
   virtual Int_t    InitGLWindow(Window_t winID) = 0;
   //winInd is the index, returned by InitGLWindow
   virtual Int_t    CreateGLContext(Int_t winInd) = 0;

   //[            Off-screen rendering part
   //create DIB section/pixmap to read GL buffer into it,
   //ctxInd is the index, returned by CreateGLContext
   virtual Bool_t   AttachOffScreenDevice(Int_t ctxInd, Int_t x, Int_t y, UInt_t w, UInt_t h) = 0;
   virtual Bool_t   ResizeOffScreenDevice(Int_t ctxInd, Int_t x, Int_t y, UInt_t w, UInt_t h) = 0;
   //analog of gVirtualX->SelectWindow(fPixmapID) => gVirtualGL->SelectOffScreenDevice(fPixmapID)
   virtual void     SelectOffScreenDevice(Int_t ctxInd) = 0;
   //Index of DIB/pixmap, valid for gVirtualX
   virtual Int_t    GetVirtualXInd(Int_t ctxInd) = 0;
   //copy pixmap into window directly
   virtual void     MarkForDirectCopy(Int_t ctxInd, Bool_t) = 0;
   //Off-screen device holds sizes for glViewport
   virtual void     ExtractViewport(Int_t ctxInd, Int_t *vp) = 0;
   //Read GL buffer into off-screen device
   virtual void     ReadGLBuffer(Int_t ctxInd) = 0;
   //]

   //Make the gl context current
   virtual Bool_t   MakeCurrent(Int_t ctxInd) = 0;
   //Swap buffers or copies DIB/pixmap (via BitBlt/XCopyArea)
   virtual void     Flush(Int_t ctxInd) = 0;
   //GL context and off-screen device deletion
   virtual void     DeleteGLContext(Int_t ctxInd) = 0;

   //functions to switch between threads in win32
   virtual Bool_t   SelectManip(TVirtualGLManip *manip, const TGLCamera *camera, const TGLRect *rect, const TGLBoundingBox *sceneBox) = 0;
   //
   virtual void     PaintSingleObject(TVirtualGLPainter *) = 0;
   virtual void     PanObject(TVirtualGLPainter *o, Int_t x, Int_t y) = 0;
   //EPS/PDF output
   virtual void     PrintViewer(TVirtualViewer3D *vv) = 0;

   virtual Bool_t   PlotSelected(TVirtualGLPainter *plot, Int_t px, Int_t py) = 0;
   virtual char    *GetPlotInfo(TVirtualGLPainter *plot, Int_t px, Int_t py) = 0;

   virtual Bool_t   HighColorFormat(Int_t ctxInd) = 0;

   static TGLManager *&Instance();

private:
   TGLManager(const TGLManager &);
   TGLManager &operator = (const TGLManager &);

   ClassDef(TGLManager, 0)// Interface for OpenGL manager
};

class TGLContext;
class TGLFormat;

class TGLPaintDevice {
   friend class TGLContext;
public:
   virtual ~TGLPaintDevice(){}
   virtual Bool_t MakeCurrent() = 0;
   virtual void SwapBuffers() = 0;
   virtual const TGLFormat *GetPixelFormat()const = 0;
   virtual const TGLContext *GetContext()const = 0;
   virtual void ExtractViewport(Int_t *vp)const = 0;

private:
   virtual void AddContext(TGLContext *ctx) = 0;
   virtual void RemoveContext(TGLContext *ctx) = 0;

   ClassDef(TGLPaintDevice, 0) // Base class for GL widgets and GL off-screen rendering
};

#ifndef __CINT__
#define gGLManager (TGLManager::Instance())
R__EXTERN TGLManager *(*gPtr2GLManager)();
#endif

#endif
