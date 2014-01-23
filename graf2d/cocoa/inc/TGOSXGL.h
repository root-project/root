#ifndef ROOT_TGOSXGL
#define ROOT_TGOSXGL

#ifndef ROOT_TVirtualGL
#include "TVirtualGL.h"
#endif

//
//TGLManager is a legacy interface to some OpenGL functions (gl-context/window management),
//which is used by different "gl painters". To make this painters work on OS X with
//Cocoa based graphics, I have to implement TGLManager for OS X - TGOSXGLManager
//('TG' part in a name prefix - is standard for ROOT).
//

class TGOSXGLManager : public TGLManager {
public:
   TGOSXGLManager();
   ~TGOSXGLManager();

   //TGLManager's final-overriders:
   Int_t    InitGLWindow(Window_t winID);
   Int_t    CreateGLContext(Int_t winInd);

   //[Off-screen rendering - not used anymore for many years (thus empty implementations on OS X).
   Bool_t   AttachOffScreenDevice(Int_t ctxInd, Int_t x, Int_t y, UInt_t w, UInt_t h);
   Bool_t   ResizeOffScreenDevice(Int_t devInd, Int_t x, Int_t y, UInt_t w, UInt_t h);
   //analog of gVirtualX->SelectWindow(fPixmapID) => gVirtualGL->SelectOffScreenDevice(fPixmapID)
   void     SelectOffScreenDevice(Int_t devInd);
   //Index of pixmap, valid for gVirtualX
   Int_t    GetVirtualXInd(Int_t devInd);
   //copy pixmap into window directly/by pad
   void     MarkForDirectCopy(Int_t devInd, Bool_t);
   //Off-screen device holds sizes for glViewport
   void     ExtractViewport(Int_t devInd, Int_t *vp);
   //Read GL buffer into pixmap
   void     ReadGLBuffer(Int_t devInd);
   //]

   //Make the gl context current
   Bool_t   MakeCurrent(Int_t devInd);
   void     Flush(Int_t ctxInd);
   void     DeleteGLContext(Int_t devInd);

   //Used by our OpenGL viewer.
   Bool_t   SelectManip(TVirtualGLManip *manip, const TGLCamera *camera, const TGLRect *rect, const TGLBoundingBox *sceneBox);
   //
   Bool_t   PlotSelected(TVirtualGLPainter *plot, Int_t px, Int_t py);
   char    *GetPlotInfo(TVirtualGLPainter *plot, Int_t px, Int_t py);
   //
   void     PaintSingleObject(TVirtualGLPainter *);
   void     PanObject(TVirtualGLPainter *o, Int_t x, Int_t y);
   void     PrintViewer(TVirtualViewer3D *vv);

   Bool_t   HighColorFormat(Int_t /*ctxInd*/){return kFALSE;}

private:
   TGOSXGLManager(const TGOSXGLManager &);
   TGOSXGLManager &operator = (const TGOSXGLManager &);

   ClassDef(TGOSXGLManager, 0) //Cocoa specific version of TGLManager.
};

#endif
