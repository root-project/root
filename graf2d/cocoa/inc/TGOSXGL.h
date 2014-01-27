#ifndef ROOT_TGOSXGL
#define ROOT_TGOSXGL

#include <map>

#ifndef ROOT_TVirtualGL
#include "TVirtualGL.h"
#endif

//
//TGLManager is a legacy interface (gl-context/window management):
//at some point we had to use OpenGL in our TCanvas/TPad classes which do not
//have direct access to low-level APIs + on Windows we had quite tricky
//mt-problems to deal with.
//TODO: in principle, we can get rid of gl-managers and work with TGLWidget,
//as it was demonstrated in glpad dev-branch 5 years ago.
//

class TGOSXGLManager : public TGLManager {
public:
   TGOSXGLManager();
   ~TGOSXGLManager();

   //TGLManager's final-overriders (window + context management):
   Int_t    InitGLWindow(Window_t winID);
   Int_t    CreateGLContext(Int_t winInd);
   void     DeleteGLContext(Int_t devInd);
   Bool_t   MakeCurrent(Int_t devInd);
   void     Flush(Int_t ctxInd);

   //In case of Cocoa 'VirtulXInd' == devInd (again, legacy).
   Int_t    GetVirtualXInd(Int_t devInd);

   //These are empty overriders, we do not have/use off-screen renreding in TCanvas/TPad anymore
   //(before we had 1) non-hardware glpixmaps/DIB sections and later 2) a hack with double buffer).
   Bool_t   AttachOffScreenDevice(Int_t ctxInd, Int_t x, Int_t y, UInt_t w, UInt_t h);
   Bool_t   ResizeOffScreenDevice(Int_t devInd, Int_t x, Int_t y, UInt_t w, UInt_t h);
   void     SelectOffScreenDevice(Int_t devInd);
   void     MarkForDirectCopy(Int_t devInd, Bool_t);
   void     ExtractViewport(Int_t devInd, Int_t *vp);
   void     ReadGLBuffer(Int_t devInd);

   //Used by our OpenGL viewer.
   //In the past we had to implement this functions to deal with mt-issues on Windows.
   Bool_t   SelectManip(TVirtualGLManip *manip, const TGLCamera *camera, const TGLRect *rect, const TGLBoundingBox *sceneBox);
   Bool_t   PlotSelected(TVirtualGLPainter *plot, Int_t px, Int_t py);
   char    *GetPlotInfo(TVirtualGLPainter *plot, Int_t px, Int_t py);
   void     PaintSingleObject(TVirtualGLPainter *);
   void     PanObject(TVirtualGLPainter *o, Int_t x, Int_t y);
   void     PrintViewer(TVirtualViewer3D *vv);

   Bool_t   HighColorFormat(Int_t /*ctxInd*/){return kFALSE;}

private:
   typedef std::map<Handle_t, Window_t> CtxToWindowMap_t;
   CtxToWindowMap_t fCtxToWin;

   TGOSXGLManager(const TGOSXGLManager &);
   TGOSXGLManager &operator = (const TGOSXGLManager &);

   ClassDef(TGOSXGLManager, 0) //Cocoa specific version of TGLManager.
};

#endif
