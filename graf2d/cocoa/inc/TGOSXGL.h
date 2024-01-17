#ifndef ROOT_TGOSXGL
#define ROOT_TGOSXGL

#include <map>

#include "TVirtualGL.h"

//
//TGLManager is a legacy interface (gl-context/window management):
//at some point we had to use OpenGL in our TCanvas/TPad classes which do not
//have direct access to low-level APIs + on Windows we had quite tricky
//mt-problems to deal with.
//

class TGOSXGLManager : public TGLManager {
public:
   TGOSXGLManager();
   ~TGOSXGLManager() override;

   //TGLManager's final-overriders (window + context management):
   Int_t    InitGLWindow(Window_t winID) override;
   Int_t    CreateGLContext(Int_t winInd) override;
   void     DeleteGLContext(Int_t devInd) override;
   Bool_t   MakeCurrent(Int_t devInd) override;
   void     Flush(Int_t ctxInd) override;

   //In case of Cocoa 'VirtulXInd' == devInd (again, legacy).
   Int_t    GetVirtualXInd(Int_t devInd) override;

   //These are empty overriders, we do not have/use off-screen renreding in TCanvas/TPad anymore
   //(before we had 1) non-hardware glpixmaps/DIB sections and later 2) a hack with double buffer).
   Bool_t   AttachOffScreenDevice(Int_t ctxInd, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   Bool_t   ResizeOffScreenDevice(Int_t devInd, Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   void     SelectOffScreenDevice(Int_t devInd) override;
   void     MarkForDirectCopy(Int_t devInd, Bool_t) override;
   void     ExtractViewport(Int_t devInd, Int_t *vp) override;
   void     ReadGLBuffer(Int_t devInd) override;

   //Used by our OpenGL viewer.
   //In the past we had to implement this functions to deal with mt-issues on Windows.
   Bool_t   SelectManip(TVirtualGLManip *manip, const TGLCamera *camera, const TGLRect *rect, const TGLBoundingBox *sceneBox) override;
   Bool_t   PlotSelected(TVirtualGLPainter *plot, Int_t px, Int_t py) override;
   char    *GetPlotInfo(TVirtualGLPainter *plot, Int_t px, Int_t py) override;
   void     PaintSingleObject(TVirtualGLPainter *) override;
   void     PanObject(TVirtualGLPainter *o, Int_t x, Int_t y) override;
   void     PrintViewer(TVirtualViewer3D *vv) override;

   Bool_t   HighColorFormat(Int_t /*ctxInd*/) override{return kFALSE;}

private:
   typedef std::map<Handle_t, Window_t> CtxToWindowMap_t;
   CtxToWindowMap_t fCtxToWin;

   TGOSXGLManager(const TGOSXGLManager &);
   TGOSXGLManager &operator = (const TGOSXGLManager &);

   ClassDefOverride(TGOSXGLManager, 0) //Cocoa specific version of TGLManager.
};

#endif
