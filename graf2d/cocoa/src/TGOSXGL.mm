#include <utility>
#include <cassert>
#include <vector>

#include <Foundation/Foundation.h>

#include "TVirtualViewer3D.h"
#include "TSeqCollection.h"
#include "TVirtualGL.h"
#include "TVirtualX.h"
#include "TGOSXGL.h"
#include "TROOT.h"
#include "TEnv.h"


ClassImp(TGOSXGLManager)

//______________________________________________________________________________
TGOSXGLManager::TGOSXGLManager()
{
   //Constructor.

   //gGLManager is a singleton, it's created by the plugin manager
   //(either from TRootCanvas or TRootEmbeddedCanvas),
   //never by user.

   assert(gGLManager == 0 && "TGOSXGLManager, gGLManager is initialized");
   gGLManager = this;

   //TODO: do we really need this?
   if (gROOT && gROOT->GetListOfSpecials())
      gROOT->GetListOfSpecials()->Add(this);
}


//______________________________________________________________________________
TGOSXGLManager::~TGOSXGLManager()
{
   //Destructor.

   //TODO: do we really need this and does ROOT ever deletes 'this'?
   if (gROOT && gROOT->GetListOfSpecials())
      gROOT->GetListOfSpecials()->Remove(this);
}


//______________________________________________________________________________
Int_t TGOSXGLManager::InitGLWindow(Window_t parentID)
{
   typedef std::pair<UInt_t, Int_t> component_type;

   std::vector<component_type> format;//Where is the hummer when you need one??? (I mean C++11 initializers '{xxx}').

   //TODO: this values actually are quite random, as it was in TX11GLManager/TGWin32GLManager,
   //find something better!

   format.push_back(component_type(Rgl::kDoubleBuffer, 1));//1 means nothing, kDoubleBuffer is enough :)
   format.push_back(component_type(Rgl::kStencil, 8));
   format.push_back(component_type(Rgl::kDepth, 32));

   if (gEnv) {
      const Int_t nSamples = gEnv->GetValue("OpenGL.Framebuffer.Multisample", 0);
      if (nSamples > 0 && nSamples <= 8) //TODO: check the 'upper limit' using API, not hardcoded 8.
         format.push_back(component_type(Rgl::kMultiSample, nSamples));
   }

   //Now, the interface is quite ugly, that's why it's called TVirtualX :)
   Int_t x = 0, y = 0;
   UInt_t width = 0, height = 0;
   gVirtualX->GetWindowSize(parentID, x, y, width, height);

   const Window_t glWin = gVirtualX->CreateOpenGLWindow(parentID, width, height, format);
   if (glWin != kNone) {
      //TRootCanvas/TRootEmbeddedCanvas never do this,
      //so ...
      gVirtualX->MapWindow(glWin);
   }

   //Window_t is long, in principle it's a potential problem: do I need a mapping?
   //But if you have billions of windows ... ;)
   return Int_t(glWin);
}


//______________________________________________________________________________
Int_t TGOSXGLManager::CreateGLContext(Int_t winID)
{
   //Called from TRootCanvas, it never shares :) -> the second parameter is kNone.
   //Handle_t is long, I'm converting to int, which can be a problem if you ...
   //have billions of gl contexts :)
   const Handle_t ctx = gVirtualX->CreateOpenGLContext(winID, kNone);
   fCtxToWin[ctx] = Window_t(winID);

   return Int_t(ctx);
}

//______________________________________________________________________________
void TGOSXGLManager::DeleteGLContext(Int_t ctxInd)
{
   //Just delegate.
   gVirtualX->DeleteOpenGLContext(ctxInd);
}

//______________________________________________________________________________
Bool_t TGOSXGLManager::MakeCurrent(Int_t ctxInd)
{
   assert(fCtxToWin.find(Handle_t(ctxInd)) != fCtxToWin.end() &&
          "MakeCurrent, window not found for a given context");

   return gVirtualX->MakeOpenGLContextCurrent(Handle_t(ctxInd), fCtxToWin[Handle_t(ctxInd)]);
}

//______________________________________________________________________________
void TGOSXGLManager::Flush(Int_t ctxInd)
{
   gVirtualX->FlushOpenGLBuffer(ctxInd);
}


//______________________________________________________________________________
Int_t TGOSXGLManager::GetVirtualXInd(Int_t ctxInd)
{
   return ctxInd;
}

//A bunch of (now) noop functions - this is a legacy from the time when
//we had a real off-screen OpenGL rendering. Nowadays we always do it "on-screen"

//______________________________________________________________________________
Bool_t TGOSXGLManager::AttachOffScreenDevice(Int_t, Int_t, Int_t, UInt_t, UInt_t)
{
   //NOOP.
   return kFALSE;
}


//______________________________________________________________________________
Bool_t TGOSXGLManager::ResizeOffScreenDevice(Int_t, Int_t, Int_t, UInt_t, UInt_t)
{
   //NOOP.
   return kFALSE;
}


//______________________________________________________________________________
void TGOSXGLManager::SelectOffScreenDevice(Int_t)
{
   //NOOP.
}


//______________________________________________________________________________
void TGOSXGLManager::MarkForDirectCopy(Int_t, Bool_t)
{
   //NOOP.
}

//______________________________________________________________________________
void TGOSXGLManager::ExtractViewport(Int_t, Int_t *)
{
   //NOOP.
}

//______________________________________________________________________________
void TGOSXGLManager::ReadGLBuffer(Int_t)
{
   //NOOP.
}

//These 'delegating' functions are legacy - were required (many years ago) on Windows.

//______________________________________________________________________________
Bool_t TGOSXGLManager::SelectManip(TVirtualGLManip *manip, const TGLCamera *camera, const TGLRect *rect, const TGLBoundingBox *sceneBox)
{
   //Why all this mess with pointers/references and not pointers/references everywhere???

   assert(manip != 0 && "SelectManip, parameter 'manip' is null");
   assert(camera != 0 && "SelectManip, parameter 'camera' is null");
   assert(rect != 0 && "SelectManip, parameter 'rect' is null");
   assert(sceneBox != 0 && "SelectManip, parameter 'sceneBox' is null");

   // Select manipulator.
   return manip->Select(*camera, *rect, *sceneBox);
}

//______________________________________________________________________________
Bool_t TGOSXGLManager::PlotSelected(TVirtualGLPainter *plot, Int_t px, Int_t py)
{
   //Analog of TObject::DistancetoPrimitive
   assert(plot != 0 && "PlotSelected, parameter 'plot' is null");

   return plot->PlotSelected(px, py);
}

//______________________________________________________________________________
char *TGOSXGLManager::GetPlotInfo(TVirtualGLPainter *plot, Int_t px, Int_t py)
{
   //Analog of TObject::GetObjectInfo
   assert(plot != 0 && "GetPlotInfo, parameter 'plot' is null");

   return plot->GetPlotInfo(px, py);
}

//______________________________________________________________________________
void TGOSXGLManager::PaintSingleObject(TVirtualGLPainter *p)
{
   // Paint a single object.
   assert(p != 0 && "PaintSingleObject, parameter 'p' is null");

   p->Paint();
}

//______________________________________________________________________________
void TGOSXGLManager::PanObject(TVirtualGLPainter *object, Int_t x, Int_t y)
{
   // Pan objects.
   assert(object != 0 && "PanObject, parameter 'object' is null");

   return object->Pan(x, y);
}

//______________________________________________________________________________
void TGOSXGLManager::PrintViewer(TVirtualViewer3D *vv)
{
   // Print viewer.
   assert(vv != 0 && "PrintViewer, parameter 'vv' is null");

   vv->PrintObjects();
}
