#include <utility>
#include <cassert>
#include <vector>

#include "TVirtualViewer3D.h"
#include "TSeqCollection.h"
#include "TVirtualGL.h"
#include "TVirtualX.h"

#include "TGOSXGL.h"
#include "TROOT.h"


ClassImp(TGOSXGLManager)

//______________________________________________________________________________
TGOSXGLManager::TGOSXGLManager()
{
   // Constructor.
   assert(gROOT != 0 && "TGOSXGLManager, gROOT is null");
   
   gGLManager = this;
   gROOT->GetListOfSpecials()->Add(this);
}


//______________________________________________________________________________
TGOSXGLManager::~TGOSXGLManager()
{
   // Destructor.
}


//______________________________________________________________________________
Int_t TGOSXGLManager::InitGLWindow(Window_t parentID)
{
   typedef std::pair<UInt_t, Int_t> component_type;

   std::vector<component_type> format;//Where is the hummer when you need one??? (I mean C++11 initializers '{xxx}').

   //TODO: this values actually are quite random, find something better!
   format.push_back(component_type(Rgl::kDoubleBuffer, 1));//1 means nothing, kDoubleBuffer is enough :)
   format.push_back(component_type(Rgl::kDepth, 32));
   format.push_back(component_type(Rgl::kMultiSample, 8));

   //Now, the interface is quite ugly :) and not very different from X11, that's why it's called TVirtualX :)
   Int_t dummyX = 0, dummyY = 0;
   UInt_t width = 0, height = 0;
   gVirtualX->GetWindowSize(parentID, dummyX, dummyY, width, height);

   //Window_t is long, so in principle it's a potential problem: do I need a mapping?
   //But billions of windows ... ;)
   return Int_t(gVirtualX->CreateOpenGLWindow(parentID, width, height, format));
}


//______________________________________________________________________________
Int_t TGOSXGLManager::CreateGLContext(Int_t winID)
{
   //This is used by TRootCanvas, it never shares :) So the second parameter is kNone.
   //Handle_t is long, I'm converting to int, which can be a problem if you ...
   //have billions of gl contexts :)
   const Handle_t ctx = gVirtualX->CreateOpenGLContext(winID, kNone);
   fCtxToWin[ctx] = Window_t(winID);
   
   return Int_t(ctx);
}


//______________________________________________________________________________
Bool_t TGOSXGLManager::MakeCurrent(Int_t ctxInd)
{
#pragma unused(ctxInd)
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
void TGOSXGLManager::DeleteGLContext(Int_t ctxInd)
{
   gVirtualX->DeleteOpenGLContext(ctxInd);
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
void TGOSXGLManager::ReadGLBuffer(Int_t)
{
   //NOOP.
}

//Context management.

//______________________________________________________________________________
Int_t TGOSXGLManager::GetVirtualXInd(Int_t ctxInd)
{
#pragma unused(ctxInd)
   //Returns an index suitable for gVirtualX.
   //NOOP?
   return 0;
}


//______________________________________________________________________________
void TGOSXGLManager::ExtractViewport(Int_t, Int_t *)
{
   //NOOP.
}

//These 'delegating' functions are legacy - were required (many years ago) on Windows.

//______________________________________________________________________________
void TGOSXGLManager::PaintSingleObject(TVirtualGLPainter *p)
{
   // Paint a single object.
   assert(p != 0 && "PaintSingleObject, parameter 'p' is null");

   p->Paint();
}


//______________________________________________________________________________
void TGOSXGLManager::PrintViewer(TVirtualViewer3D *vv)
{
   // Print viewer.
   assert(vv != 0 && "PrintViewer, parameter 'vv' is null");

   vv->PrintObjects();
}

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
void TGOSXGLManager::PanObject(TVirtualGLPainter *object, Int_t x, Int_t y)
{
   // Pan objects.
   assert(object != 0 && "PanObject, parameter 'object' is null");

   return object->Pan(x, y);
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
