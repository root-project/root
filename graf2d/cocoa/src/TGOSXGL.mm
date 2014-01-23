#include <cassert>

#include "TVirtualViewer3D.h"
#include "TSeqCollection.h"
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
Int_t TGOSXGLManager::InitGLWindow(Window_t winID)
{
#pragma unused(winID)
   return 0;
}


//______________________________________________________________________________
Int_t TGOSXGLManager::CreateGLContext(Int_t winInd)
{
#pragma unused(winInd)
   return 0;
}


//______________________________________________________________________________
Bool_t TGOSXGLManager::MakeCurrent(Int_t ctxInd)
{
#pragma unused(ctxInd)
   return false;
}

//A bunch of (now) noop functions - this is a legacy from the time when
//we had a real off-screen OpenGL rendering. Nowadays we always do it "on-screen"

//______________________________________________________________________________
void TGOSXGLManager::Flush(Int_t ctxInd)
{
#pragma unused(ctxInd)
}

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
void TGOSXGLManager::DeleteGLContext(Int_t ctxInd)
{
#pragma unused(ctxInd)
}


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
