#include <TVirtualGL.h>
#include <TVirtualX.h>
#include <TGClient.h>

#include "TGLRenderArea.h"

ClassImp(TGLWindow)
ClassImp(TGLRenderArea)

//______________________________________________________________________________
TGLWindow::TGLWindow(Window_t id, const TGWindow *parent)
               :TGCompositeFrame(gClient, id, parent), fCtx(0)
{
   fCtx = gVirtualGL->CreateContext(fId);
   //here add diagnostic
   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier, kButtonPressMask | kButtonReleaseMask, kNone, kNone);
   gVirtualX->SelectInput(fId, kKeyPressMask | kExposureMask | kPointerMotionMask | kStructureNotifyMask);
   gVirtualX->SetInputFocus(fId);
}

//______________________________________________________________________________
TGLWindow::~TGLWindow()
{
   gVirtualGL->DeleteContext(fCtx);
}

//______________________________________________________________________________
Bool_t TGLWindow::HandleConfigureNotify(Event_t *event)
{
   gVirtualX->ResizeWindow(fId, event->fWidth, event->fHeight);
   Emit("HandleConfigureNotify(Event_t*)", (Long_t)event);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLWindow::HandleButton(Event_t *event)
{
   Emit("HandleButton(Event_t*)", (Long_t)event);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLWindow::HandleKey(Event_t *event)
{
   Emit("HandleKey(Event_t*)", (Long_t)event);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLWindow::HandleMotion(Event_t *event)
{
   Emit("HandleMotion(Event_t*)", (Long_t)event);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLWindow::HandleExpose(Event_t *event)
{
   Emit("HandleExpose(Event_t*)", (Long_t)event);
   return kTRUE;
}

//______________________________________________________________________________
void TGLWindow::Refresh()
{
   gVirtualGL->SwapBuffers(fId);
}

//______________________________________________________________________________
void TGLWindow::MakeCurrent()
{
   gVirtualGL->MakeCurrent(fId, fCtx);
}

//______________________________________________________________________________
TGLRenderArea::TGLRenderArea()
                  :fArea(0)
{
}

//______________________________________________________________________________
TGLRenderArea::TGLRenderArea(Window_t wid, const TGWindow *parent)
                  :fArea(0)
{
   Window_t glWin = gVirtualGL->CreateGLWindow(wid);
   fArea = new TGLWindow(glWin, parent);
}

//______________________________________________________________________________
TGLRenderArea::~TGLRenderArea()
{
   delete fArea;
}
