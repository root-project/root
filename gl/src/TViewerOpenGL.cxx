// @(#)root/gl:$Name:  $:$Id: TViewerOpenGL.cxx,v 1.3 2004/08/09 22:11:00 rdm Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <vector>
#include <memory>

#include <TViewerOpenGL.h>
#include <TVirtualPad.h>
#include <TVirtualGL.h>
#include <TVirtualX.h>
#include <TBuffer3D.h>
#include <TGLKernel.h>
#include <TGClient.h>
#include <TGCanvas.h>
#include <Buttons.h>
#include <TAtt3D.h>
#include <TROOT.h>
#include <TMath.h>

#include "TArcBall.h"

static Float_t colors[] = {
   92.f / 255, 92.f / 255, 92.f / 255,
   122.f / 255, 122.f / 255, 122.f / 255,
   184.f / 255, 184.f / 255, 184.f / 255,
   215.f / 255, 215.f / 255, 215.f / 255,
   138.f / 255, 15.f / 255, 15.f / 255,
   184.f / 255, 20.f / 255, 20.f / 255,
   235.f / 255, 71.f / 255, 71.f / 255,
   240.f / 255, 117.f / 255, 117.f / 255,
   15.f / 255, 138.f / 255, 15.f / 255,
   20.f / 255, 184.f / 255, 20.f / 255,
   71.f / 255, 235.f / 255, 71.f / 255,
   117.f / 255, 240.f / 255, 117.f / 255,
   15.f / 255, 15.f / 255, 138.f / 255,
   20.f / 255, 20.f / 255, 184.f / 255,
   71.f / 255, 71.f / 255, 235.f / 255,
   117.f / 255, 117.f / 255, 240.f / 255,
   138.f / 255, 138.f / 255, 15.f / 255,
   184.f / 255, 184.f / 255, 20.f / 255,
   235.f / 255, 235.f / 255, 71.f / 255,
   240.f / 255, 240.f / 255, 117.f / 255,
   138.f / 255, 15.f / 255, 138.f / 255,
   184.f / 255, 20.f / 255, 184.f / 255,
   235.f / 255, 71.f / 255, 235.f / 255,
   240.f / 255, 117.f / 255, 240.f / 255,
   15.f / 255, 138.f / 255, 138.f / 255,
   20.f / 255, 184.f / 255, 184.f / 255,
   71.f / 255, 235.f / 255, 235.f / 255,
   117.f / 255, 240.f / 255, 240.f / 255
};


class TGLFaceSet : public TObject {
private:
   //non copyable class
   TGLFaceSet(const TGLFaceSet &);
   TGLFaceSet & operator = (const TGLFaceSet &);

   Int_t CheckPoints(const Int_t * source, Int_t * dest)const;
   static Bool_t Eq(const Double_t * p1, const Double_t * p2);
public:
   TGLFaceSet(const TBuffer3D & buf_initializer);

   std::vector<Double_t> fPnts;
   std::vector<Double_t> fNormals;
   std::vector<Int_t>    fPols;

   Int_t fNbPols;
   Int_t fColorInd;
};

//______________________________________________________________________________
TGLFaceSet::TGLFaceSet(const TBuffer3D & init_buf)
   : fPnts(init_buf.fPnts, init_buf.fPnts + init_buf.fNbPnts * 3),
     fNormals(init_buf.fNbPols * 3), fNbPols(init_buf.fNbPols),
     fColorInd(init_buf.fSegs[0])
{
   Int_t * segs = init_buf.fSegs;
   Int_t * pols = init_buf.fPols;
   Double_t * pnts = init_buf.fPnts;

   for (Int_t num_pol = 0, e = init_buf.fNbPols, j = 0; num_pol < e; ++num_pol) {
      ++j;
      Int_t segment_ind = pols[j] + j;
      Int_t segment_col = pols[j];
      Int_t seg1 = pols[segment_ind--];
      Int_t seg2 = pols[segment_ind--];
      Int_t np[] = {segs[seg1 * 3 + 1], segs[seg1 * 3 + 2], segs[seg2 * 3 + 1], segs[seg2 * 3 + 2]};
      Int_t n[] = {-1, -1, -1};
      Int_t normp[] = {0, 0, 0};

      np[0] != np[2] ?
               (np[0] != np[3] ?
                  (*n = *np, n[1] = np[1] == np[2] ?
                     n[2] = np[3], np[2] :
                        (n[2] = np[2], np[3])) :
                           (*n = np[1], n[1] = *np, n[2] = np[2] )) :
                              (*n = np[1], n[1] = *np, n[2] = np[3]);
     fPols.push_back(3);

     Int_t pol_size_ind = fPols.size() - 1;

     fPols.insert(fPols.end(), n, n + 3);

     Int_t check = CheckPoints(n, normp), ngood = check;

     if (check == 3)
        TMath::Normal2Plane(pnts + n[0] * 3, pnts + n[1] * 3, pnts + n[2] * 3, &fNormals[num_pol * 3]);

      while(segment_ind > j + 1) {
         seg2 = pols[segment_ind];
         np[0] = segs[seg2 * 3 + 1];
         np[1] = segs[seg2 * 3 + 2];
         if(np[0] == n[2]){
            fPols.push_back(np[1]);
            if(check != 3)
               normp[ngood ++] = np[1];
         } else{
             fPols.push_back(np[0]);
             if(check != 3)
                normp[ngood ++] = np[0];
         }

         if(check != 3 && ngood == 3) {
            check = CheckPoints(normp, normp);

            if(check == 3)
               TMath::Normal2Plane(
                                   pnts + normp[0] * 3, pnts + normp[1] * 3,
                                   pnts + normp[2] * 3, &fNormals[num_pol * 3]
                                  );
            ngood = check;
         }
         ++fPols[pol_size_ind];
         --segment_ind;
      }
      j += segment_col + 1;
   }
}

//______________________________________________________________________________
Int_t TGLFaceSet::CheckPoints(const Int_t * source, Int_t * dest) const
{
   const Double_t * p1 = &fPnts[source[0] * 3];
   const Double_t * p2 = &fPnts[source[1] * 3];
   const Double_t * p3 = &fPnts[source[2] * 3];
   Int_t ret_val = 1;

   !Eq(p1, p2) ?
      !Eq(p1, p3) ?
         !Eq(p2, p3) ?
            ret_val = 3 :
               (ret_val = 2, *dest = *source, dest[1] = source[1]) :
                  (ret_val = 2, *dest = *source, dest[1] = source[1]) :
                     !Eq(p2, p3) ?
                        ret_val = 2, *dest = *source, dest[1] = source[2] :
                           *dest = *source;

   return ret_val;
}

//______________________________________________________________________________
inline Bool_t TGLFaceSet::Eq(const Double_t * p1, const Double_t * p2)
{
   return *p1 == *p2 && p1[1] == p2[1] && p1[2] == p2[2];
}

class TGLWidget : public TGCompositeFrame {
private:
   TViewerOpenGL  *fViewer;
public:
   TGLWidget(TViewerOpenGL * c, Window_t id, const TGWindow * parent);

   Bool_t  HandleButton(Event_t * event)
   {
      return fViewer->HandleContainerButton(event);
   }
   Bool_t  HandleConfigureNotify(Event_t * event)
   {
      TGFrame::HandleConfigureNotify(event);
      return fViewer->HandleContainerConfigure(event);
   }
   Bool_t  HandleKey(Event_t * event)
   {
      return fViewer->HandleContainerKey(event);
   }
   Bool_t  HandleMotion(Event_t * event)
   {
      return fViewer->HandleContainerMotion(event);
   }
   Bool_t  HandleExpose(Event_t * event)
   {
      return fViewer->HandleContainerExpose(event);
   }
};

//______________________________________________________________________________
TGLWidget::TGLWidget(TViewerOpenGL * c, Window_t id, const TGWindow *p)
   : TGCompositeFrame(gClient, id, p)
{
   fViewer = c;
   gVirtualX->GrabButton(
                         fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask,
                         kNone, kNone
                        );
   gVirtualX->SelectInput(
                          fId,
                          kKeyPressMask | kExposureMask | kPointerMotionMask | kStructureNotifyMask
#ifndef GDK_WIN32
                         );
#else
                          | kKeyReleaseMask);
#endif
   gVirtualX->SetInputFocus(fId);
}


ClassImp(TViewerOpenGL)

//______________________________________________________________________________
TViewerOpenGL::TViewerOpenGL(TVirtualPad * vp)
   : TVirtualViewer3D(vp), TGMainFrame(gClient->GetRoot(), 600, 600),
     fCanvasWindow(0), fCanvasContainer(0), fCanvasLayout(0),
     fXc(0.), fYc(0.), fZc(0.), fRad(0.),
     fCtx(0), fGLWin(0), fPressed(kFALSE),
     fDList(1), fArcBall(0)
{
   fGLObjects.SetOwner(kTRUE);
   CreateViewer();
   Resize(600, 600);
   fArcBall = new TArcBall(600, 600);
}

//______________________________________________________________________________
void TViewerOpenGL::CreateViewer()
{
   std::auto_ptr<TGCanvas>safe_ptr1(new TGCanvas(this, GetWidth()+4, GetHeight()+4, kSunkenFrame | kDoubleBorder));

   fCanvasWindow = safe_ptr1.get();
   InitGLWindow();

   std::auto_ptr<TGLWidget>safe_ptr2(new TGLWidget(this, fGLWin, fCanvasWindow->GetViewPort()));

   fCanvasContainer = safe_ptr2.get();
   fCanvasWindow->SetContainer(fCanvasContainer);

   std::auto_ptr<TGLayoutHints> safe_ptr3(new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   fCanvasLayout = safe_ptr3.get();
   AddFrame(fCanvasWindow, fCanvasLayout);
   SetWindowName("OpenGL experimental viewer");
   SetClassHints("GLViewer", "GLViewer");
   SetMWMHints(kMWMDecorAll, kMWMFuncAll, kMWMInputModeless);
   MapSubwindows();
   Resize(GetDefaultSize());
   Show();
   safe_ptr1.release();
   safe_ptr2.release();
   safe_ptr3.release();
}

//______________________________________________________________________________
TViewerOpenGL::~TViewerOpenGL()
{
   DeleteContext();
   delete fArcBall;
   delete fCanvasContainer;
   delete fCanvasWindow;
   delete fCanvasLayout;
}

//______________________________________________________________________________
void TViewerOpenGL::InitGLWindow()
{
#ifdef GDK_WIN32
   new TGLKernel((TVirtualGLImp *)gROOT->ProcessLineFast("new TGWin32GL"));
#else
   new TGLKernel((TVirtualGLImp *)gROOT->ProcessLineFast("new TX11GL"));
#endif
   gVirtualGL->SetTrueColorMode();

   Window_t wind = fCanvasWindow->GetViewPort()->GetId();

   fGLWin = gVirtualGL->CreateGLWindow(wind);
   CreateContext();
   MakeCurrent();
}

//______________________________________________________________________________
void TViewerOpenGL::CreateContext()
{
   fCtx = gVirtualGL->CreateContext(fGLWin);
}

//______________________________________________________________________________
void TViewerOpenGL::DeleteContext()
{
   gVirtualGL->DeleteContext(fCtx);
}

//______________________________________________________________________________
void TViewerOpenGL::MakeCurrent()const
{
   gVirtualGL->MakeCurrent(fGLWin, fCtx);
}

//______________________________________________________________________________
void TViewerOpenGL::SwapBuffers()const
{
   gVirtualGL->SwapLayerBuffers(fGLWin);
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerButton(Event_t * event)
{
   if(event->fType == kButtonPress && event->fCode == kButton1)
   {
      TPoint pnt(event->fX, event->fY);

      fArcBall->Click(pnt);
      fPressed = kTRUE;
   }
   else if(event->fType == kButtonRelease && event->fCode == kButton1){
      fPressed = kFALSE;
#ifdef GDK_WIN32
      gVirtualX->SetInputFocus(fGLWin);
#endif
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerConfigure(Event_t * event)
{
   gVirtualX->ResizeWindow(fGLWin, event->fWidth, event->fHeight);
   fArcBall->SetBounds(event->fWidth, event->fHeight);
   DrawObjects();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerKey(Event_t *)
{
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerMotion(Event_t * event)
{
   if(fPressed){
      TPoint pnt(event->fX, event->fY);

      fArcBall->Drag(pnt);
      DrawObjects();
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerExpose(Event_t *)
{
   DrawObjects();
   return kTRUE;
}

//______________________________________________________________________________
void TViewerOpenGL::CreateScene(Option_t *)
{
   fGLObjects.Delete();

   TView * v = fPad->GetView();

   if(!v){
      Error("CreateScene", "view not set");
      return;
   }
   TBuffer3D * buff = fPad->GetBuffer3D();
   TObjLink * lnk = fPad->GetListOfPrimitives()->FirstLink();
   buff->fOption = TBuffer3D::kOGL;
   while(lnk){
      TObject * obj  = lnk->GetObject();
      if(obj->InheritsFrom(TAtt3D::Class()))
         obj->Paint("ogl");
      lnk = lnk->Next();
   }

   buff->fOption = TBuffer3D::kPAD;

   Double_t xdiff = fRangeX.second - fRangeX.first;
   Double_t ydiff = fRangeY.second - fRangeY.first;
   Double_t zdiff = fRangeZ.second - fRangeZ.first;
   Double_t max = xdiff > ydiff ? xdiff > zdiff ? xdiff : zdiff : ydiff > zdiff ? ydiff : zdiff;
   Double_t zfar = 3 * max;
   Double_t znear = max * 0.707;
   Double_t frp = max / 1.9;
   Float_t lmodel_amb[] = {0.5f, 0.5f, 1.f, 1.f};

   fRad = max * 1.7;
   fXc = fRangeX.first + xdiff / 2;
   fYc = fRangeY.first + ydiff / 2;
   fZc = fRangeZ.first + zdiff / 2;
   MakeCurrent();

   gVirtualGL->ClearGLColor(1.f, 1.f, 1.f, 1.f);
   gVirtualGL->ClearGLDepth(1.f);
   gVirtualGL->NewPRGL();
   gVirtualGL->FrustumGL(-frp, frp, -frp, frp, znear, zfar);
   gVirtualGL->LightModel(kLIGHT_MODEL_AMBIENT, lmodel_amb);
   gVirtualGL->LightModel(kLIGHT_MODEL_TWO_SIDE, kFALSE);
   gVirtualGL->EnableGL(kLIGHTING);
   gVirtualGL->EnableGL(kLIGHT0);
   gVirtualGL->EnableGL(kDEPTH_TEST);
   gVirtualGL->EnableGL(kCULL_FACE);
   gVirtualGL->CullFaceGL(kBACK);
   BuildGLList();
   DrawObjects();
}

//______________________________________________________________________________
void TViewerOpenGL::UpdateScene(Option_t *)
{
   TBuffer3D * buff = fPad->GetBuffer3D();

   if(buff->fOption == buff->kOGL){
      UpdateRange(buff);

      std::auto_ptr<TGLFaceSet>safe_ptr(new TGLFaceSet(*buff));

      fGLObjects.AddLast(safe_ptr.get());
      safe_ptr.release();
   }
}

//______________________________________________________________________________
void TViewerOpenGL::Show()
{
   MapRaised();
}

//______________________________________________________________________________
void TViewerOpenGL::CloseWindow()
{
   fPad->SetViewer3D(0);
   delete this;
}

//______________________________________________________________________________
void TViewerOpenGL::DrawObjects()const
{
   if(!fGLObjects.GetSize())
      return;

   MakeCurrent();
   gVirtualGL->ClearGL(0);

   Int_t cx = GetWidth() / 2, cy = GetHeight() / 2;
   Int_t d = TMath::Min(cx, cy);

   gVirtualGL->ViewportGL(cx - d, cy - d, d * 2, d * 2);
   gVirtualGL->NewMVGL();
   gVirtualGL->PushGLMatrix();
   gVirtualGL->TranslateGL(0., 0., -fRad);
   gVirtualGL->MultGLMatrix(fArcBall->GetRotMatrix());
   gVirtualGL->RotateGL(-90., 1., 0., 0.);
   gVirtualGL->TranslateGL(-fXc, -fYc, -fZc);
   gVirtualGL->RunGLList(fDList);
   gVirtualGL->PopGLMatrix();

   Float_t pos[] = {0.f, 0.f, 0.f, 1.f};

   gVirtualGL->GLLight(kLIGHT0, pos);
   SwapBuffers();
}

//______________________________________________________________________________
void TViewerOpenGL::UpdateRange(const TBuffer3D * buffer)
{
   Double_t xmin = buffer->fPnts[0], xmax = xmin, ymin = buffer->fPnts[1], ymax = ymin, zmin = buffer->fPnts[2], zmax = zmin;
   //calculate range
   for(Int_t i = 3, e = buffer->fNbPnts * 3; i < e; i += 3)
      xmin = TMath::Min(xmin, buffer->fPnts[i]), xmax = TMath::Max(xmax, buffer->fPnts[i]),
      ymin = TMath::Min(ymin, buffer->fPnts[i + 1]), ymax = TMath::Max(ymax, buffer->fPnts[i + 1]),
      zmin = TMath::Min(zmin, buffer->fPnts[i + 2]), zmax = TMath::Max(zmax, buffer->fPnts[i + 2]);

   if(!fGLObjects.GetSize()){
      fRangeX.first = xmin, fRangeX.second = xmax;
      fRangeY.first = ymin, fRangeY.second = ymax;
      fRangeZ.first = zmin, fRangeZ.second = zmax;
      return;
   }

   if(fRangeX.first > xmin)
      fRangeX.first = xmin;
   if(fRangeX.second < xmax)
      fRangeX.second = xmax;
   if(fRangeY.first > ymin)
      fRangeY.first = ymin;
   if(fRangeY.second < ymax)
      fRangeY.second = ymax;
   if(fRangeZ.first > zmin)
      fRangeZ.first = zmin;
   if(fRangeZ.second < zmax)
      fRangeZ.second = zmax;
}

//______________________________________________________________________________
void TViewerOpenGL::BuildGLList()const
{
   gVirtualGL->NewGLList(fDList, kCOMPILE);

   TObjLink * lnk = fGLObjects.FirstLink();
   GLUtesselator * t_obj = gVirtualGL->GLUNewTess();

   if(!t_obj)
      Error("DrawObjects", "No tesselator :(( ");

   while(lnk){
      TGLFaceSet * pobj = static_cast<TGLFaceSet *>(lnk->GetObject());
      Int_t * pols = &pobj->fPols[0];
      Double_t * pnts = &pobj->fPnts[0];
      Double_t * normalvector = &pobj->fNormals[0];
      //first, define the color :
      Int_t ind = pobj->fColorInd;
      Float_t rgb[] = {colors[ind * 3], colors[ind * 3 + 1], colors[ind * 3 + 2]};

      gVirtualGL->MaterialGL(kFRONT, rgb);
      gVirtualGL->MaterialGL(kFRONT, 60.);

      for(Int_t i = 0, npols = pobj->fNbPols, j = 0; i < npols; ++i){
         Int_t npoints = pols[j++];
         if(t_obj && npoints > 4){
            gVirtualGL->GLUTessCallback(t_obj);
            gVirtualGL->GLUBeginPolygon(t_obj);
            gVirtualGL->GLUNextContour(t_obj);
            gVirtualGL->SetGLNormal(normalvector + i * 3);

            for(Int_t k = 0; k < npoints; ++k, ++j)
               gVirtualGL->GLUTessVertex(t_obj, pnts + pols[j] * 3);

            gVirtualGL->GLUEndPolygon(t_obj);
         }
         else{
            gVirtualGL->BeginGL();
            gVirtualGL->SetGLNormal(normalvector + i * 3);

            for(Int_t k = 0; k < npoints; ++k, ++j)
               gVirtualGL->SetGLVertex(pnts + pols[j] * 3);
            gVirtualGL->EndGL();
         }
      }
      lnk = lnk->Next();
   }
   gVirtualGL->GLUDeleteTess(t_obj);
   gVirtualGL->EndGLList();
}
