// @(#)root/gl:$Name:  $:$Id: TViewerOpenGL.cxx,v 1.18 2004/09/14 15:15:46 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TRootHelpDialog.h"
#include "TContextMenu.h"
#include "TVirtualPad.h"
#include "TVirtualGL.h"
#include "KeySymbols.h"
#include "TVirtualX.h"
#include "TBuffer3D.h"
#include "TGLKernel.h"
#include "TGClient.h"
#include "TGCanvas.h"
#include "HelpText.h"
#include "Buttons.h"
#include "TAtt3D.h"
#include "TGMenu.h"
#include "TPoint.h"
#include "TROOT.h"
#include "TMath.h"
#include "TColor.h"
#include "TTimer.h"
///////////////////////////
#include "TGSplitter.h"
#include "TGButton.h"
#include "TGLEditor.h"
/////////////////////////////
#include "TGLSceneObject.h"
#include "TViewerOpenGL.h"
#include "TGLRenderArea.h"
#include "TGLRender.h"
#include "TGLCamera.h"
#include "TArcBall.h"


const char gHelpViewerOpenGL[] = "\
     PRESS \n\
     \tw\t--- wireframe mode\n\
     \tr\t--- hidden surface mode\n\
     \tj\t--- zoom in\n\
     \tk\t--- zoom out\n\
     HOLD the left mouse button and MOVE mouse to ROTATE object\n\n";

const Double_t gRotMatrixXOY[] = {1., 0., 0., 0., 0., 0., -1., 0.,
                                  0., 1., 0., 0., 0., 0., 0., 1.};
const Double_t gRotMatrixYOZ[] = {0., 0., -1., 0., 0., 1., 0., 0.,
                                  1., 0., 0., 0., 0., 0., 0., 1.};
const Double_t gIdentity[] = {1., 0., 0., 0., 0., 1., 0., 0.,
                              0., 0., 1., 0., 0., 0., 0., 1.};


enum EGLViewerCommands {
   kGLHelpAbout,
   kGLHelpOnViewer,
   kGLNavMode,
   kGLPickMode,
   kGLXOY,
   kGLXOZ,
   kGLYOZ,
   kGLPersp,
   kGLExit
};

ClassImp(TViewerOpenGL)

//______________________________________________________________________________
TViewerOpenGL::TViewerOpenGL(TVirtualPad * vp)
                  :TVirtualViewer3D(vp),
                   TGMainFrame(gClient->GetRoot(), 750, 600),
                   fCanvasWindow(0), fCanvasContainer(0), fCanvasLayout(0),
                   fMenuBar(0), fFileMenu(0), fModeMenu(0), fViewMenu(0), fHelpMenu(0),
                   fMenuBarLayout(0), fMenuBarItemLayout(0), fMenuBarHelpLayout(0),
                   fCamera(), fViewVolume(), fZoom(),
                   fActiveViewport(), fXc(0.), fYc(0.),
                   fZc(0.), fRad(0.), fPressed(kFALSE), fArcBall(0),
                   fSelected(0), fNbShapes(0), fConf(kPERSP), fMode(kNav),
                   fMainFrame(0), fEdFrame(0), fEditor(0),
                   fSelectedObj(0), fRGBA(), fV1(0), fV2(0),
                   fSplitter(0)
{
   static struct Init {
      Init()
      {
#ifdef GDK_WIN32
         new TGLKernel((TVirtualGLImp *)gROOT->ProcessLineFast("new TGWin32GL"));
#else
         new TGLKernel((TVirtualGLImp *)gROOT->ProcessLineFast("new TX11GL"));
#endif
      }
   }initGL;

   CreateViewer();
   Resize(750, 600);
   fArcBall = new TArcBall(600, 600);
   CalculateViewports();
}

//______________________________________________________________________________
void TViewerOpenGL::CreateViewer()
{
/////////////////////////////////////Menu Creation//////////////////////////////////////////////
   fFileMenu = new TGPopupMenu(fClient->GetRoot());
   fFileMenu->AddEntry("&Exit", kGLExit);
   fFileMenu->Associate(this);

   fModeMenu = new TGPopupMenu(fClient->GetRoot());
   fModeMenu->AddEntry("&Navigation", kGLNavMode);
   fModeMenu->AddEntry("P&icking", kGLPickMode);
   fModeMenu->Associate(this);

   fViewMenu = new TGPopupMenu(fClient->GetRoot());
   fViewMenu->AddEntry("&XOY plane", kGLXOY);
   fViewMenu->AddEntry("XO&Z plane", kGLXOZ);
   fViewMenu->AddEntry("&YOZ plane", kGLYOZ);
   fViewMenu->AddEntry("&Perspective view", kGLPersp);
   fViewMenu->Associate(this);

   fHelpMenu = new TGPopupMenu(fClient->GetRoot());
   fHelpMenu->AddEntry("&About ROOT...", kGLHelpAbout);
   fHelpMenu->AddSeparator();
   fHelpMenu->AddEntry("Help on OpenGL Viewer...", kGLHelpOnViewer);
   fHelpMenu->Associate(this);

   // Create menubar layout hints
   fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 1, 1);
   fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);
   fMenuBarHelpLayout = new TGLayoutHints(kLHintsTop | kLHintsRight);

   // Create menubar
   fMenuBar = new TGMenuBar(this, 1, 1, kHorizontalFrame | kRaisedFrame);
   fMenuBar->AddPopup("&File", fFileMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Mode", fModeMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&View", fViewMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Help",    fHelpMenu,    fMenuBarHelpLayout);
   AddFrame(fMenuBar, fMenuBarLayout);

   fMainFrame = new TGCompositeFrame(this, 100, 100, kHorizontalFrame | kRaisedFrame);
   fV1 = new TGVerticalFrame(fMainFrame, 150, 10, kSunkenFrame | kFixedWidth);
   fV2 = new TGVerticalFrame(fMainFrame, 10, 10, kSunkenFrame);
   fL1 = new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 2, 0, 2, 2);
   fMainFrame->AddFrame(fV1, fL1);
   fSplitter = new TGVSplitter(fMainFrame, 5);
   fSplitter->SetFrame(fV1, kTRUE);
   fL2 = new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 0, 0 ,0, 0);
   fMainFrame->AddFrame(fSplitter, fL2);
   fL3 = new TGLayoutHints(kLHintsRight | kLHintsExpandX | kLHintsExpandY,0,2,2,2);
   fMainFrame->AddFrame(fV2, fL3);
   //////////////////////////////////////////////////////////////////////////////////////
   fEditor = new TGLEditor(fV1, 25, 50, 75, 100);
   fL4 = new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX | kLHintsExpandY, 2, 5, 1, 2);
   fV1->AddFrame(fEditor, fL4);
   fEditor->GetButton()->Connect("Pressed()", "TViewerOpenGL", this, "ModifySelected()");
   /////////////////////////create view part/////////////////////////////////////////////
   fCanvasWindow = new TGCanvas(fV2, 10, 10, kSunkenFrame | kDoubleBorder);
   fCanvasContainer = new TGLRenderArea(fCanvasWindow->GetViewPort()->GetId(), fCanvasWindow->GetViewPort());

   TGLWindow * glWin = fCanvasContainer->GetGLWindow();

   glWin->Connect("HandleButton(Event_t*)", "TViewerOpenGL", this, "HandleContainerButton(Event_t*)");
   glWin->Connect("HandleKey(Event_t*)", "TViewerOpenGL", this, "HandleContainerKey(Event_t*)");
   glWin->Connect("HandleMotion(Event_t*)", "TViewerOpenGL", this, "HandleContainerMotion(Event_t*)");
   glWin->Connect("HandleExpose(Event_t*)", "TViewerOpenGL", this, "HandleContainerExpose(Event_t*)");
   glWin->Connect("HandleConfigureNotify(Event_t*)", "TViewerOpenGL", this, "HandleContainerConfigure(Event_t*)");

   fCanvasWindow->SetContainer(glWin);
   fCanvasLayout = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   fV2->AddFrame(fCanvasWindow, fCanvasLayout);
   AddFrame(fMainFrame, fCanvasLayout);
   ///////////////////////////////////////////////////////////////////////////////////////
   SetWindowName("OpenGL experimental viewer");
   SetClassHints("GLViewer", "GLViewer");
   SetMWMHints(kMWMDecorAll, kMWMFuncAll, kMWMInputModeless);
   MapSubwindows();
   Resize(GetDefaultSize());
   Show();
   fZoom[0] = fZoom[1] = fZoom[2] = fZoom[3] = 1.;
}

//______________________________________________________________________________
TViewerOpenGL::~TViewerOpenGL()
{
   delete fFileMenu;
   delete fModeMenu;
   delete fViewMenu;
   delete fHelpMenu;
   delete fMenuBar;
   delete fMenuBarLayout;
   delete fMenuBarHelpLayout;
   delete fMenuBarItemLayout;
   delete fArcBall;
   delete fCanvasContainer;
   delete fCanvasWindow;
   delete fCanvasLayout;
   delete fV1;
   delete fV2;
   delete fMainFrame;
   delete fSplitter;
   delete fL1;
   delete fL2;
   delete fL3;
   delete fL4;
}

//______________________________________________________________________________
void TViewerOpenGL::MakeCurrent()const
{
   fCanvasContainer->GetGLWindow()->MakeCurrent();
}

//______________________________________________________________________________
void TViewerOpenGL::SwapBuffers()const
{
   fCanvasContainer->GetGLWindow()->Refresh();
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerButton(Event_t *event)
{
   if (event->fType == kButtonPress && event->fCode == kButton1) {
      if(fMode == kNav) {
         TPoint pnt(event->fX, event->fY);
         fArcBall->Click(pnt);
         fPressed = kTRUE;
      } else {
         if ((fSelectedObj = TestSelection(event))) {
            fSelectedObj->GetColor(fRGBA[0], fRGBA[1], fRGBA[2], fRGBA[3]);
            fEditor->SetRGBA(fRGBA[0], fRGBA[1], fRGBA[2], fRGBA[3]);
            if (fConf != kPERSP) {
               fPressed = kTRUE;
               fLastPos.fX = event->fX;
               fLastPos.fY = event->fY;
            }
         } else {
            fEditor->GetButton()->SetState(kButtonDisabled);
            fEditor->Stop();
         }
      }
   } else if (event->fType == kButtonPress && event->fCode == kButton3 && fMode == kNav) {
      if ((fSelectedObj = TestSelection(event))) {
         fSelectedObj->GetColor(fRGBA[0], fRGBA[1], fRGBA[2], fRGBA[3]);
         fEditor->SetRGBA(fRGBA[0], fRGBA[1], fRGBA[2], fRGBA[3]);
      } else {
         fEditor->GetButton()->SetState(kButtonDisabled);
         fEditor->Stop();
      }
   } else if (event->fType == kButtonRelease) {
      if (event->fCode == kButton1) {
         fPressed = kFALSE;
         if (fMode == kPick) {
            MakeCurrent();
            gVirtualGL->EndMovement(&fRender);
            DrawObjects();
         }
      }
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerConfigure(Event_t *event)
{
   fArcBall->SetBounds(event->fWidth, event->fHeight);
   CalculateViewports();
   CalculateViewvolumes();
   DrawObjects();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerKey(Event_t *event)
{
   char tmp[10] = {0};
   UInt_t keysym = 0;

   gVirtualX->LookupString(event, tmp, sizeof(tmp), keysym);

   switch (keysym) {
   case kKey_Plus:
   case kKey_J:
   case kKey_j:
      fZoom[fConf] /= 1.2;
      fCamera[fConf]->Zoom(fZoom[fConf]);
      DrawObjects();
      break;
   case kKey_Minus:
   case kKey_K:
   case kKey_k:
      fZoom[fConf] *= 1.2;
      fCamera[fConf]->Zoom(fZoom[fConf]);
      DrawObjects();
      break;
   case kKey_R:
   case kKey_r:
      gVirtualGL->PolygonGLMode(kFRONT, kFILL);
      gVirtualGL->EnableGL(kCULL_FACE);
      gVirtualGL->SetGLLineWidth(1.f);
      DrawObjects();
      break;
   case kKey_W:
   case kKey_w:
      gVirtualGL->DisableGL(kCULL_FACE);
      gVirtualGL->PolygonGLMode(kFRONT_AND_BACK, kLINE);
      gVirtualGL->SetGLLineWidth(1.5f);
      DrawObjects();
      break;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerMotion(Event_t *event)
{
   if (fPressed) {
      if (fMode == kNav) {
         TPoint pnt(event->fX, event->fY);
         fArcBall->Drag(pnt);
         DrawObjects();
      } else if (fMode == kPick) {
         Double_t xshift = Double_t(event->fX - fLastPos.fX) / GetWidth() * (fRangeX.second - fRangeX.first);
         Double_t yshift = Double_t(event->fY - fLastPos.fY);
         yshift /= (GetWidth() - fMenuBar->GetHeight() - fMenuBarLayout->GetPadTop()
                  - fMenuBarLayout->GetPadBottom() - fMenuBarHelpLayout->GetPadTop()
                  - fMenuBarHelpLayout->GetPadBottom());
         yshift *= (fRangeY.second - fRangeY.first);
         MakeCurrent();
         switch (fConf) {
         case kXOY:
            gVirtualGL->MoveSelected(&fRender, xshift, yshift, 0.);
            break;
         case kXOZ:
            gVirtualGL->MoveSelected(&fRender, xshift, 0., -yshift);
            break;
         case kYOZ:
            gVirtualGL->MoveSelected(&fRender, 0., -xshift, -yshift);
            break;
	 default:
	    break;
         }

         DrawObjects();
         fLastPos.fX = event->fX;
         fLastPos.fY = event->fY;
      }
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
   TBuffer3D * buff = fPad->GetBuffer3D();
   TObjLink * lnk = fPad->GetListOfPrimitives()->FirstLink();
   buff->fOption = TBuffer3D::kOGL;
   while (lnk) {
      TObject * obj  = lnk->GetObject();
      if (obj->InheritsFrom(TAtt3D::Class()))
         obj->Paint("ogl");
      lnk = lnk->Next();
   }

   buff->fOption = TBuffer3D::kPAD;
   CalculateViewvolumes();
   MakeCurrent();
   Float_t lmodelAmb[] = {0.5f, 0.5f, 1.f, 1.f};
   gVirtualGL->LightModel(kLIGHT_MODEL_AMBIENT, lmodelAmb);
   gVirtualGL->LightModel(kLIGHT_MODEL_TWO_SIDE, kFALSE);
   gVirtualGL->EnableGL(kLIGHTING);
   gVirtualGL->EnableGL(kLIGHT0);
   gVirtualGL->EnableGL(kLIGHT1);
   gVirtualGL->EnableGL(kLIGHT2);
   gVirtualGL->EnableGL(kLIGHT3);
   gVirtualGL->EnableGL(kDEPTH_TEST);
   gVirtualGL->EnableGL(kCULL_FACE);
   gVirtualGL->CullFaceGL(kBACK);
   gVirtualGL->PolygonGLMode(kFRONT, kFILL);
   gVirtualGL->ClearGLColor(0.f, 0.f, 0.f, 1.f);
   gVirtualGL->ClearGLDepth(1.f);

   CreateCameras();
   fRender.SetActive(kPERSP);
   DrawObjects();
}

//______________________________________________________________________________
void TViewerOpenGL::UpdateScene(Option_t *)
{
   TBuffer3D * buff = fPad->GetBuffer3D();

   if (buff->fOption == buff->kOGL) {
      ++fNbShapes;
      TGLSceneObject *addObj = 0;
      TColor *color = gROOT->GetColor(buff->fColor);
      Float_t rgb[3];
      if (color) {
         rgb[0] = color->GetRed();
         rgb[1] = color->GetGreen();
         rgb[2] = color->GetBlue();
      }

      switch (buff->fType) {
      case TBuffer3D::kLINE:
         addObj = new TGLPolyLine(*buff, rgb);
  	      break;
      case TBuffer3D::kMARKER:
         addObj = new TGLPolyMarker(*buff, rgb);
         break;
      default:
         addObj = new TGLFaceSet(*buff, rgb, fNbShapes, buff->fId);
         break;
      }

      TGLSelection *box = UpdateRange(buff);
      fRender.AddNewObject(addObj, box);
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

   //DeleteWindow();
   TTimer::SingleShot(50, IsA()->GetName(), this, "ReallyDelete()");
}

//______________________________________________________________________________
void TViewerOpenGL::DrawObjects()const
{
   MakeCurrent();
   gVirtualGL->TraverseGraph(const_cast<TGLRender *>(&fRender));
   gVirtualGL->NewMVGL();

   Float_t pos[] = {0.f, 0.f, 0.f, 1.f};
   Float_t lig_prop1[] = {.4f, .4f, .4f, 1.f};

   gVirtualGL->GLLight(kLIGHT0, kPOSITION, pos);
   gVirtualGL->PushGLMatrix();
   gVirtualGL->TranslateGL(0., fRad + fYc, -fRad - fZc);
   gVirtualGL->GLLight(kLIGHT1, kPOSITION, pos);
   gVirtualGL->GLLight(kLIGHT1, kDIFFUSE, lig_prop1);
   gVirtualGL->PopGLMatrix();

   gVirtualGL->PushGLMatrix();
   gVirtualGL->TranslateGL(fRad + fXc, 0., -fRad - fZc);
   gVirtualGL->GLLight(kLIGHT2, kPOSITION, pos);
   gVirtualGL->GLLight(kLIGHT2, kDIFFUSE, lig_prop1);
   gVirtualGL->PopGLMatrix();

   gVirtualGL->TranslateGL(-fRad - fXc, 0., -fRad - fZc);
   gVirtualGL->GLLight(kLIGHT3, kPOSITION, pos);
   gVirtualGL->GLLight(kLIGHT3, kDIFFUSE, lig_prop1);

   SwapBuffers();
}

//______________________________________________________________________________
TGLSelection * TViewerOpenGL::UpdateRange(const TBuffer3D *buffer)
{
   Double_t xmin = buffer->fPnts[0], xmax = xmin, ymin = buffer->fPnts[1], ymax = ymin, zmin = buffer->fPnts[2], zmax = zmin;
   //calculate range
   for (Int_t i = 3, e = buffer->fNbPnts * 3; i < e; i += 3)
      xmin = TMath::Min(xmin, buffer->fPnts[i]), xmax = TMath::Max(xmax, buffer->fPnts[i]),
      ymin = TMath::Min(ymin, buffer->fPnts[i + 1]), ymax = TMath::Max(ymax, buffer->fPnts[i + 1]),
      zmin = TMath::Min(zmin, buffer->fPnts[i + 2]), zmax = TMath::Max(zmax, buffer->fPnts[i + 2]);

   TGLSelection *retVal = new TGLSelection(std::make_pair(xmin, xmax),
                                           std::make_pair(ymin, ymax),
                                           std::make_pair(zmin, zmax));

   if (!fRender.GetSize()) {
      fRangeX.first = xmin, fRangeX.second = xmax;
      fRangeY.first = ymin, fRangeY.second = ymax;
      fRangeZ.first = zmin, fRangeZ.second = zmax;

      return retVal;
   }

   if (fRangeX.first > xmin)
      fRangeX.first = xmin;
   if (fRangeX.second < xmax)
      fRangeX.second = xmax;
   if (fRangeY.first > ymin)
      fRangeY.first = ymin;
   if (fRangeY.second < ymax)
      fRangeY.second = ymax;
   if (fRangeZ.first > zmin)
      fRangeZ.first = zmin;
   if (fRangeZ.second < zmax)
      fRangeZ.second = zmax;

   return retVal;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   switch (GET_MSG(msg)) {
   case kC_COMMAND:
      switch (GET_SUBMSG(msg)) {
      case kCM_BUTTON:
	   case kCM_MENU:
	      switch (parm1) {
         case kGLHelpAbout: {
            char str[32];
            sprintf(str, "About ROOT %s...", gROOT->GetVersion());
            TRootHelpDialog * hd = new TRootHelpDialog(this, str, 600, 400);
            hd->SetText(gHelpAbout);
            hd->Popup();
            break;
         }
         case kGLHelpOnViewer: {
            TRootHelpDialog * hd = new TRootHelpDialog(this, "Help on GL Viewer...", 600, 400);
            hd->SetText(gHelpViewerOpenGL);
            hd->Popup();
            break;
         }
         case kGLNavMode:
            fMode = kNav;
            if (fConf != kPERSP) {
               fConf = kPERSP;
               fRender.SetActive(fConf);
               DrawObjects();
            }
            break;
         case kGLPickMode:
            fMode = kPick;
            if (fConf == kPERSP) {
               fConf = kXOZ;
               fRender.SetActive(fConf);
               DrawObjects();
            }
            break;
         case kGLXOY:

            if (fConf != kXOY) {
            //set active camera
               fConf = kXOY;
               fRender.SetActive(fConf);
               DrawObjects();
            }
            break;
         case kGLXOZ:
            if (fConf != kXOZ) {
            //set active camera
               fConf = kXOZ;
               fRender.SetActive(fConf);
               DrawObjects();
            }
            break;
         case kGLYOZ:
            if (fConf != kYOZ) {
            //set active camera
               fConf = kYOZ;
               fRender.SetActive(fConf);
               DrawObjects();
            }
            break;
         case kGLPersp:
            if (fConf != kPERSP) {
            //set active camera
               fConf = kPERSP;
               fRender.SetActive(fConf);
               DrawObjects();
               if(fMode != kNav)
                  fMode = kNav;
            }
            break;
         case kGLExit:
            CloseWindow();
            break;
	      default:
	         break;
	      }
	   default:
	      break;
      }
   default:
      break;
   }

   return kTRUE;
}

//______________________________________________________________________________
TGLSceneObject *TViewerOpenGL::TestSelection(Event_t *event)
{
   MakeCurrent();
   TGLSceneObject *obj = gVirtualGL->SelectObject(&fRender, event->fX, event->fY, fConf);
   SwapBuffers();

   return obj;
}

void TViewerOpenGL::CalculateViewports()
{
   fActiveViewport[0] = 0;
   fActiveViewport[1] = 0;
   fActiveViewport[2] = fCanvasWindow->GetWidth();
   fActiveViewport[3] = fCanvasWindow->GetHeight();
}

void TViewerOpenGL::CalculateViewvolumes()
{
   if (fRender.GetSize()) {
      Double_t xdiff = fRangeX.second - fRangeX.first;
      Double_t ydiff = fRangeY.second - fRangeY.first;
      Double_t zdiff = fRangeZ.second - fRangeZ.first;
      Double_t max = xdiff > ydiff ? xdiff > zdiff ? xdiff : zdiff : ydiff > zdiff ? ydiff : zdiff;

      Int_t w = fCanvasWindow->GetWidth() / 2;
      Int_t h = (fCanvasWindow->GetHeight()) / 2;
      Double_t frx = 1., fry = 1.;

      if (w > h)
         frx = w / double(h);
      else if (w < h)
         fry = h / double(w);

      fViewVolume[0] = max / 1.9 * frx;
      fViewVolume[1] = max / 1.9 * fry;
      fViewVolume[2] = max * 0.707;
      fViewVolume[3] = 3 * max;

      fXc = fRangeX.first + xdiff / 2;
      fYc = fRangeY.first + ydiff / 2;
      fZc = fRangeZ.first + zdiff / 2;
      fRad = max * 1.7;
   }
}

void TViewerOpenGL::CreateCameras()
{
   if (!fRender.GetSize())
      return;

   TGLSimpleTransform trXOY(gRotMatrixXOY, fRad, fXc, fYc, fZc);
   TGLSimpleTransform trXOZ(gIdentity, fRad, fXc, fYc, fZc);
   TGLSimpleTransform trYOZ(gRotMatrixYOZ, fRad, fXc, fYc, fZc);
   TGLSimpleTransform trPersp(fArcBall->GetRotMatrix(), fRad, fXc, fYc, fZc);

   fCamera[kXOY]   = new TGLOrthoCamera(fViewVolume, fActiveViewport, trXOY);
   fCamera[kXOZ]   = new TGLOrthoCamera(fViewVolume, fActiveViewport, trXOZ);
   fCamera[kYOZ]   = new TGLOrthoCamera(fViewVolume, fActiveViewport, trYOZ);
   fCamera[kPERSP] = new TGLPerspectiveCamera(fViewVolume, fActiveViewport, trPersp);

   fRender.AddNewCamera(fCamera[kXOY]);
   fRender.AddNewCamera(fCamera[kXOZ]);
   fRender.AddNewCamera(fCamera[kYOZ]);
   fRender.AddNewCamera(fCamera[kPERSP]);
}

void TViewerOpenGL::ModifySelected()
{
   fEditor->GetButton()->SetState(kButtonDisabled);
   fEditor->GetRGBA(fRGBA[0], fRGBA[1], fRGBA[2], fRGBA[3]);
   fSelectedObj->SetColor(fRGBA[0], fRGBA[1], fRGBA[2], fRGBA[3]);
   MakeCurrent();
   gVirtualGL->Invalidate(&fRender);
   DrawObjects();
}
