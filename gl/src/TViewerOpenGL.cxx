// @(#)root/gl:$Name:  $:$Id: TViewerOpenGL.cxx,v 1.41 2004/12/01 16:57:19 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include "TPluginManager.h"
#include "TRootHelpDialog.h"
#include "TContextMenu.h"
#include "TVirtualPad.h"
#include "TVirtualGL.h"
#include "KeySymbols.h"
#include "TGShutter.h"
#include "TVirtualX.h"
#include "TBuffer3D.h"
#include "TGLKernel.h"
#include "TGButton.h"
#include "TGClient.h"
#include "TGCanvas.h"
#include "HelpText.h"
#include "Buttons.h"
#include "TAtt3D.h"
#include "TGMenu.h"
#include "TPoint.h"
#include "TColor.h"
#include "TTimer.h"
#include "TROOT.h"
#include "TMath.h"

#include "TGLSceneObject.h"
#include "TViewerOpenGL.h"
#include "TGLRenderArea.h"
#include "TGLEditor.h"
#include "TGLRender.h"
#include "TGLCamera.h"
#include "TArcBall.h"

const char gHelpViewerOpenGL[] = "\
     PRESS \n\
     \tw\t--- wireframe mode\n\
     \tr\t--- filled polygons mode\n\
     \tj\t--- zoom in\n\
     \tk\t--- zoom out\n\n\
     You can ROTATE the scene by holding the left \n\
     mouse button and moving the mouse or\n\
     SELECT an object with right mouse button click.\n\
     You can select and move an object with the middle\n\
     mouse button (light sources are pickable too).\n\n\
     PROJECTIONS\n\n\
     You can select the different plane projections\n\
     in \"Projections\" menu.\n\n\
     COLOR\n\n\
     After you selected an object or a light source,\n\
     you can modify object's material and light\n\
     source color.\n\n\
     \tLIGHT SOURCES.\n\n\
     \tThere are two pickable light sources in\n\
     \tthe current implementation. They are shown as\n\
     \tspheres. Each light source has three light\n\
     \tcomponents : DIFFUSE, AMBIENT, SPECULAR.\n\
     \tEach of this components is defined by the\n\
     \tamounts of red, green and blue light it emits.\n\
     \tYou can EDIT this parameters:\n\
     \t1. Select light source sphere.\n" //hehe, too long string literal :)))
"    \t2. Select light component you want to modify\n\
     \t   by pressing one of radio buttons.\n\
     \t3. Change RGB by moving sliders\n\n\
     \tMATERIAL\n\n\
     \tObject's material is specified by the percentage\n\
     \tof red, green, blue light it reflects. A surface can\n\
     \treflect diffuse, ambient and specular light. \n\
     \tA surface has two additional parameters: EMISSION\n\
     \t- you can make surface self-luminous; SHININESS -\n\
     \tmodifying this parameter you can change surface\n\
     \thighlights.\n\
     \tSometimes changes are not visible, or light\n\
     \tsources seem not to work - you should understand\n\
     \tthe meaning of diffuse, ambient etc. light and material\n\
     \tcomponents. For example, if you define material, wich has\n\
     \tdiffuse component (1., 0., 0.) and you have a light source\n\
     \twith diffuse component (0., 1., 0.) - you surface does not\n\
     \treflect diffuse light from this source. For another example\n\
     \t- the color of highlight on the surface is specified by:\n\
     \tlight's specular component, material specular component.\n\
     \tAt the top of the color editor there is a small window\n\
     \twith sphere. When you are editing surface material,\n\
     \tyou can see this material applyed to sphere.\n\
     \tWhen edit light source, you see this light reflected\n\
     \tby sphere whith DIFFUSE and SPECULAR components\n\
     \t(1., 1., 1.).\n\n\
     OBJECT'S GEOMETRY\n\n\
     You can edit object's location and stretch it by entering\n\
     desired values in respective number entry controls.\n\n"
"    SCENE PROPERTIES\n\n\
     You can add clipping plane by clicking the checkbox and\n\
     specifying the plane's equation A*x+B*y+C*z+D=0.";


const Double_t gRotMatrixXOY[] = {1., 0., 0., 0., 0., 0., 1., 0.,
                                  0., -1., 0., 0., 0., 0., 0., 1.};

const Double_t gRotMatrixYOZ[] = {0., 0., -1., 0., 1., 0., 0., 0.,
                                  0., -1., 0., 0., 0., 0., 0., 1.};
                                
const Double_t gRotMatrixXOZ[] = {0., 1., 0., 0., 1., 0., 0., 0.,
                                  0., 0., -1., 0., 0., 0., 0., 1.};




enum EGLViewerCommands {
   kGLHelpAbout,
   kGLHelpOnViewer,
   kGLXOY,
   kGLXOZ,
   kGLYOZ,
   kGLPersp,
   kGLExit
};

ClassImp(TViewerOpenGL)

const Int_t TViewerOpenGL::fgInitX = 0;
const Int_t TViewerOpenGL::fgInitY = 0;
const Int_t TViewerOpenGL::fgInitW = 780;
const Int_t TViewerOpenGL::fgInitH = 670;

//______________________________________________________________________________
TViewerOpenGL::TViewerOpenGL(TVirtualPad * vp)
                  :TVirtualViewer3D(vp),
                   TGMainFrame(gClient->GetDefaultRoot(), fgInitW, fgInitH),
                   fCamera(), fViewVolume(), fZoom(),
                   fActiveViewport()
{
   // Create OpenGL viewer.

   //good compiler (not VC 6.0) will initialize our
   //arrays in ctor-init-list with zeroes (default initialization)
   fMainFrame = 0;
   fV2 = 0;
   fV1 = 0;
   fColorEditor = 0;
   fGeomEditor = 0;
   fSceneEditor = 0;
   fCanvasWindow = 0;
   fCanvasContainer = 0;
   fL1 = fL2 = fL3 = fL4 = 0;
   fCanvasLayout = 0;
   fMenuBar = 0;
   fFileMenu = fViewMenu = fHelpMenu = 0;
   fMenuBarLayout = fMenuBarItemLayout = fMenuBarHelpLayout = 0;
   fLightMask = 0x1b;   
   fXc = fYc = fZc = fRad = 0.;
   fPressed = kFALSE;
   fNbShapes = 0;
   fConf = kPERSP;
   fContextMenu = 0;
   fSelectedObj = 0;
   fAction = kNoAction;

   static Bool_t init = kFALSE;
   if (!init) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualGLImp"))) {
         if (h->LoadPlugin() == -1)
            return;
         TVirtualGLImp *imp = (TVirtualGLImp *) h->ExecPlugin(0);
         new TGLKernel(imp);
      }
      init = kTRUE;
   }

   CreateViewer();
   fArcBall = new TArcBall(fgInitH, fgInitH);
   CalculateViewports();
}

//______________________________________________________________________________
void TViewerOpenGL::CreateViewer()
{
   // Menus creation
   fFileMenu = new TGPopupMenu(fClient->GetRoot());
   fFileMenu->AddEntry("&Exit", kGLExit);
   fFileMenu->Associate(this);

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
   fMenuBar->AddPopup("&Projections", fViewMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Help",    fHelpMenu,    fMenuBarHelpLayout);
   AddFrame(fMenuBar, fMenuBarLayout);

   // Frames creation
   fMainFrame = new TGCompositeFrame(this, 100, 100, kHorizontalFrame | kRaisedFrame);
   fV1 = new TGVerticalFrame(fMainFrame, 150, 10, kSunkenFrame | kFixedWidth);
   fShutter = new TGShutter(fV1, kSunkenFrame | kFixedWidth);
   fShutItem1 = new TGShutterItem(fShutter, new TGHotString("Color"), 5001);
   fShutItem2 = new TGShutterItem(fShutter, new TGHotString("Object's geometry"), 5002);
   fShutItem3 = new TGShutterItem(fShutter, new TGHotString("Scene"), 5003);
   fShutItem4 = new TGShutterItem(fShutter, new TGHotString("Lights"), 5004);
   fShutter->AddItem(fShutItem1);
   fShutter->AddItem(fShutItem2);
   fShutter->AddItem(fShutItem3);
   fShutter->AddItem(fShutItem4);

   TGCompositeFrame *shutCont = (TGCompositeFrame *)fShutItem1->GetContainer();
   fColorEditor = new TGLColorEditor(shutCont, this);
   fL4 = new TGLayoutHints(kLHintsTop | kLHintsCenterX | kLHintsExpandX | kLHintsExpandY, 2, 5, 1, 2);
   shutCont->AddFrame(fColorEditor, fL4);
   fV1->AddFrame(fShutter, fL4);
   fL1 = new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 2, 0, 2, 2);
   fMainFrame->AddFrame(fV1, fL1);

   shutCont = (TGCompositeFrame *)fShutItem2->GetContainer();
   fGeomEditor = new TGLGeometryEditor(shutCont, this);
   shutCont->AddFrame(fGeomEditor, fL4);

   shutCont = (TGCompositeFrame *)fShutItem3->GetContainer();
   fSceneEditor = new TGLSceneEditor(shutCont, this);
   shutCont->AddFrame(fSceneEditor, fL4);

   shutCont = (TGCompositeFrame *)fShutItem4->GetContainer();
   fLightEditor = new TGLLightEditor(shutCont, this);
   shutCont->AddFrame(fLightEditor, fL4);

   fV2 = new TGVerticalFrame(fMainFrame, 10, 10, kSunkenFrame);
   fL3 = new TGLayoutHints(kLHintsRight | kLHintsExpandX | kLHintsExpandY,0,2,2,2);
   fMainFrame->AddFrame(fV2, fL3);

   fCanvasWindow = new TGCanvas(fV2, 10, 10, kSunkenFrame | kDoubleBorder);
   fCanvasContainer = new TGLRenderArea(fCanvasWindow->GetViewPort()->GetId(), fCanvasWindow->GetViewPort());
   fRender = new TGLRender;

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
   delete fViewMenu;
   delete fHelpMenu;
   delete fMenuBar;
   delete fMenuBarLayout;
   delete fMenuBarHelpLayout;
   delete fMenuBarItemLayout;
   delete fArcBall;
   delete fRender;
   delete fCanvasContainer;
   delete fCanvasWindow;
   delete fCanvasLayout;
   delete fV1;
   delete fV2;
   delete fMainFrame;
   delete fL1;
   delete fL2;
   delete fL3;
   delete fL4;
   delete fContextMenu;
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
   // Handle mouse button events.
   // Buttons 4 and 5 are from the mouse scroll wheel
   if (event->fType == kButtonPress) {
      if (event->fCode == kButton4) {
         // zoom out
         fZoom[fConf] *= 1.2;
         fCamera[fConf]->Zoom(fZoom[fConf]);
         DrawObjects();
         return kTRUE;
      }
      if (event->fCode == kButton5) {
         // zoom in
         fRender->SetNeedFrustum();
         fZoom[fConf] /= 1.2;
         fCamera[fConf]->Zoom(fZoom[fConf]);
         DrawObjects();
         return kTRUE;
      }
   }
   if (event->fType == kButtonPress) {
      if(event->fCode == kButton1 && fConf == kPERSP) {
         TPoint pnt(event->fX, event->fY);
         fArcBall->Click(pnt);
         fPressed = kTRUE;
         fAction = kRotating;
      } else {
         if ((fSelectedObj = TestSelection(event))) {
            fColorEditor->SetRGBA(fSelectedObj->GetColor());
            fGeomEditor->SetCenter(fSelectedObj->GetBBox()->GetData());
            if (event->fCode == kButton2) {
               fPressed = kTRUE;
               fLastPos.fX = event->fX;
               fLastPos.fY = event->fY;
               fAction = kPicking;
            } else if (TObject *ro = fSelectedObj->GetRealObject()){
               if (!fContextMenu) fContextMenu = new TContextMenu("glcm", "glcm");
               fContextMenu->Popup(event->fXRoot, event->fYRoot, ro);
            }
         } else {
            fColorEditor->Disable();
            fGeomEditor->Disable();
         }
      }
   } else if (event->fType == kButtonRelease) {
      if (event->fCode == kButton2) {
         DrawObjects();
         if (fSelectedObj)fGeomEditor->SetCenter(fSelectedObj->GetBBox()->GetData());
         fAction = kNoAction;
      }
      fPressed = kFALSE;
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
      fRender->SetNeedFrustum();
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
   case kKey_Up:
      MoveCenter(kKey_Up);
      break;
   case kKey_Down:
      MoveCenter(kKey_Down);
      break;
   case kKey_Left:
      MoveCenter(kKey_Left);
      break;
   case kKey_Right:
      MoveCenter(kKey_Right);
      break;
   case kKey_S:
   case kKey_s:
      fRender->GetStat();
      break;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TViewerOpenGL::HandleContainerMotion(Event_t *event)
{
   if (fPressed) {
      if (fAction == kRotating) {
         TPoint pnt(event->fX, event->fY);
         fArcBall->Drag(pnt);
      } else if (fAction == kPicking) {
         TGLWindow *glWin = fCanvasContainer->GetGLWindow();
         Double_t xshift = (event->fX - fLastPos.fX) / Double_t(glWin->GetWidth());
         Double_t yshift = (event->fY - fLastPos.fY) / Double_t(glWin->GetHeight());
         xshift *= fViewVolume[0] * 1.9 * fZoom[fConf];
         yshift *= fViewVolume[1] * 1.9 * fZoom[fConf];

         if (fConf != kPERSP) {
            switch (fConf) {
            case kXOY:
               fSelectedObj->Shift(xshift, -yshift, 0.);
               break;
            case kXOZ:
               fSelectedObj->Shift(-yshift, 0., xshift);
               break;
            case kYOZ:
               fSelectedObj->Shift(0., -yshift, xshift);
               break;
	         default:
	            break;
            }
         } else {
            const Double_t *rotM = fArcBall->GetRotMatrix();
            Double_t matrix[3][4] = {{rotM[0], -rotM[8], rotM[4], xshift},
                                     {rotM[1], -rotM[9], rotM[5], -yshift},
                                     {rotM[2], -rotM[10], rotM[6], 0.}};

            TToySolver tr(*matrix);
            Double_t shift[3] = {0.};
            tr.GetSolution(shift);
            fSelectedObj->Shift(shift[0], shift[1], shift[2]);
         }

         fLastPos.fX = event->fX;
         fLastPos.fY = event->fY;
      }

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
   //Calculate light sources positions and "bulb" radius
   Double_t xdiff = fRangeX.second - fRangeX.first;
   Double_t ydiff = fRangeY.second - fRangeY.first;
   Double_t zdiff = fRangeZ.second - fRangeZ.first;

   fXc = fRangeX.first + xdiff / 2;
   fYc = fRangeY.first + ydiff / 2;
   fZc = fRangeZ.first + zdiff / 2;
   fRender->SetAxes(fRangeX, fRangeY, fRangeZ);

   Float_t pos1[] = {0., fRad + fYc, -fRad - fZc, 1.f};
   Float_t pos2[] = {fRad + fXc, 0.f, -fRad - fZc, 1.f};
   Float_t pos3[] = {0.f, -fRad - fYc, -fRad - fZc, 1.f};
   Float_t pos4[] = {-fRad - fXc, 0.f, -fRad - fZc, 1.f};
   Float_t pos5[] = {0.f, 0.f, 0.f, 1.f};
   
   Float_t whiteCol[] = {.7f, .7f, .7f, 1.f};

   MakeCurrent();
   gVirtualGL->GLLight(kLIGHT4, kPOSITION, pos1);
   gVirtualGL->GLLight(kLIGHT4, kDIFFUSE, whiteCol);
   gVirtualGL->GLLight(kLIGHT1, kPOSITION, pos2);
   gVirtualGL->GLLight(kLIGHT1, kDIFFUSE, whiteCol);
   gVirtualGL->GLLight(kLIGHT2, kPOSITION, pos3);
   gVirtualGL->GLLight(kLIGHT2, kDIFFUSE, whiteCol);
   gVirtualGL->GLLight(kLIGHT3, kPOSITION, pos4);
   gVirtualGL->GLLight(kLIGHT3, kDIFFUSE, whiteCol);
   gVirtualGL->GLLight(kLIGHT0, kPOSITION, pos5);
   gVirtualGL->GLLight(kLIGHT0, kDIFFUSE, whiteCol);
   
   if (fLightMask & 1) gVirtualGL->EnableGL(kLIGHT4);
   if (fLightMask & 2) gVirtualGL->EnableGL(kLIGHT1);
   if (fLightMask & 4) gVirtualGL->EnableGL(kLIGHT2);
   if (fLightMask & 8) gVirtualGL->EnableGL(kLIGHT3);
   if (fLightMask & 16) gVirtualGL->EnableGL(kLIGHT0);

   MoveResize(fgInitX, fgInitY, fgInitW, fgInitH);
   SetWMPosition(fgInitX, fgInitY);
   CreateCameras();
   fRender->SetActive(kPERSP);

   DrawObjects();
}

//______________________________________________________________________________
void TViewerOpenGL::UpdateScene(Option_t *)
{
   TBuffer3D * buff = fPad->GetBuffer3D();

   if (buff->fOption == buff->kOGL) {
      ++fNbShapes;
      TGLSceneObject *addObj = 0;

      if (buff->fColor <= 1) buff->fColor = 42; //temporary

      Float_t colorRGB[3] = {0.f};
      TColor *rcol = gROOT->GetColor(buff->fColor);

      if (rcol) {
         rcol->GetRGB(colorRGB[0], colorRGB[1], colorRGB[2]);
      }

      switch (buff->fType) {
      case TBuffer3D::kLINE:
         addObj = new TGLPolyLine(*buff, colorRGB, fNbShapes, buff->fId);
         break;
      case TBuffer3D::kMARKER:
         addObj = new TGLPolyMarker(*buff, colorRGB, fNbShapes, buff->fId);
         break;
      case TBuffer3D::kSPHE:
         addObj = new TGLSphere(*buff, colorRGB, fNbShapes, buff->fId);
         break;
      case TBuffer3D::kTUBE:
         addObj = new TGLTube(*buff, colorRGB, fNbShapes, buff->fId);
         break;
      default:
         addObj = new TGLFaceSet(*buff, colorRGB, fNbShapes, buff->fId);
         break;
      }

      UpdateRange(addObj->GetBBox());
      fRender->AddNewObject(addObj);
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
   TTimer::SingleShot(50, IsA()->GetName(), this, "ReallyDelete()");
}

//______________________________________________________________________________
void TViewerOpenGL::DrawObjects()const
{
   MakeCurrent();
   gVirtualGL->NewMVGL();
   gVirtualGL->TraverseGraph((TGLRender *)fRender);
   SwapBuffers();
}

//______________________________________________________________________________
void TViewerOpenGL::UpdateRange(const TGLSelection *box)
{
   const Double_t *X = box->GetRangeX();
   const Double_t *Y = box->GetRangeY();
   const Double_t *Z = box->GetRangeZ();

   if (!fRender->GetSize()) {
      fRangeX.first = X[0], fRangeX.second = X[1];
      fRangeY.first = Y[0], fRangeY.second = Y[1];
      fRangeZ.first = Z[0], fRangeZ.second = Z[1];
      return;
   }

   if (fRangeX.first > X[0])
      fRangeX.first = X[0];
   if (fRangeX.second < X[1])
      fRangeX.second = X[1];
   if (fRangeY.first > Y[0])
      fRangeY.first = Y[0];
   if (fRangeY.second < Y[1])
      fRangeY.second = Y[1];
   if (fRangeZ.first > Z[0])
      fRangeZ.first = Z[1];
   if (fRangeZ.second < Z[0])
      fRangeZ.second = Z[1];
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
         case kGLXOY:
            if (fConf != kXOY) {
            //set active camera
               fConf = kXOY;
               fRender->SetActive(fConf);
               DrawObjects();
            }
            break;
         case kGLXOZ:
            if (fConf != kXOZ) {
            //set active camera
               fConf = kXOZ;
               fRender->SetActive(fConf);
               DrawObjects();
            }
            break;
         case kGLYOZ:
            if (fConf != kYOZ) {
            //set active camera
               fConf = kYOZ;
               fRender->SetActive(fConf);
               DrawObjects();
            }
            break;
         case kGLPersp:
            if (fConf != kPERSP) {
            //set active camera
               fConf = kPERSP;
               fRender->SetActive(fConf);
               DrawObjects();
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
   TGLSceneObject *obj = gVirtualGL->SelectObject(fRender, event->fX, event->fY, fConf);
   SwapBuffers();

   return obj;
}

//______________________________________________________________________________
void TViewerOpenGL::CalculateViewports()
{
   fActiveViewport[0] = 0;
   fActiveViewport[1] = 0;
   fActiveViewport[2] = fCanvasWindow->GetWidth();
   fActiveViewport[3] = fCanvasWindow->GetHeight();
}

//______________________________________________________________________________
void TViewerOpenGL::CalculateViewvolumes()
{
   if (fRender->GetSize()) {
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
      fRad = max * 1.7;
   }
}

//______________________________________________________________________________
void TViewerOpenGL::CreateCameras()
{
   if (!fRender->GetSize())
      return;

   TGLSimpleTransform trXOY(gRotMatrixXOY, fRad, &fXc, &fYc, &fZc);
   TGLSimpleTransform trXOZ(gRotMatrixXOZ, fRad, &fXc, &fYc, &fZc);
   TGLSimpleTransform trYOZ(gRotMatrixYOZ, fRad, &fXc, &fYc, &fZc);
   TGLSimpleTransform trPersp(fArcBall->GetRotMatrix(), fRad, &fXc, &fYc, &fZc);

   fCamera[kXOY]   = new TGLOrthoCamera(fViewVolume, fActiveViewport, trXOY);
   fCamera[kXOZ]   = new TGLOrthoCamera(fViewVolume, fActiveViewport, trXOZ);
   fCamera[kYOZ]   = new TGLOrthoCamera(fViewVolume, fActiveViewport, trYOZ);
   fCamera[kPERSP] = new TGLPerspectiveCamera(fViewVolume, fActiveViewport, trPersp);

   fRender->AddNewCamera(fCamera[kXOY]);
   fRender->AddNewCamera(fCamera[kXOZ]);
   fRender->AddNewCamera(fCamera[kYOZ]);
   fRender->AddNewCamera(fCamera[kPERSP]);
}

//______________________________________________________________________________
void TViewerOpenGL::ModifyScene(Int_t wid)
{
   MakeCurrent();
   switch (wid) {
   case kTBa:
      fSelectedObj->SetColor(fColorEditor->GetRGBA());
      break;
   case kTBaf:
      fRender->SetFamilyColor(fColorEditor->GetRGBA());
      break;
   case kTBa1:
      {
         Double_t c[3] = {0.};
         Double_t s[] = {1., 1., 1.};
         fGeomEditor->GetObjectData(c, s);
         fSelectedObj->Stretch(s[0], s[1], s[2]);
         fSelectedObj->Shift(c[0], c[1], c[2]);
      }
      break;
   case kTBda:
      fRender->ResetAxes();
      break;
   case kTBcp:
      if (fRender->ResetPlane()) gVirtualGL->EnableGL(kCLIP_PLANE0);
      else gVirtualGL->DisableGL(kCLIP_PLANE0);
   case kTBcpm:
      {
         Double_t eqn[4] = {0.};
         fSceneEditor->GetPlaneEqn(eqn);
         fRender->SetPlane(eqn);
      }
   case kTBTop:
      if ((fLightMask ^= 1) & 1) gVirtualGL->EnableGL(kLIGHT4);
      else gVirtualGL->DisableGL(kLIGHT4);
      break;
   case kTBRight:
      if ((fLightMask ^= 2) & 2) gVirtualGL->EnableGL(kLIGHT1);
      else gVirtualGL->DisableGL(kLIGHT1);
      break;
   case kTBBottom:
      if ((fLightMask ^= 4) & 4) gVirtualGL->EnableGL(kLIGHT2);
      else gVirtualGL->DisableGL(kLIGHT2);
      break;
   case kTBLeft:
      if ((fLightMask ^= 8) & 8) gVirtualGL->EnableGL(kLIGHT3);
      else gVirtualGL->DisableGL(kLIGHT3);
      break;
   case kTBFront:
      if ((fLightMask ^= 16) & 16) gVirtualGL->EnableGL(kLIGHT0);
      else gVirtualGL->DisableGL(kLIGHT0);
      break;
   }

   DrawObjects();
}

//______________________________________________________________________________
void TViewerOpenGL::MoveCenter(Int_t key)
{
   fRender->SetNeedFrustum();
   Double_t shift[3] = {0.};
   Double_t steps[2] = {fViewVolume[0] * fZoom[0] / 40, fViewVolume[0] * fZoom[0] / 40};

   switch (key) {
   case kKey_Left:
      shift[0] = steps[0];
      break;
   case kKey_Right:
      shift[0] = -steps[0];
      break;
   case kKey_Up:
      shift[1] = -steps[1];
      break;
   case kKey_Down:
      shift[1] = steps[1];
      break;
   }

   const Double_t *rotM = fArcBall->GetRotMatrix();
   Double_t matrix[3][4] = {{rotM[0], -rotM[8], rotM[4], shift[0]},
                            {rotM[1], -rotM[9], rotM[5], shift[1]},
                            {rotM[2], -rotM[10], rotM[6], 0.}};

   TToySolver tr(*matrix);
   tr.GetSolution(shift);
   fXc += shift[0];
   fYc += shift[1];
   fZc += shift[2];

   DrawObjects();
}
