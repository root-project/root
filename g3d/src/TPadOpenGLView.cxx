// @(#)root/g3d:$Name:  $:$Id: TPadOpenGLView.cxx,v 1.2 2000/06/05 07:28:47 brun Exp $
// Author: Valery Fine(fine@vxcern.cern.ch)   08/05/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPadOpenGLView                                                       //
//                                                                      //
// TPadOpenGLView is a window in which an OpenGL representation of a    //
// pad is viewed.                                                       //
//                                                                      //
// Seven different OpenGL lists are used to draw a 3D object:           //
//                                                                      //
// fGLList + kLightOpt   - defines the "light" options to build thr GL  //
//                         light model                                  //
// fGLList + kColorOpt   - defines the "color" options                  //
// fGLList + kScene      - defines the common object for the resize and //
//                         re-paint operations                          //
// fGLList + kProject    - defines the "Projection" transformation.     //
//                         It is affected by the rotate operations.     //
// fGLList + kView       - defines the "Viewing" transformation         //
// fGLList + kUpdateView - defines the corrections for the "Viewing"    //
//                         transformation                               //
// fGLList + kModel      - defines the "Modeling" transformations       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualGL.h"
#include "TPadOpenGLView.h"
#include "TVirtualPad.h"
#include "TView.h"
#include "TGLViewerImp.h"
#include "TGeometry.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
#include "TNode.h"
#include "TStopwatch.h"
#include "TSystem.h"

// ClassImp(TPadOpenGLView)

//____________________________________________________________________________
TPadOpenGLView::TPadOpenGLView(TVirtualPad *pad) : TPadView3D(pad)
{
   fGLViewerImp = gVirtualGL->CreateGLViewerImp(this,pad->GetTitle(), pad->GetWw(), pad->GetWh());
   fGLViewerImp->CreateContext();
   fGLViewerImp->MakeCurrent();

   Int_t i;
   for (i=0;i<3;i++) {
      fExtraAngles[i] = 0;
      fExtraTranslate[i] = 0;
      fAnglFactor[i] = 1;
   }

//    fExtraAngles[1] = -90;
    fScale = 1.0;

    fSpeedMove   = 0.02;
    fResetView   = kTRUE;
    fPerspective = kFALSE;
    fStereoFlag  = kFALSE;

    SetPad(pad);

    MapOpenGL();
//*-* Set the indentity matrix as a extra rotations
    for (i=1;i<15;i++)  fExtraRotMatrix[i] = 0;
    for (i=0;i<16;i+=5) fExtraRotMatrix[i] = 1.0;
#if 0
//*-*  Date protection
    TDatime lock;
    if (lock.GetDate() > 970530) ::exit(-1);
#endif
}

//____________________________________________________________________________
TPadOpenGLView::~TPadOpenGLView()
{
    if (!fParent) {
       delete fGLViewerImp;
       return;
    }
//*-*  Delete OpenGL lists;
   if (fGLViewerImp) {
#ifdef STEREO_GL
      if (fStereoFlag) {
         const char *turnStereo = gSystem->Getenv("OffRootStereo");
         if (turnStereo) gSystem->Exec(turnStereo);
         fStereoFlag = kFALSE;
      }
#endif
      if (fGLList) {
         // causes crash if more than one viewer is open at a time
         //gVirtualGL->DeleteGLLists(fGLList,kGLListSize);
         fGLList =0;
      }
      delete fGLViewerImp;
      fGLViewerImp = 0;
   }
}

//____________________________________________________________________________
void TPadOpenGLView::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{

    if (GetPad() && !GetPad()->IsEditable()) return;

    switch ((EEventType)event)
    {
    case kButton1Down:
//*-* Save the start coordinates
        fMouseInit=kTRUE;
        fMouseX = px;
        fMouseY = py;
        break;
    case kKeyPress :
        {
            Char_t chCharCode = (Char_t) px;
//            if (chCharCode != '+' || chCharCode != '-') chCharCode |= 0x20;
            MoveModelView(chCharCode, py);
            break;
        }
    case kButton1Up:
        if (!fMouseInit) break;
        fMouseInit=kFALSE;
        RotateView(px,py);
        fMouseX = px;
        fMouseY = py;
        PaintScene();
        break;
    case kButton1Motion:
        if (!fMouseInit) break;
        RotateView(px,py);
        fMouseX = px;
        fMouseY = py;
        PaintScene();
        break;
    default:
        break;
    };
}

//____________________________________________________________________________
void TPadOpenGLView::MapOpenGL()
{
//*-* Set the default attributes
   gVirtualGL->EnableGL(kDEPTH_TEST);
   gVirtualGL->EnableGL(kCULL_FACE);

   gVirtualGL->PolygonGLMode(kFRONT,kFILL);
   gVirtualGL->SetRootLight(kTRUE);
   gVirtualGL->FrontGLFace(kCCW);

//*-* Create a main GL list to draw the scene

   fGLList = gVirtualGL->CreateGLLists(kGLListSize);
   fGLLastList = fGLList+kGLListSize-1;


   gVirtualGL->NewGLList(fGLList+kScene,kCOMPILE);
   {
       for (UInt_t i=fGLList+kProject; i<=fGLLastList;i++) gVirtualGL->RunGLList(i);
   }
   gVirtualGL->EndGLList();
   fGLViewerImp->SetStatusText("working...",4,TGLViewerImp::kStatusPopOut);
}
//______________________________________________________________________________
void TPadOpenGLView::PushMatrix()
{
    gVirtualGL->PushGLMatrix();
}
//______________________________________________________________________________
void TPadOpenGLView::PopMatrix()
{
    gVirtualGL->PopGLMatrix();
}

//______________________________________________________________________________
void TPadOpenGLView::SetAttNode(TNode *node, Option_t *opt)
{
  if (node) SetLineAttr(node->GetLineColor(),node->GetLineWidth(),opt);
}
//______________________________________________________________________________
void TPadOpenGLView::SetLineAttr(Color_t color, Int_t width, Option_t *)
{
    gVirtualGL->SetLineAttr(color,width);
}
//______________________________________________________________________________
void TPadOpenGLView::UpdateNodeMatrix(TNode *node,Option_t *)
{
    if (node)
       UpdatePosition(node->GetX(),node->GetY(),node->GetZ(),node->GetMatrix());
}
//______________________________________________________________________________
void TPadOpenGLView::UpdatePosition(Double_t x,Double_t y,Double_t z,TRotMatrix *matrix,Option_t *)
{
    if (!gGeometry) return;
    Float_t bomb = gGeometry->GetBomb();

//*-*  Create a translation matrix

    Double_t translate[3];
    translate[0] = bomb*x;
    translate[1] = bomb*y;
    translate[2] = bomb*z;

//*-*  Create a rotation/reflection matrix

    Double_t rotate[16] = {1,0,0,0,
                           0,1,0,0,
                           0,0,1,0,
                           0,0,0,1
                          };
    if (matrix)
       matrix->GetGLMatrix(rotate);

//*-* Update the OpenGL matrix

    gVirtualGL->UpdateMatrix(translate,rotate,gGeometry->GetCurrentReflection());
}

#if 0
//______________________________________________________________________________
void TPadOpenGLView::PaintBrik(TShape *shape,Option_t *opt)
{
    gVirtualGL->PaintBrik(vertex);
}
//______________________________________________________________________________
void TPadOpenGLView::PaintCone(TShape *shape,Option_t *opt)
{
    gVirtualGL->PaintCone(vertex,-(GetNumberOfDivisions()+1),fNz);
}
#endif
//______________________________________________________________________________
void TPadOpenGLView::PaintPolyLine(TPolyLine3D *line,Option_t *opt)
{
    gVirtualGL->SetGLColorIndex(line->GetLineColor());
    gVirtualGL->SetGLLineWidth((Float_t)(line->GetLineWidth()));
    gVirtualGL->PaintPolyLine(line->GetN(), line->GetP(), opt);
}

//______________________________________________________________________________
void TPadOpenGLView::PaintPolyMarker(TPolyMarker3D *marker,Option_t *opt)
{
    gVirtualGL->SetGLColorIndex(marker->GetMarkerColor());
    gVirtualGL->SetGLPointSize((Float_t)(marker->GetMarkerSize()));
    gVirtualGL->PaintGLPoints(marker->GetN(), marker->GetP(), opt);
}
//______________________________________________________________________________
void TPadOpenGLView::PaintPoints3D(const TPoints3DABC *points,Option_t *opt)
{
  //
  // option = "L"        - connect all points with straight lines
  //          "P"        - draw the 3d points at each coordinate (by default)
  //          "LP"       - draw the 3D points and connect them with
  //                       straight lines

  if (points) gVirtualGL->PaintGLPointsObject(points,opt);
}

//______________________________________________________________________________
void TPadOpenGLView::PaintBeginModel(Option_t *)
{
//*-*   Apply OpenGL lists
    if (!fGLList) return;  // There is nothing to render

   fGLViewerImp->SetStatusText("working...",4,TGLViewerImp::kStatusPopOut);
   fGLViewerImp->MakeCurrent();

//    MakeCurrent();
   if (fResetView)
   {
       TView *view = fParent->GetView();
       if (view) {
           Float_t min[3],max[3];
           view->GetRange(min,max);

//*-*  sqrt(3) = 1.73 factor to fit all possible position of the rotated shapes

           Float_t bomb = 1.73;
           if (gGeometry) bomb = 1.73*gGeometry->GetBomb();
//           Float_t bomb = gGeometry->GetBomb();
// sqrt((max[i]-min[i])**2)/(max[i]-min[i])**2
           int i;
//*-* Set the indentity matrix as a extra rotations
           for (i=1;i<15;i++)  fExtraRotMatrix[i] = 0;
           for (i=0;i<16;i+=5) fExtraRotMatrix[i] = 1.0;
           for (i=0;i<3;i++) {
               fStep[i] = fSpeedMove*(max[0] - min[0]);
               fTranslate[i] =-(max[i]+min[i])/2;
               fExtraTranslate[i] = 0;
               fExtraAngles[i] =   0;
               fViewBoxMin[i] = (Double_t)bomb*(min[i]+fTranslate[i]);
               fViewBoxMax[i] = (Double_t)bomb*(max[i]+fTranslate[i]);
           }
// Move object backwards to allow the OpenGL perspective view
                   if (max[2] > 0) {
                           fExtraTranslate[2] = -fViewBoxMax[2] - (Double_t)bomb*TMath::Abs(max[0]-min[0]);
               fViewBoxMin[2] += fExtraTranslate[2];
               fViewBoxMax[2] += fExtraTranslate[2];
                   }
           fAngles[0] = view->GetLatitude();
           fAngles[1] = view->GetLongitude();
           fAngles[2] = view->GetPsi();
           // Create Object view
           UpdateObjectView();
       }
   }
   else
       fResetView = kTRUE;  // We can lock ResetView only at once

   UpdateProjectView();  //*-* Update/Create  kProject GL list.
   UpdateModelView();
   gVirtualGL->NewGLList(fGLList+kModel,kCOMPILE);  // Create "Model View" list
}
//______________________________________________________________________________
void TPadOpenGLView::PaintEnd(Option_t *)
{
   if (!fGLList) return;  // There is nothing to render
//*-*  Close the "model" view and start the projection one
    gVirtualGL->EndGLList();
}

//______________________________________________________________________________
void TPadOpenGLView::PaintScene(Option_t *)
{
    if (!(fGLList && fGLViewerImp) ) return;  // There is nothing to render
    fGLViewerImp->SetStatusText("working...",4,TGLViewerImp::kStatusPopOut);
    fGLViewerImp->Update();
}


#if 0
//______________________________________________________________________________
void TPadOpenGLView::HandleInput(Int_t button, Int_t x, Int_t y)
{
    switch ((EEventType)button)
    {
    case kButton1Down:
//*-* Save the start coordinates
        fMouseInit=kTRUE;
        fMouseX = x;
        fMouseY = y;
        break;
    case kButton1Up:
        if (!fMouseInit) break;
        fMouseInit=kFALSE;
        ExecuteEvent(button,x,y);
        break;
    case kButton1Motion:
        if (!fMouseInit) break;
        ExecuteEvent(button,x,y);
        break;
    case kKeyPress :
        ExecuteEvent(button,x,y);
        break;
    default:
        return;
    }
}
#endif

//______________________________________________________________________________
void TPadOpenGLView::MoveModelView(const Char_t option, Int_t count)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-* option = 'u' //*-*  Move down
//*-*          'i' //*-*  Move up
//*-*          'h' //*-*  Move right
//*-*          'l' //*-*  Move left
//*-*          'j' //*-*  Move backward
//*-*          'k' //*-*  Move foreward
//*-*
//*-*          '+' //*-*  Increase speed to move
//*-*          '-' //*-*  Decrease speed to move
//*-*
//*-*          'n' //*-*  turn "SMOOTH" color mode on
//*-*          'm' //*-*  turn "SMOOTH" color mode off
//*-*
//*-*          't' //*-*  toggle Light model
//*-*
//*-*          'p' //*-*  Perspective/Orthographic projection
//*-*          'r' //*-*  Hidden surface mode
//*-*          'w' //*-*  wireframe mode
//*-*          'c' //*-*  cull-face mode
//*-*
//*-*          's' //*-*  increase  scale factor (clip cube borders)
//*-*          'a' //*-*  decrease  scale factor (clip cube borders)
//*-*
//*-*          'x' //*-*
//*-*          'y' //*-*  rotate object along x,y,z axis
//*-*          'z' //*-*
//*-*
//*-*          'v' //*-*  toggle the stereo mode
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    if (count <= 0) count = 1;

    Int_t done = 0;

    Int_t direction = 1;       //*-* = 1 means the Positive direction has been entered
    const Char_t *fwd = "hij";  //*-* "keys" for the "positive" directions
    const Char_t *bck = "luk";  //*-* "keys" for the "negative" directions

    const Char_t *sch = fwd;
    Char_t *c = (char*)strchr(sch,option|0x20);  //*-* Check the "positive" direction"
    if (!c)
    {
        direction = -direction;
        sch = bck;
        c = (char*)strchr(sch,option|0x20);     //*-* Check the "negative" direction"
    }
    if (c)
    {
        Int_t indx = c - sch;
        fExtraTranslate[indx] += direction*count*fStep[indx];
        done = 2;
    }
    else
    {
        direction = -1;

        switch (option) {
        case 'x':
            direction = -direction;
        case 'X':
//        fExtraAngles[0] += fAxStep/rect.bottom; += count*fZStep;
          fExtraAngles[0] += direction*fAnglFactor[0];
            done = 4;
            break;
        case 'z':
            direction = -direction;
        case 'Z':
       //        fExtraAngles[1] += fAyStep/rect.bottom; += count*fZStep;
            fExtraAngles[1] += direction*fAnglFactor[1];
            done = 4;
            break;
        case 'y':
            direction = -direction;
        case 'Y':
//       fExtraAngles[3] += fAzStep/rect.bottom; += count*fZStep;
            fExtraAngles[2] += direction*fAnglFactor[2];
            done = 4;
            break;
        case '+':
            {
                fSpeedMove *= 2;
                for (Int_t i=0;i<3;i++) fStep[i] *= 2;
                break;
            }
        case '-':
            {
                fSpeedMove /= 2;
                for (Int_t i=0;i<3;i++) fStep[i] /= 2;
                break;
            }
        case 's':
        case 'S':
            if (fScale < 1.0) fScale = 1.0;
            fScale *= 1.1;
            done = 3;
            break;
        case 'a':
        case 'A':
            if (fScale > 1.0) fScale = 1.0;
            fScale *= 0.9;
            done = 3;
            break;
        case 'c':
        case 'C':
            gVirtualGL->EnableGL(kCULL_FACE);
            done = 1;
            break;
        case 'r':
        case 'R':
            gVirtualGL->PolygonGLMode(kFRONT,kFILL);
  //          gVirtualGL->PolygonGLMode(kFRONT_AND_BACK,kFILL);
            done = 1;
            break;
        case 'w':
        case 'W':
            gVirtualGL->DisableGL(kCULL_FACE);
            gVirtualGL->PolygonGLMode(kFRONT_AND_BACK,kLINE);
            done = 1;
            break;
        case 'n':    //*-*      turn "SMOOTH" shade model on
        case 'N':    //*-*      turn "SMOOTH" shade model on
            gVirtualGL->ShadeGLModel(kSMOOTH);
            done = 1;
            break;
        case 'm':    //*-*      turn "FLAT" color shade off
        case 'M':    //*-*      turn "FLAT" color shade off
            gVirtualGL->ShadeGLModel(kFLAT);
            done = 1;
            break;
        case 'v':    //*-*      toggle the stereo mode
        case 'V':    //*-*      toggle the stereo mode
#ifdef STEREO_GL
            fStereoFlag = !fStereoFlag;
            if (fStereoFlag) {
               const char *turnStereo = gSystem->Getenv("TurnRootStereo");
               if (turnStereo) gSystem->Exec(turnStereo);
            } else {
               const char *turnStereo = gSystem->Getenv("OffRootStereo");
               if (turnStereo) gSystem->Exec(turnStereo);
            }
            done = 1;
#endif
            break;
        case 't':    //*-*      toggle the light model
        case 'T':    //*-*      toggle the light model
            {
                Bool_t light = gVirtualGL->GetRootLight();
                gVirtualGL->SetRootLight(!light);
                if (light)
                    fGLViewerImp->SetStatusText("Real (\" T \")",3);
                else
                    fGLViewerImp->SetStatusText("Pseudo (\" T \")",3);
            }

            if (fParent)
            {
                fResetView = kFALSE;  // To keep the current orientations
                fScale = 1;
                fParent->Modified();
                fParent->Update();
            }
            done = 0;
            break;
        case 'p':  //*-* toggle between perspecitve and  and orthographics projections
        case 'P':  //*-* toggle between perspecitve and  and orthographics projections
            fPerspective = !fPerspective;
            done = 3;
            break;
        default:
            break;
        }
    }

    if (done)
        fGLViewerImp->SetStatusText("working...",4,TGLViewerImp::kStatusPopOut);

    switch (done) {
    case 4:
        UpdateObjectView();
        break;
    case 3:
        UpdateProjectView();
        break;
    case 2:
        UpdateModelView();
        break;
//    case 1:
//        PaintScene();
    default:
        break;
    };
    if (done)
        PaintScene();
}

//______________________________________________________________________________
void TPadOpenGLView::MoveModelView(const char *cmd, int mlsecc)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*  Emulate the "chain" of the "KeyPress" events
//*-*     Input:
//*-*     char *cmd  - any "ASCII" character defined in
//*-*                  TPadOpenGLView::MoveModelView(Char_t option, Int_t count)
//*0*
//*-*     int mlsecc - The total time desired to perform full "chain".
//*-*                  It is assumed that a single step is performed too quickly
//*-*                  and must be done slowly
//*-*              = 0;  No slowdown. It is painting as fast as the
//*-*                    present CPU allows
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
  Int_t lcmd = strlen(cmd);
  if (lcmd)  {
    Int_t i = 0;
            // Estimate the time for a single step
    Int_t quant = 0;
    TStopwatch *stopwatch=0;
    if (mlsecc > 0) {
      quant = Int_t(Double_t(mlsecc)/lcmd);
      if (quant > 0) stopwatch = new TStopwatch;
      else printf(" Display time too small !\n");
    }
           // Start stopwatch
    if (stopwatch)  stopwatch->Start();
    for (i=0;i<lcmd;i++) {
       MoveModelView(cmd[i]);
       // Estimate sleep time
      if (stopwatch) {
        Int_t sleeptime = Int_t(i*quant-stopwatch->GetRealTime()*1000);
        if (sleeptime > 0) gSystem->Sleep(sleeptime);
      }
    }
  }
}
//______________________________________________________________________________
void TPadOpenGLView::Paint(Option_t *)
{
//*-*
//*-*  This method is designed to be called from the TGLViewerImp object to paint
//*-*  the 3D scene  onto the "GL" screen
//*-*
    Int_t color;
    if (fParent)
    {
        color = fParent->GetFillColor();
        gVirtualGL->ClearColor(color);
    }
//          gVirtualGL->ClearGL(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Int_t stereo = 0;
#ifdef STEREO_GL
    if (fStereoFlag)   stereo = +1;
#endif
    gVirtualGL->ClearGL(UInt_t(stereo));
    if (fGLList) {
       gVirtualGL->RunGLList(fGLList+kScene);
#ifdef STEREO_GL
       if (stereo) {
          stereo = -stereo;

          Double_t angles[3] = {0,2,0};
          gVirtualGL->AddRotation(fExtraRotMatrix,angles);
          UpdateModelView();

          gVirtualGL->ClearGL(UInt_t(stereo));
          gVirtualGL->RunGLList(fGLList+kScene);
          // Restore angle
          angles[1] = -5;
          gVirtualGL->AddRotation(fExtraRotMatrix,angles);
          UpdateModelView();
       }
#endif
    }
}

//______________________________________________________________________________
void TPadOpenGLView::RotateView(Int_t x, Int_t y)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    Int_t fx = fMouseX-x;
    Int_t fy = fMouseY-y;
    Double_t angles[3];

    angles[0] = -fy*fAnglFactor[0];
    angles[1] = -fx*fAnglFactor[1];
    angles[2] =  0;

    for (Int_t i=0;i<3;i++) {
                fScale = 1.0;
        if (TMath::Abs(angles[i]) >= 360.0) {
            if (angles[i] > 0)
                angles[i] -= 360.0;
            else
                angles[i] += 360.0;
        }
    }

    gVirtualGL->AddRotation(fExtraRotMatrix,angles);

    UpdateModelView();
    PaintScene();
}

//______________________________________________________________________________
void TPadOpenGLView::Size(Int_t width, Int_t height)
{
   if (height) fAnglFactor[0] = 360./height;
   if (width)  fAnglFactor[1] = 360./width;
   fAnglFactor[2] = fAnglFactor[1];
//   glViewport( 0, 0, (GLint) width, (GLint)  height );
}

//______________________________________________________________________________
void TPadOpenGLView::UpdateObjectView()
{
    Double_t angles[3];
    for (Int_t i=0;i<3;i++) angles[i] = fAngles[i] + fExtraAngles[i];
    gVirtualGL->NewGLList(fGLList+kView,kCOMPILE);
       gVirtualGL->NewModelView(angles,fTranslate);
    gVirtualGL->EndGLList();
}
//______________________________________________________________________________
void TPadOpenGLView::UpdateModelView()
{
    gVirtualGL->NewGLList(fGLList+kUpdateView,kCOMPILE);
    {
      Float_t s = 0;
      for (Int_t i=0;i<3;i++) s += TMath::Abs(fExtraTranslate[i]);


      if (s != 0.0) gVirtualGL->TranslateGL(fExtraTranslate[0],
                                           fExtraTranslate[1],
                                           fExtraTranslate[2]);
      gVirtualGL->MultGLMatrix(fExtraRotMatrix);
    }
    gVirtualGL->EndGLList();

}

//______________________________________________________________________________
void TPadOpenGLView::UpdateProjectView()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-UpdateProjectView()*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*    UpdateProjectView() - defines the "Projection" transformation. It is
//*-*                   affected with the rotate operations
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    if (TMath::Abs(fScale-1.0) > 0.1)
    {
        for (Int_t i=0;i<2;i++)
        {
            Double_t shift = (fScale-1)*TMath::Abs(fViewBoxMax[i]-fViewBoxMin[i]);
            fViewBoxMin[i] -= shift;
            fViewBoxMax[i] += shift;
        }
    }
    gVirtualGL->NewGLList(fGLList+kProject,kCOMPILE);
       gVirtualGL->NewProjectionView(fViewBoxMin,fViewBoxMax,fPerspective);
    gVirtualGL->EndGLList();

    if (fGLViewerImp)
    {
        if (fPerspective)
            fGLViewerImp->SetStatusText("Perspective (\" P \")",1);
        else
            fGLViewerImp->SetStatusText("Orthographic (\" P \")",1);
#if 0
        char boxview[128];
        Double_t dnear = TMath::Abs(fViewBoxMax[0]-fViewBoxMin[0]);
        Double_t dfar = 3*(dnear + TMath::Abs(fViewBoxMax[2]-fViewBoxMin[2]));

        sprintf(boxview,"Clip: Z:%.2fx%.2f; X:%.2fx%.2f; Y:%.2fx%.2f",
            dnear,dfar,
            fViewBoxMin[0],fViewBoxMax[0],
            fViewBoxMin[1],fViewBoxMax[1]
            );
        fGLViewerImp->SetStatusText(boxview,3);
#endif
    }
}
