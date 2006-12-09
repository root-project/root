// @(#)root/gl:$Name:  $:$Id: TGLOrthoCamera.cxx,v 1.15 2006/08/31 13:42:14 couet Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualGL.h"
#include "TMath.h"

#include "TGLOrthoCamera.h"
#include "TGLIncludes.h"
#include "TGLUtil.h"


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLOrthoCamera                                                       //
//                                                                      //
// Orthographic projection camera. Currently limited to three types     //
// defined at construction time - kXOY, kXOZ, kZOY - where this refers  //
// to the viewport plane axis - e.g. kXOY has X axis horizontal, Y      //
// vertical - i.e. looking down Z axis with Y vertical.                 //
//
// The plane types restriction could easily be removed to supported     //
// arbitary ortho projections along any axis/orientation with free      //
// rotations about them.                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLOrthoCamera)

UInt_t   TGLOrthoCamera::fgZoomDeltaSens = 500;

//______________________________________________________________________________
TGLOrthoCamera::TGLOrthoCamera(EType type) :
   fType(type), fZoomMin(0.01), fZoomDefault(0.78), fZoomMax(1000.0), 
   fVolume(TGLVertex3(-100.0, -100.0, -100.0), TGLVertex3(100.0, 100.0, 100.0)),
   fZoom(1.0), fTruck(0.0, 0.0, 0.0), fMatrix()
{
   // Construct orthographic camera with 'type' defining fixed view direction
   // & orientation (in world frame):
   //
   // kXOY : X Horz. / Y Vert (looking towards +Z, Y up)
   // kXOZ : X Horz. / Z Vert (looking towards +Y, Z up)
   // kZOY : Z Horz. / Y Vert (looking towards +X, Y up)
   // 
   Setup(TGLBoundingBox(TGLVertex3(-100,-100,-100), TGLVertex3(100,100,100)));
}

//______________________________________________________________________________
TGLOrthoCamera::TGLOrthoCamera() :
   fType(kXOY), fZoomMin(0.01), fZoomDefault(0.78), fZoomMax(1000.0), 
   fVolume(TGLVertex3(-100.0, -100.0, -100.0), TGLVertex3(100.0, 100.0, 100.0)),
   fZoom(1.0), fTruck(0.0, 0.0, 0.0), fShift(0.), fCenter(),
   fVpChanged(kFALSE)
{
   // Construct orthographic camera.
   fOrthoBox[0] = 1.;
   fOrthoBox[1] = 1.;
   fOrthoBox[2] = -1.;
   fOrthoBox[3] = 1.;
}

//______________________________________________________________________________
TGLOrthoCamera::~TGLOrthoCamera()
{
   // Destroy orthographic camera
}

//______________________________________________________________________________
void TGLOrthoCamera::Setup(const TGLBoundingBox & box, Bool_t reset)
{
   // Setup camera limits suitible to view the world volume defined by 'box'
   // and call Reset() to initialise camera.
   static const Double_t rotMatrixXOY[] = { 1.,  0.,  0.,  0.,
                                            0.,  1.,  0.,  0.,
                                            0.,  0.,  1.,  0.,
                                            0.,  0.,  0.,  1. };

   static const Double_t rotMatrixXOZ[] = { 1.,  0.,  0.,  0.,
                                            0.,  0., -1.,  0.,
                                            0.,  1.,  0.,  0.,
                                            0.,  0.,  0.,  1. };

   static const Double_t rotMatrixZOY[] = { 0.,  0.,  -1.,  0.,
                                            0.,  1.,  0.,  0.,
                                            1.,  0.,  0.,  0.,
                                            0.,  0.,  0.,  1. };
	
   switch (fType) {
		// Looking down Z axis, X horz, Y vert
      case (kXOY): {
         // X -> X
         // Y -> Y
         // Z -> Z
         fVolume = box;
         fMatrix.Set(rotMatrixXOY);
         break;
      }
		// Looking down Y axis, X horz, Z vert
      case (kXOZ): {
         // X -> X
         // Z -> Y
         // Y -> Z
         fVolume.SetAligned(TGLVertex3(box.XMin(), box.ZMin(), box.YMin()), 
                            TGLVertex3(box.XMax(), box.ZMax(), box.YMax()));
         fMatrix.Set(rotMatrixXOZ);
         break;
      }
		// Looking down X axis, Z horz, Y vert
      case (kZOY): {
         // Z -> X
         // Y -> Y
         // X -> Z
         fVolume.SetAligned(TGLVertex3(box.ZMin(), box.YMin(), box.XMin()), 
                            TGLVertex3(box.ZMax(), box.YMax(), box.XMax()));
         fMatrix.Set(rotMatrixZOY);
         break;
      }
   }
   if (reset)
      Reset();
}

//______________________________________________________________________________
void TGLOrthoCamera::Reset()
{
   // Reset the camera to defaults - trucking, zooming to reframe the world volume
   // established in Setup(). Note: limits defined in Setup() are not adjusted.
   fTruck.Set(0.0, 0.0, 0.0);
   fZoom   = fZoomDefault;
   fCacheDirty = kTRUE;
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Dolly(Int_t delta, Bool_t mod1, Bool_t mod2)
{
   // Dolly the camera - 'move camera along eye line, retaining lens focal length'.
   // Arguments are:
   //
   // 'delta' - mouse viewport delta (pixels) - +ive dolly in, -ive dolly out
   // 'mod1' / 'mod2' - sensitivity modifiers - see TGLCamera::AdjustAndClampVal()
   //
   // For an orthographic camera dollying and zooming are identical and both equate 
   // logically to a rescaling of the viewport limits - without center shift. 
   // There is no perspective foreshortening or lens 'focal length'.
   //
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.
   
   // TODO: Bring all mouse handling into camera classes - would simplify interface and
   // remove these non-generic cases.
   return Zoom(delta, mod1, mod2);
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Zoom (Int_t delta, Bool_t mod1, Bool_t mod2)
{
   // Zoom the camera - 'adjust lens focal length, retaining camera position'.
   // Arguments are:
   //
   // 'delta' - mouse viewport delta (pixels) - +ive zoom in, -ive zoom out
   // 'mod1' / 'mod2' - sensitivity modifiers - see TGLCamera::AdjustAndClampVal()
   //
   // For an orthographic camera dollying and zooming are identical and both equate 
   // logically to a rescaling of the viewport limits - without center shift. 
   // There is no perspective foreshortening or lens 'focal length'.
   //
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.
   
   // TODO: Bring all mouse handling into camera classes - would simplify interface and
   // remove these non-generic cases.
   if (AdjustAndClampVal(fZoom, fZoomMin, fZoomMax, -delta*2, fgZoomDeltaSens, mod1, mod2))
   {
      fCacheDirty = kTRUE;
      return kTRUE;
   }
   else
   {
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Truck(Int_t x, Int_t y, Int_t xDelta, Int_t yDelta)
{
   // Truck the camera - 'move camera parallel to film plane'. The film 
   // plane is defined by the EyePoint() / EyeDirection() pair. Define motion 
   // using center point (x/y) and delta (xDelta/yDelta) - the mouse motion. 
   //
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.
   //
   // Note: Trucking is often mistakenly refered to as 'pan' or 'panning'. 
   // Panning is swivelling the camera on it's own axis - the eye point.
   
   //TODO: Convert TGLRect so this not required
   GLint viewport[4] = { fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height() };
   TGLVertex3 start, end;
   // Trucking done at near clipping plane
   gluUnProject(x, y, 0.0, fModVM.CArr(), fProjM.CArr(), viewport, &start.X(), &start.Y(), &start.Z());
   gluUnProject(x + xDelta, y + yDelta, 0.0, fModVM.CArr(), fProjM.CArr(), viewport, &end.X(), &end.Y(), &end.Z());
   fTruck = fTruck + (end - start);
   fCacheDirty = kTRUE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLOrthoCamera::Rotate(Int_t /*xDelta*/, Int_t /*yDelta*/)
{
   // Rotate the camera - 'swivel round the view volume center'.
   // Ignored at present for orthographic cameras - have a fixed direction. 
   // Could let the user or external code create non-axis
   // ortho projects by adjusting H/V rotations in future.
   //
   // Returns kTRUE is redraw required (camera change), kFALSE otherwise.
   
   return kFALSE;
}

//______________________________________________________________________________
void TGLOrthoCamera::Apply(const TGLBoundingBox & /*box*/, const TGLRect * pickRect) const
{
   // Apply the camera to the current GL context, setting the viewport, projection
   // and modelview matricies. After this verticies etc can be directly entered
   // in the world frame. This also updates the cached frustum values, enabling
   // all the projection, overlap tests etc defined in TGLCamera to be used.
   // 
   // Arguments are:
   // 'box' - view volume box - ignored for ortho camera. Assumed to be same
   // as one passed to Setup().
   // 'pickRect' - optional picking rect. If non-null, restrict drawing to this
   // viewport rect.
   glViewport(fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height());

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   // Load up any picking rect
   if (pickRect) {
      //TODO: Convert TGLRect so this not required
      GLint viewport[4] = { fViewport.X(), fViewport.Y(), fViewport.Width(), fViewport.Height() };
      gluPickMatrix(pickRect->X(), pickRect->Y(),
                    pickRect->Width(), pickRect->Height(),
                    viewport);
   }

   if(fViewport.Width() == 0 || fViewport.Height() == 0) {
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      return;
   }
   
   TGLVector3 extents = fVolume.Extents();
   Double_t width = extents.X();
   Double_t height = extents.Y();
   Double_t halfRange;
   if (width > height) {
      halfRange = width / 2.0;
   } else {
      halfRange = height / 2.0;
   }
   halfRange /= fZoom;

   // For near/far clipping half depth give extra slack so clip objects/manips 
   // are visible 
   Double_t halfDepth = extents.Mag();
   const TGLVertex3 & center = fVolume.Center();

   glOrtho(center.X() - halfRange, 
           center.X() + halfRange, 
           center.Y() - halfRange, 
           center.Y() + halfRange, 
           center.Z() - halfDepth, 
           center.Z() + halfDepth);


   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

   glScaled(1.0 / fViewport.Aspect(), 1.0, 1.0); 	

   // Debug aid - show current volume
   /*glDisable(GL_LIGHTING);
   glColor3d(0.0, 0.0, 1.0);
   fVolume.Draw();
   glEnable(GL_LIGHTING);*/

   glMultMatrixd(fMatrix.CArr());
   glTranslated(fTruck.X(), fTruck.Y(), fTruck.Z());

   if (fCacheDirty) {
      UpdateCache();
   }
}

//______________________________________________________________________________
void TGLOrthoCamera::Configure(Double_t left, Double_t right, 
                               Double_t top, Double_t bottom)
{
   // Configure the camera state
   Double_t width = right - left;
   Double_t height = top - bottom;

   Double_t xZoom = width/fVolume.Extents().X();
   Double_t yZoom = height/fVolume.Extents().Y();

   fZoom = (xZoom > yZoom) ? xZoom : yZoom;

   // kXOY : X Horz. / Y Vert (looking towards +Z, Y up)
   // kXOZ : X Horz. / Z Vert (looking towards +Y, Z up)
   // kZOY : Z Horz. / Y Vert (looking towards +X, Y up)
   if (fType == kXOY) {
      fTruck.X() = right - left;
      fTruck.Y() = top - bottom;
   } else if (fType == kXOZ) {
      fTruck.X() = right - left;
      fTruck.Z() = top - bottom;
   } else if (fType == kZOY) {
      fTruck.Z() = right - left;
      fTruck.Y() = top - bottom;
   }
   fCacheDirty = kTRUE;
}

//______________________________________________________________________________
void TGLOrthoCamera::SetViewport(Int_t context)
{
   //Setup viewport, if it was changed, plus reset arcball.
   Int_t vp[4] = {0};
   gGLManager->ExtractViewport(context, vp);
   if (vp[2] != Int_t(fViewport.Width()) || vp[3] != Int_t(fViewport.Height()) || 
       vp[0] != fViewport.X() || vp[1] != fViewport.Y()) 
   {
      fVpChanged = kTRUE;
      fArcBall.SetBounds(vp[2], vp[3]);
      fViewport.Set(vp[0], vp[1], vp[2], vp[3]);
   } else
      fVpChanged = kFALSE;
}

//______________________________________________________________________________
void TGLOrthoCamera::SetViewVolume(const TGLVertex3 *box)
{
   //'box' is the TGLPlotPainter's back box's coordinates.
   fCenter[0] = (box[0].X() + box[1].X()) / 2;
   fCenter[1] = (box[0].Y() + box[2].Y()) / 2;
   fCenter[2] = (box[0].Z() + box[4].Z()) / 2;
   const Double_t maxDim = box[1].X() - box[0].X();
   fOrthoBox[0] = maxDim;
   fOrthoBox[1] = maxDim;
   fOrthoBox[2] = -100 * maxDim;//100?
   fOrthoBox[3] = 100 * maxDim;
   fShift = maxDim * 1.5;
}

//______________________________________________________________________________
void TGLOrthoCamera::StartRotation(Int_t px, Int_t py)
{
   //User clicks somewhere (px, py).
   fArcBall.Click(TPoint(px, py));
}

//______________________________________________________________________________
void TGLOrthoCamera::RotateCamera(Int_t px, Int_t py)
{
   //Mouse movement.
   fArcBall.Drag(TPoint(px, py));
}

//______________________________________________________________________________
void TGLOrthoCamera::StartPan(Int_t px, Int_t py)
{
   //User clicks somewhere (px, py).
   fMousePos.fX = px;
   fMousePos.fY = fViewport.Height() - py;
}

//______________________________________________________________________________
void TGLOrthoCamera::Pan(Int_t px, Int_t py)
{
   //Pan camera.
   py = fViewport.Height() - py;
   //Extract gl matrices.
   Double_t mv[16] = {0.};
   glGetDoublev(GL_MODELVIEW_MATRIX, mv);
   Double_t pr[16] = {0.};
   glGetDoublev(GL_PROJECTION_MATRIX, pr);
   Int_t vp[] = {0, 0, fViewport.Width(), fViewport.Height()};
   //Adjust pan vector.
   TGLVertex3 start, end;
   gluUnProject(fMousePos.fX, fMousePos.fY, 1., mv, pr, vp, &start.X(), &start.Y(), &start.Z());
   gluUnProject(px, py, 1., mv, pr, vp, &end.X(), &end.Y(), &end.Z());
   fTruck += (start - end) /= 2.;
   fMousePos.fX = px;
   fMousePos.fY = py;
}

//______________________________________________________________________________
void TGLOrthoCamera::SetCamera()const
{
   //Viewport and projection.
   glViewport(0, 0, fViewport.Width(), fViewport.Height());

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(
           -fOrthoBox[0] * fZoom,
            fOrthoBox[0] * fZoom, 
           -fOrthoBox[1] * fZoom, 
            fOrthoBox[1] * fZoom, 
            fOrthoBox[2], 
            fOrthoBox[3]
          );

   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

//______________________________________________________________________________
void TGLOrthoCamera::Apply()const
{
   //Applies rotations and translations before drawing
   glTranslated(0., 0., -fShift);
   glMultMatrixd(fArcBall.GetRotMatrix());
   glRotated(45., 1., 0., 0.);
   glRotated(-45., 0., 1., 0.);
   glRotated(-90., 0., 1., 0.);
   glRotated(-90., 1., 0., 0.);
   glTranslated(-fTruck[0], -fTruck[1], -fTruck[2]);
   glTranslated(-fCenter[0], -fCenter[1], -fCenter[2]);
}

//______________________________________________________________________________
Int_t TGLOrthoCamera::GetX()const
{
   //viewport[0]
   return fViewport.X();
}

//______________________________________________________________________________
Int_t TGLOrthoCamera::GetY()const
{
   //viewport[1]
   return fViewport.Y();
}


//______________________________________________________________________________
Int_t TGLOrthoCamera::GetWidth()const
{
   //viewport[2]
   return Int_t(fViewport.Width());
}

//______________________________________________________________________________
Int_t TGLOrthoCamera::GetHeight()const
{
   //viewport[3]
   return Int_t(fViewport.Height());
}

//______________________________________________________________________________
void TGLOrthoCamera::ZoomIn()
{
   //Zoom in.
   fZoom /= 1.2;
}

//______________________________________________________________________________
void TGLOrthoCamera::ZoomOut()
{
   //Zoom out.
   fZoom *= 1.2;
}

//______________________________________________________________________________
void TGLOrthoCamera::Markup(TGLCameraMarkupStyle* ms) const
{
   // Write viewport dimensions on screen.

   static const UChar_t
      digits[][8] = {{0x38, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x38},//0
                     {0x10, 0x10, 0x10, 0x10, 0x10, 0x70, 0x10, 0x10},//1
                     {0x7c, 0x44, 0x20, 0x18, 0x04, 0x04, 0x44, 0x38},//2
                     {0x38, 0x44, 0x04, 0x04, 0x18, 0x04, 0x44, 0x38},//3
                     {0x04, 0x04, 0x04, 0x04, 0x7c, 0x44, 0x44, 0x44},//4
                     {0x7c, 0x44, 0x04, 0x04, 0x7c, 0x40, 0x40, 0x7c},//5
                     {0x7c, 0x44, 0x44, 0x44, 0x7c, 0x40, 0x40, 0x7c},//6
                     {0x20, 0x20, 0x20, 0x10, 0x08, 0x04, 0x44, 0x7c},//7
                     {0x38, 0x44, 0x44, 0x44, 0x38, 0x44, 0x44, 0x38},//8
                     {0x7c, 0x44, 0x04, 0x04, 0x7c, 0x44, 0x44, 0x7c},//9
                     {0x18, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},//.
                     {0x00, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x00, 0x00},//-
                     {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10}};//|

   TGLVector3 extents = fVolume.Extents();
   Double_t width     = extents.X()/fZoom;
   Double_t maxbarw0  = ms->Barsize()*width;

   // get 10-exponent
   Int_t exp = (Int_t) TMath::Floor(TMath::Log10(maxbarw0));

   Double_t fact = maxbarw0/TMath::Power(10, exp);
   Float_t barw;

   if (fact > 5) {
      barw = 5*TMath::Power(10, exp);
      glColor3d(1., 0., 1.0);
   }
   else if (fact > 2) { 
      barw = 2*TMath::Power(10, exp);
      glColor3d(0., 1., 1.0);
   } else {
      barw = TMath::Power(10, exp);
      glColor3d(0., 0., 1.0);
   }
   Double_t wproc = barw / width;
   TString str = Form("%.*f", (exp < 0) ? -exp : 0, barw);
   Int_t screenw = fViewport.Width();
   Int_t screenh = fViewport.Height();

   Double_t sX, sY;
   Double_t offX, offY, txtOffX, txtOffY;
   ms->Offsets(offX, offY, txtOffX, txtOffY);

   switch (ms->Position()) {
   case TGLCameraMarkupStyle::kLUp:
      sX = offX;
      sY = screenh - offY -  txtOffY - 8;
      break;
   case TGLCameraMarkupStyle::kLDn:
      sX = offX;
      sY = offY;
      break;
   case TGLCameraMarkupStyle::kRUp:
      sX = screenw -  ms->Barsize()*screenw - offX;
      sY = screenh -  offY  -  txtOffY - 8;
      break;
   case TGLCameraMarkupStyle::kRDn:
      sX = screenw -  ms->Barsize()*screenw -  offX;
      sY = offY;
      break;
   default:
      sX = 0.5*screenw;
      sY = 0.5*screenh;
      break;
   }

   glTranslatef(sX, sY, 0);

   glLineWidth(2.);
   glColor3d(1., 1., 1.);
   
   Double_t mH = 2;

   glBegin(GL_LINES);
    // horizontal static
   glVertex3d(0, 0.,0.);
   glVertex3d(ms->Barsize()*screenw, 0., 0.);
   // corner bars
   glVertex3d(ms->Barsize()*screenw,  mH, 0.);
   glVertex3d(ms->Barsize()*screenw, -mH, 0.);
   // marker cormer bar
   glColor3d(1., 0., 0.);
   glVertex3d(0.,  mH, 0.);
   glVertex3d(0., -mH, 0.);
   // marker pointer
   glVertex3d(screenw*wproc, 0., 0.);
   glVertex3d(screenw*wproc, mH, 0.);
   //marker line
   glVertex3d(0, 0.,0.);
   glVertex3d(screenw*wproc, 0., 0.);
   glEnd();

   glTranslated(-sX, -sY, 0);

   glRasterPos3d(sX + txtOffX , sY + txtOffY, -1.);
   Double_t ox = 0.;
   Double_t oy = 0.;
   for (Ssiz_t i = 0, e = str.Length(); i < e; ++i) {
      if (str[i] == '.')
         glBitmap(8, 8, ox, oy, 7.0, 0.0, digits[10]);
      else
         glBitmap(8, 8, ox, oy, 7.0, 0.0, digits[str[i] - '0']);
   }
}
