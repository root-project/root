// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveCaloLegoGL.h"
#include "TEveCalo.h"
#include "TEveRGBAPalette.h"

#include "TGLIncludes.h"

#include "TGLRnrCtx.h"
#include "TGLSelectRecord.h"
#include "TGLScene.h"
#include "TGLCamera.h"
#include "TGLContext.h"
#include "TGLUtil.h"

#include "TAxis.h"
#include "TObjString.h"

#include "TH2.h"
#include "THLimitsFinder.h"

//______________________________________________________________________________
// OpenGL renderer class for TEveCaloLego.
//

ClassImp(TEveCaloLegoGL);

//______________________________________________________________________________
TEveCaloLegoGL::TEveCaloLegoGL() :
   TGLObject(),

   fDataMax(0),
   fZAxisStep(0),

   fEtaAxis(0),
   fPhiAxis(0),

   fXAxisAtt(),
   fYAxisAtt(),
   fZAxisAtt(),
   fAxisPainter(),

   fDLCacheOK(kFALSE),
   fM(0),

   fNBinSteps(5),
   fBinSteps(0),

   fTowerPicked(-1)
{
   // Constructor.

   fDLCache = kFALSE;

   // modes for different levels of zoom in/out.
   fBinSteps = new Int_t[fNBinSteps];
   fBinSteps[0] = 1;
   fBinSteps[1] = 2;
   fBinSteps[2] = 5;
   fBinSteps[3] = 8;
   fBinSteps[4] = 20;

   fXAxisAtt.SetTMNDim(2);
   fXAxisAtt.SetTextAlign(TGLAxisAttrib::kCenterDown);
   fXAxisAtt.SetNdivisions(710);
   fXAxisAtt.SetLabelSize(0.05);
   fXAxisAtt.SetTitleSize(0.05);
   fXAxisAtt.SetTitleFontName("symbol");
   fXAxisAtt.SetTitle("h");

   fYAxisAtt = fXAxisAtt;
   fYAxisAtt.RefDir().Set(0., 1., 0.);
   fYAxisAtt.SetNdivisions(510);
   fYAxisAtt.SetTitle("f");

   fZAxisAtt.RefDir().Set(0., 0., 1.);
   fZAxisAtt.SetTextAlign(TGLAxisAttrib::kLeft);
   fZAxisAtt.SetRelativeFontSize(kTRUE);
   fZAxisAtt.SetLabelSize(0.07);
   fZAxisAtt.SetTitleSize(0.07);
}

//______________________________________________________________________________
TEveCaloLegoGL::~TEveCaloLegoGL()
{
   // Destructor.

   delete [] fBinSteps;
}

//______________________________________________________________________________
Bool_t TEveCaloLegoGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   if (SetModelCheckClass(obj, TEveCaloLego::Class())) {
      fM = dynamic_cast<TEveCaloLego*>(obj);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TEveCaloLegoGL::SetBBox()
{
   // Set bounding box.

   SetAxisAlignedBBox(((TEveCaloLego*)fExternalObj)->AssertBBox());
}

//______________________________________________________________________________
void TEveCaloLegoGL::DLCacheDrop()
{
   // Drop all display-list definitions.

   for (SliceDLMap_i i = fDLMap.begin(); i != fDLMap.end(); ++i)
      i->second = 0;

   TGLObject::DLCacheDrop();
}

//______________________________________________________________________________
void TEveCaloLegoGL::DLCachePurge()
{
   // Unregister all display-lists.

   static const TEveException eH("TEveCaloLegoGL::DLCachePurge ");

   if (fDLMap.empty()) return;

   for (SliceDLMap_i i = fDLMap.begin(); i != fDLMap.end(); ++i)
   {
      if (fScene) {
         fScene->GetGLCtxIdentity()->RegisterDLNameRangeToWipe(i->second, 1);
      } else {
         glDeleteLists(i->second, 1);
      }
      i->second = 0;
   }
   TGLObject::DLCachePurge();
}

//______________________________________________________________________________
void TEveCaloLegoGL::MakeQuad(Float_t x1, Float_t y1, Float_t z1,
                              Float_t xw, Float_t yw, Float_t h) const
{
   // Draw an axis-aligned box using quads.

   //    z
   //    |
   //    |
   //    |________y
   //   /  6-------7
   //  /  /|      /|
   // x  5-------4 |
   //    | 2-----|-3
   //    |/      |/
   //    1-------0
   //

   Float_t x2 = x1+xw;
   Float_t y2 = y1+yw;
   Float_t z2 = z1+h;

   if (x1<fM->GetEtaMin()) x1= fM->GetEtaMin();
   if (x2>fM->GetEtaMax()) x2= fM->GetEtaMax();

   if (y1<fM->GetPhiMin()) y1= fM->GetPhiMin();
   if (y2>fM->GetPhiMax()) y2= fM->GetPhiMax();


   glBegin(GL_QUADS);
   {
      // bottom 0123
      glNormal3f(0, 0, -1);
      glVertex3f(x2, y2, z1);
      glVertex3f(x2, y1, z1);
      glVertex3f(x1, y1, z1);
      glVertex3f(x1, y2, z1);
      // top 4765
      glNormal3f(0, 0, 1);
      glVertex3f(x2, y2, z2);
      glVertex3f(x1, y2, z2);
      glVertex3f(x1, y1, z2);
      glVertex3f(x2, y1, z2);

      // back 0451
      glNormal3f(1, 0, 0);
      glVertex3f(x2, y2, z1);
      glVertex3f(x2, y2, z2);
      glVertex3f(x2, y1, z2);
      glVertex3f(x2, y1, z1);
      // front 3267
      glNormal3f(-1, 0, 0);
      glVertex3f(x1, y2, z1);
      glVertex3f(x1, y1, z1);
      glVertex3f(x1, y1, z2);
      glVertex3f(x1, y2, z2);

      // left  0374
      glNormal3f(0, 1, 0);
      glVertex3f(x2, y2, z1);
      glVertex3f(x1, y2, z1);
      glVertex3f(x1, y2, z2);
      glVertex3f(x2, y2, z2);
      // right 1562
      glNormal3f(0, -1, 0);
      glVertex3f(x2, y1, z1);
      glVertex3f(x2, y1, z2);
      glVertex3f(x1, y1, z2);
      glVertex3f(x1, y1, z1);
   }
   glEnd();
}

//______________________________________________________________________________
void TEveCaloLegoGL::MakeDisplayList() const
{
   // Create display-list that draws histogram bars.
   // It is used for filled and outline passes.

   TEveCaloData::CellData_t cellData;
   Int_t   prevTower = 0;
   Float_t offset = 0;

   // ids in eta phi rng
   Int_t nSlices = fM->fData->GetNSlices();
   for (Int_t s = 0; s < nSlices; ++s)
   {
      if (fDLMap.empty() || fDLMap[s] == 0)
         fDLMap[s] = glGenLists(1);
      glNewList(fDLMap[s], GL_COMPILE);

      for (UInt_t i = 0; i < fM->fCellList.size(); ++i)
      {
         if (fM->fCellList[i].fSlice > s) continue;
         if (fM->fCellList[i].fTower != prevTower)
         {
            offset = 0;
            prevTower = fM->fCellList[i].fTower;
         }

         fM->fData->GetCellData(fM->fCellList[i], fM->fPhi, fM->fPhiOffset, cellData);
         if (s == fM->fCellList[i].fSlice)
         {
            glLoadName(i);
            MakeQuad(cellData.EtaMin(), cellData.PhiMin(), offset,
                     cellData.EtaDelta(), cellData.PhiDelta(), cellData.Value(fM->fPlotEt));
         }
         offset += cellData.Value(fM->fPlotEt);
      }
      glEndList();
   }
   fDLCacheOK=kTRUE;
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawZAxis(TGLRnrCtx &rnrCtx, Float_t azX, Float_t azY) const
{
   // Draw Z axis at given xy position;

   glPushMatrix();
   glTranslatef(azX,  azY, 0);

   // tick mark projected vector is in x dimension
   TGLMatrix modview;
   glGetDoublev(GL_MODELVIEW_MATRIX, modview.Arr());
   TGLVertex3 worldRef(azX, azY, fDataMax*0.5);

   fZAxisAtt.SetAxisColor(fM->fGridColor);
   fZAxisAtt.SetRng(0, fDataMax);
   fZAxisAtt.SetNdivisions( fM->fNZSteps*100+10);
   fZAxisAtt.RefTMOff(0) = rnrCtx.RefCamera().ViewportDeltaToWorld(worldRef, -10, 0, &modview);
   fZAxisAtt.SetTitle(Form("Et[GeV] %s", TEveUtil::FormAxisValue(fDataMax)));
   fZAxisAtt.RefTitlePos().Set(0, 0, fDataMax*1.05);
   fAxisPainter.Paint(rnrCtx, fZAxisAtt);

   glPopMatrix();
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawZScales3D(TGLRnrCtx & rnrCtx,
                                   Float_t x0, Float_t x1,
                                   Float_t y0, Float_t y1) const
{
   // Draw z-axis at the appropriate grid corner-point including
   // tick-marks and labels.

   // corner points projected
   const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();
   GLdouble mm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetIntegerv(GL_VIEWPORT, vp);
   GLdouble x[4];
   GLdouble y[4];
   GLdouble z[4];
   gluProject(x0, y0, 0, mm, pm, vp, &x[0], &y[0], &z[0]);
   gluProject(x1, y0, 0, mm, pm, vp, &x[1], &y[1], &z[1]);
   gluProject(x1, y1, 0, mm, pm, vp, &x[2], &y[2], &z[2]);
   gluProject(x0, y1, 0, mm, pm, vp, &x[3], &y[3], &z[3]);


   /**************************************************************************/

   // get pos of z axis (left most corner)

   Int_t idxLeft = 0;
   Float_t xt = x[0];
   for (Int_t i = 1; i < 4; ++i)
   {
      if (x[i] < xt)
      {
         xt  = x[i];
         idxLeft = i;
      }
   }

   Float_t azX  = 0, azY  = 0;
   switch(idxLeft)
   {
      case 0:
         azX  =  x0;      azY  =  y0;
         break;
      case 1:
         azX  =  x1;      azY  =  y0;
         break;
      case 2:
         azX  =  x1;      azY  =  y1;
         break;
      case 3:
         azX  =  x0;      azY  =  y1;
         break;
   }

   /**************************************************************************/
   // z axis body

   TGLUtil::Color(fM->fGridColor);

   if (fM->fBoxMode)
   {
      // get corner closest to eye, excluding left corner
      Double_t zt = 1.f;
      Int_t idxDepth = 0;
      for (Int_t i = 0; i < 4; ++i)
      {
         if (z[i] < zt)
         {
            zt  = z[i];
            idxDepth = i;
         }
      }
      Double_t zm = 1.f;
      Int_t idxDepthT = 0;
      for (Int_t i = 0; i < 4; ++i)
      {
         if (z[i] < zm && z[i] >= zt && i != idxDepth)
         {
            zm  = z[i];
            idxDepthT = i;
         }
      }
      if (idxDepth == idxLeft)  idxDepth =idxDepthT;

      Float_t ayX = 0; // Y position of back plane X = const
      Float_t axY = 0; // X postion of back plane  Y = const
      Float_t cX  = 0, cY  = 0; // coodinates of a point closest to eye
      switch (idxDepth)
      {
         case 0:
            axY=y1; ayX=x1;
            cX=x0; cY=y0;
            break;
         case 1:
            axY=y1; ayX=x0;
            cX= x1; cY=y0;
            break;
         case 2:
            axY=y0; ayX=x0;
            cX=x1;  cY=y1;
            break;
         case 3:
            axY=y0; ayX=x1;
            cX= x0; cY=y1;
            break;
      }

      glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
      glLineWidth(1);

      // box verticals
      glBegin(GL_LINES);
      glVertex3f(x0, axY, 0); glVertex3f(x0, axY, fDataMax);
      glVertex3f(x1, axY, 0); glVertex3f(x1, axY, fDataMax);
      glVertex3f(ayX, y0, 0); glVertex3f(ayX, y0, fDataMax);
      glVertex3f(ayX, y1, 0); glVertex3f(ayX, y1, fDataMax);
      if (fM->fBoxMode == TEveCaloLego::kFrontBack)
      {
         glVertex3f(cX, cY, 0); glVertex3f(cX, cY, fDataMax);
      }

      // box top
      glVertex3f(x0, axY, fDataMax); glVertex3f(x1, axY, fDataMax);
      glVertex3f(ayX, y0, fDataMax); glVertex3f(ayX, y1, fDataMax);
      if (fM->fBoxMode == TEveCaloLego::kFrontBack)
      {
         glVertex3f(cX, cY, fDataMax); glVertex3f(cX, axY, fDataMax);
         glVertex3f(cX, cY, fDataMax); glVertex3f(ayX, cY, fDataMax);
      }
      glEnd();
      glPopAttrib();

      // box horizontals stippled
      glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
      glLineStipple(1, 0x5555);
      glEnable(GL_LINE_STIPPLE);
      glBegin(GL_LINES);
      Int_t nhs = TMath::CeilNint(fDataMax/fZAxisStep);
      Float_t hz  = 0;
      for (Int_t i = 1; i <= nhs; ++i, hz += fZAxisStep)
      {
         glVertex3f(x0, axY, hz); glVertex3f(x1, axY, hz);
         glVertex3f(ayX, y0, hz); glVertex3f(ayX, y1, hz);
      }
      glEnd();
      glPopAttrib();
   } // draw box

   /**************************************************************************/

   DrawZAxis(rnrCtx, azX,  azY);

   if (fTowerPicked >= 0)
   {
      // left most corner of the picked tower
      TEveCaloData::CellData_t cd;
      fM->fData->GetCellData(fM->fCellList[fTowerPicked], fM->fPhi, fM->fPhiOffset, cd);
      switch(idxLeft)
      {
         case 0:
            azX  =  cd.EtaMin();      azY  =  cd.PhiMin();
            break;
         case 1:
            azX  =  cd.EtaMax();      azY  =  cd.PhiMin();
            break;
         case 2:
            azX  =  cd.EtaMax();      azY  =  cd.PhiMax();
            break;
         case 3:
            azX  =  cd.EtaMin();      azY  =  cd.PhiMax();
            break;
      }
      DrawZAxis(rnrCtx, azX,  azY);
   }
} // DrawZScales3D

//______________________________________________________________________________
void TEveCaloLegoGL::DrawXYScales(TGLRnrCtx & rnrCtx,
                                  Float_t x0, Float_t x1,
                                  Float_t y0, Float_t y1) const
{
   // Draw XY title, labels.

   // corner point closest to the eye
   const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();
   GLdouble mm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetIntegerv(GL_VIEWPORT, vp);
   GLdouble y,  z[4], x[4];
   gluProject(x0, y0, 0, mm, pm, vp, &x[0], &y, &z[0]);
   gluProject(x1, y0, 0, mm, pm, vp, &x[1], &y, &z[1]);
   gluProject(x1, y1, 0, mm, pm, vp, &x[2], &y, &z[2]);
   gluProject(x0, y1, 0, mm, pm, vp, &x[3], &y, &z[3]);

   Float_t zt = 1.f;
   Float_t zm = 0.f;
   Int_t idx = 0;
   for (Int_t i = 0; i < 4; ++i)
   {
      if (z[i] < zt)
      {
         zt  = z[i];
         idx = i;
      }
      if (z[i] > zm) zm = z[i];
   }
   if (zm - zt < 1e-2) idx = 0; // avoid flipping in front view

   // XY axis location
   Float_t axY  = 0, ayX  = 0;
   Float_t axtX = 0, aytY = 0; // title pos
   switch (idx)
   {
      case 0:
         axY  = y0; ayX  = x0;
         axtX = x1; aytY = y1;
         break;
      case 1:
         ayX  = x1; axY  = y0;
         axtX = x0; aytY = y1;
         break;
      case 2:
         ayX  = x1; axY  = y1;
         axtX = x0; aytY = y0;
         break;
      case 3:
         ayX  = x0; axY  = y1;
         axtX = x1; aytY = y0;
         break;
   }

   Float_t zOff  = -fDataMax*0.03;
   Float_t yOff  =  0.03*TMath::Sign(y1-y0, axY);
   Float_t xOff  =  0.03*TMath::Sign(x1-x0, ayX);
   Float_t rxy = (fPhiAxis->GetXmax()-fPhiAxis->GetXmin())/(fEtaAxis->GetXmax()-fEtaAxis->GetXmin());
   (rxy>1) ? yOff /= rxy : xOff *=rxy;

   if (fXAxisAtt.GetRelativeFontSize() == kFALSE)
   {
      fXAxisAtt.SetAbsLabelFontSize(fZAxisAtt.GetAbsLabelFontSize());
      fXAxisAtt.SetAbsTitleFontSize(Int_t(fZAxisAtt.GetAbsTitleFontSize()*0.8));
   }

   fXAxisAtt.SetRng(x0, x1);
   fXAxisAtt.SetAxisColor(fM->fGridColor);
   fXAxisAtt.RefTMOff(0).Set(0, yOff, 0);
   fXAxisAtt.RefTMOff(1).Set(0, 0, zOff);
   fXAxisAtt.RefTitlePos().Set(axtX, 0, 0);
   glPushMatrix();
   glTranslatef(0, axY, 0);
   fAxisPainter.Paint(rnrCtx, fXAxisAtt);
   glPopMatrix();


   fYAxisAtt.SetRng(y0, y1);
   fYAxisAtt.SetAxisColor(fM->fGridColor);
   fYAxisAtt.RefTMOff(0).Set(xOff, 0, 0);
   fYAxisAtt.RefTMOff(1).Set(0, 0, zOff);
   fYAxisAtt.SetAbsLabelFontSize(fXAxisAtt.GetAbsLabelFontSize());
   fYAxisAtt.SetAbsTitleFontSize(fXAxisAtt.GetAbsTitleFontSize());
   fYAxisAtt.RefTitlePos().Set(0, aytY, 0);
   glPushMatrix();
   glTranslatef(ayX, 0, 0);
   fAxisPainter.Paint(rnrCtx, fYAxisAtt);
   glPopMatrix();

} // DrawXYScales

//______________________________________________________________________________
Int_t TEveCaloLegoGL::GetGridStep(Int_t axId, TGLRnrCtx &rnrCtx) const
{
   // Calculate view-dependent grid density.

   GLdouble x0, y0, z0, x1, y1, z1;

   GLdouble mm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetIntegerv(GL_VIEWPORT, vp);
   const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();


   if (axId == 0)
   {
      Float_t bw = (fEtaAxis->GetXmax()-fEtaAxis->GetXmin())/fEtaAxis->GetNbins();
      for (Int_t i=0; i<fNBinSteps; ++i)
      {
         gluProject(fM->GetEta(),                 fM->GetPhi(), 0, mm, pm, vp, &x0, &y0, &z0);
         gluProject(fM->GetEta()+fBinSteps[i]*bw, fM->GetPhi(), 0, mm, pm, vp, &x1, &y1, &z1);

         Float_t gapsqr = (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1);
         if (gapsqr > fM->fBinWidth*fM->fBinWidth)
            return fBinSteps[i];
      }
   }
   else if (axId == 1)
   {
      Double_t bw =  (fPhiAxis->GetXmax()-fPhiAxis->GetXmin())/fPhiAxis->GetNbins();
      for (Int_t i=0; i<fNBinSteps; ++i)
      {
         gluProject(fM->GetEta(), fM->GetPhi(),                 0, mm, pm, vp, &x0, &y0, &z0);
         gluProject(fM->GetEta(), fM->GetPhi()+fBinSteps[i]*bw, 0, mm, pm, vp, &x1, &y1, &z1);

         Float_t  gapsqr = (x0-x1)*(x0-x1) + (y0-y1)*(y0-y1);
         if (gapsqr > fM->fBinWidth*fM->fBinWidth)
            return fBinSteps[i];
      }
   }
   return 1;
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawHistBase(TGLRnrCtx &rnrCtx) const
{
   // Draw basic histogram components: x-y grid

   using namespace TMath;

   Int_t es = GetGridStep(0, rnrCtx);
   Int_t ps = GetGridStep(1, rnrCtx);

   Float_t eta0 = fM->fEtaMin;
   Float_t eta1 = fM->fEtaMax;
   Float_t phi0 = fM->GetPhiMin();
   Float_t phi1 = fM->GetPhiMax();

   TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);
   TGLCapabilitySwitch sw_blend(GL_BLEND, kTRUE);

   TGLUtil::Color(fM->fGridColor);

   glLineWidth(1);
   glBegin(GL_LINES);

   glVertex2f(eta0, phi0);
   glVertex2f(eta0, phi1);
   glVertex2f(eta1, phi0);
   glVertex2f(eta1, phi1);

   glVertex2f(eta0, phi0);
   glVertex2f(eta1, phi0);
   glVertex2f(eta0, phi1);
   glVertex2f(eta1, phi1);

   // eta grid
   if ((es == 1 && ps == 1) || fM->fProjection == TEveCaloLego::k3D)
   {
      // original eta
      for (Int_t i=fEtaAxis->GetFirst(); i<fEtaAxis->GetLast(); i+= es)
      {
         glVertex2f(fEtaAxis->GetBinUpEdge(i), phi0);
         glVertex2f(fEtaAxis->GetBinUpEdge(i), phi1);
      }
   }
   else
   {
      // scaled eta
      Int_t   nEta = CeilNint((fEtaAxis->GetLast()-fEtaAxis->GetFirst())/es);
      Float_t etaStep = (eta1-eta0)/nEta;
      Float_t eta = eta0;
      while (eta < eta1)
      {
         glVertex2f(eta, phi0);
         glVertex2f(eta, phi1);
         eta += etaStep;
      }
   }

   // phi grid
   Int_t   nPhi = CeilNint((phi1-phi0)/(fPhiAxis->GetBinWidth(1)*ps));
   Float_t phi, phiStep;
   if (ps>1)
   {
      phiStep = (phi1-phi0)/nPhi;
      phi = phi0;
   }
   else
   {
      phiStep = fPhiAxis->GetBinWidth(1);
      phi = phiStep*CeilNint(phi0/phiStep);
   }
   while (phi < phi1)
   {
      glVertex2f(eta0, phi);
      glVertex2f(eta1, phi);
      phi+=phiStep;
   }

   glEnd();


   // scales
   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POLYGON_BIT);
   glLineWidth(2);
   if ( fM->fProjection == TEveCaloLego::k3D || rnrCtx.RefCamera().GetCamBase().GetBaseVec(1).Z() == 0)
      DrawZScales3D(rnrCtx, eta0, eta1, phi0, phi1);

   if (fM->fProjection == TEveCaloLego::k2D || rnrCtx.RefCamera().GetCamBase().GetBaseVec(1).Z() != 0)
      fXAxisAtt.SetRelativeFontSize(kTRUE);
   else
      fXAxisAtt.SetRelativeFontSize(kFALSE);
   DrawXYScales(rnrCtx, eta0, eta1, phi0, phi1);

   glPopAttrib();
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawCells3D(TGLRnrCtx & rnrCtx) const
{
   // Render the calo lego-plot with OpenGL.

   // quads
   {
      for(SliceDLMap_i i = fDLMap.begin(); i != fDLMap.end(); ++i)
      {
         TGLUtil::Color(fM->GetDataSliceColor(i->first));
         glCallList(i->second);
      }
   }
   // outlines
   {
      if (rnrCtx.SceneStyle() == TGLRnrCtx::kFill)
      {
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
         glDisable(GL_POLYGON_OFFSET_FILL);
         TGLUtil::Color(1);
         for (SliceDLMap_i i = fDLMap.begin(); i != fDLMap.end(); ++i)
            glCallList(i->second);
      }
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawCells2D(TGLRnrCtx & rnrCtx) const
{
   // Draw projected histogram.

   static const TEveException eh("TEveCaloLegoGL::DrawCells2D ");

   using namespace TMath;

   // When kValSize is used, plot gl-points at tower-centers avoid flickering.
   std::vector<TEveVector> antiFlick;

   fM->AssertPalette();

   UChar_t col[4];
   Color_t defCol = fM->GetDataSliceColor(0);  // default color is first slice color
   Int_t es = GetGridStep(0, rnrCtx);
   Int_t ps = GetGridStep(1, rnrCtx);
   if (es==1 && ps==1)
   {
      // draw in original binning
      TEveCaloData::CellData_t cellData;
      Int_t name = 0;
      Int_t prevTower = 0;
      Float_t sum = 0;
      Float_t x1=0, x2=0, y1=0, y2=0;
      TGLUtil::Color(defCol);
      for (TEveCaloData::vCellId_t::iterator it=fM->fCellList.begin(); it!=fM->fCellList.end(); it++)
      {
         fM->fData->GetCellData(*it, fM->fPhi, fM->fPhiOffset, cellData);
         glLoadName(name);
         glBegin(GL_QUADS);
         if (it->fTower != prevTower)
         {

            if (fM->f2DMode == TEveCaloLego::kValColor)
            {
               fM->fPalette->ColorFromValue(FloorNint(sum), col);
               TGLUtil::Color4ubv(col);

               x1 = Max(fM->GetEtaMin(), cellData.EtaMin());
               x2 = Min(fM->GetEtaMax(), cellData.EtaMax());

               y1 = Max(fM->GetPhiMin(), cellData.PhiMin());
               y2 = Min(fM->GetPhiMax(), cellData.PhiMax());
            }
            else if (fM->f2DMode == TEveCaloLego::kValSize)
            {
               const Float_t etaW = (cellData.EtaDelta()*sum*0.5f)/fDataMax;
               const Float_t phiW = (cellData.PhiDelta()*sum*0.5f)/fDataMax;

               x1 = Max(fM->GetEtaMin(), cellData.Eta() - etaW);
               x2 = Min(fM->GetEtaMax(), cellData.Eta() + etaW);

               y1 = Max(fM->GetPhiMin(), cellData.Phi() - phiW);
               y2 = Min(fM->GetPhiMax(), cellData.Phi() + phiW);
            }

            glVertex3f(x1, y1, sum);
            glVertex3f(x2, y1, sum);
            glVertex3f(x2, y2, sum);
            glVertex3f(x1, y2, sum);

            if (fM->f2DMode == TEveCaloLego::kValSize)
               antiFlick.push_back(TEveVector(0.5f*(x1+x2), 0.5f*(y1+y2), sum));

            sum = cellData.Value(fM->fPlotEt);
            prevTower = it->fTower;
         }
         else
         {
            sum += cellData.Value(fM->fPlotEt);
         }
         glEnd();
         ++name;
      }
   }
   else
   {
      // values in the scaled cells
      Float_t eta0 = fM->fEtaMin;
      Float_t eta1 = fM->fEtaMax;
      Float_t phi0 = fM->GetPhiMin();
      Float_t phi1 = fM->GetPhiMax();

      Int_t   nEta = CeilNint((fEtaAxis->GetLast()-fEtaAxis->GetFirst())/es);
      Int_t   nPhi = CeilNint((phi1-phi0)/(fPhiAxis->GetBinWidth(1)*ps));
      Float_t etaStep = (eta1-eta0)/nEta;
      Float_t phiStep = (phi1-phi0)/nPhi;

      std::vector<Float_t> vec;
      vec.assign((nEta)*(nPhi), 0.f);
      TEveCaloData::CellData_t cd;
      Float_t left, right, up, down; // cell corners
      for (TEveCaloData::vCellId_t::iterator it=fM->fCellList.begin(); it!=fM->fCellList.end(); it++)
      {
         fM->fData->GetCellData(*it, fM->fPhi, fM->fPhiOffset, cd);
         Int_t iMin = FloorNint((cd.EtaMin() - eta0)/etaStep);
         Int_t iMax = CeilNint ((cd.EtaMax() - eta0)/etaStep);
         Int_t jMin = FloorNint((cd.PhiMin() - phi0)/phiStep);
         Int_t jMax = CeilNint ((cd.PhiMax() - phi0)/phiStep);
         for (Int_t i=iMin; i<iMax; i++)
         {
            if (i<0 || iMax>nEta) continue;

            left  = i*etaStep;
            right = (i+1)*etaStep;

            if (i == iMin)
               left = cd.EtaMin()-eta0;

            if (i == (iMax-1))
               right = cd.EtaMax()-eta0;

            for (Int_t j=jMin; j<jMax; j++)
            {
               if (j<0 || jMax>nPhi) continue;

               down = j*phiStep;
               up = down +phiStep;

               if (j == jMin)
                  down = cd.PhiMin() - phi0;

               if (j == (jMax-1))
                  up = cd.PhiMax() - phi0;

               vec[i*nPhi+j]  += cd.Value(fM->fPlotEt)*(right-left)*(up-down);
            }
         }
      }

      Float_t surf = etaStep*phiStep;
      Float_t maxv = 0;
      for (UInt_t i =0; i<vec.size(); i++)
      {
         vec[i] /= surf;
         if (vec[i] > maxv) maxv=vec[i];
      }

      Float_t scale    = fM->fData->GetMaxVal(fM->fPlotEt)/maxv;
      Float_t logMax   = Log(maxv+1);
      Float_t scaleLog = fM->fData->GetMaxVal(fM->fPlotEt)/logMax;

      // take smallest threshold
      Float_t threshold = fM->GetDataSliceThreshold(0);
      for (Int_t s=1; s<fM->fData->GetNSlices(); s++)
      {
         if (threshold > fM->GetDataSliceThreshold(s))
            threshold = fM->GetDataSliceThreshold(s);
      }

      // draw  scaled
      Float_t etaW, phiW, sum;
      Int_t cid = 0;
      TGLUtil::Color(defCol);
      glBegin(GL_QUADS);
      for (std::vector<Float_t>::iterator it=vec.begin(); it !=vec.end(); it++)
      {
         if (*it > threshold)
         {
            Float_t logVal = Log(*it + 1);
            Float_t eta = Int_t(cid/nPhi)*etaStep + eta0;
            Float_t phi = (cid -Int_t(cid/nPhi)*nPhi)*phiStep + phi0;
            if (fM->f2DMode == TEveCaloLego::kValColor)
            {
               fM->fPalette->ColorFromValue((Int_t)(logVal*scaleLog), col);
               TGLUtil::Color4ubv(col);

               glVertex3f(eta        , phi,         (*it)*scale);
               glVertex3f(eta+etaStep, phi,         (*it)*scale);
               glVertex3f(eta+etaStep, phi+phiStep, (*it)*scale);
               glVertex3f(eta        , phi+phiStep, (*it)*scale);

            }
            else if (fM->f2DMode == TEveCaloLego::kValSize)
            {
               eta += etaStep*0.5f;
               phi += phiStep*0.5f;
               etaW = etaStep*0.5f*logVal/logMax;
               phiW = phiStep*0.5f*logVal/logMax;
               sum  = (*it)*scale;

               glVertex3f(eta - etaW, phi - phiW, sum);
               glVertex3f(eta + etaW, phi - phiW, sum);
               glVertex3f(eta + etaW, phi + phiW, sum);
               glVertex3f(eta - etaW, phi + phiW, sum);

               antiFlick.push_back(TEveVector(eta, phi, sum));
            }
         }
         ++cid;
      }
      glEnd();
   }

   if ( ! antiFlick.empty())
   {
      TGLUtil::Color(defCol);
      glPointSize(2);
      glBegin(GL_POINTS);
      for (std::vector<TEveVector>::iterator i = antiFlick.begin(); i != antiFlick.end(); ++i)
      {
         glVertex3fv(i->Arr());
      }
      glEnd();
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // Draw the object.

   fEtaAxis = fM->fData->GetEtaBins();
   fPhiAxis = fM->fData->GetPhiBins();

   // projection type
   Bool_t cells3D;
   if (fM->fProjection == TEveCaloLego::kAuto)
      cells3D = (! (rnrCtx.RefCamera().IsOrthographic() && rnrCtx.RefCamera().GetCamBase().GetBaseVec(1).Z()));
   else if (fM->fProjection == TEveCaloLego::k2D)
      cells3D = kFALSE;
   else
      cells3D = kTRUE;

   // init cached variables
   fDataMax = fM->GetMaxVal();
   Int_t ondiv;
   Double_t omin, omax;
   THLimitsFinder::Optimize(0, fDataMax, fM->fNZSteps, omin, omax, ondiv,  fZAxisStep);

   // cache
   if (fM->fCellIdCacheOK == kFALSE)
   {
      fDLCacheOK = kFALSE;
      fM->BuildCellIdCache();
   }
   if (cells3D && fDLCacheOK == kFALSE) MakeDisplayList();

   // modelview matrix
   glPushMatrix();

   Double_t em, eM, pm, pM;
   fM->fData->GetEtaLimits(em, eM);
   fM->fData->GetPhiLimits(pm, pM);

   // scale due to shortest XY axis
   Double_t unit = ((eM-em) < (pM-pm)) ? (eM-em):(pM-pm);

   // scale from rebinning
   Float_t sx = (eM-em)/fM->GetEtaRng();
   Float_t sy = (pM-pm)/fM->GetPhiRng();
   glScalef(sx/unit, sy/unit, fM->GetValToHeight());
   glTranslatef(-fM->GetEta(), -fM->fPhi, 0);

   // draw cells
   glPushAttrib(GL_LINE_BIT | GL_POLYGON_BIT);
   glLineWidth(1);
   glDisable(GL_LIGHTING);
   glEnable(GL_NORMALIZE);
   glEnable(GL_POLYGON_OFFSET_FILL);
   glPushName(0);
   glPolygonOffset(0.8, 1);
   cells3D ? DrawCells3D(rnrCtx):DrawCells2D(rnrCtx);
   glPopName();
   glPopAttrib();

   // draw histogram base
   if (rnrCtx.Selection() == kFALSE && rnrCtx.Highlight() == kFALSE)
   {
      DrawHistBase(rnrCtx);
      if (fM->fDrawHPlane)
      {
         glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT);
         glEnable(GL_BLEND);
         glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
         glDisable(GL_CULL_FACE);
         TGLUtil::ColorTransparency(fM->fPlaneColor, fM->fPlaneTransparency);
         Float_t zhp = fM->fHPlaneVal*fDataMax;
         glBegin(GL_POLYGON);
         glVertex3f(fM->fEtaMin, fM->GetPhiMin(), zhp);
         glVertex3f(fM->fEtaMax, fM->GetPhiMin(), zhp);
         glVertex3f(fM->fEtaMax, fM->GetPhiMax(), zhp);
         glVertex3f(fM->fEtaMin, fM->GetPhiMax(), zhp);
         glEnd();
	 glPopAttrib();
      }
   }

   glPopMatrix();
}

//______________________________________________________________________________
void TEveCaloLegoGL::ProcessSelection(TGLRnrCtx & /*rnrCtx*/, TGLSelectRecord & rec)
{
   // Processes secondary selection from TGLViewer.

   if (rec.GetN() < 2) return;
   Int_t cellID = rec.GetItem(1);
   TEveCaloData::CellData_t cellData;
   fM->fData->GetCellData(fM->fCellList[cellID], cellData);

   printf("Bin selected in slice %d \n", fM->fCellList[cellID].fSlice);
   cellData.Dump();
}
