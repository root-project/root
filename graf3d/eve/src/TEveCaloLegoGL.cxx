// @(#)root/eve:$Id$
// Author: Alja Mrak-Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TAxis.h"
#include "TH2.h"
#include "THLimitsFinder.h"

#include "TGLViewer.h"
#include "TGLIncludes.h"
#include "TGLPhysicalShape.h"
#include "TGLRnrCtx.h"
#include "TGLSelectRecord.h"
#include "TGLScene.h"
#include "TGLCamera.h"
#include "TGLUtil.h"
#include "TColor.h"
#include "TROOT.h"


#include "TEveCaloLegoGL.h"
#include "TEveCalo.h"
#include "TEveManager.h"
#include "TEveRGBAPalette.h"

#include <algorithm>

//______________________________________________________________________________
// OpenGL renderer class for TEveCaloLego.
//

ClassImp(TEveCaloLegoGL);

//______________________________________________________________________________
TEveCaloLegoGL::TEveCaloLegoGL() :
   TGLObject(),

   fDataMax(0),
   fGridColor(-1),
   fFontColor(-1),

   fEtaAxis(0),
   fPhiAxis(0),
   fZAxis(0),
   fM(0),
   fDLCacheOK(kFALSE),
   fCells3D(kTRUE)
{
   // Constructor.

   fDLCache = kFALSE;

   fEtaAxis = new TAxis();
   fPhiAxis = new TAxis();
   fZAxis   = new TAxis();

   fAxisPainter.SetFontMode(TGLFont::kPixmap);
}

//______________________________________________________________________________
TEveCaloLegoGL::~TEveCaloLegoGL()
{
   // Destructor.

   DLCachePurge();

   delete fEtaAxis;
   delete fPhiAxis;
   delete fZAxis;
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

   fDLCacheOK = kFALSE;
   for (SliceDLMap_i i = fDLMap.begin(); i != fDLMap.end(); ++i)
      i->second = 0;

   for (SliceDLMap_i i = fDLMapSelected.begin(); i != fDLMapSelected.end(); ++i)
      i->second = 0;

   TGLObject::DLCacheDrop();
}

//______________________________________________________________________________
void TEveCaloLegoGL::DLCachePurge()
{
   // Unregister all display-lists.

   // all lego cells
   fDLCacheOK = kFALSE;
   if (! fDLMap.empty()) {
      for (SliceDLMap_i i = fDLMap.begin(); i != fDLMap.end(); ++i) {
         if (i->second) {
            PurgeDLRange(i->second, 1);
            i->second = 0;
         }
      }
   }

   // selection cells 
   if (! fDLMapSelected.empty()) {
      for (SliceDLMap_i i = fDLMapSelected.begin(); i != fDLMapSelected.end(); ++i) {
         if (i->second) {
            PurgeDLRange(i->second, 1);
            i->second = 0;
         }
      }
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

   Float_t x2 = x1 + xw;
   Float_t y2 = y1 + yw;
   Float_t z2 = z1 + h;

   if (x1 < fM->GetEtaMin()) x1 = fM->GetEtaMin();
   if (x2 > fM->GetEtaMax()) x2 = fM->GetEtaMax();

   if (y1 < fM->GetPhiMin()) y1 = fM->GetPhiMin();
   if (y2 > fM->GetPhiMax()) y2 = fM->GetPhiMax();

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
void TEveCaloLegoGL::Make3DDisplayList(TEveCaloData::vCellId_t& cellList, SliceDLMap_t& dlMap, Bool_t selection) const
{
   // Create display-list that draws histogram bars for non-rebinned data.
   // It is used for filled and outline passes.

   TEveCaloData::CellData_t cellData;
   Int_t   prevTower = 0;
   Float_t offset = 0;

   // ids in eta phi rng
   Int_t nSlices = fM->fData->GetNSlices();
   for (Int_t s = 0; s < nSlices; ++s)
   {
      if (dlMap.empty() || dlMap[s] == 0)
         dlMap[s] = glGenLists(1);

      glNewList(dlMap[s], GL_COMPILE);

      for (UInt_t i = 0; i < cellList.size(); ++i) {
         if (cellList[i].fSlice > s) continue;
         if (cellList[i].fTower != prevTower) {
            offset = 0;
            prevTower = cellList[i].fTower;
         }

         fM->fData->GetCellData(cellList[i], cellData);
         if (s == cellList[i].fSlice) {
            if (selection)
               glLoadName(i);

            WrapTwoPi(cellData.fPhiMin, cellData.fPhiMax);
            MakeQuad(cellData.EtaMin(), cellData.PhiMin(), offset,
                     cellData.EtaDelta(), cellData.PhiDelta(), cellData.Value(fM->fPlotEt));
         }
         offset += cellData.Value(fM->fPlotEt);
      }
      glEndList();
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::Make3DDisplayListRebin(TEveCaloData::RebinData_t& rebinData, SliceDLMap_t& dlMap, Bool_t selection) const
{
   // Create display-list that draws histogram bars for rebinned data.
   // It is used for filled and outline passes.

   Int_t nSlices = fM->fData->GetNSlices();
   Float_t *vals;
   Int_t bin;
   Float_t offset;
   Float_t y0, y1;
   for (Int_t s = 0; s < nSlices; ++s)
   {
      if (dlMap.empty() || dlMap[s] == 0)
         dlMap[s] = glGenLists(1);

      glNewList(dlMap[s], GL_COMPILE);

      if (selection) glLoadName(s);
      if (selection) glPushName(0);
      for (Int_t i=1; i<= fEtaAxis->GetNbins(); ++i)
      {
         for (Int_t j=1; j <= fPhiAxis->GetNbins(); ++j)
         {
            bin = (i)+(j)*(fEtaAxis->GetNbins()+2);

            if (rebinData.fBinData[bin] !=-1)
            {
               vals = rebinData.GetSliceVals(bin);
               offset =0;
               for (Int_t t=0; t<s; t++)
                  offset+=vals[t];

               y0 = fPhiAxis->GetBinLowEdge(j);
               y1 = fPhiAxis->GetBinUpEdge(j);
               WrapTwoPi(y0, y1);
               {
                  if (selection) glLoadName(bin);
                  MakeQuad(fEtaAxis->GetBinLowEdge(i), y0, offset,
                           fEtaAxis->GetBinWidth(i), y1-y0, vals[s]);
               }
            }
         }
      }
      if (selection) glPopName();
      glEndList();
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::SetAxis3DTitlePos(TGLRnrCtx &rnrCtx, Float_t x0, Float_t x1, Float_t y0, Float_t y1) const
{
   const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();
   GLdouble mm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetIntegerv(GL_VIEWPORT, vp);
   GLdouble projX[4], projY[4], projZ[4];

   GLdouble cornerX[4];
   GLdouble cornerY[4];
   cornerX[0] = x0; cornerY[0] = y0;
   cornerX[1] = x1; cornerY[1] = y0;
   cornerX[2] = x1; cornerY[2] = y1;
   cornerX[3] = x0; cornerY[3] = y1;

   gluProject(cornerX[0], cornerY[0], 0, mm, pm, vp, &projX[0], &projY[0], &projZ[0]);
   gluProject(cornerX[1], cornerY[1], 0, mm, pm, vp, &projX[1], &projY[1], &projZ[1]);
   gluProject(cornerX[2], cornerY[2], 0, mm, pm, vp, &projX[2], &projY[2], &projZ[2]);
   gluProject(cornerX[3], cornerY[3], 0, mm, pm, vp, &projX[3], &projY[3], &projZ[3]);


   // Z axis location (left most corner)
   //
   Int_t idxLeft = 0;
   Float_t xt = projX[0];
   for (Int_t i = 1; i < 4; ++i) {
      if (projX[i] < xt) {
         xt  = projX[i];
         idxLeft = i;
      }
   }
   fZAxisTitlePos.Set(cornerX[idxLeft], cornerY[idxLeft], fDataMax* 1.05);


   // XY axis location (closest to eye) first
   //
   Float_t zt = 1.f;
   Float_t zMin = 0.f;
   Int_t idxFront = 0;
   for (Int_t i = 0; i < 4; ++i) {
      if (projZ[i] < zt) {
         zt  = projZ[i];
         idxFront = i;
      }
      if (projZ[i] > zMin) zMin = projZ[i];
   }


   Int_t xyIdx = idxFront;
   if (zMin - zt < 1e-2) xyIdx = 0; // avoid flipping in front view


   switch (xyIdx) {
      case 0:
         fXAxisTitlePos.fX = x1;
         fXAxisTitlePos.fY = y0;
         fYAxisTitlePos.fX = x0;
         fYAxisTitlePos.fY = y1;
         break;
      case 1:
         fXAxisTitlePos.fX = x0;
         fXAxisTitlePos.fY = y0;
         fYAxisTitlePos.fX = x1;
         fYAxisTitlePos.fY = y1;
         break;
      case 2:
         fXAxisTitlePos.fX = x0;
         fXAxisTitlePos.fY = y1;
         fYAxisTitlePos.fX = x1;
         fYAxisTitlePos.fY = y0;
         break;
      case 3:
         fXAxisTitlePos.fX = x1;
         fXAxisTitlePos.fY = y1;
         fYAxisTitlePos.fX = x0;
         fYAxisTitlePos.fY = y0;
         break;
   }

   // move title 5% over the axis length
   Float_t off = 0.05;
   Float_t tOffX = (x1-x0) * off; if (fYAxisTitlePos.fX > x0) tOffX = -tOffX;
   Float_t tOffY = (y1-y0) * off; if (fXAxisTitlePos.fY > y0) tOffY = -tOffY;
   fXAxisTitlePos.fX += tOffX;
   fYAxisTitlePos.fY += tOffY;


   // frame box
   //
   if (fM->fBoxMode)
   {
      // get corner closest to eye excluding left corner
      Double_t zm = 1.f;
      Int_t idxDepthT = 0;
      for (Int_t i = 0; i < 4; ++i)
      {
         if (projZ[i] < zm && projZ[i] >= zt && i != idxFront )
         {
            zm  = projZ[i];
            idxDepthT = i;
         }
      }
      if (idxFront == idxLeft)  idxFront =idxDepthT;

      switch (idxFront)
      {
         case 0:
            fBackPlaneXConst[0].Set(x1, y0, 0); fBackPlaneXConst[1].Set(x1, y1, 0);
            fBackPlaneYConst[0].Set(x0, y1, 0); fBackPlaneYConst[1].Set(x1, y1, 0);
            break;
         case 1:
            fBackPlaneXConst[0].Set(x0, y0, 0); fBackPlaneXConst[1].Set(x0, y1, 0);
            fBackPlaneYConst[0].Set(x0, y1, 0); fBackPlaneYConst[1].Set(x1, y1, 0);
            break;
         case 2:
            fBackPlaneXConst[0].Set(x0, y0, 0); fBackPlaneXConst[1].Set(x0, y1, 0);
            fBackPlaneYConst[0].Set(x0, y0, 0); fBackPlaneYConst[1].Set(x1, y0, 0);
            break;
         case 3:
            fBackPlaneXConst[0].Set(x1, y0, 0); fBackPlaneXConst[1].Set(x1, y1, 0);
            fBackPlaneYConst[0].Set(x0, y0, 0); fBackPlaneYConst[1].Set(x1, y0, 0);
            break;
      }
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawAxis3D(TGLRnrCtx & rnrCtx) const
{
   // Draw z-axis and z-box at the appropriate grid corner-point including
   // tick-marks and labels.

   // set font size first depending on size of projected axis

   TGLMatrix mm;
   GLdouble pm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX, mm.Arr());
   glGetDoublev(GL_PROJECTION_MATRIX, pm);
   glGetIntegerv(GL_VIEWPORT, vp);

   GLdouble dn[3];
   GLdouble up[3];
   gluProject(fZAxisTitlePos.fX, fZAxisTitlePos.fY, 0                , mm.Arr(), pm, vp, &dn[0], &dn[1], &dn[2]);
   gluProject(fZAxisTitlePos.fX, fZAxisTitlePos.fY, fZAxisTitlePos.fZ, mm.Arr(), pm, vp, &up[0], &up[1], &up[2]);
   Double_t len = TMath::Sqrt((up[0] - dn[0]) * (up[0] - dn[0])
                              + (up[1] - dn[1]) * (up[1] - dn[1])
                              + (up[2] - dn[2]) * (up[2] - dn[2]));

   TGLVertex3 worldRef(fZAxisTitlePos.fX, fZAxisTitlePos.fY, fZAxisTitlePos.fZ);

   fAxisPainter.RefTMOff(0) = rnrCtx.RefCamera().ViewportDeltaToWorld(worldRef, -len, 0,  &mm);
   fAxisPainter.SetLabelPixelFontSize(TMath::Nint(len*fM->GetData()->GetEtaBins()->GetLabelSize()));
   fAxisPainter.SetTitlePixelFontSize(TMath::Nint(len*fM->GetData()->GetEtaBins()->GetTitleSize()));

   // Z axis
   //
   if (fM->fData->Empty() == kFALSE)
   {
      fZAxis->SetAxisColor(fGridColor);
      fZAxis->SetLabelColor(fFontColor);
      fZAxis->SetTitleColor(fFontColor);
      fZAxis->SetNdivisions(fM->fNZSteps*100 + 10);
      fZAxis->SetLimits(0, fDataMax);
      fZAxis->SetTitle(fM->GetPlotEt() ? "Et[GeV]" : "E[GeV]");

      fAxisPainter.SetTMNDim(1);
      fAxisPainter.RefDir().Set(0., 0., 1.);
      fAxisPainter.SetLabelAlign(TGLFont::kRight, TGLFont::kCenterV);
      glPushMatrix();
      glTranslatef(fZAxisTitlePos.fX, fZAxisTitlePos.fY, 0);

      // tickmark vector = 10 pixels left
      fAxisPainter.RefTitlePos().Set(fAxisPainter.RefTMOff(0).X()*0.05,  fAxisPainter.RefTMOff(0).Y()*0.05, fZAxisTitlePos.fZ);
      fZAxis->SetLabelOffset(0.05);
      fZAxis->SetTickLength(0.05);
      fAxisPainter.PaintAxis(rnrCtx, fZAxis);
      glTranslated( fAxisPainter.RefTMOff(0).X(),  fAxisPainter.RefTMOff(0).Y(),  fAxisPainter.RefTMOff(0).Z());
      glPopMatrix();

      // draw box frame
      //
      if (fM->fBoxMode) {

         glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);

         // box verticals
         TGLUtil::LineWidth(1);
         glBegin(GL_LINES);
         TGLUtil::Color(fGridColor);

         glVertex3f(fBackPlaneXConst[0].fX   ,fBackPlaneXConst[0].fY   ,0);
         glVertex3f(fBackPlaneXConst[0].fX   ,fBackPlaneXConst[0].fY   ,fDataMax);
         glVertex3f(fBackPlaneXConst[1].fX   ,fBackPlaneXConst[1].fY   ,0);
         glVertex3f(fBackPlaneXConst[1].fX   ,fBackPlaneXConst[1].fY   ,fDataMax);


         glVertex3f(fBackPlaneYConst[0].fX   ,fBackPlaneYConst[0].fY   ,0);
         glVertex3f(fBackPlaneYConst[0].fX   ,fBackPlaneYConst[0].fY   ,fDataMax);
         glVertex3f(fBackPlaneYConst[1].fX   ,fBackPlaneYConst[1].fY   ,0);
         glVertex3f(fBackPlaneYConst[1].fX   ,fBackPlaneYConst[1].fY   ,fDataMax);

         // box top
         glVertex3f(fBackPlaneXConst[0].fX   ,fBackPlaneXConst[0].fY   ,fDataMax);
         glVertex3f(fBackPlaneXConst[1].fX   ,fBackPlaneXConst[1].fY   ,fDataMax);
         glVertex3f(fBackPlaneYConst[0].fX   ,fBackPlaneYConst[0].fY   ,fDataMax);
         glVertex3f(fBackPlaneYConst[1].fX   ,fBackPlaneYConst[1].fY   ,fDataMax);

         glEnd();

         // box horizontals stippled
         glEnable(GL_LINE_STIPPLE);
         Int_t ondiv;
         Double_t omin, omax, bw1;
         THLimitsFinder::Optimize(0, fDataMax, fM->fNZSteps, omin, omax, ondiv, bw1);

         glLineStipple(1, 0x5555);
         glBegin(GL_LINES);
         Float_t hz  = bw1;
         for (Int_t i = 1; i <= ondiv; ++i, hz += bw1) {
            glVertex3f(fBackPlaneXConst[0].fX   ,fBackPlaneXConst[0].fY   ,hz);
            glVertex3f(fBackPlaneXConst[1].fX   ,fBackPlaneXConst[1].fY   ,hz);
            glVertex3f(fBackPlaneYConst[0].fX   ,fBackPlaneYConst[0].fY   ,hz);
            glVertex3f(fBackPlaneYConst[1].fX   ,fBackPlaneYConst[1].fY   ,hz);
         }
         glEnd();

         glPopAttrib();
      }
   }

   // XY Axis
   //

   Float_t yOff  = fM->GetPhiRng();
   if (fXAxisTitlePos.fY < fM->GetPhiMax()) yOff = -yOff;

   Float_t xOff  = fM->GetEtaRng();
   if (fYAxisTitlePos.fX < fM->GetEtaMax()) xOff = -xOff;

   TAxis ax;
   ax.SetAxisColor(fGridColor);
   ax.SetLabelColor(fFontColor);
   ax.SetTitleColor(fFontColor);
   ax.SetTitleFont(fM->GetData()->GetEtaBins()->GetTitleFont());
   ax.SetLabelOffset(0.02);
   ax.SetTickLength(0.05);
   fAxisPainter.SetTMNDim(2);
   fAxisPainter.RefTMOff(1).Set(0, 0, -fDataMax);
   fAxisPainter.SetLabelAlign(TGLFont::kCenterH, TGLFont::kBottom);

   // eta
   glPushMatrix();
   fAxisPainter.RefDir().Set(1, 0, 0);
   fAxisPainter.RefTMOff(0).Set(0, yOff, 0);
   glTranslatef(0, fXAxisTitlePos.fY, 0);
   ax.SetNdivisions(710);
   ax.SetLimits(fM->GetEtaMin(), fM->GetEtaMax());
   ax.SetTitle(fM->GetData()->GetEtaBins()->GetTitle());
   fAxisPainter.RefTitlePos().Set(fXAxisTitlePos.fX, yOff*1.5*ax.GetTickLength(), -fDataMax*ax.GetTickLength());
   fAxisPainter.PaintAxis(rnrCtx, &ax);
   glPopMatrix();

   // phi
   fAxisPainter.RefDir().Set(0, 1, 0);
   fAxisPainter.RefTMOff(0).Set(xOff, 0, 0);
   ax.SetNdivisions(510);
   ax.SetLimits(fM->GetPhiMin(), fM->GetPhiMax());
   ax.SetTitle(fM->GetData()->GetPhiBins()->GetTitle());
   glPushMatrix();
   glTranslatef(fYAxisTitlePos.fX, 0, 0);
   fAxisPainter.RefTitlePos().Set( xOff*1.5*ax.GetTickLength(), fYAxisTitlePos.fY,  -fDataMax*ax.GetTickLength());
   fAxisPainter.PaintAxis(rnrCtx, &ax);
   glPopMatrix();

} // DrawAxis3D

//______________________________________________________________________________
void TEveCaloLegoGL::DrawAxis2D(TGLRnrCtx & rnrCtx) const
{
   // Draw XY axis.

   TAxis ax;
   ax.SetAxisColor(fGridColor);
   ax.SetLabelColor(fFontColor);
   ax.SetTitleColor(fFontColor);
   ax.SetTitleFont(fM->GetData()->GetEtaBins()->GetTitleFont());
   ax.SetLabelOffset(0.01);
   ax.SetTickLength(0.05);

   // set fonts
   fAxisPainter.SetAttAxis(&ax);

   // get projected length of diagonal to determine
   TGLMatrix mm;
   GLdouble pm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX, mm.Arr());
   glGetDoublev(GL_PROJECTION_MATRIX, pm);
   glGetIntegerv(GL_VIEWPORT, vp);

   GLdouble dn[3];
   GLdouble up[3];
   gluProject(fM->GetEtaMin(), fM->GetPhiMin(), 0, mm.Arr(), pm, vp, &dn[0], &dn[1], &dn[2]);
   gluProject(fM->GetEtaMax(), fM->GetPhiMax(), 0, mm.Arr(), pm, vp, &up[0], &up[1], &up[2]);
   Double_t len = TMath::Sqrt((up[0] - dn[0]) * (up[0] - dn[0])
                              + (up[1] - dn[1]) * (up[1] - dn[1])
                              + (up[2] - dn[2]) * (up[2] - dn[2]));

   // eta
   fAxisPainter.SetLabelPixelFontSize(TMath::Nint(len*fM->GetData()->GetEtaBins()->GetLabelSize()));
   fAxisPainter.SetTitlePixelFontSize(TMath::Nint(len*fM->GetData()->GetEtaBins()->GetTitleSize()));
   ax.SetNdivisions(710);
   ax.SetLimits(fM->GetEtaMin(), fM->GetEtaMax());
   ax.SetTitle(fM->GetData()->GetEtaBins()->GetTitle());
   fAxisPainter.RefTitlePos().Set(fM->GetEtaMax(), -fM->GetPhiRng()*(ax.GetTickLength()+ ax.GetLabelOffset()), 0 );
   fAxisPainter.RefDir().Set(1, 0, 0);
   fAxisPainter.RefTMOff(0).Set(0,  -fM->GetPhiRng(), 0);
   fAxisPainter.SetLabelAlign(TGLFont::kCenterH, TGLFont::kBottom);

   glPushMatrix();
   glTranslatef(0, fM->GetPhiMin(), 0);
   fAxisPainter.PaintAxis(rnrCtx, &ax);
   glPopMatrix();

   // phi
   ax.SetNdivisions(510);
   ax.SetLimits(fM->GetPhiMin(), fM->GetPhiMax());
   ax.SetTitle(fM->GetData()->GetPhiBins()->GetTitle());
   fAxisPainter.RefTitlePos().Set(-fM->GetEtaRng()*(ax.GetTickLength()+ ax.GetLabelOffset()), fM->GetPhiMax(), 0);
   fAxisPainter.RefDir().Set(0, 1, 0);
   fAxisPainter.RefTMOff(0).Set(-fM->GetEtaRng(), 0, 0);
   fAxisPainter.SetLabelAlign(TGLFont::kRight, TGLFont::kCenterV);

   glPushMatrix();
   glTranslatef(fM->GetEtaMin(), 0, 0);
   fAxisPainter.PaintAxis(rnrCtx, &ax);
   glPopMatrix();
}

//______________________________________________________________________________
Int_t TEveCaloLegoGL::GetGridStep(TGLRnrCtx &rnrCtx) const
{
   // Calculate view-dependent grid density.

   using namespace TMath;

   GLdouble x0, y0, z0, x1, y1, z1;
   GLdouble mm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetIntegerv(GL_VIEWPORT, vp);
   const GLdouble *pmx = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();

   GLdouble em, eM, pm, pM;
   fM->GetData()->GetEtaLimits(pm, pM);
   fM->GetData()->GetPhiLimits(em, eM);
   gluProject(em, pm, 0.f , mm, pmx, vp, &x0, &y0, &z0);
   gluProject(eM, pM, 0.f , mm, pmx, vp, &x1, &y1, &z1);
   Float_t d0 = Sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1));

   gluProject(em, pm, 0.f , mm, pmx, vp, &x0, &y0, &z0);
   gluProject(eM, pM, 0.f , mm, pmx, vp, &x1, &y1, &z1);
   Float_t d1 = Sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1));

   Float_t d = d1 > d0 ? d1 : d0;
   Int_t i0 = fM->fData->GetEtaBins()->FindBin(fM->GetEtaMin());
   Int_t i1 = fM->fData->GetEtaBins()->FindBin(fM->GetEtaMax());
   Int_t j0 = fM->fData->GetPhiBins()->FindBin(fM->GetPhiMin());
   Int_t j1 = fM->fData->GetPhiBins()->FindBin(fM->GetPhiMax());

   Float_t pixelsPerBin = 0.5 * d / Sqrt((i0 - i1) * (i0 - i1) + (j0 - j1) * (j0 - j1));
   Int_t ngroup = 1;
   if (fM->fAutoRebin && fM->fPixelsPerBin > pixelsPerBin)
   {
      ngroup = Nint(fM->fPixelsPerBin*0.5/pixelsPerBin);
      Int_t minN = Min(fM->fData->GetEtaBins()->GetNbins(), fM->fData->GetPhiBins()->GetNbins());
      if (ngroup * 4 > minN)
      {
         ngroup = minN/4;
      }
   }

   return ngroup;
}

//___________________________________________________________________________
void TEveCaloLegoGL::RebinAxis(TAxis *orig, TAxis *curr) const
{
   // Rebin eta, phi axis.

   Double_t center = 0.5 * (orig->GetXmin() + orig->GetXmax());
   Int_t idx0 = orig->FindBin(center);
   Double_t bc = orig->GetBinCenter(idx0);
   if (bc > center) idx0--;

   Int_t nbR = TMath::FloorNint(idx0/fM->fBinStep) + TMath::FloorNint((orig->GetNbins() - idx0)/fM->fBinStep);
   Double_t *bins = new Double_t[nbR+1];
   Int_t off = idx0 - TMath::FloorNint(idx0/fM->fBinStep)*fM->fBinStep;
   for(Int_t i = 0; i <= nbR; i++)
      bins[i] = orig->GetBinUpEdge(off + i*fM->fBinStep);

   curr->Set(nbR, bins);
   delete [] bins;
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawHistBase(TGLRnrCtx &rnrCtx) const
{
   // Draw basic histogram components: x-y grid

   Float_t eta0 = fM->fEtaMin;
   Float_t eta1 = fM->fEtaMax;
   Float_t phi0 = fM->GetPhiMin();
   Float_t phi1 = fM->GetPhiMax();

   // XY grid
   //
   TGLUtil::Color(fGridColor);
   TGLUtil::LineWidth(1);
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
   Float_t val;
   Int_t neb = fEtaAxis->GetNbins();
   for (Int_t i = 0; i<= neb; i++)
   {
      val = fEtaAxis->GetBinUpEdge(i);
      if (val > eta0 && val < eta1 )
      {
         glVertex2f(val, phi0);
         glVertex2f(val, phi1);
      }
   }

   // phi grid
   Int_t npb = fPhiAxis->GetNbins();
   for (Int_t i = 1; i <= npb; i++) {
       val = fPhiAxis->GetBinUpEdge(i);
      if (val > phi0 && val < phi1)
      {
         glVertex2f(eta0, val);
         glVertex2f(eta1, val);
      }
   }

   glEnd();

   // XYZ axes
   //
   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POLYGON_BIT);
   TGLUtil::LineWidth(2);
   if (fCells3D)
   {
      SetAxis3DTitlePos(rnrCtx, eta0, eta1, phi0, phi1);
      DrawAxis3D(rnrCtx);
   }
   else
   {
      DrawAxis2D(rnrCtx);
   }
   glPopAttrib();
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawCells3D(TGLRnrCtx & rnrCtx) const
{
   // Render the calo lego-plot with OpenGL.

   // quads
   {
      for (SliceDLMap_i i = fDLMap.begin(); i != fDLMap.end(); ++i) {
         TGLUtil::Color(fM->GetDataSliceColor(i->first));
         glCallList(i->second);
      }
   }
   // outlines
   {
      if (rnrCtx.SceneStyle() == TGLRnrCtx::kFill) {
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
         glDisable(GL_POLYGON_OFFSET_FILL);
         TGLUtil::Color(1);
         for (SliceDLMap_i i = fDLMap.begin(); i != fDLMap.end(); ++i)
            glCallList(i->second);
      }
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::PrepareCell2DData(TEveCaloData::vCellId_t& cellList, vCell2D_t& cells2D) const
{
   // Prepare cells 2D data non-rebinned for drawing.

   Int_t   max_energy_slice, cellID=0;
   Float_t sum, max_energy;

   TEveCaloData::vCellId_t::iterator currentCell = cellList.begin();
   TEveCaloData::vCellId_t::iterator nextCell    = currentCell;
   ++nextCell;

   while (currentCell != cellList.end()) {
      TEveCaloData::CellData_t currentCellData;
      TEveCaloData::CellData_t nextCellData;

      fM->fData->GetCellData(*currentCell, currentCellData);
      sum = max_energy = currentCellData.Value(fM->fPlotEt);
      max_energy_slice = currentCell->fSlice;
      while (nextCell != cellList.end() && currentCell->fTower == nextCell->fTower) {
         fM->fData->GetCellData(*nextCell, nextCellData);
         Float_t energy = nextCellData.Value(fM->fPlotEt);
         sum += energy;
         if (energy > max_energy) {
            max_energy       = energy;
            max_energy_slice = nextCell->fSlice;
         }
         ++nextCell;
         ++cellID;
      }

      WrapTwoPi(currentCellData.fPhiMin, currentCellData.fPhiMax);
      cells2D.push_back(Cell2D_t(cellID, sum, max_energy_slice));
      cells2D.back().SetGeom(currentCellData.fEtaMin, currentCellData.fPhiMin,
                             currentCellData.fEtaMax, currentCellData.fPhiMax);

      currentCell = nextCell;
      ++nextCell;
      ++cellID;
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::PrepareCell2DDataRebin(TEveCaloData::RebinData_t& rebinData, vCell2D_t& cells2D) const
{
   // Prepare cells 2D rebinned data for drawing.

   const Int_t nEta = fEtaAxis->GetNbins();
   const Int_t nPhi = fPhiAxis->GetNbins();
   std::vector<Float_t> vec;
   vec.assign((nEta + 2)*(nPhi + 2), 0.f);
   std::vector<Float_t> max_e;
   std::vector<Int_t>   max_e_slice;
   max_e.assign((nEta + 2) * (nPhi + 2), 0.f);
   max_e_slice.assign((nEta + 2) * (nPhi + 2), -1);

   for (UInt_t bin = 0; bin < rebinData.fBinData.size(); ++bin) {
      Float_t ssum = 0;
      if (rebinData.fBinData[bin] != -1) {
         Float_t *val = rebinData.GetSliceVals(bin);
         for (Int_t s = 0; s < rebinData.fNSlices; ++s) {
            ssum += val[s];
            if (val[s] > max_e[bin]) {
               max_e[bin]       = val[s];
               max_e_slice[bin] = s;
            }
         }
      }
      vec[bin] = ssum;
   }

   // smallest threshold
   Float_t threshold = fM->GetDataSliceThreshold(0);
   for (Int_t s = 1; s < fM->fData->GetNSlices(); ++s) {
      if (threshold > fM->GetDataSliceThreshold(s))
         threshold = fM->GetDataSliceThreshold(s);
   }

   // write cells
   for (Int_t i = 1; i <= fEtaAxis->GetNbins(); ++i) {
      for (Int_t j = 1; j <= fPhiAxis->GetNbins(); ++j) {
         const Int_t bin = j * (nEta + 2) + i;
         if (vec[bin] > threshold && rebinData.fBinData[bin] != -1) {
            cells2D.push_back(Cell2D_t(bin, vec[bin], max_e_slice[bin]));
            cells2D.back().SetGeom(fEtaAxis->GetBinLowEdge(i), fPhiAxis->GetBinLowEdge(j),
                                   fEtaAxis->GetBinUpEdge(i),  fPhiAxis->GetBinUpEdge(j));
         }
      }
   }
}

//--------------------------------------------------------------------------------
void TEveCaloLegoGL::DrawCells2D(TGLRnrCtx &rnrCtx, vCell2D_t& cells2D) const
{
   // Draw cells in top view.

   Float_t bws = -1; //smallest bin
   Float_t logMax = -1;

   if (fM->f2DMode == TEveCaloLego::kValColor ) {
      fM->AssertPalette();
      UChar_t col[4];

      for (UInt_t i=0; i < cells2D.size(); i++) {
         if (rnrCtx.SecSelection()) glLoadName(cells2D[i].fId);
         glBegin(GL_POLYGON);

         Float_t val = cells2D[i].fSumVal;
         fM->fPalette->ColorFromValue(TMath::FloorNint(val), col);
         TGLUtil::Color4ubv(col);
         glVertex3f(cells2D[i].fGeom[0], cells2D[i].fGeom[1], val);
         glVertex3f(cells2D[i].fGeom[2], cells2D[i].fGeom[1], val);
         glVertex3f(cells2D[i].fGeom[2], cells2D[i].fGeom[3], val);
         glVertex3f(cells2D[i].fGeom[1], cells2D[i].fGeom[3], val);
         glEnd();
      }
   }
   else {
      Float_t x, y, val;
      if (!rnrCtx.HighlightOutline())
      {
         bws = 1e5;
         for (UInt_t i=0; i< cells2D.size(); i++) {
            if ( cells2D[i].MinSize() < bws)   bws = cells2D[i].MinSize();
         }
         bws *= 0.5;

         Float_t maxv = 0;
         glBegin(GL_POINTS);
         for (UInt_t i=0; i< cells2D.size(); i++) {
            TGLUtil::Color(fM->fData->GetSliceColor(cells2D[i].fMaxSlice));
            glVertex3f(cells2D[i].X(), cells2D[i].Y() , cells2D[i].fSumVal);
            if (cells2D[i].fSumVal > maxv) maxv = cells2D[i].fSumVal;
         }
         glEnd();
         logMax = TMath::Log10(maxv + 1);
         fValToPixel =  bws/logMax;
      }

      // draw
      for (UInt_t i=0; i< cells2D.size(); i++) {
         if (rnrCtx.SecSelection())
         {
            glLoadName(cells2D[i].fMaxSlice);
            glPushName(0);
            glLoadName(cells2D[i].fId);
         }
         glBegin(GL_POLYGON);
         TGLUtil::Color(fM->fData->GetSliceColor(cells2D[i].fMaxSlice));
         val = cells2D[i].fSumVal;
         Float_t bw = fValToPixel*TMath::Log10(val+1);
         x = cells2D[i].X();
         y = cells2D[i].Y();
         glVertex3f(x - bw, y - bw, val);
         glVertex3f(x + bw, y - bw, val);
         glVertex3f(x + bw, y + bw, val);
         glVertex3f(x - bw, y + bw, val);
         glEnd();
         if (rnrCtx.SecSelection()) glPopName();
      }
   }

   if (rnrCtx.Selection() || rnrCtx.Highlight() || rnrCtx.HighlightOutline() ) return;

   // text
   TGLMatrix mm;
   GLdouble pm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX, mm.Arr());
   glGetDoublev(GL_PROJECTION_MATRIX, pm);
   glGetIntegerv(GL_VIEWPORT, vp);

   GLdouble dn[3];
   GLdouble up[3];
   gluProject(fM->GetEtaMin(), 0, 0, mm.Arr(), pm, vp, &dn[0], &dn[1], &dn[2]);
   gluProject(fM->GetEtaMax(), 0, 0, mm.Arr(), pm, vp, &up[0], &up[1], &up[2]);
   Double_t etaLenPix = up[0]-dn[0];
   Float_t sx = etaLenPix/fM->GetEtaRng();

   TGLUtil::Color(rnrCtx.ColorSet().Markup().GetColorIndex());
   TGLFont font;
   Double_t cs;
   rnrCtx.RegisterFontNoScale(fM->fCellPixelFontSize, "arial", TGLFont::kPixmap, font);

   for (UInt_t i=0; i< cells2D.size(); i++) {
      if (fM->f2DMode == TEveCaloLego::kValColor )
         cs = cells2D[i].MinSize();
      else
         cs = fValToPixel*TMath::Log10(cells2D[i].fSumVal+1);

      if (cs*sx >  fM->fDrawNumberCellPixels) {
         // can not use same format as for axis
         // space on top of towers is limited
         const char* txt;
         Float_t val = cells2D[i].fSumVal;
         if (val > 10)
            txt = Form("%d", TMath::Nint(val));
         else if (val > 1 )
            txt = Form("%.1f", val);
         else if (val > 0.01 )
            txt = Form("%.2f", 0.01*TMath::Nint(val*100));
         else
            txt = Form("~1e%d", TMath::Nint(TMath::Log10(val)));

         font.Render(txt, cells2D[i].X(), cells2D[i].Y(), val*1.2, TGLFont::kCenterH, TGLFont::kCenterV);
      }
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawHighlight(TGLRnrCtx& rnrCtx, const TGLPhysicalShape* pshp) const
{
   // Draw eta-phi range in highlight mode.

   if ( !fM->fData->GetCellsSelected().size()) return;
   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POLYGON_BIT );
   glDisable(GL_LIGHTING);
   glDisable(GL_CULL_FACE);
   glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

   TGLUtil::LineWidth(2);
   glColor4ubv(rnrCtx.ColorSet().Selection(pshp->GetSelected()).CArr());
   rnrCtx.SetHighlightOutline(kTRUE);
   TGLUtil::LockColor();

   // modelview matrix
   Double_t em, eM, pm, pM;
   fM->fData->GetEtaLimits(em, eM);
   fM->fData->GetPhiLimits(pm, pM);
   Double_t unit = ((eM - em) < (pM - pm)) ? (eM - em) : (pM - pm);
   glPushMatrix();
   Float_t sx = (eM - em) / fM->GetEtaRng();
   Float_t sy = (pM - pm) / fM->GetPhiRng();
   glScalef(sx / unit, sy / unit, fM->fData->Empty() ? 1 : fM->GetValToHeight());
   glTranslatef(-fM->GetEta(), -fM->fPhi, 0);

   // prepare rebin
   TEveCaloData::RebinData_t rebinData;
   if (fM->fBinStep> 1)  {
      fM->fData->Rebin(fEtaAxis, fPhiAxis, fM->fData->GetCellsSelected(), fM->fPlotEt, rebinData);

      Float_t scale = fM->GetMaxVal() / fMaxValRebin;
      if (fM->fNormalizeRebin) {
         for (std::vector<Float_t>::iterator it = rebinData.fSliceData.begin(); it != rebinData.fSliceData.end(); it++)
            (*it) *= scale;
      }
   }
   if (fCells3D)
   {
      if (fM->fBinStep == 1)
         Make3DDisplayList(fM->fData->GetCellsSelected(), fDLMapSelected, kFALSE);
      else
      {
         Make3DDisplayListRebin(rebinData, fDLMapSelected, kFALSE);
      }
      for (SliceDLMap_i i = fDLMapSelected.begin(); i != fDLMapSelected.end(); ++i) {
         TGLUtil::Color(fM->GetDataSliceColor(i->first));
         glCallList(i->second);
      }
   }
   else
   {
      vCell2D_t cells2D;
      if (fM->fBinStep == 1)
         PrepareCell2DData(fM->fData->GetCellsSelected(), cells2D);
      else
         PrepareCell2DDataRebin(rebinData, cells2D);

      DrawCells2D(rnrCtx, cells2D);
   }

   TGLUtil::UnlockColor();
   rnrCtx.SetHighlightOutline(kFALSE);
   glPopMatrix();
   glPopAttrib();
}

//______________________________________________________________________________
void TEveCaloLegoGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // Draw the object.

   if (! fM->fData || ! fM->fData->GetEtaBins() || ! fM->fData->GetPhiBins())
      return;

   // projection type
   if (fM->fProjection == TEveCaloLego::kAuto)
      fCells3D = (!(rnrCtx.RefCamera().IsOrthographic() && rnrCtx.RefCamera().GetCamBase().GetBaseVec(1).Z()));
   else if (fM->fProjection == TEveCaloLego::k2D)
      fCells3D = kFALSE;
   else if (fM->fProjection == TEveCaloLego::k3D)
      fCells3D = kTRUE;

   // cache max val
   fDataMax = fM->GetMaxVal();

   // modelview matrix
   Double_t em, eM, pm, pM;
   fM->fData->GetEtaLimits(em, eM);
   fM->fData->GetPhiLimits(pm, pM);
   Double_t unit = ((eM - em) < (pM - pm)) ? (eM - em) : (pM - pm);
   glPushMatrix();
   Float_t sx = (eM - em) / fM->GetEtaRng();
   Float_t sy = (pM - pm) / fM->GetPhiRng();
   glScalef(sx / unit, sy / unit, fM->fData->Empty() ? 1 : fM->GetValToHeight());
   glTranslatef(-fM->GetEta(), -fM->fPhi, 0);

   // rebin axsis , check limits, fix TwoPi cycling
   Int_t oldBinStep = fM->fBinStep;
   fM->fBinStep = GetGridStep(rnrCtx);
   if (oldBinStep != fM->fBinStep) fDLCacheOK=kFALSE;

   RebinAxis(fM->fData->GetEtaBins(), fEtaAxis);
   RebinAxis(fM->fData->GetPhiBins(), fPhiAxis);

   // cache ids
   Bool_t idCacheChanged = kFALSE;
   if (fM->fCellIdCacheOK == kFALSE) {
      fM->BuildCellIdCache();
      idCacheChanged = kTRUE;
   }

   // rebin data
   if (fDLCacheOK==kFALSE || idCacheChanged ) {
      fRebinData.fSliceData.clear();
      fRebinData.fSliceData.clear();

      if (fM->fBinStep > 1) {
         fM->fData->Rebin(fEtaAxis, fPhiAxis, fM->fCellList, fM->fPlotEt, fRebinData);
         if (fM->fNormalizeRebin) {
            //  Double_t maxVal = 0;
            fMaxValRebin = 0;
            for (UInt_t i = 0; i < fRebinData.fSliceData.size(); i += fRebinData.fNSlices) {
               Double_t sum = 0;
               for (Int_t s = 0; s < fRebinData.fNSlices; s++)
                  sum += fRebinData.fSliceData[i+s];

               if (sum > fMaxValRebin) fMaxValRebin = sum;
            }

            Float_t scale = fM->GetMaxVal() / fMaxValRebin;
            for (std::vector<Float_t>::iterator it = fRebinData.fSliceData.begin(); it != fRebinData.fSliceData.end(); it++)
               (*it) *= scale;
         }
      }
   }

   fFontColor = fM->fFontColor;
   fGridColor = fM->fGridColor;
   if (fGridColor < 0 || fFontColor < 0)
   {
      TColor* c1 = gROOT->GetColor(rnrCtx.ColorSet().Markup().GetColorIndex());
      TColor* c2 = gROOT->GetColor(rnrCtx.ColorSet().Background().GetColorIndex());
      Float_t f1, f2;
      if (fFontColor < 0) {
         f1 = 0.8; f2 = 0.2;
         fFontColor = TColor::GetColor(c1->GetRed()  *f1  + c2->GetRed()  *f2,
                                       c1->GetGreen()*f1  + c2->GetGreen()*f2,
                                       c1->GetBlue() *f1  + c2->GetBlue() *f2);
      }
      if (fGridColor < 0) {
         f1 = 0.3; f2 = 0.3;
         fGridColor = TColor::GetColor(c1->GetRed()  *f1  + c2->GetRed()  *f2,
                                       c1->GetGreen()*f1  + c2->GetGreen()*f2,
                                       c1->GetBlue() *f1  + c2->GetBlue() *f2);
      }
   }

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POLYGON_BIT);
   TGLUtil::LineWidth(1);

   if (!fM->fData->Empty()){
      glPushName(0);
      glLoadName(0);
      if (fCells3D) {
         if (fDLCacheOK == kFALSE || idCacheChanged )
         {
            if (fM->fBinStep == 1)
               Make3DDisplayList(fM->fCellList, fDLMap, kTRUE);
            else
               Make3DDisplayListRebin(fRebinData, fDLMap, kTRUE);
            fDLCacheOK = kTRUE;
         }
         glEnable(GL_NORMALIZE);
         glEnable(GL_POLYGON_OFFSET_FILL);
         glPolygonOffset(0.8, 1);

         DrawCells3D(rnrCtx);
      } else {
         glDisable(GL_LIGHTING);
         vCell2D_t cells2D;
         if (fM->fBinStep == 1)
            PrepareCell2DData(fM->fCellList, cells2D);
         else
            PrepareCell2DDataRebin(fRebinData, cells2D);

         DrawCells2D(rnrCtx, cells2D);
      }
      glPopName();;
   }

   // draw histogram base
   if (rnrCtx.Selection() == kFALSE) {
      glDisable(GL_LIGHTING);
      DrawHistBase(rnrCtx);
      if (fM->fDrawHPlane) {
         glEnable(GL_BLEND);
         glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
         glDisable(GL_CULL_FACE);
         TGLUtil::ColorTransparency(fM->fPlaneColor, fM->fPlaneTransparency);
         Float_t zhp = fM->fHPlaneVal * fDataMax;
         glBegin(GL_POLYGON);
         glVertex3f(fM->fEtaMin, fM->GetPhiMin(), zhp);
         glVertex3f(fM->fEtaMax, fM->GetPhiMin(), zhp);
         glVertex3f(fM->fEtaMax, fM->GetPhiMax(), zhp);
         glVertex3f(fM->fEtaMin, fM->GetPhiMax(), zhp);
         glEnd();
      }
   }

   glPopAttrib();
   glPopMatrix();
}

//______________________________________________________________________________
void TEveCaloLegoGL::ProcessSelection(TGLRnrCtx & /*rnrCtx*/, TGLSelectRecord & rec)
{
   // Processes tower selection from TGLViewer.

   fM->fData->GetCellsSelected().clear();

   if (rec.GetN() > 1)
   {
      Int_t slice  = rec.GetItem(1);
      Int_t cellID = rec.GetItem(2);
      if (fM->fBinStep == 1)
      {
         fM->fData->GetCellsSelected().push_back(fM->fCellList[cellID]);
      }
      else  {
         if (cellID >0)
         {
            Int_t nEta   = fEtaAxis->GetNbins();
            Int_t phiBin = Int_t(cellID/(nEta+2));
            Int_t etaBin = cellID - phiBin*(nEta+2);
            TEveCaloData::vCellId_t sl;
            fM->fData->GetCellList(fEtaAxis->GetBinCenter(etaBin), fEtaAxis->GetBinWidth(etaBin),
                                   fPhiAxis->GetBinCenter(phiBin), fPhiAxis->GetBinWidth(phiBin),
                                   sl);

            for(TEveCaloData::vCellId_i it = sl.begin(); it != sl.end(); ++it)
            {
               if ((*it).fSlice == slice )fM->fData->GetCellsSelected().push_back(*it);
            }
         }
      }
   }

   fM->fData->DataChanged();
}
