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

   fEtaAxis(0),
   fPhiAxis(0),
   fBinStep(1),

   fXAxisAtt(),
   fYAxisAtt(),
   fZAxisAtt(),
   fAxisPainter(),

   fDLCacheOK(kFALSE),
   fCells3D(kTRUE),
   fM(0)
{
   // Constructor.

   fDLCache = kFALSE;

   // modes for different levels of zoom in/out.

   fXAxisAtt.SetTMNDim(2);
   fXAxisAtt.SetTextAlign(TGLFont::kCenterDown);
   fXAxisAtt.SetNdivisions(710);
   fXAxisAtt.SetLabelSize(0.03);
   fXAxisAtt.SetTitleSize(0.03);

   fYAxisAtt = fXAxisAtt;
   fYAxisAtt.RefDir().Set(0., 1., 0.);
   fYAxisAtt.SetNdivisions(510);

   fZAxisAtt.RefDir().Set(0., 0., 1.);
   fZAxisAtt.SetTextAlign(TGLFont::kLeft);
   fZAxisAtt.SetRelativeFontSize(kTRUE);
   fZAxisAtt.SetLabelSize(0.03);
   fZAxisAtt.SetTitle("Et");
   fZAxisAtt.SetTitleUnits("GeV");
   fZAxisAtt.SetTitleSize(0.03);

   fEtaAxis = new TAxis();
   fPhiAxis = new TAxis();
}

//______________________________________________________________________________
TEveCaloLegoGL::~TEveCaloLegoGL()
{
   // Destructor.

   DLCachePurge();
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

   if ( ! fDLMap.empty())
   {
      for (SliceDLMap_i i = fDLMap.begin(); i != fDLMap.end(); ++i)
      {
         if (i->second)
         {
            PurgeDLRange(i->second, 1);
            i->second = 0;
         }
      }
   }
   TGLObject::DLCachePurge();
}

//______________________________________________________________________________
Bool_t TEveCaloLegoGL::PhiShiftInterval(Float_t &min, Float_t &max) const
{
   if (fM->GetPhiMax()>TMath::Pi() && max<=fM->GetPhiMin())
   {
      min += TMath::TwoPi();
      max += TMath::TwoPi();
   }
   else if (fM->GetPhiMin()<-TMath::Pi() && min>=fM->GetPhiMax())
   {
      min -= TMath::TwoPi();
      max -= TMath::TwoPi();
   }

   return min>=fM->GetPhiMin() && max<=fM->GetPhiMax();
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

   if (fBinStep>1)
   {
      Int_t nSlices = fM->fData->GetNSlices();
      Float_t *vals;
      Int_t bin;
      Float_t offset;
      Float_t y0, y1;
      for (Int_t s = 0; s < nSlices; ++s)
      {
         if (fDLMap.empty() || fDLMap[s] == 0)
            fDLMap[s] = glGenLists(1);

         glNewList(fDLMap[s], GL_COMPILE);
         glLoadName(s);
         glPushName(0);
         for (Int_t i=1; i<=fEtaAxis->GetNbins(); ++i)
         {
            for (Int_t j=1; j<=fPhiAxis->GetNbins(); ++j)
            {
               bin = (i)+(j)*(fEtaAxis->GetNbins()+2);

               if (fRebinData.fBinData[bin] !=-1)
               {
                  vals = fRebinData.GetSliceVals(bin);
                  offset =0;
                  for (Int_t t=0; t<s; t++)
                     offset+=vals[t];

                  y0 = fPhiAxis->GetBinLowEdge(j);
                  y1 = fPhiAxis->GetBinUpEdge(j);
                  if (PhiShiftInterval(y0, y1))
                  {
                     glLoadName(bin);
                     MakeQuad(fEtaAxis->GetBinLowEdge(i), y0, offset,
                              fEtaAxis->GetBinWidth(i), y1-y0, vals[s]);
                  }
               }
            }
         }
         glPopName();
         glEndList();
      }
   }
   else
   {
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

            fM->fData->GetCellData(fM->fCellList[i], cellData);
            if (s == fM->fCellList[i].fSlice)
            {
               glLoadName(i);
               PhiShiftInterval(cellData.fPhiMin, cellData.fPhiMax);
               MakeQuad(cellData.EtaMin(), cellData.PhiMin(), offset,
                        cellData.EtaDelta(), cellData.PhiDelta(), cellData.Value(fM->fPlotEt));
            }
            offset += cellData.Value(fM->fPlotEt);
         }
         glEndList();
      }
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
   fZAxisAtt.SetLabelColor(fM->fFontColor);
   fZAxisAtt.SetTitleColor(fM->fFontColor);
   fZAxisAtt.SetRng(0, fDataMax);
   fZAxisAtt.SetNdivisions( fM->fNZSteps*100+10);
   fZAxisAtt.RefTMOff(0) = rnrCtx.RefCamera().ViewportDeltaToWorld(worldRef, -10, 0, &modview);
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
      Int_t ondiv;
      Double_t omin, omax, bw1;
      THLimitsFinder::Optimize(0, fDataMax, fM->fNZSteps, omin, omax, ondiv, bw1);

      glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
      glLineStipple(1, 0x5555);
      glEnable(GL_LINE_STIPPLE);
      glBegin(GL_LINES);
      Float_t hz  = bw1;
      for (Int_t i = 1; i <= ondiv; ++i, hz += bw1)
      {
         glVertex3f(x0, axY, hz); glVertex3f(x1, axY, hz);
         glVertex3f(ayX, y0, hz); glVertex3f(ayX, y1, hz);
      }
      glEnd();
      glPopAttrib();
   } // draw box

   /**************************************************************************/

   DrawZAxis(rnrCtx, azX,  azY);

   if (fM->fTowerPicked >= 0)
   {
      // left most corner of the picked tower
      TEveCaloData::CellData_t cd;
      fM->fData->GetCellData(fM->fCellList[fM->fTowerPicked], cd);
      PhiShiftInterval(cd.fPhiMin, cd.fPhiMax);
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

   TAxis* ax = fM->GetData()->GetEtaBins();
   const char* titleFontXY = TGLFontManager::GetFontNameFromId(ax->GetTitleFont());

   fXAxisAtt.SetTitleFontName(titleFontXY);
   fXAxisAtt.SetRng(x0, x1);
   fXAxisAtt.SetAxisColor(fM->fGridColor);
   fXAxisAtt.SetLabelColor(fM->fFontColor);
   fXAxisAtt.SetTitleColor(fM->fFontColor);
   fXAxisAtt.RefTMOff(0).Set(0, yOff, 0);
   fXAxisAtt.RefTMOff(1).Set(0, 0, zOff);
   fXAxisAtt.RefTitlePos().Set(axtX, 0, 0);
   fXAxisAtt.SetTitle(ax->GetTitle());
   glPushMatrix();
   glTranslatef(0, axY, 0);
   fAxisPainter.Paint(rnrCtx, fXAxisAtt);
   glPopMatrix();


   ax = fM->GetData()->GetPhiBins();
   fYAxisAtt.SetTitleFontName(titleFontXY);
   fYAxisAtt.SetRng(y0, y1);
   fYAxisAtt.SetAxisColor(fM->fGridColor);
   fYAxisAtt.SetLabelColor(fM->fFontColor);
   fYAxisAtt.SetTitleColor(fM->fFontColor);
   fYAxisAtt.RefTMOff(0).Set(xOff, 0, 0);
   fYAxisAtt.RefTMOff(1).Set(0, 0, zOff);
   fYAxisAtt.SetAbsLabelFontSize(fXAxisAtt.GetAbsLabelFontSize());
   fYAxisAtt.SetAbsTitleFontSize(fXAxisAtt.GetAbsTitleFontSize());
   fYAxisAtt.RefTitlePos().Set(0, aytY, 0);
   fYAxisAtt.SetTitle(ax->GetTitle());
   glPushMatrix();
   glTranslatef(ayX, 0, 0);
   fAxisPainter.Paint(rnrCtx, fYAxisAtt);
   glPopMatrix();

} // DrawXYScales

//______________________________________________________________________________
Int_t TEveCaloLegoGL::GetGridStep(TGLRnrCtx &rnrCtx) const
{
   // Calculate view-dependent grid density.

   if (!fM->fAutoRebin) return 1;

   using namespace TMath;

   GLdouble x0, y0, z0, x1, y1, z1;
   GLdouble mm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetIntegerv(GL_VIEWPORT, vp);
   const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();

   gluProject(fM->fEtaMin, fM->GetPhiMin(), 0.f , mm, pm, vp, &x0, &y0, &z0);
   gluProject(fM->fEtaMax, fM->GetPhiMax(), 0.f , mm, pm, vp, &x1, &y1, &z1);
   Float_t d0 = Sqrt((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1) + (z0-z1)*(z0-z1));

   gluProject(fM->fEtaMax, fM->GetPhiMin(), 0.f , mm, pm, vp, &x0, &y0, &z0);
   gluProject(fM->fEtaMin, fM->GetPhiMax(), 0.f , mm, pm, vp, &x1, &y1, &z1);
   Float_t d1 = Sqrt((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1) + (z0-z1)*(z0-z1));

   Float_t d = d1>d0? d1:d0;

   Int_t i0 = fM->fData->GetEtaBins()->FindBin(fM->GetEtaMin());
   Int_t i1 = fM->fData->GetEtaBins()->FindBin(fM->GetEtaMax());
   Int_t j0 = fM->fData->GetPhiBins()->FindBin(fM->GetPhiMin());
   Int_t j1 = fM->fData->GetPhiBins()->FindBin(fM->GetPhiMax());

   Float_t ppb = d/Sqrt((i0-i1)*(i0-i1)+(j0-j1)*(j0-j1));

   Int_t ngroup;
   if (ppb < fM->fPixelsPerBin*0.5)
   {
      ngroup = 4;
   }
   else if (ppb < fM->fPixelsPerBin)
   {
      ngroup = 2;
   }
   else
   {
      ngroup = 1;
   }

   return ngroup;
}

//___________________________________________________________________________
void TEveCaloLegoGL::SetAxis(TAxis *orig, TAxis *curr) const
{
   // Calculate view-dependent grid density.

   if(orig->GetXbins()->GetSize())
      curr->Set(orig->GetNbins(), orig->GetXbins()->GetArray());
   else
      curr->Set(orig->GetNbins(), orig->GetXmin(), orig->GetXmax());


   if (fBinStep>1)
   {
      Int_t nb = curr->GetNbins();
      Int_t newbins = nb/fBinStep;
      if(curr->GetXbins()->GetSize() > 0)
      {
         // variable bin sizes
         Double_t *bins = new Double_t[newbins+1];
         for(Int_t i = 0; i <= newbins; ++i)
            bins[i] = curr->GetBinLowEdge(1+i*fBinStep);
         curr->Set(newbins,bins);
         delete [] bins;
      }
      else
      {
         curr->Set(newbins, curr->GetXmin(), curr->GetXmax());
      }
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawHistBase(TGLRnrCtx &rnrCtx) const
{
   // Draw basic histogram components: x-y grid

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
   Int_t   eFirst = fEtaAxis->FindBin(eta0);
   Int_t   bin    = eFirst;
   Float_t val    = fEtaAxis->GetBinUpEdge(bin);
   while (val < eta1)
   {
      glVertex2f(val, phi0);
      glVertex2f(val, phi1);
      ++bin;
      val = fEtaAxis->GetBinUpEdge(bin);
   }

   // phi grid
   Float_t y0, y1;
   for (Int_t i=1; i<=fPhiAxis->GetNbins(); ++i)
   {
      y0 = fPhiAxis->GetBinUpEdge(i);
      y1 = fPhiAxis->GetBinUpEdge(i);
      if (PhiShiftInterval(y0, y1))
      {
         glVertex2f(eta0, y0);
         glVertex2f(eta1, y0);
      }
   }

   glEnd();

   // XYZ axes

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POLYGON_BIT);
   glLineWidth(2);

   if (!fM->fData->Empty())
   {
      if (fM->fProjection == TEveCaloLego::k3D || rnrCtx.RefCamera().GetCamBase().GetBaseVec(1).Z() == 0)
         DrawZScales3D(rnrCtx, eta0, eta1, phi0, phi1);

      if (fM->fProjection == TEveCaloLego::k2D || rnrCtx.RefCamera().GetCamBase().GetBaseVec(1).Z() != 0)
         fXAxisAtt.SetRelativeFontSize(kTRUE);
      else
         fXAxisAtt.SetRelativeFontSize(kFALSE);
   }

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
void TEveCaloLegoGL::DrawCells2D() const
{
   // Draw projected histogram.

   static const TEveException eh("TEveCaloLegoGL::DrawCells2D ");

   using namespace TMath;

   typedef std::vector<TEveVector>            vTEveVec_t;
   typedef std::vector<TEveVector>::iterator  vTEveVec_i;
   typedef std::vector<vTEveVec_t>           vvTEveVec_t;
   typedef std::vector<vTEveVec_t>::iterator vvTEveVec_i;

   // When kValSize is used, plot gl-points at tower-centers avoid flickering.
   vvTEveVec_t antiFlick(1);
   if (fM->f2DMode == TEveCaloLego::kValSize && fM->fTopViewUseMaxColor)
   {
      antiFlick.resize(fM->GetData()->GetNSlices());
   }

   fM->AssertPalette();

   UChar_t col[4];
   Color_t defCol = fM->GetTopViewTowerColor();
   if (fBinStep == 1)
   {
      // draw in original binning
      Int_t   name = 0, max_energy_slice;
      Float_t sum, max_energy, x1=0, x2=0, y1=0, y2=0;
      TGLUtil::Color(defCol);
      TEveCaloData::vCellId_t::iterator currentCell = fM->fCellList.begin();
      TEveCaloData::vCellId_t::iterator nextCell    = currentCell;
      ++nextCell;
      while (currentCell != fM->fCellList.end())
      {
         TEveCaloData::CellData_t currentCellData;
         TEveCaloData::CellData_t nextCellData;

         fM->fData->GetCellData(*currentCell, currentCellData);
         sum = max_energy = currentCellData.Value(fM->fPlotEt);
         max_energy_slice = currentCell->fSlice;

         while (nextCell != fM->fCellList.end() && currentCell->fTower == nextCell->fTower)
         {
            fM->fData->GetCellData(*nextCell, nextCellData);
            Float_t energy = nextCellData.Value(fM->fPlotEt);
            sum += energy;
            if (fM->fTopViewUseMaxColor && energy > max_energy) {
               max_energy       = energy;
               max_energy_slice = nextCell->fSlice;
            }
            ++nextCell;
         }

         glLoadName(name);
         glBegin(GL_QUADS);
         PhiShiftInterval(currentCellData.fPhiMin,currentCellData.fPhiMax);
         if (fM->f2DMode == TEveCaloLego::kValColor)
         {
            fM->fPalette->ColorFromValue(FloorNint(sum), col);
            TGLUtil::Color4ubv(col);

            x1 = Max(fM->GetEtaMin(), currentCellData.EtaMin());
            x2 = Min(fM->GetEtaMax(), currentCellData.EtaMax());

            y1 = Max(fM->GetPhiMin(), currentCellData.PhiMin());
            y2 = Min(fM->GetPhiMax(), currentCellData.PhiMax());
         }
         else if (fM->f2DMode == TEveCaloLego::kValSize)
         {
            Double_t scaleFactor = 0;
            Double_t range = 100;
            if (range*sum/fDataMax > 1) scaleFactor = Log(range*sum/fDataMax)/Log(range);
            Float_t etaW = (currentCellData.EtaDelta()*0.5f)*scaleFactor;
            Float_t phiW = (currentCellData.PhiDelta()*0.5f)*scaleFactor;

            x1 = Max(fM->GetEtaMin(), currentCellData.Eta() - etaW);
            x2 = Min(fM->GetEtaMax(), currentCellData.Eta() + etaW);

            y1 = Max(fM->GetPhiMin(), currentCellData.Phi() - phiW);
            y2 = Min(fM->GetPhiMax(), currentCellData.Phi() + phiW);

            if (fM->fTopViewUseMaxColor)
            {
               TGLUtil::Color(fM->GetData()->GetSliceColor(max_energy_slice));
               antiFlick[max_energy_slice].push_back(TEveVector(0.5f*(x1+x2), 0.5f*(y1+y2), sum));
            }
            else
            {
               antiFlick[0].push_back(TEveVector(0.5f*(x1+x2), 0.5f*(y1+y2), sum));
            }
         }

         glVertex3f(x1, y1, sum);
         glVertex3f(x2, y1, sum);
         glVertex3f(x2, y2, sum);
         glVertex3f(x1, y2, sum);

         glEnd();

         currentCell = nextCell;
         ++nextCell;
         ++name;
      }
   }
   else
   {
      // values in the scaled cells
      const Int_t nEta = fEtaAxis->GetNbins();
      const Int_t nPhi = fPhiAxis->GetNbins();
      std::vector<Float_t> vec;
      vec.assign((nEta+2)*(nPhi+2), 0.f);
      std::vector<Float_t> max_e;
      std::vector<Int_t>   max_e_slice;
      if (fM->fTopViewUseMaxColor)
      {
         max_e.assign((nEta+2) * (nPhi+2), 0.f);
         max_e_slice.assign((nEta+2) * (nPhi+2), -1);
      }

      for (UInt_t bin=0; bin<fRebinData.fBinData.size(); ++bin)
      {
         Float_t ssum = 0;
         if (fRebinData.fBinData[bin] != -1)
         {
            Float_t *val = fRebinData.GetSliceVals(bin);
            for (Int_t s=0; s<fRebinData.fNSlices; ++s)
            {
               ssum += val[s];
               if (fM->fTopViewUseMaxColor && val[s] > max_e[bin])
               {
                  max_e[bin]       = val[s];
                  max_e_slice[bin] = s;
               }
            }
         }
         vec[bin] = ssum;
      }

      Float_t maxv = 0;
      for (UInt_t i =0; i<vec.size(); ++i)
         if (vec[i] > maxv) maxv=vec[i];

      Float_t scale    = fM->fData->GetMaxVal(fM->fPlotEt)/maxv;
      Float_t logMax   = Log(maxv+1);
      Float_t scaleLog = fM->fData->GetMaxVal(fM->fPlotEt)/logMax;

      // take smallest threshold
      Float_t threshold = fM->GetDataSliceThreshold(0);
      for (Int_t s=1; s<fM->fData->GetNSlices(); ++s)
      {
         if (threshold > fM->GetDataSliceThreshold(s))
            threshold = fM->GetDataSliceThreshold(s);
      }

      // draw  scaled
      TGLUtil::Color(defCol);
      Float_t y0, y1, eta, etaW, phi, phiW;
      for (Int_t i=1; i<=fEtaAxis->GetNbins(); ++i)
      {
         for (Int_t j=1; j<=fPhiAxis->GetNbins(); ++j)
         {
            const Int_t bin = j*(nEta+2)+i;
            if (vec[bin] > threshold && fRebinData.fBinData[bin] != -1)
            {
               y0 = fPhiAxis->GetBinLowEdge(j);
               y1 = fPhiAxis->GetBinUpEdge(j);
               if (!PhiShiftInterval(y0, y1)) continue;

               const Float_t binVal = vec[bin]*scale;
               const Float_t logVal = Log(vec[bin] + 1);

               glLoadName(bin);
               glBegin(GL_QUADS);

               if (fM->f2DMode == TEveCaloLego::kValColor)
               {
                  fM->fPalette->ColorFromValue((Int_t)(logVal*scaleLog), col);
                  TGLUtil::Color4ubv(col);

                  eta  = fEtaAxis->GetBinLowEdge(i);
                  etaW = fEtaAxis->GetBinWidth(i);

                  glVertex3f(eta     , y0, binVal);
                  glVertex3f(eta+etaW, y0, binVal);
                  glVertex3f(eta+etaW, y1, binVal);
                  glVertex3f(eta     , y1, binVal);

               }
               else if (fM->f2DMode == TEveCaloLego::kValSize)
               {
                  eta  = fEtaAxis->GetBinCenter(i);
                  etaW = fEtaAxis->GetBinWidth(i)*0.5f*logVal/logMax;
                  phi  = 0.5f*(y0 + y1);
                  phiW = 0.5f*(y1 - y0)*logVal/logMax;

                  if (fM->fTopViewUseMaxColor)
                  {
                     TGLUtil::Color(fM->GetData()->GetSliceColor(max_e_slice[bin]));
                     antiFlick[max_e_slice[bin]].push_back(TEveVector(eta, phi, binVal));
                  }
                  else
                  {
                     antiFlick[0].push_back(TEveVector(eta, phi, binVal));
                  }

                  glVertex3f(eta - etaW, phi - phiW, binVal);
                  glVertex3f(eta + etaW, phi - phiW, binVal);
                  glVertex3f(eta + etaW, phi + phiW, binVal);
                  glVertex3f(eta - etaW, phi + phiW, binVal);
               }
               glEnd();
            }
         }
      }
   }

   if (fM->f2DMode == TEveCaloLego::kValSize)
   {
      TGLUtil::Color(defCol);
      glPointSize(1);
      glBegin(GL_POINTS);
      Int_t slice = 0;
      for (vvTEveVec_i i = antiFlick.begin(); i != antiFlick.end(); ++i, ++slice)
      {
         if (fM->fTopViewUseMaxColor)
            TGLUtil::Color(fM->GetData()->GetSliceColor(slice));

         for (vTEveVec_i j = i->begin(); j != i->end(); ++j)
         {
            glVertex3fv(j->Arr());
         }
      }
      glEnd();
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // Draw the object.

   if ( ! fM->fData || ! fM->fData->GetEtaBins() || ! fM->fData->GetPhiBins())
      return;


   // projection type
   if (fM->fProjection == TEveCaloLego::kAuto)
      fCells3D = (! (rnrCtx.RefCamera().IsOrthographic() && rnrCtx.RefCamera().GetCamBase().GetBaseVec(1).Z()));
   else if (fM->fProjection == TEveCaloLego::k2D)
      fCells3D = kFALSE;
   else
      fCells3D = kTRUE;

   // cache max val
   fDataMax = fM->GetMaxVal();

   // modelview matrix
   Double_t em, eM, pm, pM;
   fM->fData->GetEtaLimits(em, eM);
   fM->fData->GetPhiLimits(pm, pM);
   Double_t unit = ((eM-em) < (pM-pm)) ? (eM-em):(pM-pm);
   glPushMatrix();
   Float_t sx = (eM-em)/fM->GetEtaRng();
   Float_t sy = (pM-pm)/fM->GetPhiRng();
   glScalef(sx/unit, sy/unit, fM->fData->Empty() ? 1: fM->GetValToHeight());
   glTranslatef(-fM->GetEta(), -fM->fPhi, 0);

   // set axis
   Int_t oldBinStep = fBinStep;
   fBinStep = GetGridStep(rnrCtx);

   SetAxis(fM->fData->GetEtaBins(), fEtaAxis);
   SetAxis(fM->fData->GetPhiBins(), fPhiAxis);

   // cache ids
   Bool_t idCacheChanged = kFALSE;
   if (fM->fCellIdCacheOK == kFALSE)
   {
      fM->BuildCellIdCache();
      idCacheChanged = kTRUE;
   }

   // rebin data
   if (oldBinStep != fBinStep || idCacheChanged)
   {
      fDLCacheOK = kFALSE;

      fRebinData.fSliceData.clear();
      fRebinData.fSliceData.clear();

      if (fBinStep > 1)
      {
         fM->fData->Rebin(fEtaAxis, fPhiAxis, fM->fCellList, fM->fPlotEt, fRebinData);
         if (fM->fNormalizeRebin)
         {
            Double_t maxVal = 0;
            for (UInt_t i=0; i<fRebinData.fSliceData.size(); i+=fRebinData.fNSlices)
            {
               Double_t sum = 0;
               for(Int_t s=0; s<fRebinData.fNSlices; s++)
                  sum += fRebinData.fSliceData[i+s];

               if (sum > maxVal) maxVal = sum;
            }

            const Float_t scale = fM->GetMaxVal() / maxVal;
            for (std::vector<Float_t>::iterator it=fRebinData.fSliceData.begin(); it!=fRebinData.fSliceData.end(); it++)
               (*it) *= scale;
         }
      }
   }

   if (!fM->fData->Empty())
   {
      glPushAttrib(GL_LINE_BIT | GL_POLYGON_BIT);
      glLineWidth(1);
      glDisable(GL_LIGHTING);
      glEnable(GL_NORMALIZE);
      glEnable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(0.8, 1);

      glPushName(0);
      glLoadName(0);
      if (fCells3D)
      {
         if (!fDLCacheOK) MakeDisplayList();
         DrawCells3D(rnrCtx);
      }
      else
      {
         DrawCells2D();
      }
      glPopName();
      glPopAttrib();
   }

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

   if (fBinStep == 1)
   {
      TEveCaloData::CellData_t cellData;
      fM->fData->GetCellData(fM->fCellList[cellID], cellData);

      if (fCells3D)
      {
         printf("Bin %d selected in slice %d val %f\n",
                fM->fCellList[cellID].fTower,
                fM->fCellList[cellID].fSlice, cellData.fValue);
      }
      else
      {
         printf("Bin %d selected\n",fM->fCellList[cellID].fTower);
      }
   }
   else
   {
      if (fCells3D)
      {
         Float_t* v = fRebinData.GetSliceVals(rec.GetItem(2));
         Int_t s = rec.GetItem(1);
         printf("Rebined bin %d selected in slice %d val %f\n", rec.GetItem(2), s, v[s]);
      }
      else
      {
         Float_t* v = fRebinData.GetSliceVals(rec.GetItem(1));
         printf("Rebined bin %d selected\n",rec.GetItem(1));
         for (Int_t s=0; s<2; s++)
         {
            printf("slice %d val %f\n", s, v[s]);
         }
      }
   }
}
