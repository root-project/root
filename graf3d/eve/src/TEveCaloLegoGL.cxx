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
   fZAxisMax(0),

   fEtaAxis(0),
   fPhiAxis(0),

   fDLCacheOK(kFALSE),
   fM(0),

   fNBinSteps(5),
   fBinSteps(0),

   fTowerPicked(-1)
{
   // Constructor.

   fDLCache = kFALSE;

   fBinSteps = new Int_t[fNBinSteps];
   fBinSteps[0] = 1;
   fBinSteps[1] = 2;
   fBinSteps[2] = 5;
   fBinSteps[3] = 8;
   fBinSteps[4] = 20;
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


   // !! This ok if master sub-classed from TAttBBox
   SetAxisAlignedBBox(((TEveCaloLego*)fExternalObj)->AssertBBox());
}

//______________________________________________________________________________
Bool_t TEveCaloLegoGL::ShouldDLCache(const TGLRnrCtx & /*rnrCtx*/) const
{
   // Determines if display-list will be used for rendering.
   // Virtual from TGLLogicalShape.

   return kFALSE;
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

         fM->fData->GetCellData(fM->fCellList[i], fM->fPhi, fM->fPhiRng, cellData);
         if (s == fM->fCellList[i].fSlice)
         {
            glLoadName(i);
            MakeQuad(cellData.EtaMin(), cellData.PhiMin(), offset,
                     cellData.EtaDelta(), cellData.PhiDelta(), cellData.Value());
         }
         offset += cellData.Value();
      }
      glEndList();
   }
   fDLCacheOK=kTRUE;
}

//______________________________________________________________________________
void TEveCaloLegoGL::RnrText(const char* txt,
                             Float_t xa, Float_t ya, Float_t za,
                             const TGLFont &font, Int_t mode) const
{
   // Render text txt at given position, font and alignmnet mode.

   glPushMatrix();
   glTranslatef(xa, ya, za);

   Float_t llx, lly, llz, urx, ury, urz;
   font.BBox(txt, llx, lly, llz, urx, ury, urz);

   glRasterPos2i(0, 0);
   switch (mode)
   {
      case 0: // xy axis title interior
         glBitmap(0, 0, 0, 0, 0, -ury*0.5f, 0);
         break;
      case 1: // xy axis title exterior
         glBitmap(0, 0, 0, 0, -urx, -ury*0.5f, 0);
         break;
      case 2: // xy labels
         if (txt[0] == '-')
            urx += (urx-llx)/strlen(txt);
         glBitmap(0, 0, 0, 0,  -urx*0.5f, -ury, 0);
         break;
      case 3:  // z labels
         glBitmap(0, 0, 0, 0, -urx, -ury*0.5f, 0);
         break;
   }
   font.Render(txt);
   glPopMatrix();
}

//______________________________________________________________________________
void TEveCaloLegoGL::SetFont(Float_t cfs, TGLRnrCtx & rnrCtx) const
{
   // Set font size for given axis length.

   Int_t fs =TGLFontManager::GetFontSize(cfs, 12, 36);

   if (fNumFont.GetMode() == TGLFont::kUndef)
   {
      rnrCtx.RegisterFont(fs, "arial",  TGLFont::kPixmap, fNumFont);
      rnrCtx.RegisterFont(fs, "symbol", TGLFont::kPixmap, fSymbolFont);
   }
   else if (fNumFont.GetSize() != fs)
   {
      rnrCtx.ReleaseFont(fNumFont);
      rnrCtx.ReleaseFont(fSymbolFont);
      rnrCtx.RegisterFont(fs, "arial",  TGLFont::kPixmap, fNumFont);
      rnrCtx.RegisterFont(fs, "symbol", TGLFont::kPixmap, fSymbolFont);
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawZScales2D(TGLRnrCtx & rnrCtx, Float_t x0, Float_t y0) const
{
   // Draw simplified z-axis for top-down view.

   GLdouble x1, y1, x2, y2, z;
   GLdouble mm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX, mm);
   glGetIntegerv(GL_VIEWPORT, vp);
   const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();
   gluProject(x0, -TMath::Pi()*0.5, 0, mm, pm, vp, &x1, &y1, &z);
   gluProject(x0,  TMath::Pi()*0.5, 0, mm, pm, vp, &x2, &y2, &z);

   Float_t etal = TMath::Sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
   SetFont(etal*0.1, rnrCtx);
   TGLUtil::Color(fM->fFontColor);
   fNumFont.PreRender(kFALSE);

   const char* txt = Form("Et[GeV] %s",TEveUtil::FormAxisValue(fZAxisMax));
   RnrText(txt, x0, y0, 0, fNumFont, 2);
   fNumFont.PostRender();
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawZAxis(TGLRnrCtx &rnrCtx, Float_t azX, Float_t azY) const
{
   // Draw Z axis at given xy position;

   glPushMatrix();
   glTranslatef(azX,  azY, 0);

   // size of tick-mark 8 pixels
   TGLVertex3 worldRef(0, 0, fZAxisMax*0.5);
   TGLVector3 off = rnrCtx.RefCamera().ViewportDeltaToWorld(worldRef, -8, 0);

   // primary tick-marks
   Int_t np = TMath::CeilNint(fZAxisMax/fZAxisStep);
   glBegin(GL_LINES);
   Float_t z =fZAxisStep;
   for (Int_t i = 0; i < np; ++i)
   {
      glVertex3f(0, 0, z);
      glVertex3f(off.X(), off.Y(), z-off.Z());
      z += fZAxisStep;
   }

   // secondary
   off  = off*0.5f;
   Double_t omin2, omax2, step2;
   Int_t div2;
   THLimitsFinder::Optimize(0, fZAxisStep, fM->GetNZSteps(), omin2, omax2, div2, step2);
   Double_t z2 = step2;
   while (z2 < fZAxisMax)
   {
      glVertex3f(0, 0, z2);
      glVertex3f(off.X(), off.Y(), z2-off.Z());
      z2 += step2;
   }
   glEnd();


   off *= 4; // label offset
   fNumFont.PreRender(kFALSE);
   TGLUtil::Color(fM->fFontColor);
   for (Int_t i = 1; i < np; ++i)
      RnrText(TEveUtil::FormAxisValue(i*fZAxisStep), off.X(), off.Y(), i*fZAxisStep+off.Z(), fNumFont, 3);

   const char* txt = TEveUtil::FormAxisValue(fZAxisMax);
   RnrText(Form("Et[GeV] %s", txt), off.X(), off.Y(), fZAxisMax+off.Z(), fNumFont, 3);

   fNumFont.PostRender();

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

   // z axis vertical
   glBegin(GL_LINES);
   glVertex3f(azX, azY, 0);
   glVertex3f(azX, azY, fZAxisMax);
   glEnd();

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
      glVertex3f(x0, axY, 0); glVertex3f(x0, axY, fZAxisMax);
      glVertex3f(x1, axY, 0); glVertex3f(x1, axY, fZAxisMax);
      glVertex3f(ayX, y0, 0); glVertex3f(ayX, y0, fZAxisMax);
      glVertex3f(ayX, y1, 0); glVertex3f(ayX, y1, fZAxisMax);
      if (fM->fBoxMode == TEveCaloLego::kFrontBack)
      {
         glVertex3f(cX, cY, 0); glVertex3f(cX, cY, fZAxisMax);
      }

      // box top
      glVertex3f(x0, axY, fZAxisMax); glVertex3f(x1, axY, fZAxisMax);
      glVertex3f(ayX, y0, fZAxisMax); glVertex3f(ayX, y1, fZAxisMax);
      if (fM->fBoxMode == TEveCaloLego::kFrontBack)
      {
         glVertex3f(cX, cY, fZAxisMax); glVertex3f(cX, axY, fZAxisMax);
         glVertex3f(cX, cY, fZAxisMax); glVertex3f(ayX, cY, fZAxisMax);
      }
      glEnd();
      glPopAttrib();

      // box horizontals stippled
      glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT);
      glLineStipple(1, 0x5555);
      glEnable(GL_LINE_STIPPLE);
      glBegin(GL_LINES);
      Int_t nhs = TMath::CeilNint(fZAxisMax/fZAxisStep);
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

   // set font (depending of size projected Z axis)
   GLdouble dn[3];
   GLdouble up[3];
   gluProject(azX, azY, 0, mm, pm, vp, &dn[0], &dn[1], &dn[2]);
   gluProject(azX, azY, fZAxisMax, mm, pm, vp, &up[0], &up[1], &up[2]);
   Float_t zAxisLength = TMath::Sqrt((  up[0]-dn[0])*(up[0]-dn[0])
                                     + (up[1]-dn[1])*(up[1]-dn[1])
                                     + (up[2]-dn[2])*(up[2]-dn[2]));
   SetFont(zAxisLength*0.08f, rnrCtx);

   DrawZAxis(rnrCtx, azX,  azY);

   if (fTowerPicked >= 0)
   {
      // left most corner of the picked tower
      TEveCaloData::CellData_t cd;
      fM->fData->GetCellData(fM->fCellList[fTowerPicked], fM->fPhi, fM->fPhiRng, cd);
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
   Float_t zOff2 =  zOff*0.5;


   Float_t rxy = (fEtaAxis->GetXmax()-fEtaAxis->GetXmin())/(fPhiAxis->GetXmax()-fPhiAxis->GetXmin());
   Float_t yOff  =  0.03*TMath::Sign(y1-y0, axY)/rxy;
   Float_t yOff2 =  yOff*0.5;

   Float_t xOff  =  0.03*TMath::Sign(x1-x0, ayX)*rxy;
   Float_t xOff2 =  xOff*0.5;

   glPushMatrix();
   glTranslatef(0, 0, zOff);
   fNumFont.PreRender(kFALSE);
   TGLUtil::Color(fM->fFontColor);

   // titles
   RnrText("h", axtX, axY+yOff, 0, fSymbolFont, (axY>0 && ayX>0) || (axY<0 && ayX<0));
   RnrText("f", ayX+xOff, aytY, 0, fSymbolFont, (ayX>0 && axY<0) || (ayX<0 && axY>0));

   // X labels
   Double_t oXmin, oXmax, oXbw;
   Int_t oXndiv;
   THLimitsFinder::Optimize(x0, x1, fM->fNZSteps, oXmin, oXmax, oXndiv, oXbw);
   for (Int_t i = 0; i <= oXndiv; ++i)
      RnrText(TEveUtil::FormAxisValue(oXmin + oXbw*i), oXmin+oXbw*i, axY + TMath::Sign(yOff*1.5f,axY), 0, fNumFont, 2);

   // Y labels
   Double_t oYmin, oYmax, oYbw;
   Int_t oYndiv;
   THLimitsFinder::Optimize(y0, y1, fM->fNZSteps, oYmin, oYmax, oYndiv, oYbw);
   for (Int_t i = 0; i <= oYndiv; ++i)
      RnrText(TEveUtil::FormAxisValue(oYmin + oYbw*i), ayX + TMath::Sign(xOff*1.5f,ayX), oYmin+oYbw*i, 0, fNumFont, 2);

   glPopMatrix();
   fNumFont.PostRender();


   /**************************************************************************/
   // draw X  axis lines

   glPushMatrix();
   glTranslatef(0, axY, 0);
   glBegin(GL_LINES);
   // body
   glVertex3f(x0, 0, 0);
   glVertex3f(x1, 0, 0);
   // tick-marks
   Double_t oXmin2, oXmax2, oXbw2;
   Int_t oXndiv2;
   THLimitsFinder::Optimize(oXmin, oXbw+oXmin, fM->fNZSteps, oXmin2, oXmax2, oXndiv2, oXbw2);
   Float_t xt = oXmin;
   Float_t xt2;
   for(Int_t i = 0; i <= oXndiv; ++i)
   {
      glVertex3f(xt, 0,    0);
      glVertex3f(xt, 0,    zOff);
      glVertex3f(xt, 0,    0);
      glVertex3f(xt, yOff, 0);
      xt2 = xt;
      for (Int_t j = 0; j <= oXndiv2; ++j)
      {
         if (xt2 >= x1) break;
         glVertex3f(xt2, 0,     0);
         glVertex3f(xt2, 0,     zOff2);
         glVertex3f(xt2, 0,     0);
         glVertex3f(xt2, yOff2, 0);
         xt2 += oXbw2;
      }
      xt += oXbw;
   }
   xt2 = oXmin;
   while (xt2 > x0)
   {
      glVertex3f(xt2, 0,     0);
      glVertex3f(xt2, 0,     zOff2);
      glVertex3f(xt2, 0,     0);
      glVertex3f(xt2, yOff2, 0);
      xt2 -= oXbw2;
   }
   glEnd();
   glPopMatrix();

   /**************************************************************************/
   // Y axis lines

   glPushMatrix();
   glTranslatef(ayX, 0, 0);
   glBegin(GL_LINES);
   // body
   glVertex3f(0, y0, 0);
   glVertex3f(0, y1, 0);
   // tick-marks
   Double_t oYmin2, oYmax2, oYbw2;
   Int_t oYndiv2;
   THLimitsFinder::Optimize(oYmin, oYbw+oYmin, fM->fNZSteps, oYmin2, oYmax2, oYndiv2, oYbw2);
   Float_t yt = oYmin;
   Float_t yt2;
   for (Int_t i = 0; i <= oYndiv; ++i)
   {
      glVertex3f(0,    yt, 0);
      glVertex3f(0,    yt, zOff);
      glVertex3f(0,    yt, 0);
      glVertex3f(xOff, yt, 0);
      yt2 = yt;
      for (Int_t j=0; j<=oYndiv2; ++j)
      {
         if (yt2 >= y1) break;
         glVertex3f(0,     yt2, 0);
         glVertex3f(0,     yt2, zOff2);
         glVertex3f(0,     yt2, 0);
         glVertex3f(xOff2, yt2, 0);
         yt2 += oYbw2;
      }
      yt += oYbw;
   }
   yt2 = oYmin;
   while (yt2 > y0)
   {
      glVertex3f(0,     yt2, 0);
      glVertex3f(0,     yt2, zOff2);
      glVertex3f(0,     yt2, 0);
      glVertex3f(xOff2, yt2, 0);
      yt2 -= oYbw2;
   }
   glEnd();
   glPopMatrix();

} // DrawXYScales

//______________________________________________________________________________
Int_t TEveCaloLegoGL::GetGridStep(Int_t axId, TGLRnrCtx &rnrCtx) const
{
   // Calculate view-dependent grid density.
   GLdouble xp0, yp0, zp0, xp1, yp1, zp1;
   GLdouble mm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetIntegerv(GL_VIEWPORT, vp);
   const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();

   Int_t firstX = fEtaAxis->GetFirst();
   Int_t lastX  = fEtaAxis->GetLast();
   Int_t firstY = fPhiAxis->GetFirst();
   Int_t lastY  = fPhiAxis->GetLast();

   if (axId == 0)
   {
      Float_t y0 = fPhiAxis->GetBinLowEdge(firstY);
      for (Int_t idx = 0; idx < fNBinSteps; ++idx)
      {
         if (firstX + fBinSteps[idx] > lastX) return 1;
         gluProject(fEtaAxis->GetBinLowEdge(firstX), y0, 0, mm, pm, vp, &xp0, &yp0, &zp0);
         gluProject(fEtaAxis->GetBinLowEdge(firstX+fBinSteps[idx]), y0, 0, mm, pm, vp, &xp1, &yp1, &zp1);
         Float_t gapsqr = (xp0-xp1)*(xp0-xp1) + (yp0-yp1)*(yp0-yp1);
         if (gapsqr > fM->fBinWidth*fM->fBinWidth)
            return fBinSteps[idx];
      }
   }
   else if (axId == 1)
   {
      Float_t x0 = fEtaAxis->GetBinLowEdge(firstX);
      for (Int_t idx = firstY; idx < fNBinSteps; ++idx)
      {
         if (firstY + fBinSteps[idx] > lastY) return 1;
         gluProject(x0, fEtaAxis->GetBinLowEdge(firstY), 0, mm, pm, vp, &xp0, &yp0, &zp0);
         gluProject(x0, fEtaAxis->GetBinLowEdge(firstY+fBinSteps[idx]), 0, mm, pm, vp, &xp1, &yp1, &zp1);
         Float_t  gapsqr = (xp0-xp1)*(xp0-xp1) + (yp0-yp1)*(yp0-yp1);
         if (gapsqr > fM->fBinWidth*fM->fBinWidth)
            return fBinSteps[idx];
      }
   }
   return 1;
}

 //______________________________________________________________________________
void TEveCaloLegoGL::DrawHistBase(TGLRnrCtx &rnrCtx) const
{
   // Draw basic histogram components: x-y grid

   Int_t es = GetGridStep(0, rnrCtx);
   Int_t ps = GetGridStep(1, rnrCtx);

   Float_t eta0 = fM->fEtaMin;
   Float_t eta1 = fM->fEtaMax;
   Float_t phi0 = fM->GetPhiMin();
   Float_t phi1 = fM->GetPhiMax();

   TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);
   TGLCapabilitySwitch sw_blend(GL_BLEND, kTRUE);

   TGLUtil::Color(fM->fGridColor);

   glBegin(GL_LINES);

   glVertex2f(eta0, phi0);
   glVertex2f(eta0, phi1);
   glVertex2f(eta1, phi0);
   glVertex2f(eta1, phi1);

   glVertex2f(eta0, phi0);
   glVertex2f(eta1, phi0);
   glVertex2f(eta0, phi1);
   glVertex2f(eta1, phi1);

   if ((es == 1 && ps == 1) || fM->fProjection == TEveCaloLego::k3D)
   {
      // original
      for (Int_t i=fEtaAxis->GetFirst(); i<fEtaAxis->GetLast(); i+= es)
      {
         glVertex2f(fEtaAxis->GetBinUpEdge(i), phi0);
         glVertex2f(fEtaAxis->GetBinUpEdge(i), phi1);
      }

      for (Int_t i=fPhiAxis->GetFirst(); i<fPhiAxis->GetLast(); i+= es)
      {
         glVertex2f(eta0, fEtaAxis->GetBinUpEdge(i));
         glVertex2f(eta1, fEtaAxis->GetBinUpEdge(i));
      }
   }
   else
   {
      // scaled
      Float_t ebw = es*(eta1-eta0)/fEtaAxis->GetNbins();
      Int_t i0 = Int_t(eta0/ebw);
      Int_t i1 = Int_t(eta1/ebw);
      for (Int_t i=i0; i<=i1; i++)
      {
         glVertex2f(i*ebw, phi0);
         glVertex2f(i*ebw, phi1);
      }

      Float_t pbw = ps*(phi1-phi0)/fPhiAxis->GetNbins();
      Int_t j0 = Int_t(phi0/pbw);
      Int_t j1 = Int_t(phi1/pbw);
      for (Int_t j=j0; j<=j1; j++)
      {
         glVertex2f(eta0, j*pbw);
         glVertex2f(eta1, j*pbw);
      }
   }
   glEnd();

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POLYGON_BIT);
   glLineWidth(2);

   // axis
   if ( fM->fProjection == TEveCaloLego::k3D || rnrCtx.RefCamera().GetCamBase().GetBaseVec(1).Z() == 0)
      DrawZScales3D(rnrCtx, eta0, eta1, phi0, phi1);
   else
      DrawZScales2D(rnrCtx, eta0, phi0);

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
         TGLUtil::Color(fM->GetPalette()->GetDefaultColor() + i->first);
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

   using namespace TMath;

   static const TEveException eh("TEveCaloLegoGL::DrawCells2D ");

   UChar_t col[4];
   Color_t defCol = fM->GetPalette()->GetDefaultColor();
   Int_t es = GetGridStep(0, rnrCtx);
   Int_t ps = GetGridStep(1, rnrCtx);
   if (es==1 && ps==1)
   {
      // draw in original binning
      TEveCaloData::CellData_t cellData;
      Int_t name = 0;
      Int_t prevTower = 0;
      Float_t sum = 0;

      for (TEveCaloData::vCellId_t::iterator it=fM->fCellList.begin(); it!=fM->fCellList.end(); it++)
      {
         fM->fData->GetCellData(*it, fM->fPhi, fM->fPhiRng, cellData);
         glLoadName(name);
         glBegin(GL_QUADS);
         if ((*it).fTower != prevTower)
         {
            if (fM->f2DMode == TEveCaloLego::kValColor)
            {
               fM->fPalette->ColorFromValue(FloorNint(sum), col);
               TGLUtil::Color4ubv(col);
               glVertex3f(cellData.EtaMin(), cellData.PhiMin(), sum);
               glVertex3f(cellData.EtaMax(), cellData.PhiMin(), sum);
               glVertex3f(cellData.EtaMax(), cellData.PhiMax(), sum);
               glVertex3f(cellData.EtaMin(), cellData.PhiMax(), sum);
            }
            else if (fM->f2DMode == TEveCaloLego::kValSize)
            {
               TGLUtil::Color(fM->fPalette->GetDefaultColor());
               Float_t etaW = (cellData.EtaDelta()*sum*0.5f)/fDataMax;
               Float_t phiW = (cellData.PhiDelta()*sum*0.5f)/fDataMax;
               glVertex3f(cellData.Eta() -etaW, cellData.Phi()-phiW, sum);
               glVertex3f(cellData.Eta() +etaW, cellData.Phi()-phiW, sum);
               glVertex3f(cellData.Eta() +etaW, cellData.Phi()+phiW, sum);
               glVertex3f(cellData.Eta() -etaW, cellData.Phi()+phiW, sum);
            }
            sum = cellData.Value();
            prevTower = (*it).fTower;
         }
         else
         {
            sum += cellData.Value();
         }
         glEnd();
         name++;
      };
   }
   else
   {
      // prepare values in the scaled cells

      Float_t eta0 = fM->fEtaMin;
      Float_t eta1 = fM->fEtaMax;
      Float_t phi0 = fM->GetPhiMin();
      Float_t phi1 = fM->GetPhiMax();

      Int_t   nEta = CeilNint(fEtaAxis->GetNbins()/es);
      Int_t   nPhi = CeilNint(fPhiAxis->GetNbins()/ps);
      Float_t etaStep = (eta1-eta0)/nEta;
      Float_t phiStep = (phi1-phi0)/nPhi;


      std::vector<Float_t> vec;
      vec.assign((nEta)*(nPhi), 0.f);

      TEveCaloData::CellData_t cd;
      Float_t left, right, up, down; // cell corners
      for (TEveCaloData::vCellId_t::iterator it=fM->fCellList.begin(); it!=fM->fCellList.end(); it++)
      {
         fM->fData->GetCellData(*it, fM->fPhi, fM->fPhiRng, cd);
         Int_t iMin = FloorNint((cd.EtaMin()- eta0)/etaStep);
         Int_t iMax = CeilNint ((cd.EtaMax()- eta0)/etaStep);
         Int_t jMin = FloorNint((cd.PhiMin()- phi0)/phiStep);
         Int_t jMax = CeilNint ((cd.PhiMax()- phi0)/phiStep);

         if (jMin <0)
            throw(eh + "cell phi %f less than set minimum %f.", cd.PhiMin(), cd.PhiMin());

         for (Int_t i=iMin; i<iMax; i++)
         {
            left  = i*etaStep;
            right = (i+1)*etaStep;

            if (i == iMin)
               left = cd.EtaMin()-eta0;

            if (i == (iMax-1))
               right = cd.EtaMax()-eta0;

            for (Int_t j=jMin; j<jMax; j++)
            {
               down = j*phiStep;
               up = down +phiStep;

               if (j == jMin)
                  down = cd.PhiMin() -phi0;

               if (j == (jMax-1))
                  up = cd.PhiMax() -phi0;

               vec[i*nPhi+j]  += cd.Value()*(right-left)*(up-down);
            }
         }
      }
      Float_t maxv = 0;
      for (std::vector<Float_t>::iterator it=vec.begin(); it !=vec.end(); it++)
         if (*it > maxv) maxv = *it;


      Float_t paletteFac = fM->fPalette->GetHighLimit()*1.f/maxv;
      Float_t logMax = Log(maxv+1);
      Float_t logPaletteFac = fM->fPalette->GetHighLimit()*1.f/logMax;

      Float_t etaW, phiW;
      Int_t cid = 0;
      glBegin(GL_QUADS);
      for (std::vector<Float_t>::iterator it=vec.begin(); it !=vec.end(); it++)
      {
         Float_t logVal = Log(*it+1);
         Float_t eta = Int_t(cid/nPhi)*etaStep + eta0;
         Float_t phi = (cid -Int_t(cid/nPhi)*nPhi)*phiStep + phi0;
         if ( fM->f2DMode == TEveCaloLego::kValColor)
         {
            fM->fPalette->ColorFromValue((Int_t)(logVal*logPaletteFac), col);
            TGLUtil::Color4ubv(col);

            {
               glVertex3f(eta        , phi,         (*it)*paletteFac);
               glVertex3f(eta+etaStep, phi,         (*it)*paletteFac);
               glVertex3f(eta+etaStep, phi+phiStep, (*it)*paletteFac);
               glVertex3f(eta        , phi+phiStep, (*it)*paletteFac);
            }
         }
         else if (fM->f2DMode == TEveCaloLego::kValSize)
         {
            TGLUtil::Color(defCol);
            eta += etaStep*0.5f;
            phi += phiStep*0.5f;
            etaW = etaStep*0.5f*logVal/logMax;
            phiW = phiStep*0.5f*logVal/logMax;
            glVertex3f(eta -etaW, phi -phiW, (*it)*paletteFac);
            glVertex3f(eta +etaW, phi -phiW, (*it)*paletteFac);
            glVertex3f(eta +etaW, phi +phiW, (*it)*paletteFac);
            glVertex3f(eta -etaW, phi +phiW, (*it)*paletteFac);
         }
         cid++;
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
   fDataMax   = fM->fData->GetMaxVal();
   Int_t ondiv;
   Double_t omin, omax;
   THLimitsFinder::Optimize(0, fDataMax, fM->fNZSteps, omin, omax, ondiv,  fZAxisStep);
   fZAxisMax = (ondiv+1)*fZAxisStep;

   // cache
   fM->AssertPalette();
   if (fM->fCacheOK == kFALSE)
   {
      fDLCacheOK = kFALSE;
      fM->ResetCache();
      fM->fData->GetCellList(fM->fPalette->GetMinVal(), fM->fPalette->GetMaxVal(),
                             fM->GetEta(), fM->GetEtaRng()*0.5,
                             fM->fPhi, fM->fPhiRng, fM->fCellList);
      fM->fCacheOK = kTRUE;
   }
   if (cells3D && fDLCacheOK == kFALSE) MakeDisplayList();


   // set modelview matrix
   glPushMatrix();
   glTranslatef(0, -fM->fPhi, 0);

   glScalef((fEtaAxis->GetXmax()-fEtaAxis->GetXmin())/fM->GetEtaRng(),
            0.5*(fPhiAxis->GetXmax()-fPhiAxis->GetXmin())/fM->fPhiRng,
            fM->fCellZScale*fM->GetDefaultCellHeight()/fZAxisMax);

   glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT | GL_POLYGON_BIT);
   glLineWidth(1);
   glDisable(GL_LIGHTING);
   glDisable(GL_CULL_FACE);
   glEnable(GL_NORMALIZE);
   glEnable(GL_POLYGON_OFFSET_FILL);


   // draw histogram base
   if (rnrCtx.Selection() == kFALSE && rnrCtx.Highlight() == kFALSE)
   {
      DrawHistBase(rnrCtx);
      if (fM->fDrawHPlane)
      {
         glEnable(GL_BLEND);
         glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
         TGLUtil::ColorTransparency(fM->fPlaneColor, fM->fPlaneTransparency);
         Float_t zhp = fM->fHPlaneVal*fZAxisMax;
         glBegin(GL_POLYGON);
         glVertex3f(fM->fEtaMin, fM->GetPhiMin(), zhp);
         glVertex3f(fM->fEtaMax, fM->GetPhiMin(), zhp);
         glVertex3f(fM->fEtaMax, fM->GetPhiMax(), zhp);
         glVertex3f(fM->fEtaMin, fM->GetPhiMax(), zhp);
         glEnd();
      }
   }

   // draw cells
   glPushName(0);
   glPolygonOffset(0.8, 1);
   cells3D ? DrawCells3D(rnrCtx):DrawCells2D(rnrCtx);
   glPopName();

   glPopMatrix();
   glPopAttrib();
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
