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
//______________________________________________________________________________
// OpenGL renderer class for TEveCaloLego.
//

ClassImp(TEveCaloLegoGL);

//______________________________________________________________________________
TEveCaloLegoGL::TEveCaloLegoGL() :
   TGLObject(),
   fDLCacheOK(kFALSE),
   fM(0),

   fFontSize(16),
   fTMSize(0.1),

   fMinBinWidth(6), // number of pixels in projected axis
   fNBinSteps(5),
   fBinSteps(0)
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

   for (std::map<Int_t, UInt_t>::iterator i=fDLMap.begin(); i!=fDLMap.end(); ++i)
      i->second = 0;

   TGLObject::DLCacheDrop();
}

//______________________________________________________________________________
void TEveCaloLegoGL::DLCachePurge()
{
   // Unregister all display-lists.

   static const TEveException eH("TEveCaloLegoGL::DLCachePurge ");

   if (fDLMap.empty()) return;

   for(std::map<Int_t, UInt_t>::iterator i=fDLMap.begin(); i!=fDLMap.end(); i++)
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

   TEveRGBAPalette& pal = *(fM->fPalette);
   TEveCaloData::CellData_t cellData;
   Int_t   prevTower = 0;
   Float_t offset = 0;

   // ids in eta phi rng
   Float_t scaleZ = fM->GetDefaultCellHeight()/(pal.GetHighLimit()- pal.GetLowLimit());  
   Int_t nSlices = fM->fData->GetNSlices(); 
   for(Int_t s=0; s<nSlices; s++) 
   {
      if (fDLMap.empty() || fDLMap[s]== 0) 
         fDLMap[s] = glGenLists(1); 

      glNewList(fDLMap[s], GL_COMPILE);
      for (UInt_t i=0; i<fM->fCellList.size(); ++i)
      {
         if (fM->fCellList[i].fSlice > s) continue;
         if (fM->fCellList[i].fTower != prevTower)
         {
            offset = 0;
            prevTower = fM->fCellList[i].fTower;
         }

         fM->fData->GetCellData(fM->fCellList[i], cellData);
         if (cellData.Value()>pal.GetMinVal() && cellData.Value()<pal.GetMaxVal())
         {
            Float_t z   = scaleZ*(cellData.Value() - pal.GetMinVal());
            if (s == fM->fCellList[i].fSlice)
            {
               glLoadName(i);
               MakeQuad(cellData.EtaMin(), cellData.PhiMin(), offset, 
                        cellData.EtaMax()-cellData.EtaMin(), cellData.EtaMax()-cellData.EtaMin(), z);
            }
            offset += z;
         }
      }
      glEndList();
   }
   fDLCacheOK=kTRUE;
}

//______________________________________________________________________________
inline void TEveCaloLegoGL::RnrText(const char* txt,
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
      case 2: // x labels
         if (txt[0] == '-')
            urx += (urx-llx)/strlen(txt);
         glBitmap(0, 0, 0, 0,  -urx*0.5f, -ury, 0);
         break;
      case 3: // y labels
         glBitmap(0, 0, 0, 0,  -urx, -ury*0.5f, 0);
         break;
      case 4:  // z labels
         glBitmap(0, 0, 0, 0, -urx, -ury*0.5f, 0);
         break;
   }
   font.Render(txt);
   glPopMatrix();
}


//______________________________________________________________________________
void TEveCaloLegoGL::SetFont(Float_t axis_len, TGLRnrCtx & rnrCtx) const
{
   // Set font size for given axis length.

   Float_t cfs = axis_len*(fM->fFontSize/100.f);
   // get match in the array
   Int_t* fsp = &TGLFontManager::GetFontSizeArray()->front();
   Int_t  nums = TGLFontManager::GetFontSizeArray()->size();  
   Int_t i = 0;
   while (i<nums)
   {
      if (cfs<fsp[i]) break;
      i++;
   }
   Int_t fs = fsp[i];
   // case the is no bigger font available
           
   if (fNumFont.GetMode() == TGLFont::kUndef || (fs != fFontSize))
   {
      fFontSize = fs;
      rnrCtx.RegisterFont(fs, "arial",  TGLFont::kPixmap, fNumFont);
      rnrCtx.RegisterFont(fs, "symbol", TGLFont::kPixmap, fSymbolFont);
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawZAxisSimplified(TGLRnrCtx & rnrCtx, Float_t x0, Float_t y0) const
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
   

   SetFont(TMath::Sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)), rnrCtx);
   TGLUtil::Color(fM->fFontColor);
   fNumFont.PreRender(kFALSE);
   RnrText(Form("Et[GeV] %d ", Int_t(fM->fData->GetMaxVal())), x0 - fTMSize*2, y0, 0, fNumFont, 4);  
   fNumFont.PostRender();
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawZAxis(TGLRnrCtx & rnrCtx,
                               Float_t x0, Float_t x1,
                               Float_t y0, Float_t y1) const
{
   // Draw z-axis at the appropriate grid corner-point including
   // tick-marks and labels.

   const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();
   GLdouble mm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetIntegerv(GL_VIEWPORT, vp);
   // get corner point closest to the eye in z-coordinate
   GLdouble x[4];
   GLdouble y[4];
   GLdouble z[4];
   gluProject(x0, y0, 0, mm, pm, vp, &x[0], &y[0], &z[0]);
   gluProject(x1, y0, 0, mm, pm, vp, &x[1], &y[1], &z[1]);
   gluProject(x1, y1, 0, mm, pm, vp, &x[2], &y[2], &z[2]);
   gluProject(x0, y1, 0, mm, pm, vp, &x[3], &y[3], &z[3]);

   // get index 
   Int_t vidx = 0;
   Float_t xt = x[0];
   for (Int_t i = 1; i < 4; ++i)
   {
      if (x[i] < xt)
      {
         xt  = x[i];
         vidx = i;
      }
   }
   // x, y location of axis
   Float_t azX, azY;
   Float_t aztX, aztY;
   if (vidx == 0) {
      azX  = x0;   azY = y0;
      aztX = -fTMSize; aztY =-fTMSize;
   }
   else if (vidx == 1) {
      azX  = x1;      azY = y0;
      aztX = +fTMSize; aztY = -fTMSize;
   }
   else if (vidx == 2) {
      azX  = x1;      azY = y1;
      aztX = +fTMSize; aztY = +fTMSize;
   }
   else {
      azX  = x0;      azY = y1;
      aztX = -fTMSize; aztY = +fTMSize;
   }

   // set font size
   {
      // project bottom and top point
      gluProject(azX, azY, 0, mm, pm, vp, &x[0], &y[0], &z[0]);
      gluProject(azX, azY, fM->GetDefaultCellHeight()*fM->fCellZScale, mm, pm, vp, &x[1], &y[1], &z[1]);
      SetFont(TMath::Sqrt((x[1]-x[0])*(x[1]-x[0])+(y[1]-y[0])*(y[1]-y[0])), rnrCtx);
   }

   TGLUtil::Color(fM->fFontColor);
   fNumFont.PreRender(kFALSE);
   Int_t tickval = 5*TMath::CeilNint(fM->fData->GetMaxVal()/(fM->fNZStep*5));
   Float_t step = tickval * fM->fCellZScale * fM->GetDefaultCellHeight()/fM->fData->GetMaxVal();
   Float_t off  = 2;

   // axis name
   for (Int_t i = 1; i < fM->fNZStep; ++i)
      RnrText(Form("%d",i*tickval), azX + aztX*off, azY + aztY*off, i*step, fNumFont, 4);
   // axis labels
   RnrText(Form("Et[GeV]  %d", fM->fNZStep*tickval), azX + aztX*off, azY + aztY*off, fM->fNZStep*step, fNumFont, 4);
   fNumFont.PostRender();

   TGLUtil::Color(fM->fGridColor);
   glBegin(GL_LINES);
   // body 
   glVertex3f(azX, azY, 0);
   glVertex3f(azX, azY, (fM->fNZStep)*step);
    // tick-marks
   Int_t ntm = (fM->fNZStep)*5;      
   step  /= 5; // number of subdevision 5
   for (Int_t i = 1; i <= ntm; ++i)
   {
      glVertex3f(azX, azY, i*step);
      if(i % 5)
         glVertex3f(azX+aztX*0.5f, azY+aztY*0.5f, i*step);
      else
         glVertex3f(azX+aztX, azY+aztY, i*step);
   }
   glEnd();
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawXYAxis(TGLRnrCtx & rnrCtx,
                                Float_t x0, Float_t x1,
                                Float_t y0, Float_t y1) const
{
   // get corner closest to projected plane
   const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();
   GLdouble mm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetIntegerv(GL_VIEWPORT, vp);

   // get corner point closest to the eye in z-coordinate
   GLdouble y;
   GLdouble z[4];
   GLdouble x[4];
   gluProject(x0, y0, 0, mm, pm, vp, &x[0], &y, &z[0]);
   gluProject(x1, y0, 0, mm, pm, vp, &x[1], &y, &z[1]);
   gluProject(x1, y1, 0, mm, pm, vp, &x[2], &y, &z[2]);
   gluProject(x0, y1, 0, mm, pm, vp, &x[3], &y, &z[3]);

   Float_t zt = 1.f;
   Float_t zm = 0.f;

   Int_t idx = 0;
   for (Int_t i=0; i<4; ++i)
   {
      if (z[i] < zt)
      {
         zt  = z[i];
         idx = i;
      }
      if (z[i] > zm) zm = z[i];
   }
   if (zm - zt < 1e-2)
      idx = 0;

   // get location of x and y axis
   Float_t axY,  ayX;
   Float_t axtX, aytY; // title pos
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
      default:
         ayX  = 0;  axY  = 0;
         axtX = 0;  aytY = 0;  
         break;
   }

   Float_t lx0 = -4;
   Int_t nX = 5;
   Float_t ly0 = -2;
   Int_t nY = 5;
   // xy labels
   {
      fNumFont.PreRender(kFALSE);
      glPushMatrix();
      glTranslatef(0, 0, -fTMSize*2.5); // translate onder the grid plane
      TGLUtil::Color(fM->fFontColor);

      RnrText("h", axtX, 1.1f*axY, 0, fSymbolFont, axY>0 && ayX>0 || axY<0 && ayX<0);
      for (Int_t i=0; i<nX; i++)
         RnrText(Form("%.0f", lx0 + 2*i), lx0 + 2*i, axY + TMath::Sign(fTMSize*3,axY), 0, fNumFont, 2);

      RnrText("f", 1.1f*ayX, aytY, 0, fSymbolFont, ayX>0 && axY<0 || ayX<0 && axY>0);
      for (Int_t i=0; i<nY; ++i)
         RnrText(Form("%.0f", ly0 + i), ayX + TMath::Sign(fTMSize*3,ayX), ly0 + i, 0, fNumFont, 3);

      glPopMatrix();
      fNumFont.PostRender();
   }
   // xy tickmarks
   TGLUtil::Color(fM->fGridColor);
   { 
      glBegin(GL_LINES);

      glVertex3f(x0, axY, 0);
      glVertex3f(x1, axY, 0);

      Float_t xs = 0.25f;
      Float_t xt = Int_t(x0/xs)*xs;
      while (xt < x1)
      {
         glVertex3f(xt, axY, 0);
         glVertex3f(xt, axY, (Int_t(xt*10) % 5) ? -fTMSize/2 : -fTMSize);
         glVertex3f(xt, axY, 0);
         glVertex3f(xt, ((Int_t(xt*10) % 5) ? 1.0125f : 1.025f)*axY, 0);
         xt += xs;
      }

      glVertex3f(ayX, y0, 0);
      glVertex3f(ayX, y1, 0);
      Float_t ys = 0.25f;
      Float_t yt = Int_t(y0/ys)*ys;
      while (yt < y1)
      {
         glVertex3f(ayX, yt, 0);
         glVertex3f(ayX, yt, (Int_t(yt*10) % 5) ? -fTMSize/2 : -fTMSize);
         glVertex3f(ayX, yt, 0);
         glVertex3f(((Int_t(yt*10) % 5) ? 1.0125f : 1.025f)*ayX, yt, 0);
         yt += ys;
      }
      glEnd();
   } 
}

//______________________________________________________________________________
Int_t TEveCaloLegoGL::GetGridStep(Int_t axId, const TAxis* ax, TGLRnrCtx &rnrCtx) const
{ 
   // Calculate view-dependent step for grid rendering.

   GLdouble xp0, yp0, zp0, xp1, yp1, zp1;
   GLdouble mm[16];
   GLint    vp[4];
   glGetDoublev(GL_MODELVIEW_MATRIX,  mm);
   glGetIntegerv(GL_VIEWPORT, vp);
   const GLdouble *pm = rnrCtx.RefCamera().RefLastNoPickProjM().CArr();

   for (Int_t idx =0; idx<fNBinSteps; ++idx)
   {
      if (axId == 0)
      {
         gluProject(ax->GetBinLowEdge(0), 0, 0, mm, pm, vp, &xp0, &yp0, &zp0);
         gluProject(ax->GetBinLowEdge(fBinSteps[idx]), 0, 0, mm, pm, vp, &xp1, &yp1, &zp1);
      }
      else
      {
         gluProject(0, ax->GetBinLowEdge(0), 0, mm, pm, vp, &xp0, &yp0, &zp0);
         gluProject(0, ax->GetBinLowEdge(fBinSteps[idx]), 0, mm, pm, vp, &xp1, &yp1, &zp1);
      }

      Float_t  gap = TMath::Sqrt((xp0-xp1)*(xp0-xp1) + (yp0-yp1)*(yp0-yp1));
      if (gap>fMinBinWidth) 
      {
         return fBinSteps[idx];
      }
   }
   return fBinSteps[fNBinSteps-1];
}

//______________________________________________________________________________
void TEveCaloLegoGL::DrawHistBase(TGLRnrCtx &rnrCtx) const
{
  // Draw basic histogram components: x-y grid and axes with
  // ticks-marks and labels.

  if (rnrCtx.Selection() || rnrCtx.Highlight())
    return;

  const TAxis* ax = fM->fData->GetEtaBins();
  const TAxis* ay = fM->fData->GetPhiBins();
  Float_t x0 = ax->GetBinLowEdge(0);
  Float_t x1 = ax->GetBinUpEdge(ax->GetNbins());
  Float_t y0 = ay->GetBinLowEdge(0);
  Float_t y1 = ay->GetBinUpEdge(ay->GetNbins());
  // look what is projected line density
  Int_t xs = GetGridStep(0, ax, rnrCtx);
  Int_t ys = GetGridStep(1, ay, rnrCtx);

  TGLCapabilitySwitch lights_off(GL_LIGHTING, kFALSE);
  TGLCapabilitySwitch sw_blend(GL_BLEND, kTRUE);
  TGLUtil::Color(fM->fGridColor);
  {
    glBegin(GL_LINES);
    // eta lines
    glVertex2f(ax->GetBinLowEdge(0), y0);
    glVertex2f(ax->GetBinLowEdge(0), y1);
    for (Int_t i=1; i<ax->GetNbins(); i+=xs)
    {
      glVertex2f(ax->GetBinUpEdge(i), y0);
      glVertex2f(ax->GetBinUpEdge(i), y1);
    }
    glVertex2f(ax->GetBinUpEdge(ax->GetNbins()), y0);
    glVertex2f(ax->GetBinUpEdge(ax->GetNbins()), y1);

    // phi lines
    glVertex2f(x0, ay->GetBinLowEdge(0));
    glVertex2f(x1, ay->GetBinLowEdge(0));
    for (Int_t i=1; i<ay->GetNbins(); i+=ys)
    {
      glVertex2f(x0, ay->GetBinUpEdge(i));
      glVertex2f(x1, ay->GetBinUpEdge(i));
    }   
    glVertex2f(x0, ay->GetBinUpEdge(ay->GetNbins()));
    glVertex2f(x1, ay->GetBinUpEdge(ay->GetNbins()));
    glEnd();
  }
  DrawZAxis(rnrCtx, x0, x1, y0, y1);
  DrawXYAxis(rnrCtx, x0, x1, y0, y1);
}
//______________________________________________________________________________
void TEveCaloLegoGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // Draw the object.

   if (!fM->fCacheOK)
   { 
      fDLCacheOK = kFALSE;
      fM->ResetCache();
      fM->fData->GetCellList((fM->fEtaMin + fM->fEtaMax)*0.5f, fM->fEtaMax - fM->fEtaMin,
                             fM->fPhi, fM->fPhiRng, fM->fThreshold, fM->fCellList);
      printf("GetCellList threshold %f\n",  fM->fThreshold);
      fM->fCacheOK = kTRUE;
   }

   fM->AssertPalette();
   if (fM->fProjection == TEveCaloLego::kAuto)
   {
      const TGLMatrix& mvm = rnrCtx.RefCamera().RefModelViewMatrix();
      TGLVector3 z (0, 0, 1);
      TGLVector3 zp(mvm.Rotate(z));
      Double_t theta = TMath::ACos(Dot(z, zp)/zp.Mag());
      if (theta > 0.15)
         Draw3D(rnrCtx);
      else
         Draw2D(rnrCtx);
   }
   else if (fM->fProjection == TEveCaloLego::k3D)
   {
      Draw3D(rnrCtx);
   }
   else {
      Draw2D(rnrCtx);
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::Draw3D(TGLRnrCtx & rnrCtx) const
{
   // Render the calo lego-plot with OpenGL.
 
   DrawHistBase(rnrCtx);

   if (!fDLCacheOK) MakeDisplayList();
   glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT);
   glPushMatrix();
   glScalef(1.f, 1.f, fM->fCellZScale);
   // cell quads
   {
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      glEnable(GL_POLYGON_OFFSET_FILL);
      glEnable(GL_NORMALIZE);

      glPolygonOffset(0.2, 1);
      glTranslatef(0, 0, 0.001);

      glPushName(0);
      for(std::map<Int_t, UInt_t>::iterator i=fDLMap.begin(); i!=fDLMap.end(); i++)
      {
         TGLUtil::Color(fM->GetPalette()->GetDefaultColor()+i->first);
         glCallList(i->second);
      }
      glPopName();
   }
   // cell outlines
   {
      if (rnrCtx.SceneStyle() == TGLRnrCtx::kFill)
      {
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
         glDisable(GL_POLYGON_OFFSET_FILL);
         TGLUtil::Color(1); 
         for(std::map<Int_t, UInt_t>::iterator i=fDLMap.begin(); i!=fDLMap.end(); i++)
            glCallList(i->second);
      }
   }
   glPopMatrix();
   glPopAttrib();
}

//______________________________________________________________________________
void TEveCaloLegoGL::Draw2DValColor(Int_t nEta, Int_t nPhi) const
{
   // Draw rebined histogram.

   using namespace TMath;
   const TAxis* ax = fM->fData->GetEtaBins();
   const TAxis* ay = fM->fData->GetPhiBins();

   std::vector<Float_t> vec;
   Float_t eta0 = ax->GetBinLowEdge(0);
   Float_t etaT = ax->GetBinUpEdge(ax->GetNbins());
   Float_t phi0 = ay->GetBinLowEdge(0);
   Float_t phiT = ay->GetBinUpEdge(ay->GetNbins());

   Float_t etaStep = (etaT-eta0)/nEta;
   Float_t phiStep = (phiT-phi0)/nPhi;

   vec.assign(nEta*nPhi, 0.f);
   TEveCaloData::CellData_t cd;
   Float_t left, right, up, down; // cell corners
   for (TEveCaloData::vCellId_t::iterator it=fM->fCellList.begin(); it!=fM->fCellList.end(); it++)
   {
      fM->fData->GetCellData(*it, cd);
      Int_t iMin = FloorNint((cd.EtaMin()- eta0)/etaStep);
      Int_t iMax = CeilNint ((cd.EtaMax()- eta0)/etaStep);
      Int_t jMin = FloorNint((cd.PhiMin()- phi0)/phiStep);
      Int_t jMax = CeilNint ((cd.PhiMax()- phi0)/phiStep);

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

   /**************************************************************************/
 
   Float_t maxv = 0;
   Float_t surf = phiStep*etaStep;
   for (std::vector<Float_t>::iterator it=vec.begin(); it !=vec.end(); it++)
   {
      (*it) /= surf;
      if (*it > maxv) maxv = *it;
   }

   UChar_t col[4];
   Float_t min = fM->fPalette->GetMinVal();
   Float_t max = fM->fPalette->GetMaxVal();
   fM->fPalette->SetMax(Int_t(maxv));
   Float_t val;
   Int_t cid = 0;
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   glBegin(GL_QUADS);
   for (std::vector<Float_t>::iterator it=vec.begin(); it !=vec.end(); it++)
   { 
      val = *it;
      if (val>min && val<max)
      {
         fM->fPalette->ColorFromValue(FloorNint(val), col);
         TGLUtil::Color4ubv(col);
	
         Float_t eta = Int_t(cid/nPhi)*etaStep + eta0;
         Float_t phi = (cid- Int_t(cid/nPhi)*nPhi)*phiStep + phi0;
         {
            glVertex2f(eta        , phi);
            glVertex2f(eta+etaStep, phi);
            glVertex2f(eta+etaStep, phi+phiStep);
            glVertex2f(eta        , phi+phiStep);
         }
      }
      cid++;
   }
   glEnd();
   fM->fPalette->SetMax(max);

   // draw grid
   TGLUtil::Color(fM->fGridColor);
   glBegin(GL_LINES);
   Float_t a = eta0;
   for (Int_t i=0; i<=nEta; i++)
   {
      glVertex2f(a, phi0);
      glVertex2f(a, phiT);
      a += etaStep;
   }
   a = phi0;
   for (Int_t i=0; i<=nPhi; i++)
   {
      glVertex2f(eta0, a);
      glVertex2f(etaT, a);
      a += phiStep;
   }
   glEnd();
}

//______________________________________________________________________________
void TEveCaloLegoGL::Draw2DValSize() const
{
   // Draw 2D histogram with original bins.

   const TAxis* ax = fM->fData->GetEtaBins();
   const TAxis* ay = fM->fData->GetPhiBins();
   Float_t eds = 0.5f*(ax->GetBinUpEdge(ax->GetNbins())- ax->GetBinLowEdge(0))/ax->GetNbins(); // standard step
   Float_t pds = 0.5f*(ay->GetBinUpEdge(ay->GetNbins())-ay->GetBinLowEdge(0))/fM->fData->GetPhiBins()->GetNbins(); // standard step
   Float_t ed, pd;
   Float_t e, p; 

   TEveCaloData::CellData_t cd;
   Float_t min = fM->fPalette->GetMinVal();
   Float_t max = fM->fPalette->GetMaxVal();
   Int_t prevTower = 0;
   Float_t sum = 0;

   for (TEveCaloData::vCellId_t::iterator it=fM->fCellList.begin(); it!=fM->fCellList.end(); it++)
   {
      fM->fData->GetCellData(*it, cd);     
      if ((*it).fTower == prevTower)
      {
         sum += cd.Value();
      }
      else 
      {
         sum /= fM->fData->GetNSlices();
         if (sum>min && sum<max)
         {
            e = cd.Eta();
            p = cd.Phi();
            if (sum > 1)
            {
               UChar_t c[4];
               fM->fPalette->ColorFromValue(TMath::Nint(sum), c);
               TGLUtil::Color4ubv(c);

               ed = cd.EtaDelta()*0.5;
               pd = cd.PhiDelta()*0.5;

               glBegin(GL_QUADS);
               glVertex2f(e-ed, p-pd);
               glVertex2f(e+ed, p-pd);
               glVertex2f(e+ed, p+pd);
               glVertex2f(e-ed, p+pd);
               glEnd();

               TGLUtil::Color(1);
               glBegin(GL_LINE_LOOP);
               glVertex2f(e-ed, p-pd);
               glVertex2f(e+ed, p-pd);
               glVertex2f(e+ed, p+pd);
               glVertex2f(e-ed, p+pd);
               glEnd();
            }
            else
            {
               TGLUtil::Color(5);
               ed = eds*0.25f;
               pd = pds*0.25f;

               glBegin(GL_QUADS);
               glVertex2f(e-ed, p-pd);
               glVertex2f(e+ed, p-pd);
               glVertex2f(e+ed, p+pd);
               glVertex2f(e-ed, p+pd);
               glEnd();

               glBegin(GL_POINTS);
               glVertex2f(e, p);
               glEnd();
            } 
         }
         sum = 0;
         prevTower = (*it).fTower;
      }
   }
}

//______________________________________________________________________________
void TEveCaloLegoGL::Draw2D(TGLRnrCtx & rnrCtx) const
{
   // Draw projected histogram.

   using namespace TMath;
   TGLMatrix   mvm;
   glGetDoublev(GL_MODELVIEW_MATRIX, mvm.Arr());
   TGLVector3 s = mvm.GetScale();
   TGLVector3 t = mvm.GetBaseVec(4);
   glPushMatrix();   
   glLoadIdentity();
   glTranslated(t.X(), t.Y(), t.Z());
   glScaled(s.X(), s.Y(), s.Z());

   const TAxis* ax = fM->fData->GetEtaBins();
   const TAxis* ay = fM->fData->GetPhiBins();
   DrawZAxisSimplified(rnrCtx, ax->GetBinLowEdge(0), ay->GetBinLowEdge(0));
   DrawXYAxis(rnrCtx, ax->GetBinLowEdge(0),ax->GetBinUpEdge(ax->GetNbins()),
              ay->GetBinLowEdge(0),ay->GetBinUpEdge(ay->GetNbins()));

   glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT);
   glDisable(GL_LIGHTING); 
   glEnable(GL_POLYGON_OFFSET_FILL);
   glPolygonOffset(0.2, 1); 
   glTranslatef(0, 0, 0.001);

   Int_t es = GetGridStep(0, ax, rnrCtx);
   Int_t ps = GetGridStep(1, ay, rnrCtx);

   if (fM->f2DMode == TEveCaloLego::kValColor)
      Draw2DValColor(ax->GetNbins()/es, ay->GetNbins()/ps);
   else
      Draw2DValSize();

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
