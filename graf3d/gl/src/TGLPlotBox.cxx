// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  31/08/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <algorithm>

#include "TColor.h"

#include "TGLIncludes.h"
#include "TGLPlotBox.h"

//______________________________________________________________________________
//
// Implementation of a box around a histogram/function for
// plot-painters.

ClassImp(TGLPlotBox)

const Int_t TGLPlotBox::fgFramePlanes[][4] =
   {
    {0, 4, 5, 1},
    {1, 5, 6, 2},
    {2, 6, 7, 3},
    {0, 3, 7, 4},
    {0, 1, 2, 3}
   };
const Double_t TGLPlotBox::fgNormals[][3] =
   {
    { 0., 1., 0.},
    {-1., 0., 0.},
    { 0.,-1., 0.},
    { 1., 0., 0.},
    { 0., 0., 1.}
   };
const Int_t TGLPlotBox::fgBackPairs[][2] =
   {
    {2, 1},
    {3, 2},
    {0, 3},
    {1, 0}
   };

const Int_t TGLPlotBox::fgFrontPairs[][2] = 
   {
    {3, 0},
    {0, 1},
    {1, 2},
    {2, 3}
   };
//______________________________________________________________________________
TGLPlotBox::TGLPlotBox(Bool_t xoy, Bool_t xoz, Bool_t yoz)
               : fFrameColor(0),
                 fXOYSelectable(xoy),
                 fXOZSelectable(xoz),
                 fYOZSelectable(yoz),
                 fSelectablePairs(),
                 fFrontPoint(0),
                 fRangeXU(1.),
                 fRangeYU(1.),
                 fRangeZU(1.),
                 fDrawBack(kTRUE),
                 fDrawFront(kTRUE)
{
   // Constructor.
   //Front point is 0.
   fSelectablePairs[0][0] = xoz;
   fSelectablePairs[0][1] = yoz;
   //Front point is 1.
   fSelectablePairs[1][0] = yoz;
   fSelectablePairs[1][1] = xoz;
   //Front point is 2.
   fSelectablePairs[2][0] = xoz;
   fSelectablePairs[2][1] = yoz;
   //Front point is 3.
   fSelectablePairs[3][0] = yoz;
   fSelectablePairs[3][1] = xoz;
}


//______________________________________________________________________________
TGLPlotBox::~TGLPlotBox()
{
   // Empty dtor to suppress g++ warnings.
}

//______________________________________________________________________________
void TGLPlotBox::DrawBack(Int_t selected, Bool_t selectionPass, const std::vector<Double_t> &zLevels,
                          Bool_t highColor)const
{
   using namespace Rgl;
   TGLDisableGuard depthTest(GL_DEPTH_TEST); //[0-0]
   glDepthMask(GL_FALSE);//[1

   if (!selectionPass) {
      glEnable(GL_BLEND);//[2
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
      glEnable(GL_LINE_SMOOTH);//[3
   }

   //Back planes are partially transparent to make their color smoother.
   Float_t backColor[] = {0.9f, 0.9f, 0.9f, 0.85f};
   if (fFrameColor)
      fFrameColor->GetRGB(backColor[0], backColor[1], backColor[2]);

   if (!selectionPass) {
      glMaterialfv(GL_FRONT, GL_DIFFUSE, backColor);
      if (selected == 1) {
         fXOYSelectable ?
                        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gGreenEmission)
                        :
                        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gRedEmission);
      }
   } else
      ObjectIDToColor(1, highColor);//Bottom plane, encoded as 1 in a selection buffer.

   DrawQuadFilled(f3DBox[0], f3DBox[1], f3DBox[2], f3DBox[3], TGLVector3(0., 0., 1.));

   if (!selectionPass) {
      if (selected == 1)
         glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gNullEmission);
      else if (selected == 2)
         fSelectablePairs[fFrontPoint][0] ? glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gGreenEmission)
                                          : glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gRedEmission);
   } else
      ObjectIDToColor(2, highColor);//Left plane, encoded as 2 in a selection buffer.

   DrawBackPlane(fgBackPairs[fFrontPoint][0], selectionPass, zLevels);

   if (!selectionPass) {
      if (selected == 2)
         glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gNullEmission);
      else if (selected == 3)
         fSelectablePairs[fFrontPoint][1] ? glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gGreenEmission)
                                          : glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gRedEmission);
   } else
      ObjectIDToColor(3, highColor); //Right plane, encoded as 3 in a selection buffer.

   DrawBackPlane(fgBackPairs[fFrontPoint][1], selectionPass, zLevels);

   glDepthMask(GL_TRUE);//1]
   if (!selectionPass) {
      if (selected == 3)
         glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gNullEmission);
      glDisable(GL_BLEND);//2]
      glDisable(GL_LINE_SMOOTH);//3]
   }
}

//______________________________________________________________________________
void TGLPlotBox::DrawFront()const
{
   using namespace Rgl;

   const TGLDisableGuard lightGuard(GL_LIGHTING);
//   const TGLEnableGuard blend(GL_BLEND);
//   const TGLEnableGuard lineSmooth(GL_LINE_SMOOTH);
   
  // glColor4d(0., 0., 0., 0.8);
   glColor3d(0., 0., 0.);
   
   const Int_t *vertInd = fgFramePlanes[fgFrontPairs[fFrontPoint][0]];   
   DrawQuadOutline(f3DBox[vertInd[0]], f3DBox[vertInd[1]], f3DBox[vertInd[2]], f3DBox[vertInd[3]]);
   
   vertInd = fgFramePlanes[fgFrontPairs[fFrontPoint][1]];
   DrawQuadOutline(f3DBox[vertInd[0]], f3DBox[vertInd[1]], f3DBox[vertInd[2]], f3DBox[vertInd[3]]);
}

//______________________________________________________________________________
void TGLPlotBox::DrawBox(Int_t selected, Bool_t selectionPass, const std::vector<Double_t> &zLevels,
                         Bool_t highColor)const
{
   // Draw back box for a plot.
   if (fDrawBack)
      DrawBack(selected, selectionPass, zLevels, highColor);

   if (fDrawFront && !selectionPass)
      DrawFront();
}


//______________________________________________________________________________
void TGLPlotBox::SetPlotBox(const Rgl::Range_t &x, const Rgl::Range_t &y, const Rgl::Range_t &z)
{
   // Set up a frame box.

   f3DBox[0].Set(x.first,  y.first,  z.first);
   f3DBox[1].Set(x.second, y.first,  z.first);
   f3DBox[2].Set(x.second, y.second, z.first);
   f3DBox[3].Set(x.first,  y.second, z.first);
   f3DBox[4].Set(x.first,  y.first,  z.second);
   f3DBox[5].Set(x.second, y.first,  z.second);
   f3DBox[6].Set(x.second, y.second, z.second);
   f3DBox[7].Set(x.first,  y.second, z.second);
}

//______________________________________________________________________________
void TGLPlotBox::SetPlotBox(const Rgl::Range_t &x, Double_t xr, const Rgl::Range_t &y, Double_t yr,
                            const Rgl::Range_t &z, Double_t zr)
{
   // Set up a frame box.
   fRangeXU = xr;
   fRangeYU = yr;
   fRangeZU = zr;

   SetPlotBox(x, y, z);
}

//______________________________________________________________________________
void TGLPlotBox::SetFrameColor(const TColor *color)
{
   // Back box color.

   fFrameColor = color;
}

namespace {

   bool Compare(const TGLVertex3 &v1, const TGLVertex3 &v2)
   {
      return v1.Z() < v2.Z();
   }

}


//______________________________________________________________________________
Int_t TGLPlotBox::FindFrontPoint()const
{
   // Convert 3d points into window coordinate system
   // and find the nearest.

   Double_t mvMatrix[16] = {0.};
   glGetDoublev(GL_MODELVIEW_MATRIX, mvMatrix);
   Double_t prMatrix[16] = {0.};
   glGetDoublev(GL_PROJECTION_MATRIX, prMatrix);
   Int_t viewport[4] = {0};
   glGetIntegerv(GL_VIEWPORT, viewport);

   const Double_t zMin = f3DBox[0].Z();
   const Double_t zMax = f3DBox[4].Z();

   const Double_t uBox[][2] = {{-fRangeXU / 2., -fRangeYU / 2.},
                               { fRangeXU / 2., -fRangeYU / 2.},
                               { fRangeXU / 2.,  fRangeYU / 2.},
                               {-fRangeXU / 2.,  fRangeYU / 2.}};
   //const Double_t uBox[][2] = {{-1., -1.}, {1., -1.}, {1., 1.}, {-1., 1.}};


   for (Int_t i = 0; i < 4; ++i) {
      gluProject(f3DBox[i].X(), f3DBox[i].Y(), zMin, mvMatrix, prMatrix, viewport,
                 &f2DBox[i].X(), &f2DBox[i].Y(), &f2DBox[i].Z());
      gluProject(f3DBox[i].X(), f3DBox[i].Y(), zMax, mvMatrix, prMatrix, viewport,
                 &f2DBox[i + 4].X(), &f2DBox[i + 4].Y(), &f2DBox[i + 4].Z());

      gluProject(uBox[i][0], uBox[i][1], -0.5, mvMatrix, prMatrix, viewport,
                 &f2DBoxU[i].X(), &f2DBoxU[i].Y(), &f2DBoxU[i].Z());
      gluProject(uBox[i][0], uBox[i][1], 0.5, mvMatrix, prMatrix, viewport,
                 &f2DBoxU[i + 4].X(), &f2DBoxU[i + 4].Y(), &f2DBoxU[i + 4].Z());
   }

   //return fFrontPoint = std::min_element(f2DBox, f2DBox + 4, Compare) - f2DBox;
   return fFrontPoint = std::min_element(f2DBoxU, f2DBoxU + 4, Compare) - f2DBoxU;
}


//______________________________________________________________________________
Int_t TGLPlotBox::GetFrontPoint()const
{
   // The nearest point.

   return fFrontPoint;
}


//______________________________________________________________________________
const TGLVertex3 *TGLPlotBox::Get3DBox()const
{
   // Get 3D box.

   return f3DBox;
}


//______________________________________________________________________________
const TGLVertex3 *TGLPlotBox::Get2DBox()const
{
   // Get 2D box.

//   return f2DBox;
   return f2DBoxU;
}


//______________________________________________________________________________
void TGLPlotBox::DrawBackPlane(Int_t plane, Bool_t selectionPass,
                               const std::vector<Double_t> &zLevels)const
{
   //Draw back plane with number 'plane'
   using namespace Rgl;
   const Int_t *vertInd = fgFramePlanes[plane];
   DrawQuadFilled(f3DBox[vertInd[0]], f3DBox[vertInd[1]], f3DBox[vertInd[2]],
                  f3DBox[vertInd[3]], fgNormals[plane]);
   //Antialias back plane outline.
   if (!selectionPass) {
      const TGLDisableGuard lightGuard(GL_LIGHTING);
      glColor3d(0., 0., 0.);
      DrawQuadOutline(f3DBox[vertInd[0]], f3DBox[vertInd[1]],
                      f3DBox[vertInd[2]], f3DBox[vertInd[3]]);
      //draw grid.
      const TGLEnableGuard stippleGuard(GL_LINE_STIPPLE);//[1-1]
      const UShort_t stipple = 0x5555;
      glLineStipple(1, stipple);

      Double_t lineCaps[][4] =
      {
         {f3DBox[1].X(), f3DBox[0].Y(), f3DBox[0].X(), f3DBox[0].Y()},
         {f3DBox[1].X(), f3DBox[0].Y(), f3DBox[1].X(), f3DBox[2].Y()},
         {f3DBox[1].X(), f3DBox[2].Y(), f3DBox[0].X(), f3DBox[3].Y()},
         {f3DBox[0].X(), f3DBox[3].Y(), f3DBox[0].X(), f3DBox[0].Y()}
      };

      for (UInt_t i = 0; i < zLevels.size(); ++i) {
         glBegin(GL_LINES);
         glVertex3d(lineCaps[plane][0], lineCaps[plane][1], zLevels[i]);
         glVertex3d(lineCaps[plane][2], lineCaps[plane][3], zLevels[i]);
         glEnd();
      }

   }
}
