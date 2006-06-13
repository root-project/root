#include <algorithm>

#include "TError.h"
#include "TPoint.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TMath.h"
#include "TH1.h"


#include "TGLPlotPainter.h"
#include "TGLIncludes.h"
#include "TGLQuadric.h"
#include "TGLUtil.h"

ClassImp(TGLPlotPainter)
ClassImp(TGLPlotFrame)

const Int_t TGLPlotFrame::fFramePlanes[][4] = {
                                               {0, 4, 5, 1},
                                               {1, 5, 6, 2},
                                               {2, 6, 7, 3},
                                               {0, 3, 7, 4},
                                               {0, 1, 2, 3}
                                              };

const Double_t TGLPlotFrame::fFrameNormals[][3] = {
                                                   { 0., 1., 0.},
                                                   {-1., 0., 0.},
                                                   { 0.,-1., 0.},
                                                   { 1., 0., 0.},
                                                   { 0., 0., 1.}
                                                  };

const Int_t TGLPlotFrame::fBackPairs[][2] = {
                                             {2, 1},
                                             {3, 2},
                                             {0, 3},
                                             {1, 0}
                                            };

//______________________________________________________________________________
TGLPlotFrame::TGLPlotFrame(Bool_t logX, Bool_t logY, Bool_t logZ)
                : fLogX(logX),
                  fLogY(logY),
                  fLogZ(logZ),
                  fScaleX(1.),
                  fScaleY(1.),
                  fScaleZ(1.),
                  fFrontPoint(0),
                  fViewport(),
                  fZoom(1.),
                  fFrustum(),
                  fShift(0.),
                  fCenter(),
                  fFactor(1.)
{
}

//______________________________________________________________________________
TGLPlotFrame::~TGLPlotFrame()
{
}

namespace {

   Double_t FindMinBinWidth(const TAxis *axis)
   {
      Int_t currBin = axis->GetFirst();
      Double_t width = axis->GetBinWidth(currBin);

      if (!axis->IsVariableBinSize())//equal bins
         return width;

      ++currBin;
      //variable size bins
      for (const Int_t lastBin = axis->GetLast(); currBin <= lastBin; ++currBin)
         width = TMath::Min(width, axis->GetBinWidth(currBin));

      return width;
   }

}

//______________________________________________________________________________
Bool_t TGLPlotFrame::ExtractAxisInfo(const TAxis *axis, Bool_t log, BinRange_t &bins, Range_t &range)
{
   //"Generic" function, can be used for X/Y/Z axis.
   //[low edge of first ..... up edge of last]
   //If log is true, at least up edge of last MUST be positive or function fails (1).
   //If log is true and low edge is negative, try to find bin with positive low edge, bin number
   //must be less or equal to last (2). If no such bin, function failes.
   //When looking for a such bin, I'm trying to find value which is 0.01 of
   //MINIMUM bin width (3) (if bins are equidimensional, first's bin width is OK).
   //But even such lookup can fail, so, it's a stupid idea to have negative ranges
   //and logarithmic scale :)

   bins.first = axis->GetFirst(), bins.second = axis->GetLast();
   range.first = axis->GetBinLowEdge(bins.first), range.second = axis->GetBinUpEdge(bins.second);

   if (log) {
      if (range.second <= 0.)
         return kFALSE;//(1)

      range.second = TMath::Log10(range.second);

      if (range.first <= 0.) {//(2)
         Int_t bin = axis->FindFixBin(FindMinBinWidth(axis) * 0.01);//(3)
         //Overflow or something stupid.
         if (bin > bins.second)
            return kFALSE;
         
         if (axis->GetBinLowEdge(bin) <= 0.) {
            ++bin;
            if (bin > bins.second)//Again, something stupid.
               return kFALSE;
         }

         bins.first = bin;
         range.first = axis->GetBinLowEdge(bin);
      }

      range.first = TMath::Log10(range.first);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLPlotFrame::ExtractAxisZInfo(TH1 *hist, Bool_t logZ, const BinRange_t &xBins, 
                                        const BinRange_t &yBins, Range_t &zRange)
{
   //First, look through hist to find minimum and maximum values.
   const Bool_t minimum = hist->GetMinimumStored() != -1111;
   const Bool_t maximum = hist->GetMaximumStored() != -1111;
   const Double_t margin = gStyle->GetHistTopMargin();

   zRange.second = hist->GetCellContent(xBins.first, yBins.first), zRange.first = zRange.second;
   Double_t summ = 0.;

   for (Int_t i = xBins.first; i <= xBins.second; ++i) {
      for (Int_t j = yBins.first; j <= yBins.second; ++j) {
         Double_t val = hist->GetCellContent(i, j);
         zRange.second = TMath::Max(val, zRange.second);
         zRange.first = TMath::Min(val, zRange.first);
         summ += val;
      }
   }

   if (hist->GetMaximumStored() != -1111) 
      zRange.second = hist->GetMaximumStored();
   if (hist->GetMinimumStored() != -1111) 
      zRange.first = hist->GetMinimumStored();

   if (logZ && zRange.second <= 0.)
      return kFALSE;//cannot setup logarithmic scale
   
   if (zRange.first >= zRange.second)
      zRange.first = 0.001 * zRange.second;

   //Here is a strange (for me) magic with factor. This code is the same as in THistPainter,
   //to get the same behavior. But (IMHO) it's incorrect, because summ can be negative
   //and we can still have positive bins, which will be "truncated" by negative zMax.
   fFactor = hist->GetNormFactor() > 0. ? hist->GetNormFactor() : summ;
   if (summ) fFactor /= summ;
   if (!fFactor) fFactor = 1.;
   if (fFactor < 0.)
      Warning("TGLPlotPainter::ExtractAxisZInfo", 
              "Negative factor, negative ranges - possible incorrect behavior");

   zRange.second *= fFactor;
   zRange.first *= fFactor;

   if (logZ) {
      if (zRange.first <= 0.)
         zRange.first = TMath::Min(1., 0.001 * zRange.second);
      zRange.first = TMath::Log10(zRange.first);
      if (!minimum) 
         zRange.first += TMath::Log10(0.5);
      zRange.second = TMath::Log10(zRange.second);
      if (!maximum)
         zRange.second += TMath::Log10(2*(0.9/0.95));//This magic numbers are from THistPainter.
      return kTRUE;
   }

   if (!maximum)
      zRange.second += margin * (zRange.second - zRange.first);
   if (!minimum) {
      if (gStyle->GetHistMinimumZero())
         zRange.first >= 0 ? zRange.first = 0. : zRange.first -= margin * (zRange.second - zRange.first);
      else 
         zRange.first >= 0 && zRange.first - margin * (zRange.second - zRange.first) <= 0 ?
            zRange.first = 0 : zRange.first -= margin * (zRange.second - zRange.first);
   }

   return kTRUE;
}

//______________________________________________________________________________
void TGLPlotFrame::AdjustShift(const TPoint &p1, const TPoint &p2, TGLVector3 &shiftVec, 
                               const Int_t *viewport)
{
   //Extract gl matrices.
   Double_t mv[16] = {0.};
   glGetDoublev(GL_MODELVIEW_MATRIX, mv);
   Double_t pr[16] = {0.};
   glGetDoublev(GL_PROJECTION_MATRIX, pr);
   //Adjust pan vector.
   TGLVertex3 start, end;
   gluUnProject(p1.fX, p1.fY, 1., mv, pr, viewport, &start.X(), &start.Y(), &start.Z());
   gluUnProject(p2.fX, p2.fY, 1., mv, pr, viewport, &end.X(), &end.Y(), &end.Z());
   shiftVec += (start - end) /= 2.;
}

//______________________________________________________________________________
void TGLPlotFrame::CalculateGLCameraParams(const Range_t &x, const Range_t &y, const Range_t &z)
{
   //Finds the maximum dimension and adjust scale coefficients
   const Double_t xRange = x.second - x.first;
   const Double_t yRange = y.second - y.first;
   const Double_t zRange = z.second - z.first;
   const Double_t maxDim = TMath::Max(TMath::Max(xRange, yRange), zRange);

   fScaleX = maxDim / xRange;
   fScaleY = maxDim / yRange;
   fScaleZ = maxDim / zRange;

   const Double_t xMin = x.first * fScaleX, xMax = x.second * fScaleX;
   const Double_t yMin = y.first * fScaleY, yMax = y.second * fScaleY;
   const Double_t zMin = z.first * fScaleZ/*z.first > 0. ? 0. : z.first * fScaleZ*/, zMax = z.second * fScaleZ;

   fFrame[0].Set(xMin, yMin, zMin);
   fFrame[1].Set(xMax, yMin, zMin);
   fFrame[2].Set(xMax, yMax, zMin);
   fFrame[3].Set(xMin, yMax, zMin);
   fFrame[4].Set(xMin, yMin, zMax);
   fFrame[5].Set(xMax, yMin, zMax);
   fFrame[6].Set(xMax, yMax, zMax);
   fFrame[7].Set(xMin, yMax, zMax);

   fCenter[0] = x.first + xRange / 2;
   fCenter[1] = y.first + yRange / 2;
   fCenter[2] = z.first + zRange / 2;

   fFrustum[0] = maxDim;
   fFrustum[1] = maxDim;
   fFrustum[2] = -100 * maxDim;
   fFrustum[3] = 100 * maxDim;
   fShift = maxDim * 1.5;

 //  std::cout<<"scales "<<fScaleX<<' '<<fScaleY<<' '<<fScaleZ<<std::endl;
}

namespace {

   bool Compare(const TGLVertex3 &v1, const TGLVertex3 &v2)
   {
      return v1.Z() < v2.Z();
   }

}

//______________________________________________________________________________
void TGLPlotFrame::FindFrontPoint()
{
   //Convert 3d points into window coordinate system
   //and find the nearest.
   Double_t mvMatrix[16] = {0.};
   glGetDoublev(GL_MODELVIEW_MATRIX, mvMatrix);
   Double_t prMatrix[16] = {0.};
   glGetDoublev(GL_PROJECTION_MATRIX, prMatrix);

   const Double_t zMin = fFrame[0].Z();
   const Double_t zMax = fFrame[4].Z();

   for (Int_t i = 0; i < 4; ++i) {
      gluProject(fFrame[i].X(), fFrame[i].Y(), zMin, mvMatrix, prMatrix, fViewport,
                 &f2DAxes[i].X(), &f2DAxes[i].Y(), &f2DAxes[i].Z());
      gluProject(fFrame[i].X(), fFrame[i].Y(), zMax, mvMatrix, prMatrix, fViewport,
                 &f2DAxes[i + 4].X(), &f2DAxes[i + 4].Y(), &f2DAxes[i + 4].Z());
   }

   fFrontPoint = std::min_element(f2DAxes, f2DAxes + 4, ::Compare) - f2DAxes;
}

//______________________________________________________________________________
void TGLPlotFrame::SetTransformation()
{
   //Applies rotations and translations before drawing
   glTranslated(0., 0., -fShift);
   glMultMatrixd(fArcBall.GetRotMatrix());
   glRotated(45., 1., 0., 0.);
   glRotated(-45., 0., 1., 0.);
   glRotated(-90., 0., 1., 0.);
   glRotated(-90., 1., 0., 0.);
   glTranslated(-fPan[0], -fPan[1], -fPan[2]);
   glTranslated(-fCenter[0] * fScaleX, -fCenter[1] * fScaleY, -fCenter[2] * fScaleZ);
}

//______________________________________________________________________________
void TGLPlotFrame::SetCamera()
{
   //Viewport and projection.
   glViewport(fViewport[0], fViewport[1], fViewport[2], fViewport[3]);

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(
           -fFrustum[0] * fZoom,
            fFrustum[0] * fZoom, 
           -fFrustum[1] * fZoom, 
            fFrustum[1] * fZoom, 
            fFrustum[2], 
            fFrustum[3]
          );
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

namespace RootGL
{
   //______________________________________________________________________________
   void DrawCylinder(TGLQuadric *quadric, Double_t xMin, Double_t xMax, Double_t yMin, 
                     Double_t yMax, Double_t zMin, Double_t zMax)
   {
      //Cylinder for lego3.
      GLUquadric *quad = quadric->Get();

      if (quad) {
         if (zMin > zMax)
            std::swap(zMin, zMax);
         const Double_t xCenter = xMin + (xMax - xMin) / 2;
         const Double_t yCenter = yMin + (yMax - yMin) / 2;
         const Double_t radius = TMath::Min((xMax - xMin) / 2, (yMax - yMin) / 2);

         glPushMatrix();
         glTranslated(xCenter, yCenter, zMin);
         gluCylinder(quad, radius, radius, zMax - zMin, 40, 1);
         glPopMatrix();
         glPushMatrix();
         glTranslated(xCenter, yCenter, zMax);
         gluDisk(quad, 0., radius, 40, 1);
         glPopMatrix();
         glPushMatrix();
         glTranslated(xCenter, yCenter, zMin);
         glRotated(180., 0., 1., 0.);
         gluDisk(quad, 0., radius, 40, 1);
         glPopMatrix();
      }
   }

   //______________________________________________________________________________
   void DrawQuadOutline(const TGLVertex3 &v1, const TGLVertex3 &v2, 
                        const TGLVertex3 &v3, const TGLVertex3 &v4)
   {
      //Outline.
      glBegin(GL_LINE_LOOP);
      glVertex3dv(v1.CArr());
      glVertex3dv(v2.CArr());
      glVertex3dv(v3.CArr());
      glVertex3dv(v4.CArr());
      glEnd();
   }

   //______________________________________________________________________________
   void DrawQuadFilled(const TGLVertex3 &v0, const TGLVertex3 &v1, const TGLVertex3 &v2,
                       const TGLVertex3 &v3, const TGLVertex3 &normal)
   {
      //Draw quad face.
      glBegin(GL_POLYGON);
      glNormal3dv(normal.CArr());
      glVertex3dv(v0.CArr());
      glVertex3dv(v1.CArr());
      glVertex3dv(v2.CArr());
      glVertex3dv(v3.CArr());
      glEnd();
   }

   const Int_t    gBoxFrontQuads[][4] = {{0, 1, 2, 3}, {4, 0, 3, 5}, {4, 5, 6, 7}, {7, 6, 2, 1}};
   const Double_t gBoxFrontNormals[][3] = {{-1., 0., 0.}, {0., -1., 0.}, {1., 0., 0.}, {0., 1., 0.}};
   const Int_t    gBoxFrontPlanes[][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

   //______________________________________________________________________________
   void DrawBoxFront(Double_t xMin, Double_t xMax, Double_t yMin, Double_t yMax, 
                     Double_t zMin, Double_t zMax, Int_t fp)
   {
      //Draws lego's bar as a 3d box
      if (zMax < zMin) 
         std::swap(zMax, zMin);
      //Top and bottom are always drawn.
      glBegin(GL_POLYGON);
      glNormal3d(0., 0., 1.);
      glVertex3d(xMax, yMin, zMax);
      glVertex3d(xMax, yMax, zMax);
      glVertex3d(xMin, yMax, zMax);
      glVertex3d(xMin, yMin, zMax);
      glEnd();

      glBegin(GL_POLYGON);
      glNormal3d(0., 0., -1.);
      glVertex3d(xMax, yMin, zMin);
      glVertex3d(xMin, yMin, zMin);
      glVertex3d(xMin, yMax, zMin);
      glVertex3d(xMax, yMax, zMin);
      glEnd();
      //Draw two visible front planes.
      const Double_t box[][3] = {{xMin, yMin, zMax}, {xMin, yMax, zMax}, {xMin, yMax, zMin}, {xMin, yMin, zMin},
                                 {xMax, yMin, zMax}, {xMax, yMin, zMin}, {xMax, yMax, zMin}, {xMax, yMax, zMax}};
      const Int_t *verts = gBoxFrontQuads[gBoxFrontPlanes[fp][0]];

      glBegin(GL_POLYGON);
      glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[fp][0]]);
      glVertex3dv(box[verts[0]]);
      glVertex3dv(box[verts[1]]);
      glVertex3dv(box[verts[2]]);
      glVertex3dv(box[verts[3]]);
      glEnd();
      
      verts = gBoxFrontQuads[gBoxFrontPlanes[fp][1]];

      glBegin(GL_POLYGON);
      glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[fp][1]]);
      glVertex3dv(box[verts[0]]);
      glVertex3dv(box[verts[1]]);
      glVertex3dv(box[verts[2]]);
      glVertex3dv(box[verts[3]]);
      glEnd();
   }

   //______________________________________________________________________________
   void DrawBoxFrontTextured(Double_t x1, Double_t x2, Double_t y1, Double_t y2, Double_t z1, 
                             Double_t z2, Double_t texMin, Double_t texMax, Int_t frontPoint)
   {
      //Draws lego's bar as a textured box
      if (z2 < z1) {
         std::swap(z2, z1);
         std::swap(texMin, texMax);
      }

      //Top and bottom are always drawn.
      glBegin(GL_POLYGON);
      glNormal3d(0., 0., 1.);
      glTexCoord1d(texMax); glVertex3d(x2, y1, z2);
      glTexCoord1d(texMax); glVertex3d(x2, y2, z2);
      glTexCoord1d(texMax); glVertex3d(x1, y2, z2);
      glTexCoord1d(texMax); glVertex3d(x1, y1, z2);
      glEnd();

      glBegin(GL_POLYGON);
      glNormal3d(0., 0., -1.);
      glTexCoord1d(texMin); glVertex3d(x2, y1, z1);
      glTexCoord1d(texMin); glVertex3d(x1, y1, z1);
      glTexCoord1d(texMin); glVertex3d(x1, y2, z1);
      glTexCoord1d(texMin); glVertex3d(x2, y2, z1);
      glEnd();

      const Double_t box[][3] = {{x1, y1, z2}, {x1, y2, z2}, {x1, y2, z1}, {x1, y1, z1},
                                 {x2, y1, z2}, {x2, y1, z1}, {x2, y2, z1}, {x2, y2, z2}};
      const Double_t z[] = {texMax, texMax, texMin, texMin, texMax, texMin, texMin, texMax};
      const Int_t *verts = gBoxFrontQuads[gBoxFrontPlanes[frontPoint][0]];

      glBegin(GL_POLYGON);
      glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[frontPoint][0]]);
      glTexCoord1d(z[verts[0]]), glVertex3dv(box[verts[0]]);
      glTexCoord1d(z[verts[1]]), glVertex3dv(box[verts[1]]);
      glTexCoord1d(z[verts[2]]), glVertex3dv(box[verts[2]]);
      glTexCoord1d(z[verts[3]]), glVertex3dv(box[verts[3]]);
      glEnd();
      
      verts = gBoxFrontQuads[gBoxFrontPlanes[frontPoint][1]];

      glBegin(GL_POLYGON);
      glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[frontPoint][1]]);
      glTexCoord1d(z[verts[0]]), glVertex3dv(box[verts[0]]);
      glTexCoord1d(z[verts[1]]), glVertex3dv(box[verts[1]]);
      glTexCoord1d(z[verts[2]]), glVertex3dv(box[verts[2]]);
      glTexCoord1d(z[verts[3]]), glVertex3dv(box[verts[3]]);
      glEnd();
   }

   void DrawTrapezoid(const Double_t ver[][2], Double_t zMin, Double_t zMax, Bool_t color)
   {
      //In polar coordinates, box became trapezoid.
      //Four faces need normal calculations.
      if (zMin > zMax)
         std::swap(zMin, zMax);
      //top
      glBegin(GL_POLYGON);
      glNormal3d(0., 0., 1.);
      glVertex3d(ver[0][0], ver[0][1], zMax);
      glVertex3d(ver[1][0], ver[1][1], zMax);
      glVertex3d(ver[2][0], ver[2][1], zMax);
      glVertex3d(ver[3][0], ver[3][1], zMax);
      glEnd();
      //bottom
      glBegin(GL_POLYGON);
      glNormal3d(0., 0., -1.);
      glVertex3d(ver[0][0], ver[0][1], zMin);
      glVertex3d(ver[3][0], ver[3][1], zMin);
      glVertex3d(ver[2][0], ver[2][1], zMin);
      glVertex3d(ver[1][0], ver[1][1], zMin);
      glEnd();
      //

      Double_t trapezoid[][3] = {{ver[0][0], ver[0][1], zMin}, {ver[1][0], ver[1][1], zMin},
                                 {ver[2][0], ver[2][1], zMin}, {ver[3][0], ver[3][1], zMin},
                                 {ver[0][0], ver[0][1], zMax}, {ver[1][0], ver[1][1], zMax},
                                 {ver[2][0], ver[2][1], zMax}, {ver[3][0], ver[3][1], zMax}};
      Double_t normal[3] = {0.};
      glBegin(GL_POLYGON);
      if (color) {
         TMath::Normal2Plane(trapezoid[1], trapezoid[2], trapezoid[6], normal);
         glNormal3dv(normal);
      }
      glVertex3dv(trapezoid[1]);
      glVertex3dv(trapezoid[2]);
      glVertex3dv(trapezoid[6]);
      glVertex3dv(trapezoid[5]);
      glEnd();

      glBegin(GL_POLYGON);
      if (color) {
         TMath::Normal2Plane(trapezoid[0], trapezoid[4], trapezoid[7], normal);
         glNormal3dv(normal);
      }
      glVertex3dv(trapezoid[0]);
      glVertex3dv(trapezoid[4]);
      glVertex3dv(trapezoid[7]);
      glVertex3dv(trapezoid[3]);
      glEnd();

      glBegin(GL_POLYGON);
      if (color) {
         TMath::Normal2Plane(trapezoid[0], trapezoid[1], trapezoid[5], normal);
         glNormal3dv(normal);
      }
      glVertex3dv(trapezoid[0]);
      glVertex3dv(trapezoid[1]);
      glVertex3dv(trapezoid[5]);
      glVertex3dv(trapezoid[4]);
      glEnd();

      glBegin(GL_POLYGON);
      if (color) {
         TMath::Normal2Plane(trapezoid[3], trapezoid[7], trapezoid[6], normal);
         glNormal3dv(normal);
      }
      glVertex3dv(trapezoid[3]);
      glVertex3dv(trapezoid[7]);
      glVertex3dv(trapezoid[6]);
      glVertex3dv(trapezoid[2]);
      glEnd();
   }

   //______________________________________________________________________________
   void DrawTrapezoid(const Double_t ver[][3])
   {
      Double_t normal[3] = {0.};

      glBegin(GL_POLYGON);
      TMath::Normal2Plane(ver[0], ver[1], ver[2], normal);
      glNormal3dv(normal);
      glVertex3dv(ver[0]);
      glVertex3dv(ver[1]);
      glVertex3dv(ver[2]);
      glVertex3dv(ver[3]);
      glEnd();
      //bottom
      glBegin(GL_POLYGON);
      TMath::Normal2Plane(ver[4], ver[7], ver[6], normal);
      glNormal3dv(normal);
      glVertex3dv(ver[4]);
      glVertex3dv(ver[7]);
      glVertex3dv(ver[6]);
      glVertex3dv(ver[5]);
      glEnd();
      //

      glBegin(GL_POLYGON);
      TMath::Normal2Plane(ver[0], ver[3], ver[7], normal);
      glNormal3dv(normal);
      glVertex3dv(ver[0]);
      glVertex3dv(ver[3]);
      glVertex3dv(ver[7]);
      glVertex3dv(ver[4]);
      glEnd();

      glBegin(GL_POLYGON);
      TMath::Normal2Plane(ver[3], ver[2], ver[6], normal);
      glNormal3dv(normal);
      glVertex3dv(ver[3]);
      glVertex3dv(ver[2]);
      glVertex3dv(ver[6]);
      glVertex3dv(ver[7]);
      glEnd();

      glBegin(GL_POLYGON);
      TMath::Normal2Plane(ver[5], ver[6], ver[2], normal);
      glNormal3dv(normal);
      glVertex3dv(ver[5]);
      glVertex3dv(ver[6]);
      glVertex3dv(ver[2]);
      glVertex3dv(ver[1]);
      glEnd();

      glBegin(GL_POLYGON);
      TMath::Normal2Plane(ver[0], ver[4], ver[5], normal);
      glNormal3dv(normal);
      glVertex3dv(ver[0]);
      glVertex3dv(ver[4]);
      glVertex3dv(ver[5]);
      glVertex3dv(ver[1]);
      glEnd();
   }

   //______________________________________________________________________________
   void DrawTrapezoidTextured(const Double_t ver[][2], Double_t zMin, Double_t zMax,
                              Double_t texMin, Double_t texMax)
   {
      //In polar coordinates, box became trapezoid.
      //Four faces need normal calculations.
      const Double_t trapezoid[][3] = {{ver[0][0], ver[0][1], zMin}, {ver[1][0], ver[1][1], zMin},
                                       {ver[2][0], ver[2][1], zMin}, {ver[3][0], ver[3][1], zMin},
                                       {ver[0][0], ver[0][1], zMax}, {ver[1][0], ver[1][1], zMax},
                                       {ver[2][0], ver[2][1], zMax}, {ver[3][0], ver[3][1], zMax}};
      if (zMin > zMax) {
         std::swap(zMin, zMax);
         std::swap(texMin, texMax);
      }
      //top
      glBegin(GL_POLYGON);
      glNormal3d(0., 0., 1.);
      glTexCoord1d(texMax), glVertex3dv(trapezoid[4]);
      glTexCoord1d(texMax), glVertex3dv(trapezoid[5]);
      glTexCoord1d(texMax), glVertex3dv(trapezoid[6]);
      glTexCoord1d(texMax), glVertex3dv(trapezoid[7]);
      glEnd();
      //bottom
      glBegin(GL_POLYGON);
      glNormal3d(0., 0., -1.);
      glTexCoord1d(texMin), glVertex3dv(trapezoid[0]);
      glTexCoord1d(texMin), glVertex3dv(trapezoid[3]);
      glTexCoord1d(texMin), glVertex3dv(trapezoid[2]);
      glTexCoord1d(texMin), glVertex3dv(trapezoid[1]);
      glEnd();
      //
      glBegin(GL_POLYGON);
      Double_t normal[3] = {};
      TMath::Normal2Plane(trapezoid[1], trapezoid[2], trapezoid[6], normal);
      glNormal3dv(normal);
      glTexCoord1d(texMin), glVertex3dv(trapezoid[1]);
      glTexCoord1d(texMin), glVertex3dv(trapezoid[2]);
      glTexCoord1d(texMax), glVertex3dv(trapezoid[6]);
      glTexCoord1d(texMax), glVertex3dv(trapezoid[5]);
      glEnd();

      glBegin(GL_POLYGON);
      TMath::Normal2Plane(trapezoid[0], trapezoid[4], trapezoid[7], normal);
      glNormal3dv(normal);
      glTexCoord1d(texMin), glVertex3dv(trapezoid[0]);
      glTexCoord1d(texMax), glVertex3dv(trapezoid[4]);
      glTexCoord1d(texMax), glVertex3dv(trapezoid[7]);
      glTexCoord1d(texMin), glVertex3dv(trapezoid[3]);
      glEnd();

      glBegin(GL_POLYGON);
      TMath::Normal2Plane(trapezoid[0], trapezoid[1], trapezoid[5], normal);
      glNormal3dv(normal);
      glTexCoord1d(texMin), glVertex3dv(trapezoid[0]);
      glTexCoord1d(texMin), glVertex3dv(trapezoid[1]);
      glTexCoord1d(texMax), glVertex3dv(trapezoid[5]);
      glTexCoord1d(texMax), glVertex3dv(trapezoid[4]);
      glEnd();

      glBegin(GL_POLYGON);
      TMath::Normal2Plane(trapezoid[3], trapezoid[7], trapezoid[6], normal);
      glNormal3dv(normal);
      glTexCoord1d(texMin), glVertex3dv(trapezoid[3]);
      glTexCoord1d(texMax), glVertex3dv(trapezoid[7]);
      glTexCoord1d(texMax), glVertex3dv(trapezoid[6]);
      glTexCoord1d(texMin), glVertex3dv(trapezoid[2]);
      glEnd();
   }

   //______________________________________________________________________________
   void DrawTrapezoidTextured(const Double_t ver[][3], Double_t texMin, Double_t texMax)
   {
      Double_t normal[3] = {};
      if (texMin > texMax)
         std::swap(texMin, texMax);
      const Double_t tex[] = {texMin, texMin, texMax, texMax, texMin, texMin, texMax, texMax};
      glBegin(GL_POLYGON);
      TMath::Normal2Plane(ver[0], ver[1], ver[2], normal);
      glNormal3dv(normal);
      glTexCoord1d(tex[0]), glVertex3dv(ver[0]);
      glTexCoord1d(tex[1]), glVertex3dv(ver[1]);
      glTexCoord1d(tex[2]), glVertex3dv(ver[2]);
      glTexCoord1d(tex[3]), glVertex3dv(ver[3]);
      glEnd();
      glBegin(GL_POLYGON);
      TMath::Normal2Plane(ver[4], ver[7], ver[6], normal);
      glNormal3dv(normal);
      glTexCoord1d(tex[4]), glVertex3dv(ver[4]);
      glTexCoord1d(tex[7]), glVertex3dv(ver[7]);
      glTexCoord1d(tex[6]), glVertex3dv(ver[6]);
      glTexCoord1d(tex[5]), glVertex3dv(ver[5]);
      glEnd();
      glBegin(GL_POLYGON);
      TMath::Normal2Plane(ver[0], ver[3], ver[7], normal);
      glNormal3dv(normal);
      glTexCoord1d(tex[0]), glVertex3dv(ver[0]);
      glTexCoord1d(tex[3]), glVertex3dv(ver[3]);
      glTexCoord1d(tex[7]), glVertex3dv(ver[7]);
      glTexCoord1d(tex[4]), glVertex3dv(ver[4]);
      glEnd();
      glBegin(GL_POLYGON);
      TMath::Normal2Plane(ver[3], ver[2], ver[6], normal);
      glNormal3dv(normal);
      glTexCoord1d(tex[3]), glVertex3dv(ver[3]);
      glTexCoord1d(tex[2]), glVertex3dv(ver[2]);
      glTexCoord1d(tex[6]), glVertex3dv(ver[6]);
      glTexCoord1d(tex[7]), glVertex3dv(ver[7]);
      glEnd();
      glBegin(GL_POLYGON);
      TMath::Normal2Plane(ver[5], ver[6], ver[2], normal);
      glNormal3dv(normal);
      glTexCoord1d(tex[5]), glVertex3dv(ver[5]);
      glTexCoord1d(tex[6]), glVertex3dv(ver[6]);
      glTexCoord1d(tex[2]), glVertex3dv(ver[2]);
      glTexCoord1d(tex[1]), glVertex3dv(ver[1]);
      glEnd();
      glBegin(GL_POLYGON);
      TMath::Normal2Plane(ver[0], ver[4], ver[5], normal);
      glNormal3dv(normal);
      glTexCoord1d(tex[0]), glVertex3dv(ver[0]);
      glTexCoord1d(tex[4]), glVertex3dv(ver[4]);
      glTexCoord1d(tex[5]), glVertex3dv(ver[5]);
      glTexCoord1d(tex[1]), glVertex3dv(ver[1]);
      glEnd();
   }

   //______________________________________________________________________________
   void DrawTrapezoidTextured2(const Double_t ver[][2], Double_t zMin, Double_t zMax,
                               Double_t texMin, Double_t texMax)
   {
      //In polar coordinates, box became trapezoid.
      if (zMin > zMax) {
         std::swap(zMin, zMax);
         std::swap(texMin, texMax);
      }

      const Double_t trapezoid[][3] = {{ver[0][0], ver[0][1], zMin}, {ver[1][0], ver[1][1], zMin},
                                       {ver[2][0], ver[2][1], zMin}, {ver[3][0], ver[3][1], zMin},
                                       {ver[0][0], ver[0][1], zMax}, {ver[1][0], ver[1][1], zMax},
                                       {ver[2][0], ver[2][1], zMax}, {ver[3][0], ver[3][1], zMax}};
      const Double_t tex[] = {texMin, texMax, texMax, texMin, texMin, texMax, texMax, texMin};
      //top
      glBegin(GL_POLYGON);
      glNormal3d(0., 0., 1.);
      glTexCoord1d(tex[4]), glVertex3dv(trapezoid[4]);
      glTexCoord1d(tex[5]), glVertex3dv(trapezoid[5]);
      glTexCoord1d(tex[6]), glVertex3dv(trapezoid[6]);
      glTexCoord1d(tex[7]), glVertex3dv(trapezoid[7]);
      glEnd();
      //bottom
      glBegin(GL_POLYGON);
      glNormal3d(0., 0., -1.);
      glTexCoord1d(tex[0]), glVertex3dv(trapezoid[0]);
      glTexCoord1d(tex[3]), glVertex3dv(trapezoid[3]);
      glTexCoord1d(tex[2]), glVertex3dv(trapezoid[2]);
      glTexCoord1d(tex[1]), glVertex3dv(trapezoid[1]);
      glEnd();
      //
      glBegin(GL_POLYGON);
      Double_t normal[3] = {};
      TMath::Normal2Plane(trapezoid[1], trapezoid[2], trapezoid[6], normal);
      glNormal3dv(normal);
      glTexCoord1d(tex[1]), glVertex3dv(trapezoid[1]);
      glTexCoord1d(tex[2]), glVertex3dv(trapezoid[2]);
      glTexCoord1d(tex[6]), glVertex3dv(trapezoid[6]);
      glTexCoord1d(tex[5]), glVertex3dv(trapezoid[5]);
      glEnd();

      glBegin(GL_POLYGON);
      TMath::Normal2Plane(trapezoid[0], trapezoid[4], trapezoid[7], normal);
      glNormal3dv(normal);
      glTexCoord1d(tex[0]), glVertex3dv(trapezoid[0]);
      glTexCoord1d(tex[4]), glVertex3dv(trapezoid[4]);
      glTexCoord1d(tex[7]), glVertex3dv(trapezoid[7]);
      glTexCoord1d(tex[3]), glVertex3dv(trapezoid[3]);
      glEnd();

      glBegin(GL_POLYGON);
      TMath::Normal2Plane(trapezoid[0], trapezoid[1], trapezoid[5], normal);
      glNormal3dv(normal);
      glTexCoord1d(tex[0]), glVertex3dv(trapezoid[0]);
      glTexCoord1d(tex[1]), glVertex3dv(trapezoid[1]);
      glTexCoord1d(tex[5]), glVertex3dv(trapezoid[5]);
      glTexCoord1d(tex[4]), glVertex3dv(trapezoid[4]);
      glEnd();

      glBegin(GL_POLYGON);
      TMath::Normal2Plane(trapezoid[3], trapezoid[7], trapezoid[6], normal);
      glNormal3dv(normal);
      glTexCoord1d(tex[3]), glVertex3dv(trapezoid[3]);
      glTexCoord1d(tex[7]), glVertex3dv(trapezoid[7]);
      glTexCoord1d(tex[6]), glVertex3dv(trapezoid[6]);
      glTexCoord1d(tex[2]), glVertex3dv(trapezoid[2]);
      glEnd();
   }
}
