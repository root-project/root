#ifdef GDK_WIN32
#include <Windows4Root.h>
#endif

#include <GL/gl.h>
#include <GL/glu.h>

#include <TAttMarker.h>
#include <TBuffer3D.h>
#include <TError.h>

#include "TGLSceneObject.h"

static GLUtriangulatorObj *GetTesselator()
{
   static struct Init {
      Init()
      {
#ifdef GDK_WIN32
         typedef void (CALLBACK *tessfuncptr_t)();
#else
         typedef void (*tessfuncptr_t)();
#endif
         fTess = gluNewTess();
         
         if (!fTess) {
            Error("GetTesselator::Init", "could not create tesselation object");
         } else {
            gluTessCallback(fTess, (GLenum)GLU_BEGIN, (tessfuncptr_t)glBegin);
            gluTessCallback(fTess, (GLenum)GLU_END, (tessfuncptr_t)glEnd);
            gluTessCallback(fTess, (GLenum)GLU_VERTEX, (tessfuncptr_t)glVertex3dv);
         }
      }
      ~Init()
      {
         if(fTess)
            gluDeleteTess(fTess);
      }
      GLUtriangulatorObj *fTess;
   }singleton;

   return singleton.fTess;
}

static GLUquadric *GetQuadric()
{
   static struct Init {
      Init()
      {
         fQuad = gluNewQuadric();
         if (!fQuad) {
            Error("GetQuadric::Init", "could not create quadric object");
         } else {
            gluQuadricOrientation(fQuad, (GLenum)GLU_OUTSIDE);
            gluQuadricDrawStyle(fQuad,   (GLenum)GLU_FILL);
            gluQuadricNormals(fQuad,     (GLenum)GLU_FLAT);
         }
      }
      ~Init()
      {
         if(fQuad)
            gluDeleteQuadric(fQuad);
      }
      GLUquadric *fQuad;
   }singleton;

   return singleton.fQuad;
}

static GLenum gLightNames[] = {GL_LIGHT0, GL_LIGHT1, GL_LIGHT2, GL_LIGHT3,
                               GL_LIGHT4, GL_LIGHT5, GL_LIGHT6, GL_LIGHT7};

//______________________________________________________________________________
TGLSceneObject::TGLSceneObject(const Float_t *color, UInt_t glname)
                   :fColor(), fGLName(glname), fNextT(0) 
{
   if (color) {
      fColor[0] = color[0];
      fColor[1] = color[1];
      fColor[2] = color[2];
   } else {
      fColor[0] = 
      fColor[1] = 
      fColor[2] = 1.f;
   }
   fColor[3] = 1.f;
}

//______________________________________________________________________________
Bool_t TGLSceneObject::IsTransparent()const
{
   return kFALSE;
}

//______________________________________________________________________________
void TGLSceneObject::ResetTransparency()
{
}

//______________________________________________________________________________
void TGLSceneObject::Shift(Double_t, Double_t, Double_t)
{
}

//______________________________________________________________________________
TObject *TGLSceneObject::GetRealObject()const
{
   return 0;
}
//______________________________________________________________________________
TGLFaceSet::TGLFaceSet(const TBuffer3D & buff, const Float_t *color, 
                       UInt_t glname, TObject *realobj)
               :TGLSceneObject(color, glname), 
                fVertices(buff.fPnts, buff.fPnts + 3 * buff.fNbPnts),
                fNormals(3 * buff.fNbPols)
{
   fRealObj = realobj;
   fIsTransparent = kFALSE;
   fNbPols = buff.fNbPols;

   Int_t * segs = buff.fSegs;
   Int_t * pols = buff.fPols;
   Double_t * pnts = buff.fPnts;

   for (Int_t numPol = 0, e = buff.fNbPols, j = 0; numPol < e; ++numPol) {
      ++j;
      Int_t segmentInd = pols[j] + j;
      Int_t segmentCol = pols[j];
      Int_t seg1 = pols[segmentInd--];
      Int_t seg2 = pols[segmentInd--];
      Int_t np[] = {segs[seg1 * 3 + 1], segs[seg1 * 3 + 2], segs[seg2 * 3 + 1], segs[seg2 * 3 + 2]};
      Int_t n[] = {-1, -1, -1};
      Int_t normp[] = {0, 0, 0};

      np[0] != np[2] ?
               (np[0] != np[3] ?
                  (*n = *np, n[1] = np[1] == np[2] ?
                     n[2] = np[3], np[2] :
                        (n[2] = np[2], np[3])) :
                           (*n = np[1], n[1] = *np, n[2] = np[2] )) :
                              (*n = np[1], n[1] = *np, n[2] = np[3]);
      fPolyDesc.push_back(3);
      Int_t sizeInd = fPolyDesc.size() - 1;
      fPolyDesc.insert(fPolyDesc.end(), n, n + 3);
      Int_t check = CheckPoints(n, normp), ngood = check;

      if (check == 3)
         TMath::Normal2Plane(pnts + n[0] * 3, pnts + n[1] * 3, pnts + n[2] * 3, &fNormals[numPol * 3]);

      while (segmentInd > j + 1) {
         seg2 = pols[segmentInd];
         np[0] = segs[seg2 * 3 + 1];
         np[1] = segs[seg2 * 3 + 2];
         if (np[0] == n[2]) {
            fPolyDesc.push_back(np[1]);
            if (check != 3)
               normp[ngood ++] = np[1];
         } else {
             fPolyDesc.push_back(np[0]);
             if (check != 3)
                normp[ngood ++] = np[0];
         }

         if (check != 3 && ngood == 3) {
            check = CheckPoints(normp, normp);
            if (check == 3)
               TMath::Normal2Plane(pnts + normp[0] * 3, pnts + normp[1] * 3,
                                   pnts + normp[2] * 3, &fNormals[numPol * 3]);
            ngood = check;
         }
         ++fPolyDesc[sizeInd];
         --segmentInd;
      }
      j += segmentCol + 1;
   }
}

//______________________________________________________________________________
Bool_t TGLFaceSet::IsTransparent()const
{
   return fIsTransparent;
}

//______________________________________________________________________________
void TGLFaceSet::ResetTransparency()
{
   fIsTransparent = !fIsTransparent;
}

//______________________________________________________________________________
void TGLFaceSet::GLDraw()const
{
   glMaterialfv(GL_FRONT, GL_SPECULAR, fColor);
   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);
   glMaterialf(GL_FRONT, GL_SHININESS, 60.f);
   
   GLUtriangulatorObj *tessObj = GetTesselator();
   const Double_t *pnts = &fVertices[0];
   const Double_t *normals = &fNormals[0];
   const Int_t *pols = &fPolyDesc[0];

   if (IsTransparent()) {
      glEnable(GL_BLEND);
      glDepthMask(GL_FALSE);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE);
   }

   glLoadName(GetGLName());
      
   for (UInt_t i = 0, j = 0; i < fNbPols; ++i) {
      Int_t npoints = pols[j++];
      if (tessObj && npoints > 4) {
         gluBeginPolygon(tessObj);
         gluNextContour(tessObj, (GLenum)GLU_UNKNOWN);
         glNormal3dv(normals + i * 3);
         
         for (Int_t k = 0; k < npoints; ++k, ++j)
            gluTessVertex(tessObj, (Double_t *)pnts + pols[j] * 3, (Double_t *)pnts + pols[j] * 3);
         
         gluEndPolygon(tessObj);
      } else {
         glBegin(GL_POLYGON);
         glNormal3dv(normals + i * 3);
         
         for (Int_t k = 0; k < npoints; ++k, ++j)
            glVertex3dv(pnts + pols[j] * 3);
         glEnd();
      }
   }
  
   if (IsTransparent()) {
      glDepthMask(GL_TRUE);
      glDisable(GL_BLEND);
   } 
}

//______________________________________________________________________________
void TGLFaceSet::SetColor(const Float_t *newcolor)
{
   if (newcolor) {
      fColor[0] = newcolor[0];
      fColor[1] = newcolor[1];
      fColor[2] = newcolor[2];
   } else {
      fColor[0] =
      fColor[1] =
      fColor[2] = 1.f;
   }  
}

//______________________________________________________________________________
void TGLFaceSet::Shift(Double_t x, Double_t y, Double_t z)
{
   for (UInt_t i = 0, e = fVertices.size(); i < e; i +=3) {
      fVertices[i] += x;
      fVertices[i + 1] += y;
      fVertices[i + 2] += z;
   }
}

//______________________________________________________________________________
Int_t TGLFaceSet::CheckPoints(const Int_t * source, Int_t *dest) const
{
   const Double_t * p1 = &fVertices[source[0] * 3];
   const Double_t * p2 = &fVertices[source[1] * 3];
   const Double_t * p3 = &fVertices[source[2] * 3];
   Int_t retVal = 1;

   !Eq(p1, p2) ?
      !Eq(p1, p3) ?
         !Eq(p2, p3) ?
            retVal = 3 :
               (retVal = 2, *dest = *source, dest[1] = source[1]) :
                  (retVal = 2, *dest = *source, dest[1] = source[1]) :
                     !Eq(p2, p3) ?
                        retVal = 2, *dest = *source, dest[1] = source[2] :
                           *dest = *source;

   return retVal;
}

//______________________________________________________________________________
TGLPolyMarker::TGLPolyMarker(const TBuffer3D &buff, const Float_t *color)
                  :TGLSceneObject(color),
                   fVertices(buff.fPnts, buff.fPnts + 3 * buff.fNbPnts),
                   fStyle(7)
{
   //TAttMarker is not TObject descendant, so I need dynamic_cast
   if (TAttMarker *realObj = dynamic_cast<TAttMarker *>(buff.fId))
      fStyle = realObj->GetMarkerStyle();
}

//______________________________________________________________________________
void TGLPolyMarker::GLDraw()const 
{
   const Double_t *vertices = &fVertices[0];
   UInt_t size = fVertices.size();
   Int_t stacks = 6, slices = 6;
   Float_t pointSize = 6.f;
   Double_t top_radius = 5.;
   GLUquadric *quadObj = GetQuadric();   

   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);

   switch (fStyle) {
   case 27:
      stacks = 2, slices = 4;
   case 4:case 8:case 20:case 24:
      if (quadObj) { //VC6.0 broken for scope
         for (UInt_t i = 0; i < size; i += 3) {
            glPushMatrix();
            glTranslated(vertices[i], vertices[i + 1], vertices[i + 2]);
            gluSphere(quadObj, 5., slices, stacks);
            glPopMatrix();
         }
      }
      break;
   case 22:case 26:
      top_radius = 0.;
   case 21:case 25:
      if (quadObj) { //VC6.0 broken for scope
         for (UInt_t i = 0; i < size; i += 3) {
            glPushMatrix();
            glTranslated(vertices[i], vertices[i + 1], vertices[i + 2]);
            gluCylinder(quadObj, 5., top_radius, 5., 4, 1);
            glPopMatrix();
         }
      }
      break;
   case 23:
      if (quadObj) {//VC6.0 broken for scope
         for (UInt_t i = 0; i < size; i += 3) {
            glPushMatrix();
            glTranslated(vertices[i], vertices[i + 1], vertices[i + 2]);
            glRotated(180, 1., 0., 0.);
            gluCylinder(quadObj, 5., 0., 5., 4, 1);
            glPopMatrix();
         }
      }
      break;
   case 3: case 2: case 5:
      DrawStars();
      break;
   case 1: case 9: case 10: case 11: default:{
      glBegin(GL_POINTS);
      for (UInt_t i = 0; i < size; i += 3)
         glVertex3dv(vertices + i);
      glEnd();
   }
   break;
   case 6:
      pointSize = 3.f;
   case 7:
      glPointSize(pointSize);
      glBegin(GL_POINTS);
      for (UInt_t i = 0; i < size; i += 3)
         glVertex3dv(vertices + i);
      glEnd();
      glPointSize(1.f);
   }
}

//______________________________________________________________________________
void TGLPolyMarker::DrawStars()const
{
   for (UInt_t i = 0; i < fVertices.size(); i += 3) {
      Double_t x = fVertices[i];
      Double_t y = fVertices[i + 1];
      Double_t z = fVertices[i + 2];
      glBegin(GL_LINES);
      if (fStyle == 2 || fStyle == 3) {
         glVertex3d(x - 2., y, z);
         glVertex3d(x + 2., y, z);
         glVertex3d(x, y, z - 2.);
         glVertex3d(x, y, z + 2.);
         glVertex3d(x, y - 2., z);
         glVertex3d(x, y + 2., z);
      }
      if(fStyle != 2) {
         glVertex3d(x - 1.4, y - 1.4, z - 1.4);
         glVertex3d(x + 1.4, y + 1.4, z + 1.4);
         glVertex3d(x - 1.4, y - 1.4, z + 1.4);
         glVertex3d(x + 1.4, y + 1.4, z - 1.4);
         glVertex3d(x - 1.4, y + 1.4, z - 1.4);
         glVertex3d(x + 1.4, y - 1.4, z + 1.4);
         glVertex3d(x - 1.4, y + 1.4, z + 1.4);
         glVertex3d(x + 1.4, y - 1.4, z - 1.4);
      }
      glEnd();
   }
}

//______________________________________________________________________________
TGLPolyLine::TGLPolyLine(const TBuffer3D &buff, const Float_t *color)
                :TGLSceneObject(color),
                 fVertices(buff.fPnts, buff.fPnts + 3 * buff.fNbPnts)
{
}

//______________________________________________________________________________
void TGLPolyLine::GLDraw()const
{
   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);
   glBegin(GL_LINE_STRIP);
   
   for (UInt_t i = 0; i < fVertices.size(); i += 3)
      glVertex3d(fVertices[i], fVertices[i + 1], fVertices[i + 2]);
   
   glEnd();
}

//______________________________________________________________________________
TGLSimpleLight::TGLSimpleLight(UInt_t glname, UInt_t lightname, const Float_t *pos, Bool_t dir)
                   :TGLSceneObject(0, glname), fLightName(lightname)
{
   glEnable(gLightNames[lightname]);
   fPosition[0] = pos[0];
   fPosition[1] = pos[1];
   fPosition[2] = pos[2];
   dir ? fPosition[3] = 1.f : fPosition[3] = 0.f;
}

//______________________________________________________________________________
void TGLSimpleLight::GLDraw()const
{
   glLightfv(gLightNames[fLightName], GL_POSITION, fPosition);
}

//______________________________________________________________________________
void TGLSimpleLight::Shift(Double_t x, Double_t y, Double_t z)
{
   fPosition[0] += x;
   fPosition[1] += y;
   fPosition[2] += z;
}

//______________________________________________________________________________
TGLSelection::TGLSelection(const PDD_t &x, const PDD_t &y, const PDD_t &z)
                 :fXRange(x), fYRange(y), fZRange(z)
{
}

//______________________________________________________________________________
void TGLSelection::GLDraw()const
{
   Double_t xmin = fXRange.first, xmax = fXRange.second;
   Double_t ymin = fYRange.first, ymax = fYRange.second;
   Double_t zmin = fZRange.first, zmax = fZRange.second;

   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);
   glBegin(GL_LINE_LOOP);
   glVertex3d(xmin, ymin, zmin);
   glVertex3d(xmin, ymax, zmin);
   glVertex3d(xmax, ymax, zmin);
   glVertex3d(xmax, ymin, zmin);
   glEnd();
   glBegin(GL_LINE_LOOP);
   glVertex3d(xmin, ymin, zmax);
   glVertex3d(xmin, ymax, zmax);
   glVertex3d(xmax, ymax, zmax);
   glVertex3d(xmax, ymin, zmax);
   glEnd();
   glBegin(GL_LINES);
   glVertex3d(xmin, ymin, zmin);
   glVertex3d(xmin, ymin, zmax);
   glVertex3d(xmin, ymax, zmin);
   glVertex3d(xmin, ymax, zmax);
   glVertex3d(xmax, ymax, zmin);
   glVertex3d(xmax, ymax, zmax);
   glVertex3d(xmax, ymin, zmin);
   glVertex3d(xmax, ymin, zmax);
   glEnd();
}

//______________________________________________________________________________
void TGLSelection::Shift(Double_t x, Double_t y, Double_t z)
{
   fXRange.first += x, fXRange.second += x;
   fYRange.first += y, fYRange.second += y;
   fZRange.first += z, fZRange.second += z;
}

