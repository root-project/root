// @(#)root/gl:$Name:  $:$Id: TGFrame.h,v 1.59 2005/01/12 18:39:29 brun Exp $
// Author: Timur Pocheptsov 18/08/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include <cassert>

#ifdef WIN32
#include "Windows4root.h"
#endif

#include <GL/gl.h>
#include <GL/glu.h>

#include "TVirtualGL.h"
#include "TVirtualPad.h"
#include "TError.h"
#include "TBuffer3D.h"
#include "TColor.h"
#include "TMath.h"
#include "TSystem.h"
#include "TGLPixmap.h"
#include "TArcBall.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TObjArray.h"
#include "Buttons.h"

#include "TPad.h"
#include "KeySymbols.h"

ClassImp(TGLPixmap)

class GLSelection {
private:
   Double_t fBBox[6];

public:
   GLSelection();
   GLSelection(const Double_t *bbox);
   GLSelection(Double_t xmin, Double_t xmax, Double_t ymin,
                Double_t ymax, Double_t zmin, Double_t zmax);

   void SetBBox(const Double_t *newBbox);
   void SetBBox(Double_t xmin, Double_t xmax, Double_t ymin,
               Double_t ymax, Double_t zmin, Double_t zmax);

   const Double_t *GetRangeX()const{return fBBox;}
   const Double_t *GetRangeY()const{return fBBox + 2;}
   const Double_t *GetRangeZ()const{return fBBox + 4;}
};

class GLSceneObject : public TObject {
protected:
   std::vector<Double_t> fVertices;
   Float_t               fColor[17];
   GLSelection           fSelectionBox;
   Bool_t                fIsSelected;

private:
   UInt_t                fGLName;
   GLSceneObject        *fNextT;
   TObject              *fRealObject;

public:
   GLSceneObject(const TBuffer3D &buffer, Int_t verticesReserve,
                  const Float_t *color = 0, UInt_t glName = 0, TObject *realObj = 0);
   GLSceneObject(const TBuffer3D &buffer,
                  const Float_t *color = 0, UInt_t glName = 0, TObject *realObj = 0);
   GLSceneObject(UInt_t glName, const Float_t *color, Short_t trans, TObject *realObj);

   virtual Bool_t IsTransparent()const;

   virtual void GLDraw()const = 0;

   GLSelection *GetBBox(){return &fSelectionBox;}
   const GLSelection *GetBBox()const{return &fSelectionBox;}

   void SetNextT(GLSceneObject *next){fNextT = next;}
   GLSceneObject *GetNextT()const{return fNextT;}

   UInt_t GetGLName()const{return fGLName;}
   TObject *GetRealObject()const{return fRealObject;}

   const Float_t *GetColor()const{return fColor;}
   void SetColor(const Float_t *newColor, Bool_t fromCtor = kFALSE);

   void Select(Bool_t select = kTRUE){fIsSelected = select;}

   void SetBBox();
private:
   GLSceneObject(const GLSceneObject &);
   GLSceneObject & operator = (const GLSceneObject &);

   void SetBBox(const TBuffer3D & buffer);

};

class GLFaceSet : public GLSceneObject {
private:
   std::vector<Double_t> fNormals;
   std::vector<Int_t>    fPolyDesc;
   UInt_t                fNbPols;

public:
   GLFaceSet(const TBuffer3D &buff, const Float_t *color,
              UInt_t glName, TObject *realObj);

   void GLDraw()const;
   void GLDrawPolys()const;

private:
   Int_t CheckPoints(const Int_t *source, Int_t *dest)const;
   static Bool_t Eq(const Double_t *p1, const Double_t *p2);
   void CalculateNormals();
};

static GLUtriangulatorObj *getTesselator()
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
            Error("getTesselator::Init", "could not create tesselation object");
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

//______________________________________________________________________________
GLSelection::GLSelection()
{
   fBBox[0] = fBBox[1] = fBBox[2] =
   fBBox[3] = fBBox[4] = fBBox[5] = 0.;
}

//______________________________________________________________________________
GLSelection::GLSelection(const Double_t *bbox)
{
   for (Int_t i= 0; i < 6; ++i) fBBox[i] = bbox[i];
}

//______________________________________________________________________________
GLSelection::GLSelection(Double_t xmin, Double_t xmax, Double_t ymin,
                           Double_t ymax, Double_t zmin, Double_t zmax)
{
   fBBox[0] = xmin, fBBox[1] = xmax;
   fBBox[2] = ymin, fBBox[3] = ymax;
   fBBox[4] = zmin, fBBox[5] = zmax;
}

//______________________________________________________________________________
void GLSelection::SetBBox(const Double_t *newBBox)
{
   for (Int_t i= 0; i < 6; ++i) fBBox[i] = newBBox[i];
}

//______________________________________________________________________________
void GLSelection::SetBBox(Double_t xmin, Double_t xmax, Double_t ymin,
                          Double_t ymax, Double_t zmin, Double_t zmax)
{
   fBBox[0] = xmin, fBBox[1] = xmax;
   fBBox[2] = ymin, fBBox[3] = ymax;
   fBBox[4] = zmin, fBBox[5] = zmax;
}

//______________________________________________________________________________
GLSceneObject::GLSceneObject(const TBuffer3D &buffer, const Float_t *color,
                               UInt_t glName, TObject *obj) :
   fVertices(buffer.fPnts, buffer.fPnts + 3 * buffer.NbPnts()),
   fColor(),
   fIsSelected(kFALSE),
   fGLName(glName),
   fNextT(0),
   fRealObject(obj)
{
   SetColor(color, kTRUE);
   fColor[3] = 1.f - buffer.fTransparency / 100.f;
   SetBBox(buffer);
}

//______________________________________________________________________________
GLSceneObject::GLSceneObject(const TBuffer3D &buffer, Int_t verticesReserve,
                               const Float_t *color, UInt_t glName, TObject *obj) :
   fVertices(verticesReserve, 0.),
   fColor(),
   fIsSelected(kFALSE),
   fGLName(glName),
   fNextT(0),
   fRealObject(obj)
{
   SetColor(color, kTRUE);
   fColor[3] = 1.f - buffer.fTransparency / 100.f;
   SetBBox(buffer);
}

//______________________________________________________________________________
GLSceneObject::GLSceneObject(UInt_t glName, const Float_t *color, Short_t trans, TObject *obj)
   : fColor(), fIsSelected(kFALSE), fGLName(glName), fNextT(0), fRealObject(obj)
{
   SetColor(color, kTRUE);
   fColor[3] = 1.f - trans / 100.f;
}

//______________________________________________________________________________
Bool_t GLSceneObject::IsTransparent()const
{
   return fColor[3] < 1.f;
}

//______________________________________________________________________________
void GLSceneObject::SetColor(const Float_t *color, Bool_t fromCtor)
{
   if (!fromCtor) {
      for (Int_t i = 0; i < 17; ++i) fColor[i] = color[i];
   } else {
      if (color) {
         //diffuse and specular
         fColor[0] = color[0];
         fColor[1] = color[1];
         fColor[2] = color[2];
      } else {
         //for (Int_t i = 0; i < 12; ++i) fColor[i] = 1.f;
         fColor[0] = 1.f;
         fColor[1] = .3f;
         fColor[2] = .0f;
      }
      //ambient
      fColor[4] = fColor[5] = fColor[6] = 0.f;
      //specular
      fColor[8] = fColor[9] = fColor[10] = 0.7f;
      //emission
      fColor[12] = fColor[13] = fColor[14] = 0.f;
      //alpha
      fColor[3] = fColor[7] = fColor[11] = fColor[15] = 1.f;
      //shininess
      if (color) fColor[16] = 60.f;
      else fColor[16] = 10.f;
   }
}

//______________________________________________________________________________
void GLSceneObject::SetBBox(const TBuffer3D & buffer)
{
   Double_t xmin = buffer.fPnts[0], xmax = xmin;
   Double_t ymin = buffer.fPnts[1], ymax = ymin;
   Double_t zmin = buffer.fPnts[2], zmax = zmin;

   for (UInt_t nv = 3; nv < buffer.NbPnts()*3; nv += 3) {
      xmin = TMath::Min(xmin, buffer.fPnts[nv]);
      xmax = TMath::Max(xmax, buffer.fPnts[nv]);
      ymin = TMath::Min(ymin, buffer.fPnts[nv + 1]);
      ymax = TMath::Max(ymax, buffer.fPnts[nv + 1]);
      zmin = TMath::Min(zmin, buffer.fPnts[nv + 2]);
      zmax = TMath::Max(zmax, buffer.fPnts[nv + 2]);
   }

   fSelectionBox.SetBBox(xmin, xmax, ymin, ymax, zmin, zmax);
}

//______________________________________________________________________________
void GLSceneObject::SetBBox()
{
   // Use the buffer bounding box if provided

   if (fVertices.size() >= 3) {
      Double_t xmin = fVertices[0], xmax = xmin;
      Double_t ymin = fVertices[1], ymax = ymin;
      Double_t zmin = fVertices[2], zmax = zmin;

      for (UInt_t nv = 3; nv < fVertices.size(); nv += 3) {
         xmin = TMath::Min(xmin, fVertices[nv]);
         xmax = TMath::Max(xmax, fVertices[nv]);
         ymin = TMath::Min(ymin, fVertices[nv + 1]);
         ymax = TMath::Max(ymax, fVertices[nv + 1]);
         zmin = TMath::Min(zmin, fVertices[nv + 2]);
         zmax = TMath::Max(zmax, fVertices[nv + 2]);
      }

      fSelectionBox.SetBBox(xmin, xmax, ymin, ymax, zmin, zmax);
   }
}

//______________________________________________________________________________
GLFaceSet::GLFaceSet(const TBuffer3D & buff, const Float_t *color, UInt_t glname, TObject *realobj)
               :GLSceneObject(buff, color, glname, realobj),
                fNormals(3 * buff.NbPols())
{
   fNbPols = buff.NbPols();

   Int_t *segs = buff.fSegs;
   Int_t *pols = buff.fPols;
   Int_t shiftInd = buff.fReflection ? 1 : -1;

   Int_t descSize = 0;

   for (UInt_t i = 0, j = 1; i < fNbPols; ++i, ++j)
   {
      descSize += pols[j] + 1;
      j += pols[j] + 1;
   }

   fPolyDesc.resize(descSize);

   for (UInt_t numPol = 0, currInd = 0, j = 1; numPol < fNbPols; ++numPol) {
      Int_t segmentInd = shiftInd < 0 ? pols[j] + j : j + 1;
      Int_t segmentCol = pols[j];
      Int_t s1 = pols[segmentInd];
      segmentInd += shiftInd;
      Int_t s2 = pols[segmentInd];
      segmentInd += shiftInd;
      Int_t segEnds[] = {segs[s1 * 3 + 1], segs[s1 * 3 + 2],
                         segs[s2 * 3 + 1], segs[s2 * 3 + 2]};
      Int_t numPnts[3] = {0};

      if (segEnds[0] == segEnds[2]) {
         numPnts[0] = segEnds[1], numPnts[1] = segEnds[0], numPnts[2] = segEnds[3];
      } else if (segEnds[0] == segEnds[3]) {
         numPnts[0] = segEnds[1], numPnts[1] = segEnds[0], numPnts[2] = segEnds[2];
      } else if (segEnds[1] == segEnds[2]) {
         numPnts[0] = segEnds[0], numPnts[1] = segEnds[1], numPnts[2] = segEnds[3];
      } else {
         numPnts[0] = segEnds[0], numPnts[1] = segEnds[1], numPnts[2] = segEnds[2];
      }

      fPolyDesc[currInd] = 3;
      Int_t sizeInd = currInd++;
      fPolyDesc[currInd++] = numPnts[0];
      fPolyDesc[currInd++] = numPnts[1];
      fPolyDesc[currInd++] = numPnts[2];
      Int_t lastAdded = numPnts[2];

      Int_t end = shiftInd < 0 ? j + 1 : j + segmentCol;
      for (; segmentInd != end; segmentInd += shiftInd) {
         segEnds[0] = segs[pols[segmentInd] * 3 + 1];
         segEnds[1] = segs[pols[segmentInd] * 3 + 2];
         if (segEnds[0] == lastAdded) {
            fPolyDesc[currInd++] = segEnds[1];
            lastAdded = segEnds[1];
         } else {
            fPolyDesc[currInd++] = segEnds[0];
            lastAdded = segEnds[0];
         }
         ++fPolyDesc[sizeInd];
      }
      j += segmentCol + 2;
   }

   CalculateNormals();
}

//______________________________________________________________________________
void GLFaceSet::GLDraw()const
{
   static float spec[] = {.4f, .4f, .4f, 1.f};

   glMaterialfv(GL_FRONT, GL_DIFFUSE, fColor);
   glMaterialfv(GL_FRONT, GL_SPECULAR, spec);
   glMaterialf(GL_FRONT, GL_SHININESS, 20);

   glEnable(GL_POLYGON_OFFSET_FILL);
   glPolygonOffset(1.f, 1.f);

   if (IsTransparent()) {
      glEnable(GL_BLEND);
      glDepthMask(GL_FALSE);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   }

   glLoadName(GetGLName());

   GLDrawPolys();

   if (IsTransparent()) {
      glDepthMask(GL_TRUE);
      glDisable(GL_BLEND);
   }

   glDisable(GL_POLYGON_OFFSET_FILL);
   glDisable(GL_LIGHTING);
   glColor3d(0., 0., 0.);
   glPolygonMode(GL_FRONT, GL_LINE);
   GLDrawPolys();
   glEnable(GL_LIGHTING);
   glPolygonMode(GL_FRONT, GL_FILL);
}

//______________________________________________________________________________
void GLFaceSet::GLDrawPolys()const
{
  GLUtriangulatorObj *tessObj = getTesselator();
  const Double_t *pnts = &fVertices[0];
  const Double_t *normals = &fNormals[0];
  const Int_t *pols = &fPolyDesc[0];

   for (UInt_t i = 0, j = 0; i < fNbPols; ++i) {
      Int_t npoints = pols[j++];

      if (tessObj && npoints > 4) {
         gluBeginPolygon(tessObj);
         gluNextContour(tessObj, (GLenum)GLU_UNKNOWN);
         glNormal3dv(normals + i * 3);

         for (Int_t k = 0; k < npoints; ++k, ++j) {
            gluTessVertex(tessObj, (Double_t *)pnts + pols[j] * 3, (Double_t *)pnts + pols[j] * 3);
         }
         gluEndPolygon(tessObj);
      } else {
         glBegin(GL_POLYGON);
         glNormal3dv(normals + i * 3);

         for (Int_t k = 0; k < npoints; ++k, ++j) {
            glVertex3dv(pnts + pols[j] * 3);
         }
         glEnd();
      }
   }
}

//______________________________________________________________________________
Int_t GLFaceSet::CheckPoints(const Int_t *source, Int_t *dest) const
{
   const Double_t * p1 = &fVertices[source[0] * 3];
   const Double_t * p2 = &fVertices[source[1] * 3];
   const Double_t * p3 = &fVertices[source[2] * 3];
   Int_t retVal = 1;

   if (Eq(p1, p2)) {
      dest[0] = source[0];
      if (!Eq(p1, p3) ) {
         dest[1] = source[2];
         retVal = 2;
      }
   } else if (Eq(p1, p3)) {
      dest[0] = source[0];
      dest[1] = source[1];
      retVal = 2;
   } else {
      dest[0] = source[0];
      dest[1] = source[1];
      retVal = 2;
      if (!Eq(p2, p3)) {
         dest[2] = source[2];
         retVal = 3;
      }
   }

   return retVal;
}

//______________________________________________________________________________
Bool_t GLFaceSet::Eq(const Double_t *p1, const Double_t *p2)
{
   Double_t dx = TMath::Abs(p1[0] - p2[0]);
   Double_t dy = TMath::Abs(p1[1] - p2[1]);
   Double_t dz = TMath::Abs(p1[2] - p2[2]);
   return dx < 1e-10 && dy < 1e-10 && dz < 1e-10;
}

//______________________________________________________________________________
void GLFaceSet::CalculateNormals()
{
   Double_t *pnts = &fVertices[0];
   for (UInt_t i = 0, j = 0; i < fNbPols; ++i) {
      Int_t polEnd = fPolyDesc[j] + j + 1;
      Int_t norm[] = {fPolyDesc[j + 1], fPolyDesc[j + 2], fPolyDesc[j + 3]};
      j += 4;
      Int_t check = CheckPoints(norm, norm), ngood = check;
      if (check == 3) {
         TMath::Normal2Plane(pnts + norm[0] * 3, pnts + norm[1] * 3,
                             pnts + norm[2] * 3, &fNormals[i * 3]);
         j = polEnd;
         continue;
      }
      while (j < (UInt_t)polEnd) {
         norm[ngood++] = fPolyDesc[j++];
         if (ngood == 3) {
            ngood = CheckPoints(norm, norm);
            if (ngood == 3) {
               TMath::Normal2Plane(pnts + norm[0] * 3, pnts + norm[1] * 3,
                                   pnts + norm[2] * 3, &fNormals[i * 3]);
               j = polEnd;
               break;
            }
         }
      }
   }
}

class GLCamera : public TObject{
protected:
   const Double_t *fViewVolume;
   const Int_t *fViewPort;
   Double_t fZoom;
   Bool_t fDrawFrame;

public:
   GLCamera(const Double_t *viewvolume, const Int_t *viewport);
   const Int_t *GetViewport()const
   {
      return fViewPort;
   }
   virtual void TurnOn()const = 0;
   virtual void TurnOn(Int_t x, Int_t y)const = 0;
   void Zoom(Double_t zoom)
   {
      fZoom = zoom;
   }
   void Select()
   {
      fDrawFrame = kTRUE;
   }
private:
   GLCamera(const GLCamera &);
   GLCamera & operator = (const GLCamera &);
};

class GLTransformation {
public:
   virtual ~GLTransformation();
   virtual void Apply()const = 0;
};


class GLSimpleTransform : public GLTransformation {
private:
   const Double_t *fRotMatrix;
   Double_t       fShift;
   //modifications
   const Double_t *fX;
   const Double_t *fY;
   const Double_t *fZ;
public:
   GLSimpleTransform(const Double_t *rm, Double_t s, const Double_t *x,
                      const Double_t *y, const Double_t *z);
   void Apply()const;
};

class GLPerspectiveCamera : public GLCamera {
private:
   GLSimpleTransform fTransformation;
public:
   GLPerspectiveCamera(const Double_t *vv, const Int_t *vp,
                        const GLSimpleTransform &tr);
   void TurnOn()const;
   void TurnOn(Int_t x, Int_t y)const;
};

GLCamera::GLCamera(const Double_t *vv, const Int_t *vp)
              :fViewVolume(vv), fViewPort(vp),
               fZoom(1.), fDrawFrame(kFALSE)
{
}

GLTransformation::~GLTransformation()
{
}

GLSimpleTransform::GLSimpleTransform(const Double_t *rm, Double_t s, const Double_t *x,
                                       const Double_t *y, const Double_t *z)
                        :fRotMatrix(rm), fShift(s),
                         fX(x), fY(y), fZ(z)
{
}

void GLSimpleTransform::Apply()const
{
   glTranslated(0., 0., -fShift);
   glMultMatrixd(fRotMatrix);
   glRotated(-90., 1., 0., 0.);
   glTranslated(-*fX, -*fY, -*fZ);
}

GLPerspectiveCamera::GLPerspectiveCamera(const Double_t *vv, const Int_t *vp,
                                           const GLSimpleTransform &tr)
                         :GLCamera(vv, vp),
                          fTransformation(tr)
{
}

void GLPerspectiveCamera::TurnOn()const
{
   glViewport(fViewPort[0], fViewPort[1], fViewPort[2], fViewPort[3]);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   Double_t frx = fViewVolume[0] * fZoom;
   Double_t fry = fViewVolume[1] * fZoom;

   glFrustum(-frx, frx, -fry, fry, fViewVolume[2], fViewVolume[3]);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   fTransformation.Apply();
}

void GLPerspectiveCamera::TurnOn(Int_t x, Int_t y)const
{
   gluPickMatrix(x, fViewPort[3] - y, 1., 1., (Int_t *)fViewPort);
   Double_t frx = fViewVolume[0] * fZoom;
   Double_t fry = fViewVolume[1] * fZoom;

   glFrustum(-frx, frx, -fry, fry, fViewVolume[2], fViewVolume[3]);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   fTransformation.Apply();
}


class TGLRender {
   friend class TGLPixmap;
private:
   TObjArray      fGLObjects;
   TObjArray      fGLCameras;

   Bool_t         fGLInit;
   Bool_t         fAllActive;
   Int_t          fActiveCam;

   GLSceneObject *fFirstT;
   GLSceneObject *fSelectedObj;

public:
   TGLRender();
   virtual ~TGLRender();
   void Traverse();
   void SetAllActive(){fAllActive = kTRUE;}
   void SetActive(UInt_t cam);
   void AddNewObject(GLSceneObject *newObject);
   void RemoveAllObjects();
   void AddNewCamera(GLCamera *newCamera);
   GLSceneObject *SelectObject(Int_t x, Int_t y, Int_t);

   Int_t GetSize()const{return fGLObjects.GetEntriesFast();}

private:
   void DrawScene();
   void DrawAxes();

   void Init();
};

//______________________________________________________________________________
TGLRender::TGLRender()
{
   fGLObjects.SetOwner(kTRUE);
   fGLCameras.SetOwner(kTRUE);

   fGLInit = kFALSE;
   fAllActive = kTRUE;
   fActiveCam = 0;
   fFirstT = 0;
   fSelectedObj = 0;
}

//______________________________________________________________________________
TGLRender::~TGLRender()
{
}

//______________________________________________________________________________
void TGLRender::Traverse()
{
   if (!fGLInit) {
      fGLInit = kTRUE;
      Init();
   }

   Int_t start = 0, end = fGLCameras.GetEntriesFast();
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   if (end == 0) {
      return;
   }

   if (!fAllActive) {
      start = fActiveCam;
      end = start + 1;
   }

   for (; start < end; ++start) {
      GLCamera *currCam = (GLCamera *)fGLCameras.At(start);
      currCam->TurnOn();

      DrawScene();
   }
}

//______________________________________________________________________________
void TGLRender::SetActive(UInt_t ncam)
{
   fActiveCam = ncam;
   fAllActive = kFALSE;
}

//______________________________________________________________________________
void TGLRender::AddNewObject(GLSceneObject *newobject)
{
   fGLObjects.AddLast(newobject);
}

//______________________________________________________________________________
void TGLRender::RemoveAllObjects()
{
   fGLObjects.Delete();
   fSelectedObj = 0;
   assert(fGLObjects.GetEntriesFast() == 0);
}

//______________________________________________________________________________
void TGLRender::AddNewCamera(GLCamera *newcamera)
{
   fGLCameras.AddLast(newcamera);
}

//______________________________________________________________________________
GLSceneObject *TGLRender::SelectObject(Int_t x, Int_t y, Int_t cam)
{
   GLCamera *actCam = (GLCamera *)fGLCameras.At(cam);
   std::vector<UInt_t>selectBuff(fGLObjects.GetEntriesFast() * 4);
   std::vector<std::pair<UInt_t, Int_t> >objNames;

   glSelectBuffer(selectBuff.size(), &selectBuff[0]);
   glRenderMode(GL_SELECT);
   glInitNames();
   glPushName(0);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   actCam->TurnOn(x, y);

   DrawScene();

   Int_t hits = glRenderMode(GL_RENDER);

   if (hits < 0) {
      Error("TGLRender::SelectObject", "selection buffer overflow");
   } else if (hits > 0) {
      objNames.resize(hits);
      for (Int_t i = 0; i < hits; ++i) {
         //object's "depth"
         objNames[i].first = selectBuff[i * 4 + 1];
         //object's name
         objNames[i].second = selectBuff[i * 4 + 3];
      }
      std::sort(objNames.begin(), objNames.end());
      UInt_t chosen = 0;
      GLSceneObject *hitObject = 0;
      for (Int_t j = 0; j < hits; ++j) {
         chosen = objNames[j].second;
         hitObject = (GLSceneObject *)fGLObjects.At(chosen);
         if (!hitObject->IsTransparent())
            break;
      }
      if (hitObject->IsTransparent()) {
         chosen = objNames[0].second;
         hitObject = (GLSceneObject *)fGLObjects.At(chosen);
      }
      if (hitObject != fSelectedObj) {
         if (fSelectedObj) fSelectedObj->Select(kFALSE);
         fSelectedObj = hitObject;
         fSelectedObj->Select();
      }
   } else if (fSelectedObj) {
      fSelectedObj->Select(kFALSE);
      fSelectedObj = 0;
   }

   return fSelectedObj;
}

//______________________________________________________________________________
void TGLRender::Init()
{
   glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
   Float_t lmodelAmb[] = {0.5f, 0.5f, 1.f, 1.f};
   glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodelAmb);
   glEnable(GL_LIGHTING);
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);
   glClearDepth(1.);
}

//______________________________________________________________________________
void TGLRender::DrawScene()
{

   for (Int_t i = 0, e = fGLObjects.GetEntriesFast(); i < e; ++i) {
      GLSceneObject *currObj = (GLSceneObject *)fGLObjects.At(i);
      if (currObj->IsTransparent()) {
         currObj->SetNextT(fFirstT);
         fFirstT = currObj;
      } else {
         currObj->GLDraw();
      }
   }

   while (fFirstT) {
      fFirstT->GLDraw();
      fFirstT = fFirstT->GetNextT();
   }
}


////////////////////////////////////////////////////////////
//______________________________________________________________________________
TGLPixmap::TGLPixmap(TPad * pad, Int_t devInd, Int_t x, Int_t y, UInt_t w, UInt_t h) :
                   fCamera(), fViewVolume(), fZoom(),
                   fActiveViewport(), fBuildingScene(kFALSE),
                   fPad(pad), fFirstScene(kTRUE)
{
   x_ = x, y_ = y, w_ = w, h_ = h;
   devInd_ = devInd;

   fLightMask = 0x1b;
   fXc = fYc = fZc = fRad = 0.;
   fPressed = kFALSE;
   fNbShapes = 0;
   fSelectedObj = 0;
   fAction = kNoAction;

   CreateViewer();
   fArcBall = new TArcBall(w, h);
   CalculateViewports();
}

//______________________________________________________________________________
void TGLPixmap::CreateViewer()
{
   fZoom[0] = fZoom[1] = fZoom[2] = fZoom[3] = 1.;
   fRender = new TGLRender;
}

//______________________________________________________________________________
TGLPixmap::~TGLPixmap()
{
   delete fArcBall;
   delete fRender;
}

//______________________________________________________________________________
void TGLPixmap::MakeCurrent()const
{
   gGLManager->MakeCurrent(devInd_);
}

//______________________________________________________________________________
void TGLPixmap::SwapBuffers()const
{
}

//______________________________________________________________________________
void TGLPixmap::DrawObjects()const
{
   MakeCurrent();
   const_cast<TGLPixmap *>(this)->CalculateViewports();
   //new MVGL!!!
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

   Float_t pos1[] = {0., fRad + fYc, -fRad - fZc, 1.f};
   Float_t pos2[] = {fRad + fXc, 0.f, -fRad - fZc, 1.f};
   Float_t pos3[] = {0.f, -fRad - fYc, -fRad - fZc, 1.f};
   Float_t pos4[] = {-fRad - fXc, 0.f, -fRad - fZc, 1.f};
   Float_t pos5[] = {0.f, 0.f, 0.f, 1.f};

   Float_t whiteCol[] = {.7f, .7f, .7f, 1.f};
   glLightfv(GL_LIGHT4, GL_POSITION, pos1);
   glLightfv(GL_LIGHT4, GL_DIFFUSE, whiteCol);
   glLightfv(GL_LIGHT1, GL_POSITION, pos2);
   glLightfv(GL_LIGHT1, GL_DIFFUSE, whiteCol);
   glLightfv(GL_LIGHT2, GL_POSITION, pos3);
   glLightfv(GL_LIGHT2, GL_DIFFUSE, whiteCol);
   glLightfv(GL_LIGHT3, GL_POSITION, pos4);
   glLightfv(GL_LIGHT3, GL_DIFFUSE, whiteCol);
   glLightfv(GL_LIGHT0, GL_POSITION, pos5);
   glLightfv(GL_LIGHT0, GL_DIFFUSE, whiteCol);

   glEnable(GL_LIGHT4);
   glEnable(GL_LIGHT1);
   glEnable(GL_LIGHT2);
   glEnable(GL_LIGHT3);
   glEnable(GL_LIGHT0);

   fRender->Traverse();

   glFlush();
   gGLManager->Flush(devInd_);
}

//______________________________________________________________________________
void TGLPixmap::UpdateRange(const GLSelection *box)
{
   assert(fBuildingScene);
   const Double_t *X = box->GetRangeX();
   const Double_t *Y = box->GetRangeY();
   const Double_t *Z = box->GetRangeZ();

   if (!fRender->GetSize()) {
      fRangeX.first = X[0], fRangeX.second = X[1];
      fRangeY.first = Y[0], fRangeY.second = Y[1];
      fRangeZ.first = Z[0], fRangeZ.second = Z[1];
      return;
   }

   if (fRangeX.first > X[0])
      fRangeX.first = X[0];
   if (fRangeX.second < X[1])
      fRangeX.second = X[1];
   if (fRangeY.first > Y[0])
      fRangeY.first = Y[0];
   if (fRangeY.second < Y[1])
      fRangeY.second = Y[1];
   if (fRangeZ.first > Z[0])
      fRangeZ.first = Z[0];
   if (fRangeZ.second < Z[1])
      fRangeZ.second = Z[1];
}


//______________________________________________________________________________
TObject *TGLPixmap::SelectObject(Int_t x, Int_t y)
{
   MakeCurrent();
   CalculateViewvolumes();

   Int_t tmpVal = fActiveViewport[1];
   fActiveViewport[1] = 0;
   GLSceneObject *obj = fRender->SelectObject(x, y, 0);
   fActiveViewport[1] = tmpVal;

   if (obj) {
      return obj->GetRealObject();
   }

   return 0;
}

Int_t TGLPixmap::DistancetoPrimitive(Int_t x, Int_t y)
{
   TObject *selection = gGLManager->Select(this, x, y);
   if(selection) gPad->SetSelected(selection);
   else gPad->SetSelected(this);

   return 0;
}

//______________________________________________________________________________
void TGLPixmap::CalculateViewports()
{
   gGLManager->ExtractViewport(devInd_, fActiveViewport);
}

//______________________________________________________________________________
void TGLPixmap::CalculateViewvolumes()
{
   CalculateViewports();

   if (fRender->GetSize()) {
      Double_t xdiff = fRangeX.second - fRangeX.first;
      Double_t ydiff = fRangeY.second - fRangeY.first;
      Double_t zdiff = fRangeZ.second - fRangeZ.first;
      Double_t max = xdiff > ydiff ? xdiff > zdiff ? xdiff : zdiff : ydiff > zdiff ? ydiff : zdiff;

      Double_t frx = 1., fry = 1.;

      if (fActiveViewport[2] > fActiveViewport[3])
         frx = fActiveViewport[2] / double(fActiveViewport[3]);
      else if (fActiveViewport[2] < fActiveViewport[3])
         fry = fActiveViewport[3] / double(fActiveViewport[2]);

      fViewVolume[0] = max / 1.9 * frx;
      fViewVolume[1] = max / 1.9 * fry;
      fViewVolume[2] = max * 0.707;
      fViewVolume[3] = 3 * max;
      fRad = max * 1.7;
   }
}

//______________________________________________________________________________
void TGLPixmap::CreateCameras()
{
   if (!fRender->GetSize())
      return;

   GLSimpleTransform trPersp(fArcBall->GetRotMatrix(), fRad, &fXc, &fYc, &fZc);

   fCamera = new GLPerspectiveCamera(fViewVolume, fActiveViewport, trPersp);

   fRender->AddNewCamera(fCamera);
}

//______________________________________________________________________________
Bool_t TGLPixmap::PreferLocalFrame() const
{
   // Not at present - but in the future....
   return kFALSE;
}

void TGLPixmap::BeginScene()
{
   // Scene builds can't be nested
   if (fBuildingScene) {
      assert(kFALSE);
      return;
   }

   // Clear any existing scene contents
   fRender->RemoveAllObjects();
   fNbShapes = 0;
   fBuildingScene = kTRUE;
}

//______________________________________________________________________________
void TGLPixmap::EndScene()
{
   CalculateViewvolumes();

   if (fFirstScene) {
      // Calculate light sources positions
      Double_t xdiff = fRangeX.second - fRangeX.first;
      Double_t ydiff = fRangeY.second - fRangeY.first;
      Double_t zdiff = fRangeZ.second - fRangeZ.first;

      fXc = fRangeX.first + xdiff / 2;
      fYc = fRangeY.first + ydiff / 2;
      fZc = fRangeZ.first + zdiff / 2;

      CreateCameras();
      fFirstScene = kFALSE;
   }

   fBuildingScene = kFALSE;
   gGLManager->DrawViewer(this);
}

Int_t TGLPixmap::AddObject(const TBuffer3D &buffer, Bool_t *addChildren)
{
   if (addChildren) {
      *addChildren = kTRUE;
   }

   UInt_t reqSections = TBuffer3D::kCore|TBuffer3D::kRawSizes|TBuffer3D::kRaw;

   if (!buffer.SectionsValid(reqSections)) {
      return reqSections;
   }

   Float_t rgba[] = {.4f, .4f, .4f, 1.f};
   TColor *c = gROOT->GetColor(buffer.fColor);

      if (c && buffer.fColor > 1) {
         c->GetRGB(rgba[0], rgba[1], rgba[2]);
      }


   GLFaceSet *newPix = new GLFaceSet(buffer, rgba, fNbShapes ++, buffer.fID);
   fRender->AddNewObject(newPix);
   UpdateRange(newPix->GetBBox());

   return TBuffer3D::kNone;
}

Bool_t TGLPixmap::OpenComposite(const TBuffer3D &, Bool_t *)
{
   return kFALSE;
}

void TGLPixmap::CloseComposite()
{
}

void TGLPixmap::AddCompositeOp(UInt_t)
{
}

void TGLPixmap::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   switch(event) {
   case kButton1Down:
      //fix arc ball first
      CalculateViewports();
      fArcBall->SetBounds(fActiveViewport[2], fActiveViewport[3]);
      fArcBall->Click(TPoint(px, py));
      gGLManager->MarkForDirectCopy(devInd_, kTRUE);
      break;
   case kButton1Up:
      gGLManager->MarkForDirectCopy(devInd_, kFALSE);
      break;
   case kButton1Motion:
      fArcBall->Drag(TPoint(px, py));
      gGLManager->DrawViewer(this);
      break;
   case kMouseMotion:
      gPad->SetCursor(kRotate);
      break;
   case kKeyPress:
      if (py == kKey_J || py == kKey_j) {
         gGLManager->MarkForDirectCopy(devInd_, kTRUE);
         fZoom[0] /= 1.2;
         fCamera->Zoom(fZoom[0]);
         gGLManager->DrawViewer(this);
         gGLManager->MarkForDirectCopy(devInd_, kFALSE);
      } else if (py == kKey_K || py == kKey_k) {
         gGLManager->MarkForDirectCopy(devInd_, kTRUE);
         fZoom[0] *= 1.2;
         fCamera->Zoom(fZoom[0]);
         gGLManager->DrawViewer(this);
         gGLManager->MarkForDirectCopy(devInd_, kFALSE);
      }

   }
}

void TGLPixmap::DrawViewer()
{
   gGLManager->MakeCurrent(devInd_);
   fRender->Init();

   Color_t backColor = fPad->GetFillColor();
   Float_t rgb[] = {1.f, 1.f, 1.f};//white will be default
   TColor *c = gROOT->GetColor(backColor);

   if (c && backColor > 1) {
      c->GetRGB(rgb[0], rgb[1], rgb[2]);
   }

   glClearColor(rgb[0], rgb[1], rgb[2], 1.f);

   DrawObjects();
}

void TGLPixmap::ZoomIn()
{
   fZoom[0] /= 2;
   fCamera->Zoom(fZoom[0]);
}

void TGLPixmap::ZoomOut()
{
   fZoom[0] *= 2;
   fCamera->Zoom(fZoom[0]);
}
