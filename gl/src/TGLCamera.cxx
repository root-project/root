#ifdef GDK_WIN32
#include <WIndows4Root.h>
#endif

#include <iostream>

#include <GL/gl.h>
#include <GL/glu.h>

#include "TGLCamera.h"

TGLCamera::TGLCamera(const Double_t *vv, const Int_t *vp)
              :fViewVolume(vv), fViewPort(vp),
               fZoom(1.), fDrawFrame(kFALSE)
{
}

TGLTransformation::~TGLTransformation()
{
}

TGLSimpleTransform::TGLSimpleTransform(const Double_t *rm, Double_t s, Double_t x,
                                       Double_t y, Double_t z)
                        :fRotMatrix(rm), fShift(s),
                         fX(x), fY(y), fZ(z)
{
}

void TGLSimpleTransform::Apply()const
{
   glTranslated(0., 0., -fShift);
   glMultMatrixd(fRotMatrix);
   glRotated(-90., 1., 0., 0.);
   glTranslated(-fX, -fY, -fZ);
}

TGLPerspectiveCamera::TGLPerspectiveCamera(const Double_t *vv, const Int_t *vp,
                                           const TGLSimpleTransform &tr)
                         :TGLCamera(vv, vp),
                          fTransformation(tr) 
{
}

void TGLPerspectiveCamera::TurnOn()const
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

void TGLPerspectiveCamera::TurnOn(Int_t x, Int_t y)const
{
   gluPickMatrix(x, fViewPort[3] - y, 1., 1., (Int_t *)fViewPort);
   Double_t frx = fViewVolume[0] * fZoom;
   Double_t fry = fViewVolume[1] * fZoom;

   glFrustum(-frx, frx, -fry, fry, fViewVolume[2], fViewVolume[3]);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   fTransformation.Apply();
}

TGLOrthoCamera::TGLOrthoCamera(const Double_t *vv, const Int_t *vp,
                               const TGLSimpleTransform &tr)
                   :TGLCamera(vv, vp), 
                    fTransformation(tr)
{
}

void TGLOrthoCamera::TurnOn()const
{
   glViewport(fViewPort[0], fViewPort[1], fViewPort[2], fViewPort[3]);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   Double_t frx = fViewVolume[0] * fZoom;
   Double_t fry = fViewVolume[1] * fZoom;

   glOrtho(-frx, frx, -fry, fry, fViewVolume[2], fViewVolume[3]);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   fTransformation.Apply();
}

void TGLOrthoCamera::TurnOn(Int_t x, Int_t y)const
{
   Int_t viewport[4] = {};
   glGetIntegerv(GL_VIEWPORT, viewport);
   gluPickMatrix(x, fViewPort[3] - y, 1., 1., (Int_t *)fViewPort);
   Double_t frx = fViewVolume[0] * fZoom;
   Double_t fry = fViewVolume[1] * fZoom;

   glOrtho(-frx, frx, -fry, fry, fViewVolume[2], fViewVolume[3]);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   fTransformation.Apply();
}
