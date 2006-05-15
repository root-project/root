// @(#)root/gl:$Name:  $:$Id: TGLKernel.cxx,v 1.41 2006/01/26 11:59:41 brun Exp $
// Author: Valery Fine(fine@vxcern.cern.ch)   05/03/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLKernel                                                            //
//                                                                      //
// The TGLKernel implementation of TVirtualGL class.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TError.h"
#include "TGLKernel.h"
#include "TView.h"
#include "TGeometry.h"
#include "TROOT.h"
#include "TColor.h"
#include "TPoints3DABC.h"
#include "TGLViewer.h"
#include "TGLManip.h"
#include "TGLOutput.h"
#include "TGLRenderArea.h"
#include "TSystem.h"

#include <GL/gl.h>

#include "gl2ps.h"

#ifndef   ColorOffset
#define  ColorOffset 0
#endif

//ClassImp(TGLKernel)

GLenum GLCommand[] = { GLConstants(GL_)  };


//______________________________________________________________________________
TGLKernel::TGLKernel(TVirtualGLImp *imp) : TVirtualGL(imp), fQuad(0), fTessObj(0)
{
   // Constructor.

   gVirtualGL = this;
   gROOT->GetListOfSpecials()->Add(this);
}


//______________________________________________________________________________
TGLKernel::TGLKernel(const char *name) : TVirtualGL(name), fQuad(0), fTessObj(0)
{
   // Constructor.

   gVirtualGL = this;
   gROOT->GetListOfSpecials()->Add(this);
}


//______________________________________________________________________________
TGLKernel::~TGLKernel()
{
   // Destructor.

   gROOT->GetListOfSpecials()->Remove(this);
   gluDeleteQuadric(fQuad);
}


//______________________________________________________________________________
void TGLKernel::ClearColor(Int_t color)
{
   // Clear GL color.

   if (GetTrueColorMode()) {
      Float_t red;
      Float_t green;
      Float_t blue;
      Float_t alpha = 0.0;

      TColor *c = gROOT->GetColor(color);
      if (!c) c= gROOT->GetColor(1);
      c->GetRGB(red,green,blue);
      glClearColor(red, green, blue, alpha);
   } else {
      glClearIndex(color+ColorOffset);
   }
}


//______________________________________________________________________________
void TGLKernel::ClearGLColor(Float_t red, Float_t green, Float_t blue, Float_t alpha)
{
   // Clear GL color.

   glClearColor(red, green, blue, alpha);
}


//______________________________________________________________________________
void TGLKernel::ClearGLColor(Float_t *colors)
{
   // Clear GL color.

   GLclampf red   = colors[0];
   GLclampf green = colors[1];
   GLclampf blue  = colors[2];
   GLclampf alpha = colors[3];
   glClearColor(red, green, blue, alpha);
}


//______________________________________________________________________________
void TGLKernel::ClearGL(UInt_t stereo)
{
   // Clear GL.

#ifdef STEREO_GL
   if (stereo) {
      if (Int_t(stereo) < 0)
         glDrawBuffer(GL_BACK_LEFT);
      else
         glDrawBuffer(GL_BACK_RIGHT);
   }
#endif
   if (stereo) { }
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}


//______________________________________________________________________________
void TGLKernel::DisableGL(EG3D2GLmode mode)
{
   // Disable GL.

   glDisable(GLCommand[mode]);
}


//______________________________________________________________________________
void TGLKernel::EnableGL(EG3D2GLmode mode)
{
   // Enable GL.

   glEnable(GLCommand[mode]);
}


//______________________________________________________________________________
void TGLKernel::FlushGL()
{
   // FLush GL buffer.

   glFlush();
}


//______________________________________________________________________________
void TGLKernel::FrontGLFace(EG3D2GLmode faceflag)
{
   // GL front face.

   fFaceFlag = faceflag;
   glFrontFace(GLCommand[faceflag]);
}


//______________________________________________________________________________
void TGLKernel::NewGLList(UInt_t ilist, EG3D2GLmode mode)
{
   // New GL list.

   glNewList(ilist, GLCommand[mode]);
}


//______________________________________________________________________________
void TGLKernel::NewGLModelView(Int_t ilist)
{
   // New GL mode view.

   glNewList(ilist, GL_COMPILE);
}


//______________________________________________________________________________
void TGLKernel::GetGL(EG3D2GLmode mode, UChar_t *params)
{
   // Get GL.

   glGetBooleanv(GLCommand[mode], params);
}


//______________________________________________________________________________
void TGLKernel::GetGL(EG3D2GLmode mode, Double_t *params)
{
   // Get GL.

   glGetDoublev(GLCommand[mode], params);
}


//______________________________________________________________________________
void TGLKernel::GetGL(EG3D2GLmode mode, Float_t *params)
{
   // Get GL.

   glGetFloatv(GLCommand[mode], params);
}


//______________________________________________________________________________
void TGLKernel::GetGL(EG3D2GLmode mode, Int_t *params)
{
   // Get GL.

   glGetIntegerv(GLCommand[mode], params);
}


//______________________________________________________________________________
Int_t TGLKernel::GetGLError()
{
   // Get GL error.

   return glGetError();
}


//______________________________________________________________________________
void TGLKernel::EndGLList()
{
   // End GL list.

   glEndList();
}


//______________________________________________________________________________
void TGLKernel::BeginGLCmd(EG3D2GLmode mode)
{
   // Begin GL.

   glBegin(GLCommand[mode]);
}


//______________________________________________________________________________
void TGLKernel::EndGLCmd()
{
   // End GL.

   glEnd();
}


//______________________________________________________________________________
void TGLKernel::PushGLMatrix()
{
   // Push GL matrix.

   glPushMatrix();
}


//______________________________________________________________________________
void TGLKernel::PopGLMatrix()
{
   // Pop GL matrix.

   glPopMatrix();
}


//______________________________________________________________________________
void TGLKernel::RotateGL(Double_t *direction, Int_t mode)
{
   // GL rotate.

   if (mode) {
      Double_t angle = direction[0];
      Double_t x     = direction[1];
      Double_t y     = direction[2];
      Double_t z     = direction[3];
      RotateGL(angle,x,y,z);
   } else {
      // Double_t theta   - polar angle for the axis x`
      // Double_t phi     - azimutal angle for the axis x`
      // Double_t psi     - azimutal angle for the axis y`
      Double_t theta = direction[0];
      Double_t phi   = direction[1];
      Double_t psi   = direction[2];
      RotateGL(theta,phi,psi);
   }
}


//______________________________________________________________________________
void TGLKernel::RotateGL(Double_t angle, Double_t x,Double_t y,Double_t z)
{
   // The RotateGL function computes a matrix that performs a counterclockwise
   // rotation of angle degrees about the vector from the origin through
   // the point (x, y, z).

   glRotated(angle,x,y,z);
}


//______________________________________________________________________________
void TGLKernel::RotateGL(Double_t theta, Double_t phi, Double_t psi)
{
   // Rotate GL.
   //
   // Double_t theta   - polar angle for the axis x`
   // Double_t phi     - azimutal angle for the axis x`
   // Double_t psi     - azimutal angle for the axis y`

   if (TMath::Abs(90-theta) > 0.00001) RotateGL(theta-90, 1, 0, 0);
   if (TMath::Abs(90-psi)   > 0.00001) RotateGL(90-psi,0, 1, 0);
   if (TMath::Abs(phi)      > 0.00001) RotateGL(phi, 0, 0, 1);
}


//______________________________________________________________________________
void TGLKernel::TranslateGL(Double_t x,Double_t y,Double_t z)
{
   // GL Translate.

   glTranslated(x,y,z);
}


//______________________________________________________________________________
void TGLKernel::TranslateGL(Double_t *xyz)
{
   // GL Translate.

   TranslateGL(xyz[0], xyz[1],xyz[2]);
}


//______________________________________________________________________________
void TGLKernel::MultGLMatrix(Double_t *mat)
{
   // Mult GL matrix.

   glMultMatrixd(mat);
}


//______________________________________________________________________________
void TGLKernel::SetGLColor(Float_t *rgb)
{
   // Set GL color.

   glColor3fv(rgb);
}


//______________________________________________________________________________
void TGLKernel::SetGLVertex(Float_t *vertex)
{
   // Set GL vertex.

   glVertex3fv(vertex);
}


//______________________________________________________________________________
void TGLKernel::SetGLVertex(const Double_t *vert)
{
   // Set GL vertex.

   glVertex3dv(vert);
}


//______________________________________________________________________________
void TGLKernel::SetGLColorIndex(Int_t color)
{
   // Set GL color index.

   if (color != -1) fColorIndx = color;

   // Change BLACK color to GRAY to make it visible
   if (fColorIndx == 1 && gVirtualGL->GetTrueColorMode()) fColorIndx=19;
   SetCurrentColor(fColorIndx);
}


//______________________________________________________________________________
void TGLKernel::SetCurrentColor(Int_t color)
{

   // Set current color.

   if (gVirtualGL->GetTrueColorMode()) {
      Float_t rgb[3];
      Float_t red;
      Float_t green;
      Float_t blue;

      TColor *c = gROOT->GetColor(color);
      if (!c) c = gROOT->GetColor(17);
      c->GetRGB(red,green,blue);
      rgb[0] = red;
      rgb[1] = green;
      rgb[2] = blue;
      SetGLColor(rgb);
   } else {
      glIndexi(color+ColorOffset);
   }
}


//______________________________________________________________________________
void TGLKernel::SetGLPointSize(Float_t size)
{
   // The SetGLPointSize function specifies the diameter of rasterized points.

   glPointSize(size);
}


//______________________________________________________________________________
void TGLKernel::SetGLLineWidth(Float_t width)
{
   // The SetGLLineWidth function specifies the width of rasterized lines.

   glLineWidth(width);
}


//______________________________________________________________________________
void TGLKernel::SetRootLight(Bool_t flag)
{
   // Set Root light.

   if (flag != fRootLight) {
      fRootLight = flag;
      if (fRootLight) {
         glDisable(GL_LIGHT0);
         glDisable(GL_LIGHTING);
         glDisable(GL_COLOR_MATERIAL);
      } else {
         glEnable(GL_LIGHT0);
         glEnable(GL_LIGHTING);
         glEnable(GL_COLOR_MATERIAL);
      #ifdef STEREO_GL
         glEnable(GL_STEREO);
      #endif
      }
   }
}


//______________________________________________________________________________
void TGLKernel::DeleteGLLists(Int_t ilist, Int_t range)
{
   // Delete GL list.

   glDeleteLists(ilist, range);
}


//______________________________________________________________________________
Int_t TGLKernel::CreateGLLists(Int_t range)
{
   // Create GL list.

   return glGenLists(range);
}


//______________________________________________________________________________
void TGLKernel::RunGLList(Int_t list)
{
   // Run GL listi (call list).

   glCallList(list);
}


//______________________________________________________________________________
void TGLKernel::NewProjectionView(Double_t min[], Double_t max[], Bool_t perspective)
{
   // Define new Projection view.

#if 0
   if (kTRUE) {
      glEnable(GL_LIGHT0);
      glEnable(GL_LIGHTING);
      glEnable(GL_COLOR_MATERIAL);
      glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,0);
      glColorMaterial(GL_FRONT,GL_AMBIENT_AND_DIFFUSE);
   } else {
      ::SetRootLight(kTRUE);
   }
#endif

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   Double_t dnear = TMath::Abs(max[0]-min[0]);
   Double_t dfar = 3*(dnear + TMath::Abs(max[2]-min[2]));

   if (perspective)
      glFrustum(min[0],max[0],min[1],max[1],dnear,dfar);
   else
      glOrtho  (min[0],max[0],min[1],max[1],dnear,dfar);

   glCullFace(GL_BACK);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();

#if 0
   if (!fRootLight && fTrueColorMode) {
      glPushMatrix();
      Float_t rgb[] = {0.8,0.8,0.8};
      SetGLColor(rgb);
      glRotated(75,1,0,0);
      glNormal3f(0.0,0.0,1.0);
      glRectd(min[0],min[1],max[0],max[1]);
      glPopMatrix();
   }
#endif
}


//______________________________________________________________________________
void TGLKernel::PolygonGLMode(EG3D2GLmode face , EG3D2GLmode mode)
{
   // Set GL polygon mode.

   glPolygonMode(GLCommand[face], GLCommand[mode]);
}


//______________________________________________________________________________
void TGLKernel::SetStack(Double_t *matrix)
{
   // SetStack(Double_t *matrix) method replaces the matrix on the top of the
   //                            stack of the matrix with a new one supplied or
   //                            with the indentity.
   // Parameters:
   // ----------
   // Double_t *matrix - the pointer to a new matrix to updtae stack
   //                      = 0 the indentity matrix must be applied

   glPopMatrix();
   if (!matrix) glLoadIdentity();
   else glLoadMatrixd(matrix);

   glPushMatrix();
}


//______________________________________________________________________________
void TGLKernel::ShadeGLModel(EG3D2GLmode model)
{
   // Set GL shade model.

   glShadeModel(GLCommand[model]);
}


//______________________________________________________________________________
void TGLKernel::AddRotation(Double_t *rotmatrix, Double_t *angles)
{
   // Add Rotation.

   GLint mode;
   glPushMatrix();
   glGetIntegerv(GL_MATRIX_MODE,&mode);

   glLoadIdentity();
   if (TMath::Abs(angles[0]) > 0.00001) RotateGL(angles[0],1,0,0);
   if (TMath::Abs(angles[1]) > 0.00001) RotateGL(angles[1],0,1,0);
   if (TMath::Abs(angles[2]) > 0.00001) RotateGL(angles[2],0,0,1);
   glMultMatrixd(rotmatrix);

   switch (mode) {
   case GL_MODELVIEW:
      glGetDoublev(GL_MODELVIEW_MATRIX, rotmatrix);
      break;
   case GL_PROJECTION:
      glGetDoublev(GL_PROJECTION_MATRIX,rotmatrix);
      printf(" projection \n");
      break;
   case GL_TEXTURE:
      glGetDoublev(GL_TEXTURE_MATRIX,   rotmatrix);
      printf(" texture \n");
      break;
   default:
      Error("TGLKernel::AddRotation", "unknown matrix!");
      return;
   }

   glPopMatrix();
}


//______________________________________________________________________________
void TGLKernel::NewModelView(Double_t *angles,Double_t *delta )
{
   // New model view.

   RotateGL(-(-90+angles[0]), -( 90+angles[1]), 90+angles[2]);

   // Move model to the center of the "view" box
   TranslateGL(delta[0], delta[1], delta[2]);
}


//______________________________________________________________________________
void TGLKernel::PaintGLPointsObject(const TPoints3DABC *points, Option_t *option)
{
   // PaintGLPointsObject - draws the points
   //
   // option = "L"        - connect all points with straight lines
   //          "P"        - draw the 3d points at each coordinate (by default)
   //          "LP"       - draw the 3D points and connect them with
   //                       straight lines

   if (!points) return;
   Int_t n = points->Size();

   if (n <=0) return;

   GLenum mode = GL_POINTS;
   Int_t pass = 0;
   if (option && strchr(option,'P')) { pass++;}
   if (option && strchr(option,'L')) { mode = GL_LINE_STRIP; pass++;}

   while (pass >= 0) {
      pass--;
      glBegin(mode);
      for (int i=0; i < n; i++) {
         glVertex3f(GLfloat(points->GetX(i)),
                      GLfloat(points->GetY(i)),
                      GLfloat(points->GetZ(i)));
      }
      glEnd();
      mode=GL_POINTS;
   }
}


//______________________________________________________________________________
void TGLKernel::PaintGLPoints(Int_t n, Float_t *p, Option_t *)
{
   // Paint GL points.

   if (n <= 0 || p == 0) return;
   GLfloat *point = p;

   glBegin(GL_POINTS);

   for (int i=0; i < n; i++, point+=3) glVertex3fv(point);
   glEnd();
}


//______________________________________________________________________________
void TGLKernel::PaintPolyLine(Int_t n, Float_t *p, Option_t *)
{
   // Paint Polyline via OpenGL.

   if (n <= 0 || p == 0) return;
   GLfloat *line = p;

   glBegin(GL_LINE_STRIP);

   for (int i=0; i < n; i++, line+=3) glVertex3fv(line);
   glEnd();
}


//______________________________________________________________________________
void TGLKernel::PaintPolyLine(Int_t n, Double_t *p, Option_t *)
{
   // Paint Polyline via OpenGL.

   if (n <= 0 || p == 0) return;
   GLdouble *line = p;

   glBegin(GL_LINE_STRIP);

   for (int i=0; i < n; i++, line+=3) glVertex3dv(line);
   glEnd();
}


//______________________________________________________________________________
void TGLKernel::PaintBrik(Float_t vertex[24])
{
   // Paint BRIK via OpenGL.

#define vert(i) &vertex[3*(i)]
   Int_t i;
   Float_t normal[3];

   if (vertex) {

      // The counterclockwise polygons are taken to be front-facing by default

      if (fRootLight) LightIndex(0);

      glBegin(GL_QUADS);
      //*-*  "Top"  of TBRIK
      if (!fRootLight)
         glNormal3fv(TMath::Normal2Plane(vert(7),vert(4),vert(6),normal));
      for (i=4;i<8;i++)
         glVertex3fv(vert(i));

      // "Bottom"  of TBRIK
      if (!fRootLight)
         glNormal3fv(TMath::Normal2Plane(vert(0),vert(3),vert(1),normal));
      for (i=3;i>-1;i--) {
         glVertex3fv(vert(i));
      }

      // "Walls"  of TBRIK
      for (i=0;i<3;i++) {
         if (fRootLight)
            LightIndex(i+1);
         else
            glNormal3fv(TMath::Normal2Plane(vert(i),vert(i+1),vert(i+4),normal));

         glVertex3fv(vert(i));
         glVertex3fv(vert(i+1));
         glVertex3fv(vert(i+5));
         glVertex3fv(vert(i+4));
      }

      // The last "wall" to close the brik
      if (fRootLight)
         LightIndex(i+1);
      else
         glNormal3fv(TMath::Normal2Plane(vert(i),vert(0),vert(i+4),normal));

      i = 0;
      glVertex3fv(vert(i));
      glVertex3fv(vert(i+4));
      i = 3;
      glVertex3fv(vert(i+4));
      glVertex3fv(vert(i));

      glEnd();

      if (fRootLight)
         LightIndex(0);  //reset the original color
   }
#undef vert
}


//______________________________________________________________________________
void TGLKernel::PaintXtru(Float_t *vertex, Int_t nxy, Int_t nz)
{
   // Paint Xtru shape via OpenGL.

   Float_t frontnorm[3] = {0,0,1};  // top (normally max z)
   Float_t backnorm[3]  = {0,0,-1}; // bottom (normally min z)
   Float_t normal[3]    = {0,0,0};

   // OpenGL doesn't handle concave polygons correctly
   // it isn't required to do anything more than the convex hull of all vertices
   Float_t *p = vertex;
   Float_t *start = vertex;
   Int_t ixy = 0;

   // the front face should go around counter clockwise
   // while the back go around clockwise

   // Front or top face
   if (fRootLight) LightIndex(0);
   else            glNormal3fv(frontnorm);

   glBegin(GL_POLYGON);
   p  = vertex+3*(nz-1)*nxy;

   start = p;

   for (ixy=0; ixy<nxy; glVertex3fv(p), ixy++,p+=3);

   glVertex3fv(start);
   glEnd();

   // Back face
   // go around in given order to keep outward normal
   if (fRootLight) LightIndex(0);
   else            glNormal3fv(backnorm);

   glBegin(GL_POLYGON);
   p     = vertex+3*(nxy-1);
   start = p;

   for (ixy=0; ixy<nxy; glVertex3fv(p), ixy++,  p-=3);

   glVertex3fv(start);
   glEnd();

   // The sides are given as a QUAD_STRIP list but care must be taken
   // to ensure that the normals are "outward" going.  Which way to
   // specify the points isn't quite clear.  This sequence has been
   // empirically determined to give the right direction.  Be wary
   // of changing it or risk turning the volume inside out when drawing
   // it in sold form.   The filling of the points buffer has taken
   // care of ordering the points in counterclockwise, increasing z order.

   Int_t cindex[] = { -1, 1, 2, -1};
   Int_t ncol = ((nxy&1) == 1) ? 3 : 2;

   Float_t *key = vertex;
   for (Int_t iz=0; iz<nz-1; iz++) {
      glBegin(GL_QUAD_STRIP);

      for (Int_t ixy=nxy; ixy > -1; ixy--) {

         Float_t *p1 = key + 3*(ixy%nxy); // reconnect back to first
         Float_t *p2 = p1  + 3*nxy;

         // one more point is needed to calculate the normal
         // take next point along in CCW order so that normal is _out_
         Float_t *px = key + 3*((ixy+1)%nxy);

         if (fRootLight) {
            // pick light indices such that adjacent quads don't
            // have the same color (nor do they match the ends)
            LightIndex(cindex[ixy%ncol+iz%2]);
         } else
            glNormal3fv(TMath::Normal2Plane(p1,px,p2,normal));

         glVertex3fv(p1);
         glVertex3fv(p2);
      }
      key += 3*nxy;
      glEnd();
   }

   if (fRootLight)  LightIndex(0);  //reset the original color
}


//______________________________________________________________________________
Float_t *Normal2Line(Float_t *p1, Float_t *p2, Float_t *normal)
{
   // Compute a normal vector to the conic surface
   //
   //  Principal chord:
   //  ----------------
   //     p1 = (a1,b1,c1);
   //     p2 = (a2,b2,c2);
   //
   //     v      = (a2-a1,b2-b1,c2-c1);
   //     normal = (a2-0,b2-0,c2-Z);
   //
   //     v*normal = 0;
   //
   //          a2*(a2-a1) + b2*(b2-b1) + c2*(c2-c1)
   //     z = -------------------------------------
   //                         c2-c1

   Float_t z = 0;
   Int_t i;

   for(i=0;i<3;i++) {
      normal[i] = p2[i];
      z += p2[i]*(p2[i]-p1[i]);
   }
   z /= p2[2]-p1[2];

   normal[2] -= z;

   TMath::Normalize(normal);

   return normal;
}


//______________________________________________________________________________
void TGLKernel::PaintCone(Float_t *vertex,Int_t nseg,Int_t nstacks)
{
   // Paint Cone.
   //
   // vertex  - array of the 3d coordinates
   // nseg    - number of segments (precision)
   //           < 0 means the shape is segmented
   //           > 0 means the shape is closed
   // nstacks -  number of stack sections
   //
   // Vertex[2*(3*ndiv*nstacks)]
   //  i - the division number
   //  j - the stack number
   //  k = 0 internal points
   //      1 external points

#define vert(i,j,k) (&vertex[3*((i)+2*ndiv*((j)+(k)))])
   Int_t i,j;
   Float_t normal[3];
   GLfloat *nextv, *exnextv;
   Int_t ndiv = TMath::Abs(nseg);
   Int_t pt3 = 3*ndiv;
   Int_t pt6 = 2*pt3;
   Float_t onenorm[3] = {0,0,1};
   Float_t backone[3] = {0,0,-1};
   if (vertex) {
      // The counter clockwise polygons are taken to be front-facing by default

      // "Top"  of TPCON
      if (fRootLight)
         LightIndex(0);
      else
         glNormal3fv(onenorm);

      // Check internal radius
      nextv = vert(0,nstacks-1,0);
      if (*(nextv+1) == 0.0 && *nextv == 0.0) { // The inner radius is ZERO

         // Draw the "Triangle fan"
         glBegin(GL_TRIANGLE_FAN);
         glVertex3f(0.0,0.0,*(nextv+2)); //*-* Set the center of the fan
         nextv += pt3;
         for (i=0;i<ndiv;i++) {
            glVertex3fv(nextv);
            nextv += 3;
         }
         if (nseg > 0)
         glVertex3fv(nextv-pt3);
      } else {
         // Draws the series of the quadrilaterals
         glBegin(GL_QUAD_STRIP);
         exnextv = nextv + pt3;
         for (i=0;i<ndiv;i++) {
            glVertex3fv(nextv);
            glVertex3fv(exnextv);
            nextv += 3;
            exnextv += 3;
         }
         if (nseg > 0 ) {
            glVertex3fv(nextv  - pt3);
            glVertex3fv(exnextv- pt3);
         }

      }
      glEnd();

      // "Bottom"  of TPCON
      if (!fRootLight)
         glNormal3fv(backone);

      nextv = vert(0,0,0);
      if (*(nextv+1) == 0.0 && *nextv == 0.0 ) {
         // Draw the "Triangle fan"
         glBegin(GL_TRIANGLE_FAN);
         glVertex3f(0.0,0.0,*(nextv+2)); //*-* Set the center of the fan
         nextv += pt6;
         for (i=0;i<ndiv;i++) {
            nextv -= 3;
            glVertex3fv(nextv);
         }
         if (nseg > 0)
            glVertex3fv(nextv+pt3-3);
      } else {
         //*-*  Draws the series of the quadrilaterals
         glBegin(GL_QUAD_STRIP);
         exnextv = nextv + pt3;
         for (i=0;i<ndiv;i++) {
            glVertex3fv(exnextv);
            glVertex3fv(nextv);
            nextv += 3;
            exnextv +=3;
         }
         if (nseg > 0) {
            glVertex3fv(exnextv- pt3);
            glVertex3fv(nextv  - pt3);
         }
      }
      glEnd();

      // "Walls"  of TPCON
      // Create the external walls
      nextv = vertex + pt3;
      for (i=0;i<nstacks-1;i++) {
         glBegin(GL_QUAD_STRIP);
         {
            // nextv = vert(0,i,1);
            exnextv = nextv+pt6;
            for(j=0;j<ndiv;j++) {
               if (fRootLight)
                  LightIndex(j>>1);
               else
                  glNormal3fv(Normal2Line(exnextv,nextv,normal));

               glVertex3fv(exnextv);
               glVertex3fv(nextv);
               nextv += 3;
               exnextv += 3;
            }
            if (nseg > 0) {
               // To "close" shape we have to add on extra "wall"
               if (fRootLight)
                  LightIndex(j>>1);
               else
                  glNormal3fv(Normal2Line(exnextv-pt3,nextv-pt3,normal));

               glVertex3fv(exnextv - pt3);
               glVertex3fv(nextv   - pt3);
            }
            nextv += 3*ndiv;
         }
         glEnd();
      }

      // Create the internal walls
      exnextv = vertex;
      for (i=0;i<nstacks-1;i++) {
         if (*(exnextv+1) == 0.0 && *exnextv == 0.0) continue; // No internal wall at all

         glBegin(GL_QUAD_STRIP);
         {
            nextv = exnextv+pt6;
            for(j=0;j<ndiv;j++) {
               if (fRootLight)
                  LightIndex(j>>1);
               else
                  glNormal3fv(Invert(Normal2Line(nextv,exnextv,normal)));

               glVertex3fv(exnextv);
               glVertex3fv(nextv);
               nextv += 3;
               exnextv += 3;
            }
            if (nseg > 0) {
               // To "close" shape we have to add on extra "wall"
               if (fRootLight)
                  LightIndex(j>>1);
               else
                  glNormal3fv(Invert(Normal2Line(nextv-pt3,exnextv-pt3,normal)));

               glVertex3fv(exnextv - pt3);
               glVertex3fv(nextv   - pt3);
            }
            exnextv += pt3;
         }
         glEnd();
      }
      if (nseg < 0 ) {

         // Close the side holes
         Float_t oz[3]   = {0.0,0.0,1.0};
         Float_t base[3] = {0.0,0.0,0.0};

         // First hole
         nextv = vertex;
         base[0] = *(nextv+pt3);
         base[1] = *(nextv+pt3+1);
         if (fRootLight) {
            LightIndex(2>>1);
         } else {
            TMath::NormCross(base,oz,normal);
            glNormal3fv(normal);
         }

         glBegin(GL_QUAD_STRIP);
         {
            for (i=0;i<nstacks;i++) {
               glVertex3fv(nextv);

               nextv += pt3;
               glVertex3fv(nextv);

               nextv += pt3;
            }
         }
         glEnd();

         // Second one
         nextv = vertex + 3*(ndiv-1);
         base[0] = *(nextv+pt3);
         base[1] = *(nextv+pt3+1);

         if (fRootLight) {
            LightIndex(ndiv>>1);
         } else {
            TMath::NormCross(oz,base,normal);
            glNormal3fv(normal);
         }

         glBegin(GL_QUAD_STRIP);
         {
            for (i=0;i<nstacks;i++) {
               glVertex3fv(nextv+pt3);
               glVertex3fv(nextv);
               nextv += 6*ndiv;
            }
         }
         glEnd();
      }
      if (fRootLight)
         LightIndex(0);  //Reset the original color
   }
#undef vert
}


//______________________________________________________________________________
void TGLKernel::SetLineAttr(Color_t color, Int_t width)
{
   // Set GL line attributes.

   Color_t c = color;

   if (GetRootLight()) {
      c = ((c % 8) - 1) * 4;
      if (c<0) c = 0;
   }

   SetGLColorIndex(c);
   SetGLLineWidth((Float_t)width);
}


//______________________________________________________________________________
void TGLKernel::UpdateMatrix(Double_t *translate, Double_t *rotate, Bool_t isreflection)
{
   // Update matrix.

   if (translate) TranslateGL(translate);

   if (rotate) {
      if (isreflection)
         FrontGLFace(kCW);  // kCW stands for the CLOCKWISE
      else
         FrontGLFace(kCCW); // kCCW stands for the COUNTERCLOCKWISE

      MultGLMatrix(rotate);
   }
}


//______________________________________________________________________________
void TGLKernel::ClearGLDepth(Float_t val)
{
   // Clear GL depth buffer.

   glClearDepth(val);
}


//______________________________________________________________________________
void TGLKernel::MatrixModeGL(EG3D2GLmode mode)
{
   // Set GL matrix mode.

   glMatrixMode(GLCommand[mode]);
   glLoadIdentity();
}


//______________________________________________________________________________
void TGLKernel::FrustumGL( Double_t xmin, Double_t xmax, Double_t ymin,
                                 Double_t ymax, Double_t znear, Double_t zfar)
{
   // GL Frustrum.

   glFrustum(xmin, xmax, ymin, ymax, znear, zfar);
}


//______________________________________________________________________________
void TGLKernel::GLLight(EG3D2GLmode name, EG3D2GLmode prop, const Float_t * lig_mat)
{
   // GL light.

   glLightfv(GLCommand[name], GLCommand[prop], lig_mat);
}


//______________________________________________________________________________
void TGLKernel::LightModel(EG3D2GLmode name, const Float_t * lig_mat)
{
   // Set GL light model.

   glLightModelfv(GLCommand[name], lig_mat);
}


//______________________________________________________________________________
void TGLKernel::LightModel(EG3D2GLmode name, Int_t prop)
{
   // Set GL light model.

   glLightModeli(GLCommand[name], prop);
}


//______________________________________________________________________________
void TGLKernel::CullFaceGL(EG3D2GLmode face)
{
   // Cull GL faces.

   glCullFace(GLCommand[face]);
}


//______________________________________________________________________________
void TGLKernel::ViewportGL(Int_t x, Int_t y, Int_t w, Int_t h)
{
   // Define GL viewport.

   glViewport(x, y, w, h);
}


//______________________________________________________________________________
void TGLKernel::MaterialGL(EG3D2GLmode face, const Float_t * mat_prop)
{
   // Define GL material.

   glMaterialfv(GLCommand[face], GL_SPECULAR, mat_prop);
   glMaterialfv(GLCommand[face], GL_DIFFUSE, mat_prop);
}


//______________________________________________________________________________
void TGLKernel::MaterialGL(EG3D2GLmode face, Float_t mat_prop)
{
   // Define GL material.

   glMaterialf(GLCommand[face], GL_SHININESS, mat_prop);
}


//______________________________________________________________________________
void TGLKernel::BeginGL(EG3D2GLmode mode)
{
   // Begin GL.

   glBegin(GLCommand[mode]);
}


//______________________________________________________________________________
void TGLKernel::EndGL()
{
   // End with GL.

   glEnd();
}


//______________________________________________________________________________
void TGLKernel::SetGLNormal(const Double_t *normal)
{
   // Set GL normal.

   glNormal3dv(normal);
}


//______________________________________________________________________________
void TGLKernel::NewMVGL()
{
   // New MVGL.

   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}


//______________________________________________________________________________
void TGLKernel::NewPRGL()
{
   // New PRGL.

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
}


//______________________________________________________________________________
void TGLKernel::PaintPolyMarker(const Double_t * vertices, Style_t marker_style, UInt_t size)
{
   // Paint Polymarker.

   if(!fQuad && (fQuad = gluNewQuadric()))
   {
      gluQuadricOrientation(fQuad, (GLenum)GLU_OUTSIDE);
      gluQuadricDrawStyle(fQuad,   (GLenum)GLU_FILL);
      gluQuadricNormals(fQuad,     (GLenum)GLU_FLAT);
   }

   Int_t stacks = 6, slices = 6;
   Float_t point_size = 6.f;
   Double_t top_radius = 5.;

   switch (marker_style) {
   case 27:
      stacks = 2, slices = 4;
   case 4:case 8:case 20:case 24:
      if (fQuad) { //VC6.0 broken for scope
         for (UInt_t i = 0; i < size; i += 3) {
            glPushMatrix();
            glTranslated(vertices[i], vertices[i + 1], vertices[i + 2]);
            gluSphere(fQuad, 5., slices, stacks);
            glPopMatrix();
         }
      }
      break;
   case 22:case 26:
      top_radius = 0.;
   case 21:case 25:
      if (fQuad) { //VC6.0 broken for scope
         for (UInt_t i = 0; i < size; i += 3) {
            glPushMatrix();
            glTranslated(vertices[i], vertices[i + 1], vertices[i + 2]);
            gluCylinder(fQuad, 5., top_radius, 5., 4, 1);
            glPopMatrix();
         }
      }
      break;
   case 23:
      if (fQuad) {//VC6.0 broken for scope
         for (UInt_t i = 0; i < size; i += 3) {
            glPushMatrix();
            glTranslated(vertices[i], vertices[i + 1], vertices[i + 2]);
            glRotated(180, 1., 0., 0.);
            gluCylinder(fQuad, 5., 0., 5., 4, 1);
            glPopMatrix();
         }
      }
      break;
   case 3: case 2: case 5:
      DrawStars(vertices, marker_style, size);
      break;
   case 1: case 9: case 10: case 11: default:{
      glBegin(GL_POINTS);
      for (UInt_t i = 0; i < size; i += 3)
         glVertex3dv(vertices + i);
      glEnd();
   }
   break;
   case 6:
      point_size = 3.f;
   case 7:
      glPointSize(point_size);
      glBegin(GL_POINTS);
      for (UInt_t i = 0; i < size; i += 3)
         glVertex3dv(vertices + i);
      glEnd();
      glPointSize(1.f);
   }
}


//______________________________________________________________________________
void TGLKernel::DrawStars(const Double_t * vert, Style_t style, UInt_t size)
{
   // Draw stars.

   for (Int_t i = 0; i < (Int_t)size; i += 3) {
      Double_t x = vert[i], y = vert[i + 1], z = vert[i + 2];
      glBegin(GL_LINES);
      if (style == 2 || style == 3) {
         glVertex3d(x - 2., y, z);
         glVertex3d(x + 2., y, z);
         glVertex3d(x, y, z - 2.);
         glVertex3d(x, y, z + 2.);
         glVertex3d(x, y - 2., z);
         glVertex3d(x, y + 2., z);
      }
      if(style != 2) {
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
void TGLKernel::DrawSelectionBox( Double_t xmin, Double_t xmax,
                                  Double_t ymin, Double_t ymax,
                                  Double_t zmin, Double_t zmax)
{
   // Draw selection box.

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
void TGLKernel::EnterSelectionMode(UInt_t * buff, Int_t size, Event_t * event, Int_t * viewport)
{
   // Enter selection mode.

   glGetIntegerv(GL_VIEWPORT, viewport);
   glSelectBuffer(size, buff);
   glRenderMode(GL_SELECT);
   glInitNames();
   glPushName(0);
   glMatrixMode(GL_PROJECTION);
   glPushMatrix();
   glLoadIdentity();
   gluPickMatrix(GLdouble(event->fX), GLdouble(viewport[3] - event->fY), 1., 1., viewport);
}


//______________________________________________________________________________
Int_t TGLKernel::ExitSelectionMode()
{
   // Exit selection mode.

   glMatrixMode(GL_PROJECTION);
   glPopMatrix();
   return glRenderMode(GL_RENDER);
}


//______________________________________________________________________________
void TGLKernel::GLLoadName(UInt_t name)
{
   // Load name.

   glLoadName(name);
}


//______________________________________________________________________________
void TGLKernel::DrawFaceSet(const Double_t * pnts, const Int_t * pols, const Double_t * normals,
                            const Float_t * mat, UInt_t size)
{
   // Draw face set.

#if defined(R__WIN32)
   typedef void (CALLBACK * funcptr)();
#elif defined(R__AIXGCC)
   typedef void (*funcptr)(...);
#else
   typedef void (*funcptr)();
#endif
   glMaterialfv(GL_FRONT, GL_SPECULAR, mat);
   glMaterialfv(GL_FRONT, GL_DIFFUSE, mat);
   glMaterialf(GL_FRONT, GL_SHININESS, 60.f);

   if (!fTessObj)
      fTessObj = gluNewTess();

   for (UInt_t i = 0, j = 0; i < size; ++i) {
      Int_t npoints = pols[j++];
      if (fTessObj && npoints > 4) {
         gluTessCallback(fTessObj, (GLenum)GLU_BEGIN, (funcptr)glBegin);
         gluTessCallback(fTessObj, (GLenum)GLU_END, (funcptr)glEnd);
         gluTessCallback(fTessObj, (GLenum)GLU_VERTEX, (funcptr)glVertex3dv);

         gluBeginPolygon(fTessObj);
         gluNextContour(fTessObj, (GLenum)GLU_UNKNOWN);
         glNormal3dv(normals + i * 3);

         for (Int_t k = 0; k < npoints; ++k, ++j)
            gluTessVertex(fTessObj, (Double_t *)pnts + pols[j] * 3, (Double_t *)pnts + pols[j] * 3);

         gluEndPolygon(fTessObj);
      } else {
         glBegin(GL_POLYGON);
         glNormal3dv(normals + i * 3);

         for (Int_t k = 0; k < npoints; ++k, ++j)
            glVertex3dv(pnts + pols[j] * 3);
         glEnd();
      }
   }
}


//______________________________________________________________________________
void TGLKernel::DrawViewer(TGLViewer *viewer)
{
   // Draw viewer.

   if (gDebug>3) {
      Info("TGLKernel::DrawViewer", "got request to draw viewer = %d", viewer);
   }
   viewer->DoDraw();
}


//______________________________________________________________________________
Bool_t TGLKernel::SelectViewer(TGLViewer *viewer, const TGLRect * rect)
{
   // Select viewer.
   return viewer->DoSelect(*rect);
}


//______________________________________________________________________________
Bool_t TGLKernel::SelectManip(TGLManip *manip, const TGLCamera * camera, const TGLRect * rect, const TGLBoundingBox * sceneBox)
{
   // Slect manipulator.

   return manip->Select(*camera, *rect, *sceneBox);
}


//______________________________________________________________________________
static GLUquadric *GetQuadric1()
{
   // GLU quadric.

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


//______________________________________________________________________________
void TGLKernel::DrawSphere(const Float_t *rgba)
{
   // Draw a sphere.

   const Float_t whiteColor[] = {1.f, 1.f, 1.f, 1.f};
   const Float_t nullColor[] = {0.f, 0.f, 0.f, 1.f};
   if (rgba[16] < 0.f) {
      glLightfv(GL_LIGHT0, GL_DIFFUSE, rgba);

      glLightfv(GL_LIGHT0, GL_AMBIENT, rgba + 4);
      glLightfv(GL_LIGHT0, GL_SPECULAR, rgba + 8);
      glMaterialfv(GL_FRONT, GL_DIFFUSE, whiteColor);
      glMaterialfv(GL_FRONT, GL_AMBIENT, nullColor);
      glMaterialfv(GL_FRONT, GL_SPECULAR, whiteColor);
      glMaterialfv(GL_FRONT, GL_EMISSION, nullColor);
      glMaterialf(GL_FRONT, GL_SHININESS, 60.f);
   } else {
      glLightfv(GL_LIGHT0, GL_DIFFUSE, whiteColor);
      glLightfv(GL_LIGHT0, GL_AMBIENT, nullColor);
      glLightfv(GL_LIGHT0, GL_SPECULAR, whiteColor);
      glMaterialfv(GL_FRONT, GL_DIFFUSE, rgba);
      glMaterialfv(GL_FRONT, GL_AMBIENT, rgba + 4);
      glMaterialfv(GL_FRONT, GL_SPECULAR, rgba + 8);
      glMaterialfv(GL_FRONT, GL_EMISSION, rgba + 12);
      glMaterialf(GL_FRONT, GL_SHININESS, rgba[16]);
   }

   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   GLUquadric * quad = GetQuadric1();
   if (quad) {
      glRotated(-90., 1., 0., 0.);
      gluSphere(quad, 1., 100, 100);
   }
   glDisable(GL_BLEND);
}


//______________________________________________________________________________
void TGLKernel::CaptureViewer(TGLViewer * viewer, Int_t format, const char * filePath)
{
   // Capture viewer.

   TGLOutput::Capture(*viewer, TGLOutput::EFormat(format), filePath);
}
