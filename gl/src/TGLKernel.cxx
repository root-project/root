// @(#)root/gl:$Name:  $:$Id: TGLKernel.cxx,v 1.6 2002/02/23 10:15:22 brun Exp $
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
// The TGLKernel class defines the interface to OpenGL.                 //
// All interactions with OpenGL should go via this class.               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGLKernel.h"
#include "TMath.h"
#include "TVirtualX.h"
#include "TPadOpenGLView.h"
#include "TView.h"
#include "TGeometry.h"
#include "TROOT.h"
#include "TColor.h"
#include "TError.h"
#include "TPoints3DABC.h"

#ifndef   ColorOffset
#ifdef WIN32
# define  ColorOffset 0
#else
# define  ColorOffset 0
#endif
#endif

GLenum GLCommand[] = { GLConstants(GL_)  };


//______________________________________________________________________________
TGLKernel::TGLKernel()
{
   // GLKernel ctor. First delete TVirtualGL instance then reassign
   // global to this.

   delete gVirtualGL;
   gVirtualGL = this;
}

//______________________________________________________________________________
TGLKernel::~TGLKernel()
{
   gVirtualGL = new TVirtualGL;
}

//______________________________________________________________________________
void TGLKernel::ClearColor(Int_t color)
{
    if (gVirtualGL->GetTrueColorMode())
    {
        Float_t red;
        Float_t green;
        Float_t blue;
        Float_t alpha = 0.0;

        TColor *c = gROOT->GetColor(color);
        if (!c) c= gROOT->GetColor(1);
        c->GetRGB(red,green,blue);
        TGLKernel::ClearGLColor(red, green, blue, alpha);
    }
    else
        glClearIndex(color+ColorOffset);
}

//______________________________________________________________________________
void TGLKernel::ClearGLColor(Float_t red, Float_t green, Float_t blue, Float_t alpha)
{
    glClearColor(red, green, blue, alpha);
}

//______________________________________________________________________________
void TGLKernel::ClearGLColor(Float_t *colors)
{
    GLclampf red   = colors[0];
    GLclampf green = colors[1];
    GLclampf blue  = colors[2];
    GLclampf alpha = colors[3];
    glClearColor(red, green, blue, alpha);

}

//______________________________________________________________________________
void TGLKernel::ClearGL(UInt_t stereo) {
#ifdef STEREO_GL
   if (stereo) {
      if (Int_t(stereo) < 0)
         glDrawBuffer(GL_BACK_LEFT);
      else
         glDrawBuffer(GL_BACK_RIGHT);
   }
#endif
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

//______________________________________________________________________________
TPadView3D *TGLKernel::CreatePadGLView(TVirtualPad *c)
{ return new TPadOpenGLView(c); }

//______________________________________________________________________________
void TGLKernel::DisableGL(EG3D2GLmode mode){glDisable(GLCommand[mode]);}

//______________________________________________________________________________
void TGLKernel::EnableGL(EG3D2GLmode mode){glEnable(GLCommand[mode]);}

//______________________________________________________________________________
void TGLKernel::FlushGL(){ glFlush();}

//______________________________________________________________________________
void TGLKernel::FrontGLFace(EG3D2GLmode faceflag){
    fFaceFlag = faceflag;
    glFrontFace(GLCommand[faceflag]);
}
//______________________________________________________________________________
void TGLKernel::NewGLList(UInt_t ilist, EG3D2GLmode mode)
{  glNewList(ilist,GLCommand[mode]); }
//______________________________________________________________________________
void TGLKernel::NewGLModelView(Int_t ilist)
{
    glNewList(ilist,GL_COMPILE);
//--    glMatrixMode(GL_MODELVIEW);
//--    glLoadIdentity();
}

//______________________________________________________________________________
void TGLKernel::GetGL(EG3D2GLmode mode, void  *params, EGLTypes type)
{
    switch (type) {
    case kBoolType:
        TGLKernel::GetGL(mode,(UChar_t *)params);
        break;
    case kDoubleType:
        TGLKernel::GetGL(mode,(Double_t *)params);
        break;
    case kFloatType:
        TGLKernel::GetGL(mode,(Float_t *)params);
        break;
    case kIntegerType:
        TGLKernel::GetGL(mode,(Int_t *)params);
        break;
    default:
        break;
    };
}
//______________________________________________________________________________
void TGLKernel::GetGL(EG3D2GLmode mode, UChar_t *params)
{glGetBooleanv(GLCommand[mode],params);}

//______________________________________________________________________________
void TGLKernel::GetGL(EG3D2GLmode mode, Double_t *params)
{glGetDoublev(GLCommand[mode],params);}

//______________________________________________________________________________
void TGLKernel::GetGL(EG3D2GLmode mode, Float_t *params)
{glGetFloatv(GLCommand[mode],params);}

//______________________________________________________________________________
void TGLKernel::GetGL(EG3D2GLmode mode, Int_t *params)
{glGetIntegerv(GLCommand[mode],params);}


//______________________________________________________________________________
Int_t TGLKernel::GetGLError()
{ return glGetError(); }

//______________________________________________________________________________
void TGLKernel::EndGLList() { glEndList();}

//______________________________________________________________________________
void TGLKernel::BeginGLCmd(EG3D2GLmode mode)
{ glBegin(GLCommand[mode]); }

//______________________________________________________________________________
void TGLKernel::EndGLCmd(){ glEnd(); }

//______________________________________________________________________________
void TGLKernel::PushGLMatrix() {glPushMatrix();}

//______________________________________________________________________________
void TGLKernel::PopGLMatrix() {glPopMatrix();}

//______________________________________________________________________________
void TGLKernel::RotateGL(Double_t *direction, Int_t mode)
{
    if (mode)
    {
        Double_t angle = direction[0];
        Double_t x     = direction[1];
        Double_t y     = direction[2];
        Double_t z     = direction[3];
        TGLKernel::RotateGL(angle,x,y,z);
    }
    else
    {
//*-* Double_t Theta   - polar angle for the axis x`
//*-* Double_t Phi     - azimutal angle for the axis x`
//*-* Double_t Psi     - azimutal angle for the axis y`
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        Double_t Theta = direction[0];
        Double_t Phi   = direction[1];
        Double_t Psi   = direction[2];
        TGLKernel::RotateGL(Theta,Phi,Psi);
    }
}
//______________________________________________________________________________
void TGLKernel::RotateGL(Double_t angle, Double_t x,Double_t y,Double_t z)
{
//*-* The RotateGL function computes a matrix that performs a counterclockwise
//*-* rotation of angle degrees about the vector from the origin through
//*-* the point (x, y, z).
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    glRotated(angle,x,y,z);
}

//______________________________________________________________________________
void TGLKernel::RotateGL(Double_t Theta, Double_t Phi, Double_t Psi)
{
//*-* Double_t Theta   - polar angle for the axis x`
//*-* Double_t Phi     - azimutal angle for the axis x`
//*-* Double_t Psi     - azimutal angle for the axis y`
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    if (TMath::Abs(90-Theta) > 0.00001)
                                        TGLKernel::RotateGL(Theta-90,1,0,0);
    if (TMath::Abs(90-Psi)   > 0.00001)
                                        TGLKernel::RotateGL(90-Psi,  0,1,0);
    if (TMath::Abs(Phi)      > 0.00001)
                                        TGLKernel::RotateGL(Phi,     0,0,1);
}
//______________________________________________________________________________
void TGLKernel::TranslateGL(Double_t x,Double_t y,Double_t z){glTranslated(x,y,z);}

//______________________________________________________________________________
void TGLKernel::TranslateGL(Double_t *xyz) {TGLKernel::TranslateGL(xyz[0], xyz[1],xyz[2]);}

//______________________________________________________________________________
void TGLKernel::MultGLMatrix(Double_t *mat){ glMultMatrixd(mat); }

//______________________________________________________________________________
void TGLKernel::SetGLColor(Float_t *rgb){ glColor3fv(rgb); }

//______________________________________________________________________________
void TGLKernel::SetGLVertex(Float_t *vertex) { glVertex3fv(vertex);}

//______________________________________________________________________________
void TGLKernel::SetGLColorIndex(Int_t color)
{
    if (color != -1) fColorIndx = color;
    // We have to change BLACK color to GRAY to make it visible
    if (fColorIndx == 1 && gVirtualGL->GetTrueColorMode()) fColorIndx=19;
    SetCurrentColor(fColorIndx);
}

//______________________________________________________________________________
void TGLKernel::SetCurrentColor(Int_t color)
{
//    if (fRootLight)
    if (gVirtualGL->GetTrueColorMode())
    {
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
        TGLKernel::SetGLColor(rgb);
    }
    else
        glIndexi(color+ColorOffset);
}

//______________________________________________________________________________
void TGLKernel::SetGLPointSize(Float_t size){
//*-*
//*-* The SetGLPointSize function specifies the diameter of rasterized points.
//*-*
        glPointSize(size);
}

//______________________________________________________________________________
void TGLKernel::SetGLLineWidth(Float_t width){
//*-*
//*-* The SetGLLineWidth function specifies the width of rasterized lines.
//*-*
        glLineWidth(width);
}

//______________________________________________________________________________
void TGLKernel::SetRootLight(Bool_t flag)
{
    if (flag != fRootLight) {
        fRootLight = flag;
        if (fRootLight)
        {
            glDisable(GL_LIGHT0);
            glDisable(GL_LIGHTING);
            glDisable(GL_COLOR_MATERIAL);
        }
        else
        {
            glEnable(GL_LIGHT0);
            glEnable(GL_LIGHTING);
            glEnable(GL_COLOR_MATERIAL);
#ifdef STEREO_GL
            glEnable(GL_STEREO);
#endif
//            glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,0);
//          glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE);

//          glColorMaterial(GL_FRONT,GL_AMBIENT_AND_DIFFUSE);
      }
   }
}
//______________________________________________________________________________
void TGLKernel::DeleteGLLists(Int_t ilist, Int_t range){glDeleteLists(ilist,range);}

//______________________________________________________________________________
Int_t TGLKernel::CreateGLLists(Int_t range) { return glGenLists(range);}

//______________________________________________________________________________
void TGLKernel::RunGLList(Int_t list){ glCallList(list); }

//______________________________________________________________________________
void TGLKernel::NewProjectionView(Double_t min[],Double_t max[],Bool_t perspective)
{
#if 0
    if (kTRUE)
    {
       glEnable(GL_LIGHT0);
       glEnable(GL_LIGHTING);
       glEnable(GL_COLOR_MATERIAL);
       glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,0);
       glColorMaterial(GL_FRONT,GL_AMBIENT_AND_DIFFUSE);

    }
    else
        SetRootLight(kTRUE);
#endif
//       glMaterial(GL_FRONT,
//       glLightModel();


    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    Double_t dnear = TMath::Abs(max[0]-min[0]);
    Double_t dfar = 3*(dnear + TMath::Abs(max[2]-min[2]));

    if (perspective)
        glFrustum(min[0],max[0],min[1],max[1],dnear,dfar);
    else
        glOrtho  (min[0],max[0],min[1],max[1],dnear,dfar);

//       RotateGL(-(-90+view->GetLatitude()),-(90+view->GetLongitude()),view->GetPsi()+90);

    glCullFace(GL_BACK);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

#if 0
    if (!fRootLight && fTrueColorMode)
    {
        glPushMatrix();
        Float_t rgb[] = {0.8,0.8,0.8};
        TGLKernel::SetGLColor(rgb);
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
    glPolygonMode(GLCommand[face], GLCommand[mode]);
}

//______________________________________________________________________________
void TGLKernel::SetStack(Double_t *matrix)
{
//
// SetStack(Double_t *matrix) method replaces the matrix on the top of the stack
//                            of the matrix with a new one supplied or with the
//                            indentity.
// Parameters:
// ----------
// Double_t *matrix - the pointer to a new matrix to updtae stack
//                      = 0 the indentity matrix must be applied
//
        glPopMatrix();
        if (!matrix) glLoadIdentity();
        else
                glLoadMatrixd(matrix);
    glPushMatrix();
 }

//______________________________________________________________________________
void TGLKernel::ShadeGLModel(EG3D2GLmode model)
{
     glShadeModel(GLCommand[model]);
}

//______________________________________________________________________________
void TGLKernel::AddRotation(Double_t *rotmatrix, Double_t *angles)
{
   GLint mode;
   glPushMatrix();
   glGetIntegerv(GL_MATRIX_MODE,&mode);

    glLoadIdentity();
    if (TMath::Abs(angles[0]) > 0.00001)
       TGLKernel::RotateGL(angles[0],1,0,0);
    if (TMath::Abs(angles[1]) > 0.00001)
       TGLKernel::RotateGL(angles[1],0,1,0);
    if (TMath::Abs(angles[2]) > 0.00001)
       TGLKernel::RotateGL(angles[2],0,0,1);
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
       Error("TGLKernel::AddRotation", "unknown matrix !");
       return;
    }

    glPopMatrix();
}

//______________________________________________________________________________
void TGLKernel::NewModelView(Double_t *angles,Double_t *delta )
{

    TGLKernel::RotateGL(-(-90+angles[0]),
                        -( 90+angles[1]),
                           90+angles[2]);
//*-*  Move model to the center of the "view" box ??????????
    TGLKernel::TranslateGL(delta[0],delta[1],delta[2]);
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

//    if (fRootLight)
//           LightIndex(0);  //reset the original color

    GLenum mode = GL_POINTS;
    Int_t pass = 0;
    if (option && strchr(option,'P')) { pass++;}
    if (option && strchr(option,'L')) { mode = GL_LINE_STRIP; pass++;}
    while (pass >= 0) {
      pass--;
      glBegin(mode);
         for (int i=0; i < n; i++)
             glVertex3f(GLfloat(points->GetX(i))
                       ,GLfloat(points->GetY(i))
                       ,GLfloat(points->GetZ(i)));
      glEnd();
      mode=GL_POINTS;
    }
}

//______________________________________________________________________________
void TGLKernel::PaintGLPoints(Int_t n, Float_t *p, Option_t *)
{
    if (n <= 0 || p == 0) return;
    GLfloat *point = p;

//    if (fRootLight)
//           LightIndex(0);  //reset the original color

    glBegin(GL_POINTS);
      for (int i=0; i < n; i++, point+=3)   glVertex3fv(point);
    glEnd();
}

//______________________________________________________________________________
void TGLKernel::PaintPolyLine(Int_t n, Float_t *p, Option_t *)
{
    if (n <= 0 || p == 0) return;
    GLfloat *line = p;

//    if (fRootLight)
//           LightIndex(0);  //reset the original color

    glBegin(GL_LINE_STRIP);
      for (int i=0; i < n; i++, line+=3)   glVertex3fv(line);
    glEnd();
}

//______________________________________________________________________________
void TGLKernel::PaintBrik(Float_t vertex[24])
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*Paint BRIK via OpenGL *-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                            =====================

#define vert(i) &vertex[3*(i)]
    Int_t i;
    Float_t normal[3];

    if (vertex) {

//*-* The counterclockwise polygons are taken to be front-facing by default

        if (fRootLight) LightIndex(0);
        glBegin(GL_QUADS);
       //*-*  "Top"  of TBRIK
        if (!fRootLight)
            glNormal3fv(TMath::Normal2Plane(vert(7),vert(4),vert(6),normal));
        for (i=4;i<8;i++)
           glVertex3fv(vert(i));

//*-*   "Bottom"  of TBRIK
        if (!fRootLight)
            glNormal3fv(TMath::Normal2Plane(vert(0),vert(3),vert(1),normal));
        for (i=3;i>-1;i--) {
           glVertex3fv(vert(i));
        }

//*-*   "Walls"  of TBRIK
        for (i=0;i<3;i++)
        {
            if (fRootLight)
                LightIndex(i+1);
            else
                glNormal3fv(TMath::Normal2Plane(vert(i),vert(i+1),vert(i+4),normal));

            glVertex3fv(vert(i));
            glVertex3fv(vert(i+1));
            glVertex3fv(vert(i+5));
            glVertex3fv(vert(i+4));
        }
//*-*  The last "wall" to close the brik

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
// Paint Xtru shape via OpenGL

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
     p     = vertex+3*(nz-1)*nxy;
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
           }
           else
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
//*-* The calculation  a normal vector to the conic surface
//*-*
//*-*  Principial chord:
//*-*  ----------------
//*-*     p1 = (a1,b1,c1);
//*-*     p2 = (a2,b2,c2);
//*-*
//*-*     v      = (a2-a1,b2-b1,c2-c1);
//*-*     normal = (a2-0,b2-0,c2-Z);
//*-*
//*-*     v*normal = 0;
//*-*
//*-*          a2*(a2-a1) + b2*(b2-b1) + c2*(c2-c1)
//*-*     Z = -------------------------------------
//*-*                         c2-c1
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  Float_t Z = 0;
  Int_t i;
  for(i=0;i<3;i++)
  {
      normal[i] = p2[i];
      Z += p2[i]*(p2[i]-p1[i]);
  }
  Z /= p2[2]-p1[2];

  normal[2] -= Z;

  TMath::Normalize(normal);

  return normal;

}

//______________________________________________________________________________
void TGLKernel::PaintCone(Float_t *vertex,Int_t nseg,Int_t nstacks)
{
    //*-*  vertex  - array of the 3d coordinates
    //*-*  nseg    - number of segments (precision)
    //*-*            < 0 means the shape is segmented
    //*-*            > 0 means the shape is closed
    //*-*  nstacks -  number of stack sections
    //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    //*-*  Vertex[2*(3*ndiv*nstacks)]
    //*-*   i - the division number
    //*-*   j - the stack number
    //*-*   k = 0 internal points
    //*-*       1 external points
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
//*-* The counterclockwise polygons are taken to be front-facing by deafult

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*  "Top"  of TPCON
        if (fRootLight)
            LightIndex(0);
        else
            glNormal3fv(onenorm);

//*-*  Check internal radius
        nextv = vert(0,nstacks-1,0);
        if (*(nextv+1) == 0.0 && *nextv == 0.0) // The inner radius is ZERO
        {
//*-*  Draw the "Triangle fan"
            glBegin(GL_TRIANGLE_FAN);
            glVertex3f(0.0,0.0,*(nextv+2)); //*-* Set the center of the fan
            nextv += pt3;
            for (i=0;i<ndiv;i++)
            {
                glVertex3fv(nextv);
                nextv += 3;
            }
            if (nseg > 0)
                glVertex3fv(nextv-pt3);
        }
        else
        {
//*-*  Draws the series of the quadrilaterals
            glBegin(GL_QUAD_STRIP);
            exnextv = nextv + pt3;
            for (i=0;i<ndiv;i++)
            {
                glVertex3fv(nextv);
                glVertex3fv(exnextv);
                nextv += 3;
                exnextv += 3;
            }
            if (nseg > 0 )
            {
                glVertex3fv(nextv  - pt3);
                glVertex3fv(exnextv- pt3);
            }

        }
        glEnd();
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*   "Bottom"  of TPCON
        if (!fRootLight)
            glNormal3fv(backone);

        nextv = vert(0,0,0);
        if (*(nextv+1) == 0.0 && *nextv == 0.0 )
        {
//*-*  Draw the "Triangle fan"
            glBegin(GL_TRIANGLE_FAN);
            glVertex3f(0.0,0.0,*(nextv+2)); //*-* Set the center of the fan
            nextv += pt6;
            for (i=0;i<ndiv;i++)
            {
                nextv -= 3;
                glVertex3fv(nextv);
            }
            if (nseg > 0)
                glVertex3fv(nextv+pt3-3);
        }
        else
        {
            //*-*  Draws the series of the quadrilaterals
            glBegin(GL_QUAD_STRIP);
            exnextv = nextv + pt3;
            for (i=0;i<ndiv;i++)
            {
                glVertex3fv(exnextv);
                glVertex3fv(nextv);
                nextv += 3;
                exnextv +=3;
            }
            if (nseg > 0)
            {
                glVertex3fv(exnextv- pt3);
                glVertex3fv(nextv  - pt3);
            }
        }
        glEnd();


//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*   "Walls"  of TPCON
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*    Create the external walls
        nextv = vertex + pt3;
        for (i=0;i<nstacks-1;i++)
        {
            glBegin(GL_QUAD_STRIP);
            {
//                nextv = vert(0,i,1);
                exnextv = nextv+pt6;
                for(j=0;j<ndiv;j++)
                {
                    if (fRootLight)
                        LightIndex(j>>1);
                    else
                        glNormal3fv(Normal2Line(exnextv,nextv,normal));

                    glVertex3fv(exnextv);
                    glVertex3fv(nextv);
                    nextv += 3;
                    exnextv += 3;
                }
                if (nseg > 0)
                {
//*-* To "close" shape we have to add on extra "wall"
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
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*    Create the internal walls
        exnextv = vertex;
        for (i=0;i<nstacks-1;i++)
        {
          if (*(exnextv+1) == 0.0 && *exnextv == 0.0) continue; // No internal wall at all

          glBegin(GL_QUAD_STRIP);
          {
              nextv = exnextv+pt6;
              for(j=0;j<ndiv;j++)
              {
                  if (fRootLight)
                      LightIndex(j>>1);
                  else
                      glNormal3fv(Invert(Normal2Line(nextv,exnextv,normal)));

                  glVertex3fv(exnextv);
                  glVertex3fv(nextv);
                  nextv += 3;
                  exnextv += 3;
              }
              if (nseg > 0)
              {
//*-* To "close" shape we have to add on extra "wall"
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
        if (nseg < 0 )
        {

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*  Close the side holes
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
            Float_t oz[3]   = {0.0,0.0,1.0};
            Float_t base[3] = {0.0,0.0,0.0};

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*   First hole
            nextv = vertex;
            base[0] = *(nextv+pt3);
            base[1] = *(nextv+pt3+1);
            if (fRootLight)
                      LightIndex(2>>1);
            else {
                TMath::NormCross(base,oz,normal);
                glNormal3fv(normal);
            }

            glBegin(GL_QUAD_STRIP);
            {
                for (i=0;i<nstacks;i++)
                {
                    glVertex3fv(nextv);

                    nextv += pt3;
                    glVertex3fv(nextv);

                    nextv += pt3;
                }
            }
            glEnd();

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*   Second one
            nextv = vertex + 3*(ndiv-1);
            base[0] = *(nextv+pt3);
            base[1] = *(nextv+pt3+1);

            if (fRootLight)
                LightIndex(ndiv>>1);
            else {
                TMath::NormCross(oz,base,normal);
                glNormal3fv(normal);
            }

            glBegin(GL_QUAD_STRIP);
            {
                for (i=0;i<nstacks;i++)
                {
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
    Color_t c = color;
    if (TGLKernel::GetRootLight())
    {
        c = ((c % 8) - 1) * 4;
        if (c<0) c = 0;
    }

    TGLKernel::SetGLColorIndex(c);
    TGLKernel::SetGLLineWidth((Float_t)width);
}

//______________________________________________________________________________
void TGLKernel::UpdateMatrix(Double_t *translate, Double_t *rotate, Bool_t isreflection)
{
    if (translate) TGLKernel::TranslateGL(translate);

    if (rotate)
    {
        if (isreflection)
            TGLKernel::FrontGLFace(kCW);  // kCW stands for the CLOCKWISE
        else
            TGLKernel::FrontGLFace(kCCW); // kCCW stands for the COUNTERCLOCKWISE

        TGLKernel::MultGLMatrix(rotate);
    }
}

#if 0
The gluProject() function maps object coordinates to window coordinates.

  glGetDoublev(GL_MODELVIEW_MATRIX,matrix);
  glGetDoublev(GL_PROJECTION_MATRIX,matrix);
  glGetIntegerv(GL_VIEWPORT,viewport);

The gluUnProject() function maps window coordinates to object coordinates.

#endif
