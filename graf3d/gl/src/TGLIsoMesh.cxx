// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  28/07/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLIncludes.h"

#include "TGLPlotPainter.h"
#include "TGLIsoMesh.h"

namespace Rgl {

//Functions for TGLTF3/TGLIso/TGL5DPainter.
////////////////////////////////////////////////////////////////////////////////
///Surface with material and lighting.

template<class V>
void DrawMesh(GLenum type, const std::vector<V> &vs, const std::vector<V> &ns,
              const std::vector<UInt_t> &fTS)
{
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);
   glVertexPointer(3, type, 0, &vs[0]);
   glNormalPointer(type, 0, &ns[0]);
   glDrawElements(GL_TRIANGLES, fTS.size(), GL_UNSIGNED_INT, &fTS[0]);
   glDisableClientState(GL_NORMAL_ARRAY);
   glDisableClientState(GL_VERTEX_ARRAY);
}

////////////////////////////////////////////////////////////////////////////////
///Call function-template.

void DrawMesh(const std::vector<Float_t> &vs, const std::vector<Float_t> &ns,
              const std::vector<UInt_t> &ts)
{
   DrawMesh(GL_FLOAT, vs, ns, ts);
}

////////////////////////////////////////////////////////////////////////////////
///Call function-template.

void DrawMesh(const std::vector<Double_t> &vs, const std::vector<Double_t> &ns,
              const std::vector<UInt_t> &ts)
{
   DrawMesh(GL_DOUBLE, vs, ns, ts);
}

////////////////////////////////////////////////////////////////////////////////
///Only vertices, no normal (no lighting and material).

template<class V>
void DrawMesh(GLenum type, const std::vector<V> &vs, const std::vector<UInt_t> &fTS)
{
   glEnableClientState(GL_VERTEX_ARRAY);
   glVertexPointer(3, type, 0, &vs[0]);
   glDrawElements(GL_TRIANGLES, fTS.size(), GL_UNSIGNED_INT, &fTS[0]);
   glDisableClientState(GL_VERTEX_ARRAY);
}

////////////////////////////////////////////////////////////////////////////////
///Call function-template.

void DrawMesh(const std::vector<Float_t> &vs, const std::vector<UInt_t> &ts)
{
   DrawMesh(GL_FLOAT, vs, ts);
}

////////////////////////////////////////////////////////////////////////////////
///Call function-template.

void DrawMesh(const std::vector<Double_t> &vs, const std::vector<UInt_t> &ts)
{
   DrawMesh(GL_DOUBLE, vs, ts);
}

////////////////////////////////////////////////////////////////////////////////
///Mesh with cut.
///Material and lighting are enabled.

template<class V, class GLN, class GLV>
void DrawMesh(GLN normal3, GLV vertex3, const std::vector<V> &vs,
              const std::vector<V> &ns, const std::vector<UInt_t> &fTS,
              const TGLBoxCut &box)
{
   glBegin(GL_TRIANGLES);

   for (UInt_t i = 0, e = fTS.size() / 3; i < e; ++i) {
      const UInt_t * t = &fTS[i * 3];
      if (box.IsInCut(&vs[t[0] * 3]))
         continue;
      if (box.IsInCut(&vs[t[1] * 3]))
         continue;
      if (box.IsInCut(&vs[t[2] * 3]))
         continue;

      normal3(&ns[t[0] * 3]);
      vertex3(&vs[t[0] * 3]);

      normal3(&ns[t[1] * 3]);
      vertex3(&vs[t[1] * 3]);

      normal3(&ns[t[2] * 3]);
      vertex3(&vs[t[2] * 3]);
   }

   glEnd();
}

////////////////////////////////////////////////////////////////////////////////
///Call function-template.

void DrawMesh(const std::vector<Float_t> &vs, const std::vector<Float_t> &ns,
              const std::vector<UInt_t> &ts, const TGLBoxCut &box)
{
   DrawMesh(&glNormal3fv, &glVertex3fv, vs,  ns, ts, box);
}

////////////////////////////////////////////////////////////////////////////////
///Call function-template.

void DrawMesh(const std::vector<Double_t> &vs, const std::vector<Double_t> &ns,
              const std::vector<UInt_t> &ts, const TGLBoxCut &box)
{
   DrawMesh(&glNormal3dv, &glVertex3dv, vs, ns, ts, box);
}

////////////////////////////////////////////////////////////////////////////////
///Mesh with cut.
///No material and lighting.

template<class V, class GLV>
void DrawMesh(GLV vertex3, const std::vector<V> &vs, const std::vector<UInt_t> &fTS,
              const TGLBoxCut &box)
{
   glBegin(GL_TRIANGLES);

   for (UInt_t i = 0, e = fTS.size() / 3; i < e; ++i) {
      const UInt_t * t = &fTS[i * 3];
      if (box.IsInCut(&vs[t[0] * 3]))
         continue;
      if (box.IsInCut(&vs[t[1] * 3]))
         continue;
      if (box.IsInCut(&vs[t[2] * 3]))
         continue;

      vertex3(&vs[t[0] * 3]);
      vertex3(&vs[t[1] * 3]);
      vertex3(&vs[t[2] * 3]);
   }

   glEnd();
}

////////////////////////////////////////////////////////////////////////////////
///Call function-template.

void DrawMesh(const std::vector<Float_t> &vs, const std::vector<UInt_t> &ts, const TGLBoxCut &box)
{
   DrawMesh(&glVertex3fv, vs, ts, box);
}

////////////////////////////////////////////////////////////////////////////////
///Call function-template.

void DrawMesh(const std::vector<Double_t> &vs, const std::vector<UInt_t> &ts, const TGLBoxCut &box)
{
   DrawMesh(&glVertex3dv, vs, ts, box);
}

////////////////////////////////////////////////////////////////////////////////
///NormalToColor generates a color from a given normal

void NormalToColor(Double_t *rfColor, const Double_t *n)
{
   const Double_t x = n[0];
   const Double_t y = n[1];
   const Double_t z = n[2];
   rfColor[0] = (x > 0. ? x : 0.) + (y < 0. ? -0.5 * y : 0.) + (z < 0. ? -0.5 * z : 0.);
   rfColor[1] = (y > 0. ? y : 0.) + (z < 0. ? -0.5 * z : 0.) + (x < 0. ? -0.5 * x : 0.);
   rfColor[2] = (z > 0. ? z : 0.) + (x < 0. ? -0.5 * x : 0.) + (y < 0. ? -0.5 * y : 0.);
}

////////////////////////////////////////////////////////////////////////////////
///Colored mesh with lighting disabled.

void DrawMapleMesh(const std::vector<Double_t> &vs, const std::vector<Double_t> &ns,
                   const std::vector<UInt_t> &fTS)
{
   Double_t color[] = {0., 0., 0., 0.15};

   glBegin(GL_TRIANGLES);

   for (UInt_t i = 0, e = fTS.size() / 3; i < e; ++i) {
      const UInt_t *t = &fTS[i * 3];
      const Double_t * n = &ns[t[0] * 3];
      //
      NormalToColor(color, n);
      glColor4dv(color);
      glVertex3dv(&vs[t[0] * 3]);
      //
      n = &ns[t[1] * 3];
      NormalToColor(color, n);
      glColor4dv(color);
      glVertex3dv(&vs[t[1] * 3]);
      //
      n = &ns[t[2] * 3];
      NormalToColor(color, n);
      glColor4dv(color);
      glVertex3dv(&vs[t[2] * 3]);
   }

   glEnd();
}

////////////////////////////////////////////////////////////////////////////////
///Colored mesh with cut and disabled lighting.

void DrawMapleMesh(const std::vector<Double_t> &vs, const std::vector<Double_t> &ns,
                   const std::vector<UInt_t> &fTS, const TGLBoxCut & box)
{
   Double_t color[] = {0., 0., 0., 0.15};

   glBegin(GL_TRIANGLES);

   for (UInt_t i = 0, e = fTS.size() / 3; i < e; ++i) {
      const UInt_t *t = &fTS[i * 3];
      if (box.IsInCut(&vs[t[0] * 3]))
         continue;
      if (box.IsInCut(&vs[t[1] * 3]))
         continue;
      if (box.IsInCut(&vs[t[2] * 3]))
         continue;
      const Double_t * n = &ns[t[0] * 3];
      //
      NormalToColor(color, n);
      glColor4dv(color);
      glVertex3dv(&vs[t[0] * 3]);
      //
      n = &ns[t[1] * 3];
      NormalToColor(color, n);
      glColor4dv(color);
      glVertex3dv(&vs[t[1] * 3]);
      //
      n = &ns[t[2] * 3];
      NormalToColor(color, n);
      glColor4dv(color);
      glVertex3dv(&vs[t[2] * 3]);
   }

   glEnd();
}

}
