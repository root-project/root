#include "ROOT/TEveGluTess.hxx"

#include <math.h>
#include <stdlib.h>

#include <stdexcept>

// This is a minimal-change import of GLU libtess from:
//   git://anongit.freedesktop.org/git/mesa/glu
//
// Changes to make it build in the horrible one-file-hack way:
// - remove include gl.h from glu.h (and replace with the mini-gl defs below);
// - comment out clashing typedef EdgePair in tess.c;
// - use -Wno-unused-parameter for this cxx file.

namespace ROOT { namespace Experimental { namespace GLU
{

//==============================================================================
// Slurp-in of glu/libtess
//==============================================================================

#include "glu/GL/glu.h"

#include "glu/dict.c"
#include "glu/geom.c"
#undef Swap
#include "glu/memalloc.c"
#include "glu/mesh.c"
// #include "glu/priorityq-heap.c" - included in priorityq.c
#include "glu/priorityq.c"
#include "glu/normal.c" // has to be after priorityq.c (don't ask, may defines be with you)
#include "glu/render.c"
#include "glu/sweep.c"
#include "glu/tess.c"
#include "glu/tessmono.c"


//==============================================================================
// TriangleCollector
//==============================================================================

typedef void (*tessfuncptr_t)();

TriangleCollector::TriangleCollector() :
   fNTriangles(0), fNVertices(0), fV0(-1), fV1(-1), fType(GL_NONE)
{
   fTess = gluNewTess();
   if ( ! fTess) throw std::bad_alloc();

   gluTessCallback(fTess, (GLenum)GLU_TESS_BEGIN_DATA,   (tessfuncptr_t) tess_begin);
   gluTessCallback(fTess, (GLenum)GLU_TESS_VERTEX_DATA,  (tessfuncptr_t) tess_vertex);
   gluTessCallback(fTess, (GLenum)GLU_TESS_COMBINE_DATA, (tessfuncptr_t) tess_combine);
   gluTessCallback(fTess, (GLenum)GLU_TESS_END_DATA,     (tessfuncptr_t) tess_end);
}

TriangleCollector::~TriangleCollector()
{
   gluDeleteTess(fTess);
}

//------------------------------------------------------------------------------

void TriangleCollector::tess_begin(GLenum type, TriangleCollector* tc)
{
   tc->fNVertices = 0;
   tc->fV0 = tc->fV1 = -1;
   tc->fType = type;
}

void TriangleCollector::tess_vertex(Int_t* vi, TriangleCollector* tc)
{
   tc->process_vertex(*vi);
}

void TriangleCollector::tess_combine(GLdouble /*coords*/[3], void* /*vertex_data*/[4],
                                     GLfloat  /*weight*/[4], void** /*outData*/,
                                     TriangleCollector* /*tc*/)
{
   throw std::runtime_error("GLU::TriangleCollector tesselator requested vertex combining -- not supported yet.");
}

void TriangleCollector::tess_end(TriangleCollector* tc)
{
   tc->fType = GL_NONE;
}

//------------------------------------------------------------------------------

void TriangleCollector::add_triangle(Int_t v0, Int_t v1, Int_t v2)
{
   fPolyDesc.push_back(3);
   fPolyDesc.push_back(v0);
   fPolyDesc.push_back(v1);
   fPolyDesc.push_back(v2);
   ++fNTriangles;
}

void TriangleCollector::process_vertex(Int_t vi)
{
   ++fNVertices;

   if (fV0 == -1) {
      fV0 = vi;
      return;
   }
   if (fV1 == -1) {
      fV1 = vi;
      return;
   }

   switch (fType)
   {
      case GL_TRIANGLES:
      {
         add_triangle(fV0, fV1, vi);
         fV0 = fV1 = -1;
         break;
      }
      case GL_TRIANGLE_STRIP:
      {
         if (fNVertices % 2 == 0)
            add_triangle(fV1, fV0, vi);
         else
            add_triangle(fV0, fV1, vi);
         fV0 = fV1;
         fV1 = vi;
         break;
      }
      case GL_TRIANGLE_FAN:
      {
         add_triangle(fV0, fV1, vi);
         fV1 = vi;
         break;
      }
      default:
      {
         throw std::runtime_error("GLU::TriangleCollector unexpected type in tess_vertex callback.");
      }
   }
}

void TriangleCollector::ProcessData(const std::vector<Double_t>& verts,
                                    const std::vector<Int_t>   & polys,
                                    const Int_t                  n_polys)
{
   const Double_t *pnts = &verts[0];
   const Int_t    *pols = &polys[0];

   for (Int_t i = 0, j = 0; i < n_polys; ++i)
   {
      Int_t n_points = pols[j++];

      gluTessBeginPolygon(fTess, this);
      gluTessBeginContour(fTess);

      for (Int_t k = 0; k < n_points; ++k, ++j)
      {
         gluTessVertex(fTess, (Double_t*) pnts + pols[j] * 3, (GLvoid*) &pols[j]);
      }

      gluTessEndContour(fTess);
      gluTessEndPolygon(fTess);
   }
}

}}}
