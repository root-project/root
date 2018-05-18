#ifndef ROOT_TEveGluTess_hxx
#define ROOT_TEveGluTess_hxx

#include "Rtypes.h"

#include <vector>

namespace ROOT { namespace Experimental { namespace GLU
{
//==============================================================================
// Mini gl.h
//==============================================================================

/* Datatypes */
typedef unsigned int    GLenum;
typedef unsigned char   GLboolean;
typedef unsigned int    GLbitfield;
typedef void            GLvoid;
typedef signed char     GLbyte;         /* 1-byte signed */
typedef short           GLshort;        /* 2-byte signed */
typedef int             GLint;          /* 4-byte signed */
typedef unsigned char   GLubyte;        /* 1-byte unsigned */
typedef unsigned short  GLushort;       /* 2-byte unsigned */
typedef unsigned int    GLuint;         /* 4-byte unsigned */
typedef int             GLsizei;        /* 4-byte signed */
typedef float           GLfloat;        /* single precision float */
typedef float           GLclampf;       /* single precision float in [0,1] */
typedef double          GLdouble;       /* double precision float */
typedef double          GLclampd;       /* double precision float in [0,1] */

/* Boolean values */
constexpr int GL_FALSE                               = 0;
constexpr int GL_TRUE                                = 1;
constexpr int GL_NONE                                = 0;

/* Primitives */
constexpr int GL_POINTS                              = 0x0000;
constexpr int GL_LINES                               = 0x0001;
constexpr int GL_LINE_LOOP                           = 0x0002;
constexpr int GL_LINE_STRIP                          = 0x0003;
constexpr int GL_TRIANGLES                           = 0x0004;
constexpr int GL_TRIANGLE_STRIP                      = 0x0005;
constexpr int GL_TRIANGLE_FAN                        = 0x0006;
constexpr int GL_QUADS                               = 0x0007;
constexpr int GL_QUAD_STRIP                          = 0x0008;
constexpr int GL_POLYGON                             = 0x0009;


//==============================================================================
// TriangleCollector
//==============================================================================

class GLUtesselator;

class TriangleCollector
{
protected:
   GLUtesselator     *fTess;
   Int_t              fNTriangles;
   Int_t              fNVertices;
   Int_t              fV0, fV1;
   Int_t              fType;
   std::vector<Int_t> fPolyDesc;

   void add_triangle(Int_t v0, Int_t v1, Int_t v2);
   void process_vertex(Int_t vi);

   static void tess_begin(GLenum type, TriangleCollector* tc);
   static void tess_vertex(Int_t* vi, TriangleCollector* tc);
   static void tess_combine(GLdouble coords[3], void*  vertex_data[4],
                            GLfloat  weight[4], void** outData,
                            TriangleCollector* tc);
   static void tess_end(TriangleCollector* tc);

public:
   TriangleCollector();
   ~TriangleCollector();

   // Process polygons
   void ProcessData(const std::vector<Double_t>& verts,
                    const std::vector<Int_t>   & polys,
                    const Int_t                  n_polys);

   // Get output
   Int_t               GetNTrianlges() { return fNTriangles; }
   std::vector<Int_t>& RefPolyDesc()   { return fPolyDesc; }
};

}}}

#endif
