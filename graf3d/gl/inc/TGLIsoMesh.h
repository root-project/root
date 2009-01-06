#ifndef ROOT_TGLIsoMesh
#define ROOT_TGLIsoMesh

#include <vector>

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

namespace Rgl {
namespace Mc {

/*
TIsoMesh - set of vertices, per-vertex normals, "triangles". 
Each "triangle" is a triplet of indices, pointing into vertices 
and normals arrays. For example, triangle t = {1, 4, 6}
has vertices &fVerts[1 * 3], &fVerts[4 * 3], &fVerts[6 * 3];
and normals &fNorms[1 * 3], &fNorms[4 * 3], &fNorms[6 * 3]
"V" parameter should be Float_t or Double_t (or some
integral type?).

Prefix "T" in a class name only for code-style checker.
*/

template<class V>
class TIsoMesh {
public:
   UInt_t AddVertex(const V *v)
   {
      const UInt_t index = UInt_t(fVerts.size() / 3);
      fVerts.push_back(v[0]);
      fVerts.push_back(v[1]);
      fVerts.push_back(v[2]);

      return index;
   }

   void AddNormal(const V *n)
   {
      fNorms.push_back(n[0]);
      fNorms.push_back(n[1]);
      fNorms.push_back(n[2]);
   }

   UInt_t AddTriangle(const UInt_t *t)
   {
      const UInt_t index = UInt_t(fTris.size() / 3);
      fTris.push_back(t[0]);
      fTris.push_back(t[1]);
      fTris.push_back(t[2]);

      return index;
   }

   void Swap(TIsoMesh &rhs)
   {
      std::swap(fVerts, rhs.fVerts);
      std::swap(fNorms, rhs.fNorms);
      std::swap(fTris, rhs.fTris);
   }

   void ClearMesh()
   {
      fVerts.clear();
      fNorms.clear();
      fTris.clear();
   }

   std::vector<V>      fVerts;
   std::vector<V>      fNorms;
   std::vector<UInt_t> fTris;
};

}//namespace Mc
}//namespace Rgl

#endif
