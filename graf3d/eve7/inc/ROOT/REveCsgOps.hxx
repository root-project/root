// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  01/04/2005

#ifndef ROOT7_REveCsgOps
#define ROOT7_REveCsgOps

#include "Rtypes.h"

#include <memory>

class TBuffer3D;
class TGeoCompositeShape;
class TGeoMatrix;

namespace ROOT {
namespace Experimental {
namespace EveCsg {

class TBaseMesh {
public:
   virtual ~TBaseMesh() {}
   virtual Int_t NumberOfPolys() const = 0;
   virtual Int_t NumberOfVertices() const = 0;
   virtual Int_t SizeOfPoly(Int_t polyIndex) const = 0;
   virtual const Double_t *GetVertex(Int_t vertNum) const = 0;
   virtual Int_t GetVertexIndex(Int_t polyNum, Int_t vertNum) const = 0;
};

// TBaseMesh *BuildFromCompositeShape(TGeoCompositeShape *cshape, Int_t n_seg);

std::unique_ptr<TBaseMesh> BuildFromCompositeShapeNew(TGeoCompositeShape *cshape, Int_t n_seg);


} // namespace EveCsg
} // namespace Experimental
} // namespace ROOT

#endif
