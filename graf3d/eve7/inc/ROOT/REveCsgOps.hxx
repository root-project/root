// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  01/04/2005

#ifndef ROOT7_REveCsgOps
#define ROOT7_REveCsgOps

#include "Rtypes.h"

class TBuffer3D;
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


TBaseMesh *ConvertToMesh(const TBuffer3D &buff, TGeoMatrix *matr = nullptr);
TBaseMesh *BuildUnion(const TBaseMesh *leftOperand, const TBaseMesh *rightOperand);
TBaseMesh *BuildIntersection(const TBaseMesh *leftOperand, const TBaseMesh *rightOperand);
TBaseMesh *BuildDifference(const TBaseMesh *leftOperand, const TBaseMesh *rightOperand);


} // namespace EveCsg
} // namespace Experimental
} // namespace ROOT

#endif
