// @(#)root/csg:$Id$
// Author:  Timur Pocheptsov  01/04/2005

#ifndef ROOT_CsgOps
#define ROOT_CsgOps

#if !defined(ROOT_CsgOps_cxx) && !defined(G__DICTIONARY) && !defined(__ROOTCLING__) && !defined(__CLING__)
#warning "This header and the TBaseMesh class are deprecated and will be removed after ROOT 6.44, together with the RCsg target. Use instead the 'TGeoTesselated' public interface"
#endif

#include "RtypesCore.h"

class TBuffer3D;

namespace RootCsg {

// I need TBaseMesh to have an opaque pointer
// to hidden representation of resulting mesh.

class TBaseMesh {
public:
   TBaseMesh() = default;
   virtual ~TBaseMesh() = default;

   virtual UInt_t NumberOfPolys() const = 0;
   virtual UInt_t NumberOfVertices() const = 0;
   virtual UInt_t SizeOfPoly(UInt_t polyIndex) const = 0;
   virtual const Double_t *GetVertex(UInt_t vertNum) const = 0;
   virtual Int_t GetVertexIndex(UInt_t polyNum, UInt_t vertNum) const = 0;
};

TBaseMesh *ConvertToMesh(const TBuffer3D &buff);
TBaseMesh *BuildUnion(const TBaseMesh *leftOperand, const TBaseMesh *rightOperand);
TBaseMesh *BuildIntersection(const TBaseMesh *leftOperand, const TBaseMesh *rightOperand);
TBaseMesh *BuildDifference(const TBaseMesh *leftOperand, const TBaseMesh *rightOperand);

} // namespace RootCsg

#endif
