// @(#)root/csg:$Id$
// Author:  Timur Pocheptsov  01/04/2005

#ifndef ROOT_CsgOps
#define ROOT_CsgOps

namespace RootCsg {

// I need TBaseMesh to have an opaque pointer
// to hidden representation of resulting mesh.

class TBaseMesh {
public:
   TBaseMesh() = default;
   virtual ~TBaseMesh() = default;

   virtual unsigned int NumberOfPolys() const = 0;
   virtual unsigned int NumberOfVertices() const = 0;
   virtual unsigned int SizeOfPoly(unsigned int polyIndex) const = 0;
   virtual const double *GetVertex(unsigned int vertNum) const = 0;
   virtual int GetVertexIndex(unsigned int polyNum, unsigned int vertNum) const = 0;
};

TBaseMesh *ConvertToMesh(double *pnts, const int *segs, const int *pols, unsigned int nbPnts, unsigned int nbSegs,
                         unsigned int nbPols);
TBaseMesh *BuildUnion(const TBaseMesh *leftOperand, const TBaseMesh *rightOperand);
TBaseMesh *BuildIntersection(const TBaseMesh *leftOperand, const TBaseMesh *rightOperand);
TBaseMesh *BuildDifference(const TBaseMesh *leftOperand, const TBaseMesh *rightOperand);

} // namespace RootCsg

#endif
