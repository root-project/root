// @(#)root/gl:$Name:  $:$Id: CsgOps.h,v 1.25 2005/03/09 18:19:26 brun Exp $
// Author:  Timur Pocheptsov  01/04/2005
   
#ifndef ROOT_CsgOps
#define ROOT_CsgOps

#ifndef ROOT_Rtype
#include "Rtypes.h"
#endif

class TBuffer3D;

namespace RootCsg {

	/*
		I need BaseMesh to have an opaque pointer
		to hidden representatioin of resulting mesh.
	*/

	class BaseMesh {
	public:
		virtual ~BaseMesh(){}
      virtual UInt_t NumberOfPolys()const = 0;
      virtual UInt_t NumberOfVertices()const = 0;
      virtual UInt_t SizeOfPoly(UInt_t polyIndex)const = 0;
      virtual const Double_t *GetVertex(UInt_t vertNum)const = 0;
      virtual Int_t GetVertexIndex(UInt_t polyNum, UInt_t vertNum)const = 0;
	};

   BaseMesh *ConvertToMesh(const TBuffer3D &buff);
   BaseMesh *BuildUnion(const BaseMesh *leftOperand, const BaseMesh *rightOperand);
   BaseMesh *BuildIntersection(const BaseMesh *leftOperand, const BaseMesh *rightOperand);
   BaseMesh *BuildDifference(const BaseMesh *leftOperand, const BaseMesh *rightOperand);
   	
}

#endif
