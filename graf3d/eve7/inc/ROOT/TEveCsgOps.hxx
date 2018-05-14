// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  01/04/2005

#ifndef ROOT_TEveCsgOps_hxx
#define ROOT_TEveCsgOps_hxx

#include "Rtypes.h"
#include "TPad.h"
#include "TVirtualViewer3D.h"

class TBuffer3D;
class TGeoCompositeShape;

namespace ROOT { namespace Experimental { namespace Csg
{

class TBaseMesh
{
public:

   virtual ~TBaseMesh(){}
   virtual UInt_t NumberOfPolys()const = 0;
   virtual UInt_t NumberOfVertices()const = 0;
   virtual UInt_t SizeOfPoly(UInt_t polyIndex)const = 0;
   virtual const Double_t *GetVertex(UInt_t vertNum)const = 0;
   virtual Int_t GetVertexIndex(UInt_t polyNum, UInt_t vertNum)const = 0;
};

TBaseMesh *ConvertToMesh(const TBuffer3D &buff);
TBaseMesh *BuildUnion(const TBaseMesh *leftOperand, const TBaseMesh *rightOperand);
TBaseMesh *BuildIntersection(const TBaseMesh *leftOperand, const TBaseMesh *rightOperand);
TBaseMesh *BuildDifference(const TBaseMesh *leftOperand, const TBaseMesh *rightOperand);


//==============================================================================
//==============================================================================

// Internal pad class overriding handling of updates and 3D-viewers.

class TCsgPad : public TPad
{
public:
   TCsgPad();
   virtual ~TCsgPad() {}

   Bool_t    IsBatch() const override { return kTRUE; }

   void      Update() override {}

   TVirtualViewer3D *GetViewer3D(Option_t * /*type*/ = "") override { return fViewer3D; }
};

//------------------------------------------------------------------------------

// Internal VV3D for extracting composite shapes.

class TCsgVV3D : public TVirtualViewer3D
{

   // Composite shape specific
   typedef std::pair<UInt_t, TBaseMesh*> CSPart_t;

   std::vector<CSPart_t>   fCSTokens;
   Int_t                   fCSLevel;
   mutable bool            fCompositeOpen;

   TBaseMesh* BuildComposite();

public:
   std::unique_ptr<TBaseMesh> fResult;

   TCsgVV3D();
   virtual ~TCsgVV3D() {}

   // virtual stuff that is used.
   Int_t  AddObject(const TBuffer3D& buffer, Bool_t* addChildren = 0) override;
   Bool_t OpenComposite(const TBuffer3D& buffer, Bool_t* addChildren = 0) override;
   void   CloseComposite() override;
   void   AddCompositeOp(UInt_t operation) override;

   // virtual crap that needs to be defined but is not used/needed.
   Int_t  AddObject(UInt_t, const TBuffer3D &, Bool_t * = 0) override { return -1; }
   Bool_t CanLoopOnPrimitives() const override { return kTRUE; }
   void   PadPaint(TVirtualPad*) override {}
   void   ObjectPaint(TObject*, Option_t* = "") override {}

   Int_t  DistancetoPrimitive(Int_t, Int_t) override { return 9999; }
   void   ExecuteEvent(Int_t, Int_t, Int_t) override {}

   Bool_t PreferLocalFrame() const override { return kTRUE; }

   void   BeginScene() override {}
   Bool_t BuildingScene() const override { return kTRUE; }
   void   EndScene() override {}
};

//------------------------------------------------------------------------------

TBaseMesh *BuildFromCompositeShape(TGeoCompositeShape *cshape, Int_t n_seg);

}}}

#endif
