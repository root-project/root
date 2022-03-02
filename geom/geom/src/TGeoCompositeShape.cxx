// @(#)root/geom:$Id$
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TGeoCompositeShape
\ingroup Shapes_classes

Composite shapes are Boolean combinations of two or more shape
components. The supported Boolean operations are union (+), intersection
(\*) and subtraction(-). Composite shapes derive from the base
**`TGeoShape`** class, therefore providing all shape features:
computation of bounding box, finding if a given point is inside or
outside the combination, as well as computing the distance to
entering/exiting. They can be directly used for creating volumes or used
in the definition of other composite shapes.

Composite shapes are provided in order to complement and extend the set
of basic shape primitives. They have a binary tree internal structure,
therefore all shape-related geometry queries are signals propagated from
top level down to the final leaves, while the provided answers are
assembled and interpreted back at top. This `CSG`
`(composite solid geometry)` hierarchy is effective for small number of
components, while performance drops dramatically for large structures.
Building a complete geometry in this style is virtually possible but
highly not recommended.

#### The Structure of Composite Shapes

A composite shape can always be looked as the result of a Boolean
operation between only two shape components. All information identifying
these two components as well as their positions with respect to the
frame of the composite is represented by an object called Boolean node.
A composite shape has a pointer to such a Boolean node. Since the shape
components may also be composites, they will also contain binary Boolean
nodes branching out other two shapes in the hierarchy. Any such branch
ends-up when the final leaves are no longer composite shapes, but basic
primitives. The figure shows the composite shapes structure.

\image html geom_composite_shape001.png "The composite shapes structure" width=600px

Suppose that A, B, C and D represent basic shapes, we will illustrate
how the internal representation of few combinations look like. We do
this only for understanding how to create them in a proper way, since
the user interface for this purpose is in fact very simple. We will
ignore for the time being the positioning of components. The definition
of a composite shape takes an expression where the identifiers are shape
names. The expression is parsed and decomposed in 2 sub-expressions and
the top-level Boolean operator.

1. Union: `A+B+C`

Just to illustrate the Boolean expression parsing and the composite
shape structure, let's take a simple example. We will describe the union
of A, B and C. Both union operators are at the same level. Since:

`A+B+C = (A+B)+C = A+(B+C)`

The first` (+)` is taken as separator, hence the expression split in:
`A` and `(B+C)`. A Boolean node of type **`TGeoUnion`**`("A","B+C")` is
created. This tries to replace the 2 expressions by actual pointers to
corresponding shapes. The first expression (A) contains no operators
therefore is interpreted as representing a shape. The shape named "A" is
searched into the list of shapes handled by the manager class and stored
as the "left" shape in the Boolean union node. Since the second
expression is not yet fully decomposed, the "right" shape in the
combination is created as a new composite shape. This will split at its
turn B+C into B and C and create a **`TGeoUnion`**`("B","C")`. The B and
C identifiers will be looked for and replaced by the pointers to the
actual shapes into the new node. Finally, the composite "`A+B+C`" will
be represented as shown in Fig.17-23.**

\image html geom_composite_shape002.png "Representation of A+B+C" width=600px

To build this composite shape:

~~~{.cpp}
TGeoCompositeShape *cs1 = new TGeoCompositeShape("CS1","A+B+C");
~~~

Any shape entering a Boolean combination can be prior positioned. In
order to do so, one has to attach a matrix name to the shape name by
using a colon (:). As for shapes, the named matrix has to be prior
defined:

~~~{.cpp}
TGeoMatrix *mat;
// ... code creating some geometrical transformation
mat->SetName("mat1");
mat->RegisterYourself();  // see Geometrical transformations
~~~

An identifier `shape:matrix` have the meaning: `shape` is translated or
rotated with `matrix` with respect to the Boolean combination it enters
as operand. Note that in the expression A+B+C no matrix identifier was
provided, therefore the identity matrix was used for positioning the
shape components. The next example will illustrate a more complex case.

2. `(A:m1+B):m2-(C:m3*D:m4):m5`

Let's try to understand the expression above. This expression means:
subtract the intersection of **C** and **D** from the union of **A** and
**B**. The usage of parenthesis to force the desired precedence is
always recommended. One can see that not only the primitive shapes have
some geometrical transformations, but also their intermediate
compositions.


\image html geom_composite_shape003.png "Internal representation for composite shapes" width=600px

~~~{.cpp}
TGeoCompositeShape *cs2 = new TGeoCompositeShape("CS2",
"(A:m1+B):m2-(C:m3*D:m4):m5");
~~~

Building composite shapes as in the first example is not always quite
useful since we were using un-positioned shapes. When supplying just
shape names as identifiers, the created Boolean nodes will assume that
the shapes are positioned with an identity transformation with respect
to the frame of the created composite. In order to provide some
positioning of the combination components, we have to attach after each
shape identifier the name of an existing transformation, separated by a
colon. Obviously all transformations created for this purpose have to be
objects with unique names in order to be properly substituted during
parsing.

#### Composite Shape Example

One should have in mind that the same shape or matrix identifiers can be
used many times in the same expression, as in the following example:

~~~{.cpp}
{
   TCanvas *c = new TCanvas("c", "c",0,0,600,600);
   const Double_t sq2 = TMath::Sqrt(2.);
   TGeoManager *mgr =
      new TGeoManager("Geom","composite shape example");
   TGeoMedium *medium = 0;
   TGeoVolume *top = mgr->MakeBox("TOP",medium,100,250,250);
   mgr->SetTopVolume(top);

   // make shape components
   TGeoBBox *sbox  = new TGeoBBox("B",100,125*sq2,125*sq2);
   TGeoTube *stub  = new TGeoTube("T",0,100,250);
   TGeoPgon *spgon = new TGeoPgon("P",0.,360.,6,2);
   spgon->DefineSection(0,-250,0,80);
   spgon->DefineSection(1,250,0,80);

   // define some rotations
   TGeoRotation *r1 = new TGeoRotation("r1",90,0,0,180,90,90);
   r1->RegisterYourself();
   TGeoRotation *r2 = new TGeoRotation("r2",90,0,45,90,45,270);
   r2->RegisterYourself();
   // create a composite
   TGeoCompositeShape *cs = new TGeoCompositeShape("cs", "((T+T:r1)-(P+P:r1))*B:r2");
   TGeoVolume *comp = new TGeoVolume("COMP",cs);
   comp->SetLineColor(kRed);

   // put it in the top volume
   top->AddNode(comp,1);
   mgr->CloseGeometry();
   // visualize it with ray tracing
   top->Raytrace();
}
~~~

\image html geom_composite_shape004.png "A composite shape example" width=400px


Composite shapes can be subsequently used for defining volumes.
Moreover, these volumes contain other volumes, following the general
criteria. Volumes created based on composite shapes cannot be divided.

*/

#include <iostream>
#include "TRandom3.h"

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoBoolNode.h"

#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

#include "TGeoCompositeShape.h"
ClassImp(TGeoCompositeShape);

////////////////////////////////////////////////////////////////////////////////
/// Needed just for cleanup.

void TGeoCompositeShape::ClearThreadData() const
{
   if (fNode) fNode->ClearThreadData();
}

////////////////////////////////////////////////////////////////////////////////
/// Needed just for cleanup.

void TGeoCompositeShape::CreateThreadData(Int_t nthreads)
{
   if (fNode) fNode->CreateThreadData(nthreads);
}

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TGeoCompositeShape::TGeoCompositeShape()
                   :TGeoBBox(0, 0, 0)
{
   SetShapeBit(TGeoShape::kGeoComb);
   fNode  = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TGeoCompositeShape::TGeoCompositeShape(const char *name, const char *expression)
                   :TGeoBBox(0, 0, 0)
{
   SetShapeBit(TGeoShape::kGeoComb);
   SetName(name);
   fNode  = 0;
   MakeNode(expression);
   if (!fNode) {
      Error("ctor", "Composite %s: cannot parse expression: %s", name, expression);
      return;
   }
   ComputeBBox();
}

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TGeoCompositeShape::TGeoCompositeShape(const char *expression)
                   :TGeoBBox(0, 0, 0)
{
   SetShapeBit(TGeoShape::kGeoComb);
   fNode  = 0;
   MakeNode(expression);
   if (!fNode) {
      TString message = TString::Format("Composite (no name) could not parse expression %s", expression);
      Error("ctor", "%s", message.Data());
      return;
   }
   ComputeBBox();
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with a Boolean node

TGeoCompositeShape::TGeoCompositeShape(const char *name, TGeoBoolNode *node)
                   :TGeoBBox(0,0,0)
{
   SetName(name);
   fNode = node;
   if (!fNode) {
      Error("ctor", "Composite shape %s has null node", name);
      return;
   }
   ComputeBBox();
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TGeoCompositeShape::~TGeoCompositeShape()
{
   if (fNode) delete fNode;
}

////////////////////////////////////////////////////////////////////////////////
/// Computes capacity of this shape [length^3] by sampling with 1% error.

Double_t TGeoCompositeShape::Capacity() const
{
   Double_t pt[3];
   if (!gRandom) gRandom = new TRandom3();
   Double_t vbox = 8*fDX*fDY*fDZ; // cm3
   Int_t igen=0;
   Int_t iin = 0;
   while (iin<10000) {
      pt[0] = fOrigin[0]-fDX+2*fDX*gRandom->Rndm();
      pt[1] = fOrigin[1]-fDY+2*fDY*gRandom->Rndm();
      pt[2] = fOrigin[2]-fDZ+2*fDZ*gRandom->Rndm();
      igen++;
      if (Contains(pt)) iin++;
   }
   Double_t capacity = iin*vbox/igen;
   return capacity;
}

////////////////////////////////////////////////////////////////////////////////
/// compute bounding box of the sphere

void TGeoCompositeShape::ComputeBBox()
{
   if(fNode) fNode->ComputeBBox(fDX, fDY, fDZ, fOrigin);
}

////////////////////////////////////////////////////////////////////////////////
/// Computes normal vector in POINT to the composite shape.

void TGeoCompositeShape::ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm)
{
   if (fNode) fNode->ComputeNormal(point,dir,norm);
}

////////////////////////////////////////////////////////////////////////////////
/// Tests if point is inside the shape.

Bool_t TGeoCompositeShape::Contains(const Double_t *point) const
{
   if (fNode) return fNode->Contains(point);
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Compute closest distance from point px,py to each corner.

Int_t TGeoCompositeShape::DistancetoPrimitive(Int_t px, Int_t py)
{
   const Int_t numPoints = GetNmeshVertices();
   return ShapeDistancetoPrimitive(numPoints, px, py);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute distance from outside point to this composite shape.
/// Check if the bounding box is crossed within the requested distance

Double_t TGeoCompositeShape::DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact,
                                      Double_t step, Double_t *safe) const
{
   Double_t sdist = TGeoBBox::DistFromOutside(point,dir, fDX, fDY, fDZ, fOrigin, step);
   if (sdist>=step) return TGeoShape::Big();
   if (fNode) return fNode->DistFromOutside(point, dir, iact, step, safe);
   return TGeoShape::Big();
}

////////////////////////////////////////////////////////////////////////////////
/// Compute distance from inside point to outside of this composite shape.

Double_t TGeoCompositeShape::DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact,
                                      Double_t step, Double_t *safe) const
{
   if (fNode) return fNode->DistFromInside(point, dir, iact, step, safe);
   return TGeoShape::Big();
}

////////////////////////////////////////////////////////////////////////////////
/// Divide all range of iaxis in range/step cells

TGeoVolume *TGeoCompositeShape::Divide(TGeoVolume  * /*voldiv*/, const char * /*divname*/, Int_t /*iaxis*/,
                                       Int_t /*ndiv*/, Double_t /*start*/, Double_t /*step*/)
{
   Error("Divide", "Composite shapes cannot be divided");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns numbers of vertices, segments and polygons composing the shape mesh.

void TGeoCompositeShape::GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const
{
   nvert = GetNmeshVertices();
   nsegs = 0;
   npols = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// print shape parameters

void TGeoCompositeShape::InspectShape() const
{
   printf("*** TGeoCompositeShape : %s = %s\n", GetName(), GetTitle());
   printf(" Bounding box:\n");
   TGeoBBox::InspectShape();
}

////////////////////////////////////////////////////////////////////////////////
/// Make a boolean node according to the top level boolean operation of expression.
/// Propagates signal to branches until expression is fully decomposed.
///   printf("Making node for : %s\n", expression);

void TGeoCompositeShape::MakeNode(const char *expression)
{
   if (fNode) delete fNode;
   fNode = 0;
   SetTitle(expression);
   TString sleft, sright, smat;
   Int_t boolop;
   boolop = TGeoManager::Parse(expression, sleft, sright, smat);
   if (boolop<0) {
      // fail
      Error("MakeNode", "parser error");
      return;
   }
   if (smat.Length())
      Warning("MakeNode", "no geometrical transformation allowed at this level");
   switch (boolop) {
      case 0:
         Error("MakeNode", "Expression has no boolean operation");
         return;
      case 1:
         fNode = new TGeoUnion(sleft.Data(), sright.Data());
         return;
      case 2:
         fNode = new TGeoSubtraction(sleft.Data(), sright.Data());
         return;
      case 3:
         fNode = new TGeoIntersection(sleft.Data(), sright.Data());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint this composite shape into the current 3D viewer
/// Returns bool flag indicating if the caller should continue to
/// paint child objects

Bool_t TGeoCompositeShape::PaintComposite(Option_t *option) const
{
   Bool_t addChildren = kTRUE;

   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   TVirtualViewer3D * viewer = gPad->GetViewer3D();
   if (!painter || !viewer) return kFALSE;

   if (fNode) {
      // Fill out the buffer for the composite shape - nothing extra
      // over TGeoBBox
      Bool_t preferLocal = viewer->PreferLocalFrame();
      if (TBuffer3D::GetCSLevel()) preferLocal = kFALSE;
      static TBuffer3D buffer(TBuffer3DTypes::kComposite);
      FillBuffer3D(buffer, TBuffer3D::kCore|TBuffer3D::kBoundingBox,
                   preferLocal);

      Bool_t paintComponents = kTRUE;

      // Start a composite shape, identified by this buffer
      if (!TBuffer3D::GetCSLevel())
         paintComponents = viewer->OpenComposite(buffer, &addChildren);

      TBuffer3D::IncCSLevel();

      // Paint the boolean node - will add more buffers to viewer
      TGeoHMatrix *matrix = (TGeoHMatrix*)TGeoShape::GetTransform();
      TGeoHMatrix backup(*matrix);
      if (preferLocal) matrix->Clear();
      if (paintComponents) fNode->Paint(option);
      if (preferLocal) *matrix = backup;
      // Close the composite shape
      if (!TBuffer3D::DecCSLevel())
         viewer->CloseComposite();
   }

   return addChildren;
}

////////////////////////////////////////////////////////////////////////////////
/// Register the shape and all components to TGeoManager class.

void TGeoCompositeShape::RegisterYourself()
{
   if (gGeoManager->GetListOfShapes()->FindObject(this)) return;
   gGeoManager->AddShape(this);
   TGeoMatrix *matrix;
   TGeoShape  *shape;
   TGeoCompositeShape *comp;
   if (fNode) {
      matrix = fNode->GetLeftMatrix();
      if (!matrix->IsRegistered()) matrix->RegisterYourself();
      else if (!gGeoManager->GetListOfMatrices()->FindObject(matrix)) {
         gGeoManager->GetListOfMatrices()->Add(matrix);
      }
      matrix = fNode->GetRightMatrix();
      if (!matrix->IsRegistered()) matrix->RegisterYourself();
      else if (!gGeoManager->GetListOfMatrices()->FindObject(matrix)) {
         gGeoManager->GetListOfMatrices()->Add(matrix);
      }
      shape = fNode->GetLeftShape();
      if (!gGeoManager->GetListOfShapes()->FindObject(shape)) {
         if (shape->IsComposite()) {
            comp = (TGeoCompositeShape*)shape;
            comp->RegisterYourself();
         } else {
            gGeoManager->AddShape(shape);
         }
      }
      shape = fNode->GetRightShape();
      if (!gGeoManager->GetListOfShapes()->FindObject(shape)) {
         if (shape->IsComposite()) {
            comp = (TGeoCompositeShape*)shape;
            comp->RegisterYourself();
         } else {
            gGeoManager->AddShape(shape);
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// computes the closest distance from given point to this shape, according
/// to option. The matching point on the shape is stored in spoint.

Double_t TGeoCompositeShape::Safety(const Double_t *point, Bool_t in) const
{
   if (fNode) return fNode->Safety(point,in);
   return 0.;
}

////////////////////////////////////////////////////////////////////////////////
/// Save a primitive as a C++ statement(s) on output stream "out".

void TGeoCompositeShape::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   if (TObject::TestBit(kGeoSavePrimitive)) return;
   if (fNode) fNode->SavePrimitive(out,option);
   out << "   // Shape: " << GetName() << " type: " << ClassName() << std::endl;
   out << "   TGeoShape *" << GetPointerName() << " = new TGeoCompositeShape(\"" << GetName() << "\", pBoolNode);" << std::endl;
   if (strlen(GetTitle())) out << "   " << GetPointerName() << "->SetTitle(\"" << GetTitle() << "\");" << std::endl;
   TObject::SetBit(TGeoShape::kGeoSavePrimitive);
}

////////////////////////////////////////////////////////////////////////////////
/// create points for a composite shape

void TGeoCompositeShape::SetPoints(Double_t *points) const
{
   if (fNode) fNode->SetPoints(points);
}

////////////////////////////////////////////////////////////////////////////////
/// create points for a composite shape

void TGeoCompositeShape::SetPoints(Float_t *points) const
{
   if (fNode) fNode->SetPoints(points);
}

////////////////////////////////////////////////////////////////////////////////
/// compute size of this 3D object

void TGeoCompositeShape::Sizeof3D() const
{
   if (fNode) fNode->Sizeof3D();
}

////////////////////////////////////////////////////////////////////////////////
/// Return number of vertices of the mesh representation

Int_t TGeoCompositeShape::GetNmeshVertices() const
{
   if (!fNode) return 0;
   return fNode->GetNpoints();
}

////////////////////////////////////////////////////////////////////////////////
/// Check the inside status for each of the points in the array.
/// Input: Array of point coordinates + vector size
/// Output: Array of Booleans for the inside of each point

void TGeoCompositeShape::Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const
{
   for (Int_t i=0; i<vecsize; i++) inside[i] = Contains(&points[3*i]);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute the normal for an array o points so that norm.dot.dir is positive
/// Input: Arrays of point coordinates and directions + vector size
/// Output: Array of normal directions

void TGeoCompositeShape::ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize)
{
   for (Int_t i=0; i<vecsize; i++) ComputeNormal(&points[3*i], &dirs[3*i], &norms[3*i]);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute distance from array of input points having directions specified by dirs. Store output in dists

void TGeoCompositeShape::DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t* step) const
{
   for (Int_t i=0; i<vecsize; i++) dists[i] = DistFromInside(&points[3*i], &dirs[3*i], 3, step[i]);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute distance from array of input points having directions specified by dirs. Store output in dists

void TGeoCompositeShape::DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t* step) const
{
   for (Int_t i=0; i<vecsize; i++) dists[i] = DistFromOutside(&points[3*i], &dirs[3*i], 3, step[i]);
}

////////////////////////////////////////////////////////////////////////////////
/// Compute safe distance from each of the points in the input array.
/// Input: Array of point coordinates, array of statuses for these points, size of the arrays
/// Output: Safety values

void TGeoCompositeShape::Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const
{
   for (Int_t i=0; i<vecsize; i++) safe[i] = Safety(&points[3*i], inside[i]);
}
