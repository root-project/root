// @(#)root/geom:$Name:  $:$Id: TGeoCompositeShape.cxx,v 1.12 2003/07/31 20:19:32 brun Exp $
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


///////////////////////////////////////////////////////////////////////////////
// TGeoCompositeShape - class handling Boolean composition of shapes
//
//   Composite shapes are Boolean combination of two or more shape
// components. The supported boolean operations are union (+), intersection (*) 
// and subtraction. Composite shapes derive from the base TGeoShape class, 
// therefore providing all shape features : computation of bounding box, finding 
// if a given point is inside or outside the combination, as well as computing the
// distance to entering/exiting. It can be directly used for creating volumes or
// used in the definition of other composite shapes. 
//   Composite shapes are provided in order to complement and extend the set of 
// basic shape primitives. They have a binary tree internal structure, therefore
// all shape-related geometry queries are signals propagated from top level down
// to the final leaves, while the provided answers are assembled and interpreted
// back at top. This CSG hierarchy is effective for small number of components,
// while performance drops dramatically for large structures. Building a complete
// geometry in this style is virtually possible but highly not recommended.
//
//   Structure of composite shapes
//
//   A composite shape can always be regarded as the result of a Boolean operation
// between only two shape components. All information identifying these two
// components as well as their positions with respect to the frame of the composite
// is represented by an object called Boolean node. A composite shape just have
// a pointer to such a Boolean node. Since the shape components may also be 
// composites, they will also contain binary Boolean nodes branching other two
// shapes in the hierarcy. Any such branch ends-up when the final leaves are no
// longer composite shapes, but basic primitives.
//
//Begin_Html
/*
<img src="gif/t_booltree.jpg">
*/
//End_Html
//
//   Suppose that A, B, C and D represent basic shapes, we will illustrate
// how the internal representation of few combinations look like. We do this
// only for the sake of understanding how to create them in a proper way, since
// the user interface for this purpose is in fact very simple. We will ignore 
// for the time being the positioning of components. The definition of a composite
// shape takes an expression where the identifiers are shape names. The 
// expression is parsed and decomposed in 2 sub-expressions and the top-level
// Boolean operator.
//
// 1.     A+B+C
//   This represent the union of A, B and C. Both union operators are at the
// same level. Since:
//        A+B+C = (A+B)+C = A+(B+C)
// the first (+) is taken as separator, hence the expression splitted:
//        A and B+C
// A Boolean node of type TGeoUnion("A", "B+C") is created. This tries to replace
// the 2 expressions by actual pointers to corresponding shapes.
// The first expression (A) contains no operators therefore is interpreted as
// representing a shape. The shape named "A" is searched into the list of shapes
// handled by the manager class and stored as the "left" shape in the Boolean
// union node. Since the second expression is not yet fully decomposed, the "right"
// shape in the combination is created as a new composite shape. This will split
// at its turn B+C into B and C and create a TGeoUnion("B","C"). The B and C
// identifiers will be looked for and replaced by the pointers to the actual shapes
// into the new node. Finally, the composite "A+B+C" will be represented as:
//
//                 A
//                |
//   [A+B+C] = (+)             B
//                |           |
//                 [B+C] = (+)
//                            |
//                             C
//
// where [] is a composite shape, (+) is a Boolean node of type union and A, B,
// C are pointers to the corresponding shapes.
//   Building this composite shapes takes the following line :
//      TGeoCompositeShape *cs1 = new TGeoCompositeShape("CS1", "A+B+C");
//
// 2.      (A+B)\(C+D)
//   This expression means: subtract the union of C and D from the union of A and
// B. The usage of paranthesys to force operator precedence is always recommended.
// The representation of the corresponding composite shape looks like:
//
//                                   A
//                                  |
//                       [A+B] = (+)
//                      |           |
//   [(A+B)\(C+D)] = (\)           C B
//                      |         |  
//                       [C+D]=(+)
//                                |
//                                 D
//
//      TGeoCompositeShape *cs2 = new TGeoCompositeShape("CS2", "(A+B)\(C+D)");
//   
//   Building composite shapes as in the 2 examples above is not always quite
// usefull since we were using unpositioned shapes. When suplying just shape
// names as identifiers, the created boolean nodes will assume that the shapes
// are positioned with an identity transformation with respect to the frame of
// the created composite. In order to provide some positioning of the combination
// components, we have to attach after each shape identifier the name of an 
// existing transformation, separated by a colon. Obviously all transformations 
// created for this purpose have to be objects with unique names in order to be 
// properly substituted during parsing. 
//   Let's look at the code implementing the second example :
//
//      TGeoTranslation *t1 = new TGeoTranslation("T1",0,0,-20);
//      TGeoTranslation *t2 = new TGeoTranslation("T2",0,0, 20);
//      TGeoRotation *r1 = new TGeoRotation("R1"); // transformations need names
//      r1->SetAngles(90,30,90,120,0,0); // rotation with 30 degrees about Z
//      TGeoTube *a = new TGeoTube(0, 10,20);
//      a->SetName("A");                 // shapes need names too
//      TGeoTube *b = new TGeoTube(0, 20,20);
//      b->SetName("B");
//      TGeoBBox *c = new TGeoBBox(10,10,50);
//      c->SetName("C");              
//      TGeoBBox *d = new TGeoBBox(50,10,10);
//      d->SetName("D");
//      
//      TGeoCompositeShape *cs;
//      cs = new TGeoCompositeShape("CS", "(A:t1+B:t2)\(C+D:r1)");
//
//   The newly created composite looks like 2 cylinders of different radii sitting
// one on top of the other and having 2 rectangular holes : a longitudinal one
// along Z axis corresponding to C and an other one in the XY plane due to D.
//   One should have in mind that the same shape or matrix identifier can be
// used many times in the same expression. For instance:
//
//      (A:t1-A:t2)*B:t1
//
// is a valid expression. Expressions that cannot be parsed or identifiers that
// cannot be substituted by existing objects generate error messages.
//   Composite shapes can be subsequently used for defining volumes. Moreover,
// these volumes may have daughters but these have to obbey overlapping/extruding
// rules (see TGeoVolume). Volumes created based on composite shapes cannot be
// divided. Visualization of such volumes is currently not implemented.

//#include "TROOT.h"

#include "TGeoManager.h"
#include "TGeoBoolNode.h"
#include "TVirtualGeoPainter.h"

#include "TGeoCompositeShape.h"
ClassImp(TGeoCompositeShape)

//_____________________________________________________________________________
TGeoCompositeShape::TGeoCompositeShape()
                   :TGeoBBox(0, 0, 0)
{
// Default constructor
   SetShapeBit(TGeoShape::kGeoComb);
   fNode  = 0;
}   

//_____________________________________________________________________________
TGeoCompositeShape::TGeoCompositeShape(const char *name, const char *expression)
                   :TGeoBBox(0, 0, 0)
{
// Default constructor
   SetShapeBit(TGeoShape::kGeoComb);
   SetName(name);
   fNode  = 0;
   MakeNode(expression);
   if (!fNode) {
      char message[256];
      sprintf(message, "could not build expression %s", expression);
      Error("ctor", message);
      return;
   }
   ComputeBBox();
}  

//_____________________________________________________________________________
TGeoCompositeShape::TGeoCompositeShape(const char *expression)
                   :TGeoBBox(0, 0, 0)
{
// Default constructor
   SetShapeBit(TGeoShape::kGeoComb);
   fNode  = 0;
   MakeNode(expression);
   if (!fNode) {
      char message[256];
      sprintf(message, "could not build expression %s", expression);
      Error("ctor", message);
      return;
   }
   ComputeBBox();
}  

//_____________________________________________________________________________
TGeoCompositeShape::~TGeoCompositeShape()
{
// destructor
   if (fNode) delete fNode;
}

//_____________________________________________________________________________
void TGeoCompositeShape::ComputeBBox()
{
// compute bounding box of the sphere
   if(fNode) fNode->ComputeBBox(fDX, fDY, fDZ, fOrigin);
}   

//_____________________________________________________________________________
Bool_t TGeoCompositeShape::Contains(Double_t *point) const
{
// test if point is inside this sphere
   if (fNode) return fNode->Contains(point);
   return kFALSE;
}

//_____________________________________________________________________________
Double_t TGeoCompositeShape::DistToIn(Double_t *point, Double_t *dir, Int_t iact,
                                      Double_t step, Double_t *safe) const
{
// Compute distance from outside point to this composite shape.
   if (!CouldBeCrossed(point, dir)) return kBig;
   if (fNode) return fNode->DistToIn(point, dir, iact, step, safe);
   return kBig;
}   

//_____________________________________________________________________________
Double_t TGeoCompositeShape::DistToOut(Double_t *point, Double_t *dir, Int_t iact,
                                      Double_t step, Double_t *safe) const
{
// Compute distance from inside point to outside of this composite shape.
   if (fNode) return fNode->DistToOut(point, dir, iact, step, safe);
   return kBig;
}   

//_____________________________________________________________________________
TGeoVolume *TGeoCompositeShape::Divide(TGeoVolume  * /*voldiv*/, const char * /*divname*/, Int_t /*iaxis*/, 
                                       Int_t /*ndiv*/, Double_t /*start*/, Double_t /*step*/) 
{
// Divide all range of iaxis in range/step cells 
   Error("Divide", "Composite shapes cannot be divided");
   return 0;
}      

//_____________________________________________________________________________
void TGeoCompositeShape::InspectShape() const
{
// print shape parameters
   printf("*** TGeoCompositeShape : %s = %s\n", GetName(), GetTitle());
   TGeoBBox::InspectShape();
}

//_____________________________________________________________________________
void TGeoCompositeShape::MakeNode(const char *expression)
{
// Make a booleann node according to the top level boolean operation of expression.
// Propagates signal to branches until expression is fully decomposed.
   printf("Making node for : %s\n", expression);
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

//_____________________________________________________________________________
void TGeoCompositeShape::Paint(Option_t *option)
{
// paint this shape according to option
   TGeoHMatrix *glmat = gGeoManager->GetCurrentMatrix();
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return;
   PaintNext(glmat, option);
}

//_____________________________________________________________________________
void TGeoCompositeShape::PaintNext(TGeoHMatrix *glmat, Option_t *option)
{
// paint this shape according to option
   if (fNode) fNode->PaintNext(glmat, option);
}

//_____________________________________________________________________________
Double_t TGeoCompositeShape::Safety(Double_t * /*point*/, Bool_t /*in*/) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   return kBig;
}

//_____________________________________________________________________________
void TGeoCompositeShape::SetPoints(Double_t *buff) const
{
// create points for a composite shape
   TGeoBBox::SetPoints(buff);
}

//_____________________________________________________________________________
void TGeoCompositeShape::SetPoints(Float_t *buff) const
{
// create points for a composite shape
   TGeoBBox::SetPoints(buff);
}

//_____________________________________________________________________________
void TGeoCompositeShape::Sizeof3D() const
{
// compute size of this 3D object
   if (fNode) fNode->Sizeof3D();
}
