// @(#)geom/geocad:$Id$
// Author: Cinzia Luzzi   5/5/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// TGeoToOCC Class                                                              //
// --------------------                                                       //
//                                                                            //
//   This class contains implementation of converting ROOT's                  //
//   geometry shapes to OpenCascade shapes.                                   //                                                                        //
//   Each ROOT shape is translated in the corrispondent OCC shape using the   //
//   following methods:                                                       //
//                                                                            //
// TGeoBBox               ->           OCC_Box(..)                            //
// TGeoSphere             ->           OCC_Sphere(..)                         //
// TGeoArb8               ->           OCC_Arb8(..)                           //
// TGeoConeSeg            ->           OCC_Cones(..)                          //
// TGeoCone               ->           OCC_Cones(..)                          //
// TGeoPara               ->           OCC_ParaTrap(..)                       //
// TGeoTrap               ->           OCC_ParaTrap(..)                       //
// TGeoGtra               ->           OCC_ParaTrap(..)                       //
// TGeoTrd1               ->           OCC_Trd(..)                            //
// TGeoTrd2               ->           OCC_Trd(..)                            //
// TGeoTubeSeg            ->           OCC_Tube(..)                           //
// TGeoCtub               ->           OCC_Cuttub(..)                         //
// TGeoTube               ->           OCC_TubeSeg(..)                        //
// TGeoPcon               ->           OCC_Pcon(..)                           //
// TGeoTorus              ->           OCC_Torus(..)                          //
// TGeoPgon               ->           OCC_Pgon(..)                           //
// TGeoEltu               ->           OCC_Eltu(..)                           //
// TGeoHype               ->           OCC_Hype(..)                           //
// TGeoXtru               ->           OCC_Xtru(..)                           //
// TGeoCompositeShape     ->           OCC_CompositeShape(..)                 //
//                                                                            //
// A log file is created in /tmp/TGeoCad.log                                  //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#include "TGeoToOCC.h"


//Cascade

#include <TopoDS.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Solid.hxx>
#include <gp_Pnt.hxx>
#include <gp_Ax1.hxx>
#include <gp_Circ.hxx>
#include <gp_Dir.hxx>
#include <gp_Hypr.hxx>
#include <gp_Pln.hxx>
#include <GC_MakeArcOfCircle.hxx>
#include <GC_MakeEllipse.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeShell.hxx>
#include <BRepBuilderAPI_MakeSolid.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepAlgo_Section.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeCone.hxx>
#include <BRepPrimAPI_MakeTorus.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeWedge.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <TopExp_Explorer.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgo_Cut.hxx>
#include <Geom_Plane.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepGProp.hxx>
#include <GProp_GProps.hxx>
#include <TColgp_HArray1OfPnt.hxx>
#include <ShapeFix_ShapeTolerance.hxx>

//ROOT
#include "TString.h"
#include "TClass.h"
#include "TGeoBoolNode.h"
#include "TGeoShapeAssembly.h"
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#include "TGeoArb8.h"
#include "TGeoPara.h"
#include "TGeoTorus.h"
#include "TGeoCone.h"
#include "TGeoTube.h"
#include "TGeoEltu.h"
#include "TGeoSphere.h"
#include "TGeoPcon.h"
#include "TGeoPgon.h"
#include "TGeoHype.h"
#include "TGeoPolygon.h"
#include "TGeoMatrix.h"

#include <exception>


TGeoToOCC::TGeoToOCC():fOccShape()
{

}

TGeoToOCC::~TGeoToOCC()
{

}

TopoDS_Shape TGeoToOCC::OCC_SimpleShape(TGeoShape *TG_Shape)
{
   TString type = TG_Shape->IsA()->GetName();
   out.open("/tmp/TGeoCad.log",ios::app);
   out<<"Translating: "<<type<<endl;
   out.close();
   if(TG_Shape->IsA()==TGeoTube::Class()) {
      TGeoTube* TG_Tube=(TGeoTube*)TG_Shape;
      return OCC_Tube(TG_Tube->GetRmin(), TG_Tube->GetRmax(),TG_Tube->GetDz(),0, 0);
   } else if(TG_Shape->IsA()==TGeoTubeSeg::Class()) {
      TGeoTubeSeg* TG_TubeSeg=(TGeoTubeSeg*)TG_Shape;
      Double_t r = (TG_TubeSeg->GetPhi2()-TG_TubeSeg->GetPhi1());
      return OCC_Tube(TG_TubeSeg->GetRmin(), TG_TubeSeg->GetRmax(),TG_TubeSeg->GetDz(),(TG_TubeSeg->GetPhi1())*M_PI/180., r*M_PI/180.);
   } else if(TG_Shape->IsA()==TGeoEltu::Class()) {
      TGeoEltu* TG_Eltu=(TGeoEltu*)TG_Shape;
      return OCC_EllTube(TG_Eltu->GetA(),TG_Eltu->GetB() , TG_Eltu->GetDz());
   } else if(TG_Shape->IsA()==TGeoCtub::Class()) {
      TGeoCtub* TG_Ctub=(TGeoCtub*)TG_Shape;
      Double_t r = (TG_Ctub->GetPhi2()-TG_Ctub->GetPhi1());
      return OCC_Cuttub(TG_Ctub->GetRmin(), TG_Ctub->GetRmax(),  TG_Ctub->GetDz(),
      TG_Ctub->GetPhi1()*M_PI/180.,r*M_PI/180.,TG_Ctub->GetNlow(),TG_Ctub->GetNhigh());
   } else if(TG_Shape->IsA()==TGeoCone::Class()) {
      TGeoCone* TG_Cone=(TGeoCone*)TG_Shape;
      return OCC_Cones(TG_Cone->GetRmin1(),TG_Cone->GetRmax1(),TG_Cone->GetRmin2(), TG_Cone->GetRmax2(),TG_Cone->GetDz(), 0, 2*M_PI);
   } else if(TG_Shape->IsA()==TGeoConeSeg::Class()) {
      TGeoConeSeg* TG_ConeSeg=(TGeoConeSeg*)TG_Shape;
      Double_t r  = (TG_ConeSeg->GetPhi2()-TG_ConeSeg->GetPhi1());
      return OCC_Cones(TG_ConeSeg->GetRmin1(), TG_ConeSeg->GetRmax1(),TG_ConeSeg->GetRmin2(), TG_ConeSeg->GetRmax2(),
      TG_ConeSeg->GetDz(), (TG_ConeSeg->GetPhi1())*M_PI/180., r*M_PI/180.);
   } else if(TG_Shape->IsA()==TGeoTorus::Class()) {
      TGeoTorus* TG_Torus=(TGeoTorus*)TG_Shape;
      Double_t DPhi=(Double_t)TG_Torus->GetDphi()-TG_Torus->GetPhi1();
      if (DPhi<0)
      DPhi=(Double_t)TG_Torus->GetPhi1()-TG_Torus->GetDphi();
      Double_t Phi1= (Double_t)TG_Torus->GetPhi1();
      return OCC_Torus((Double_t)TG_Torus->GetRmin(),(Double_t)TG_Torus->GetRmax(),(Double_t)TG_Torus->GetR(),
      Phi1*M_PI/180., DPhi*M_PI/180.);
   } else if(TG_Shape->IsA()==TGeoSphere::Class()) {
      TGeoSphere* TG_Sphere=(TGeoSphere*)TG_Shape;
      Double_t DPhi = (TG_Sphere->GetPhi2()-TG_Sphere->GetPhi1());
      Double_t DTheta = (TG_Sphere->GetTheta2()-TG_Sphere->GetTheta1());
      return OCC_Sphere(TG_Sphere->GetRmin(), TG_Sphere->GetRmax(),(TG_Sphere->GetPhi1())*M_PI/180., DPhi*M_PI/180.,
      TG_Sphere->GetTheta1()*M_PI/180., DTheta*M_PI/180.);
   } else if(TG_Shape->IsA()==TGeoPcon::Class()) {
      TGeoPcon* TG_Pcon=(TGeoPcon*)TG_Shape;
      return OCC_Pcon((TG_Pcon->GetPhi1())*M_PI/180.,
      (TG_Pcon->GetDphi())*M_PI/180.,TG_Pcon->GetNz(),TG_Pcon->GetRmin(),TG_Pcon->GetRmax(),TG_Pcon->GetZ());
   } else if(TG_Shape->IsA()==TGeoPgon::Class()) {
      TGeoPgon* TG_Pgon=(TGeoPgon*)TG_Shape;
      Int_t numpoints=TG_Pgon->GetNmeshVertices();
      Double_t *p = new Double_t[3*numpoints];
      TG_Pgon->SetPoints(p);
      return OCC_Pgon(TG_Pgon->GetNsegments(),TG_Pgon->GetNz(),p,TG_Pgon->GetPhi1(),TG_Pgon->GetDphi(),numpoints*3);
   } else if(TG_Shape->IsA()==TGeoHype::Class()) {
      TGeoHype* TG_Hype=(TGeoHype*)TG_Shape;
      return OCC_Hype(TG_Hype->GetRmin(), TG_Hype->GetRmax(), TG_Hype->GetStIn(), TG_Hype->GetStOut(),TG_Hype->GetDz());
   } else if(TG_Shape->IsA()==TGeoXtru::Class()) {
      return OCC_Xtru((TGeoXtru*)TG_Shape);
   } else if (TG_Shape->IsA()==TGeoBBox::Class()) {
      TGeoBBox * TG_Box=(TGeoBBox*)TG_Shape;
      const Double_t * Origin = TG_Box->GetOrigin();
      return OCC_Box(TG_Box->GetDX(),TG_Box->GetDY(),TG_Box->GetDZ(),Origin[0],Origin[1],Origin[2]);
   } else if (TG_Shape->IsA()==TGeoTrd1::Class()) {
      TGeoTrd1 * TG_Trd1=(TGeoTrd1*)TG_Shape;
      return OCC_Trd(TG_Trd1->GetDx1(),TG_Trd1->GetDx2(),TG_Trd1->GetDy(),TG_Trd1->GetDy(),TG_Trd1->GetDz());
   } else if (TG_Shape->IsA()==TGeoTrd2::Class()) {
      TGeoTrd2 * TG_Trd2=(TGeoTrd2*)TG_Shape;
      return OCC_Trd(TG_Trd2->GetDx1(),TG_Trd2->GetDx2(),TG_Trd2->GetDy1(),TG_Trd2->GetDy2(),TG_Trd2->GetDz());
   } else if (TG_Shape->IsA()==TGeoArb8::Class()) {
      TGeoArb8 * TG_Arb8=(TGeoArb8*)TG_Shape;
     Double_t vertex[24];
      TG_Shape->SetPoints(vertex);
      return OCC_Arb8(TG_Arb8->GetDz(),TG_Arb8->GetVertices(),vertex);
   } else if (TG_Shape->IsA()==TGeoShapeAssembly::Class()) {
      TGeoBBox * TG_Ass=(TGeoBBox*)TG_Shape;
      return OCC_Box(TG_Ass->GetDX(),TG_Ass->GetDY(),TG_Ass->GetDZ(),0,0,0);
   } else if (TG_Shape->IsA()==TGeoPara::Class()) {
      //TGeoPara * TG_Para=(TGeoPara*)TG_Shape;
      Double_t vertex[24];
      TG_Shape->SetPoints(vertex);
      return OCC_ParaTrap(vertex);
   }  else if (TG_Shape->IsA()==TGeoTrap::Class()) {
      //TGeoTrap * TG_Trap=(TGeoTrap*)TG_Shape;
      Double_t vertex[24];
      TG_Shape->SetPoints(vertex);
      return OCC_ParaTrap(vertex);
   } else if (TG_Shape->IsA()==TGeoGtra::Class()) {
      //TGeoGtra * TG_Tra=(TGeoGtra*)TG_Shape;
      Double_t vertex[24];
      TG_Shape->SetPoints(vertex);
      return OCC_ParaTrap(vertex);
   } else {
      throw std::domain_error("Unknown Shape");
   }
}

TopoDS_Shape TGeoToOCC::OCC_CompositeShape(TGeoCompositeShape *comp, TGeoHMatrix m)
{
   Double_t const *t;
   Double_t const *r;
   gp_Trsf Transl;
   gp_Trsf Transf;
   out.open("/tmp/TGeoCad.log",ios::app);
   TopoDS_Shape leftOCCShape;
   TopoDS_Shape rightOCCShape;
   TopoDS_Shape result;
   GProp_GProps System;
   GProp_GProps System2;
   TGeoBoolNode *boolNode=comp->GetBoolNode();
   TGeoShape *rightShape=boolNode->GetRightShape();
   TString rightSName = rightShape->IsA()->GetName();
   TGeoShape *leftShape=boolNode->GetLeftShape();
   TString leftSName = leftShape->IsA()->GetName();
   TGeoMatrix *leftMtx=boolNode->GetLeftMatrix();
   TGeoMatrix *rightMtx=boolNode->GetRightMatrix();
   TGeoHMatrix  leftGlobMatx=m*(*leftMtx);
   if(leftSName == "TGeoCompositeShape") {
      leftOCCShape=OCC_CompositeShape((TGeoCompositeShape*)leftShape, leftGlobMatx);
   } else {
      t=leftGlobMatx.GetTranslation();
      r=leftGlobMatx.GetRotationMatrix();
      Transl.SetTranslation(gp_Vec(t[0],t[1],t[2]));
      Transf.SetValues(r[0],r[1],r[2],0,
                       r[3],r[4],r[5],0,
                       r[6],r[7],r[8],0
#if OCC_VERSION_MAJOR == 6 && OCC_VERSION_MINOR < 8
                       ,0,1
#endif
                       );
      BRepBuilderAPI_Transform Transformation(Transf);
      BRepBuilderAPI_Transform Translation(Transl);
      Transformation.Perform(OCC_SimpleShape(leftShape),true);
      TopoDS_Shape shapeTransf = Transformation.Shape();
      Translation.Perform(shapeTransf, Standard_True);
      leftOCCShape = Translation.Shape();
   }
   TGeoHMatrix  rightGlobMatx=m*(*rightMtx);
   if(rightSName == "TGeoCompositeShape" ) {
      rightOCCShape=OCC_CompositeShape((TGeoCompositeShape*)rightShape, leftGlobMatx);
   } else {
      t=rightGlobMatx.GetTranslation();
      r=rightGlobMatx.GetRotationMatrix();
      Transl.SetTranslation(gp_Vec(t[0],t[1],t[2]));
      Transf.SetValues(
                     r[0],r[1],r[2],0,
                     r[3],r[4],r[5],0,
                     r[6],r[7],r[8],0
#if OCC_VERSION_MAJOR == 6 && OCC_VERSION_MINOR < 8
                       ,0,1
#endif
                       );
      BRepBuilderAPI_Transform Transformation(Transf);
      BRepBuilderAPI_Transform Translation(Transl);
      TopoDS_Shape sh=OCC_SimpleShape(rightShape);
      Transformation.Perform(sh,true);
      TopoDS_Shape shapeTransf = Transformation.Shape();
      Translation.Perform(shapeTransf, Standard_True);
      rightOCCShape = Translation.Shape();
   }
   TGeoBoolNode::EGeoBoolType boolOper=boolNode->GetBooleanOperator();
   if(TGeoBoolNode::kGeoUnion == boolOper){
      if (leftOCCShape.IsNull())out<<"leftshape is null"<<endl;
      if (rightOCCShape.IsNull())out<<"rightshape is null"<<endl;
      leftOCCShape.Closed(true);
      rightOCCShape.Closed(true);
      BRepAlgoAPI_Fuse Result(leftOCCShape, rightOCCShape);
      Result.Build();
      result=Result.Shape();
      result.Closed(true);
      return Reverse(result);
   } else if(TGeoBoolNode::kGeoIntersection == boolOper) {
      BRepAlgoAPI_Common Result(rightOCCShape,leftOCCShape);
      Result.Build();
      result=Result.Shape();
      result.Closed(true);
      return Reverse(result);
   } else if(TGeoBoolNode::kGeoSubtraction ==boolOper) {
      if (leftOCCShape.IsNull())out<<"leftshape is null"<<endl;
      if (rightOCCShape.IsNull())out<<"rightshape is null"<<endl;
      out.close();
      BRepGProp::VolumeProperties(rightOCCShape, System);
      if (System.Mass() < 0.0) rightOCCShape.Reverse();
      BRepGProp::VolumeProperties(leftOCCShape, System2);
      if (System2.Mass() < 0.0) leftOCCShape.Reverse();
      BRepAlgoAPI_Cut Result(leftOCCShape,rightOCCShape);
      Result.Build();
      result=Result.Shape();
      return Reverse(result);
   } else {
     throw std::domain_error( "Unknown operation" );
   }
}

TopoDS_Shape TGeoToOCC::OCC_EllTube(Double_t a, Double_t b, Double_t dz)
{
   gp_Pnt p (0.,0.,-dz);
   gp_Dir d (0,0,1);
   gp_Ax2 ax2 (p,d);
   TopoDS_Edge e;
   TopoDS_Wire w;
   TopoDS_Face f;
   if(a>b)
      e=BRepBuilderAPI_MakeEdge(GC_MakeEllipse (ax2, a, b));
   else
      e=BRepBuilderAPI_MakeEdge(GC_MakeEllipse (ax2, b, a));
   w=BRepBuilderAPI_MakeWire(e);
   f=BRepBuilderAPI_MakeFace(w);
   gp_Vec v(0 , 0 , dz*2);
   fOccShape = BRepPrimAPI_MakePrism(f , v);
   if(a<b) {
      gp_Trsf t;
      t.SetRotation(gp::OZ(), M_PI/2.);
      BRepBuilderAPI_Transform brepT(fOccShape , t);
      fOccShape = brepT.Shape();
   }
   return Reverse(fOccShape);
}

TopoDS_Shape TGeoToOCC::OCC_Torus(Double_t Rmin, Double_t Rmax, Double_t Rtor,
                            Double_t SPhi, Double_t DPhi)
{
   TopoDS_Solid torMin;
   TopoDS_Solid torMax;
   TopoDS_Shape tor;
   gp_Trsf t;
   if (Rmin==0) Rmin=0.000001;
   if (Rmax==0) Rmax=0.000001;
   if (Rtor==0) Rtor=0.000001;
   torMin = BRepPrimAPI_MakeTorus(Rtor,Rmin,DPhi);
   torMax = BRepPrimAPI_MakeTorus(Rtor,Rmax,DPhi);
   BRepAlgoAPI_Cut cutResult(torMax, torMin);
   cutResult.Build();
   tor=cutResult.Shape();
   TopExp_Explorer anExp1 (tor, TopAbs_SOLID);
   if (anExp1.More()) {
      TopoDS_Shape aTmpShape = anExp1.Current();
      tor = TopoDS::Solid (aTmpShape);
   }
   t.SetRotation(gp::OZ(), SPhi);
   BRepBuilderAPI_Transform brepT(tor , t);
   fOccShape = brepT.Shape();
   return  Reverse(fOccShape);
}


TopoDS_Shape TGeoToOCC::OCC_Sphere(Double_t rmin, Double_t rmax,
                                    Double_t phi1, Double_t Dphi,
                                    Double_t theta1, Double_t Dtheta)
{
   TopoDS_Edge eO;
   TopoDS_Edge e1;
   TopoDS_Edge e2;
   TopoDS_Edge eI;
   TopoDS_Face f;
   TopoDS_Wire w;


   if(rmin==0&&phi1==0&&Dphi==2*M_PI&&theta1==0&&Dtheta==M_PI) {
      TopoDS_Solid s= BRepPrimAPI_MakeSphere(rmax);
      return s;
   }
   Handle(Geom_TrimmedCurve) arcO =   GC_MakeArcOfCircle (gp_Circ (gp_Ax2 (gp_Pnt(0., 0., 0.),
                                                              gp_Dir (0, 1,0)),rmax),theta1,
                                                               theta1+Dtheta, true);
   BRepBuilderAPI_MakeEdge makeEO(arcO);
   eO = TopoDS::Edge(makeEO.Shape());
   if(rmin>0) {
      Handle(Geom_TrimmedCurve) arcI =   GC_MakeArcOfCircle (gp_Circ (gp_Ax2 (gp_Pnt(0.,0., 0.),
                                                                                   gp_Dir (0,1, 0)),rmin),
                                                                  theta1, theta1+Dtheta,true);

      BRepBuilderAPI_MakeEdge makeEI(arcI);
      eI = TopoDS::Edge(makeEI.Shape());
      e1 = BRepBuilderAPI_MakeEdge(makeEO.Vertex1(), makeEI.Vertex1());
      e2 = BRepBuilderAPI_MakeEdge(makeEO.Vertex2(), makeEI.Vertex2());
      w = BRepBuilderAPI_MakeWire(eO , e2 , eI, e1);
      f = BRepBuilderAPI_MakeFace(w);
   } else {
      gp_Pnt pZero(0.,0.,0.);
      TopoDS_Vertex vZero = BRepBuilderAPI_MakeVertex(pZero);
      e1 = BRepBuilderAPI_MakeEdge(makeEO.Vertex1(),vZero );
      e2 = BRepBuilderAPI_MakeEdge(makeEO.Vertex2(),vZero );
      w = BRepBuilderAPI_MakeWire(eO , e2 ,  e1);
      f = BRepBuilderAPI_MakeFace(w);
   }
   gp_Trsf t;
   t.SetRotation(gp::OZ(), phi1);
   BRepBuilderAPI_Transform brepT(f , t);
   fOccShape= brepT.Shape();
   fOccShape = BRepPrimAPI_MakeRevol(fOccShape,gp::OZ(),Dphi);
   return Reverse(fOccShape);
}

TopoDS_Shape TGeoToOCC::OCC_Tube(Double_t rmin, Double_t rmax,
                           Double_t dz, Double_t phi1,
                           Double_t phi2)
{
   TopoDS_Solid innerCyl;
   TopoDS_Solid outerCyl;
   TopoDS_Shape tubs;
   TopoDS_Shape  tubsT;
   gp_Trsf TT;
   gp_Trsf TR;
   if (rmin==0) rmin=rmin+0.00001;
   if (rmax==0) rmax=rmax+0.00001;
   if (phi1==0&&phi2==0) {
     innerCyl = BRepPrimAPI_MakeCylinder(rmin,dz*2);
     outerCyl = BRepPrimAPI_MakeCylinder(rmax,dz*2);
   } else {
      innerCyl = BRepPrimAPI_MakeCylinder(rmin,dz*2,phi2);
      outerCyl = BRepPrimAPI_MakeCylinder(rmax,dz*2,phi2);
   }
   BRepAlgoAPI_Cut cutResult(outerCyl, innerCyl);
   cutResult.Build();
   tubs=cutResult.Shape();
   TopExp_Explorer anExp1 (tubs, TopAbs_SOLID);
   if (anExp1.More()) {
      TopoDS_Shape aTmpShape = anExp1.Current();
      tubs = TopoDS::Solid (aTmpShape);
   }
   TT.SetRotation(gp_Ax1(gp_Pnt(0.,0.,0.), gp_Vec(0., 0., 1.)), phi1);
   BRepBuilderAPI_Transform theTT(TT);
   theTT.Perform(tubs, Standard_True);
   tubsT=theTT.Shape();
   TR.SetTranslation(gp_Vec(0,0,-dz ));
   BRepBuilderAPI_Transform theTR(TR);
   theTR.Perform(tubsT, Standard_True);
   fOccShape=theTR.Shape();
   return  Reverse(fOccShape);
}

TopoDS_Shape TGeoToOCC::OCC_Cones(Double_t rmin1, Double_t rmax1, Double_t rmin2, Double_t rmax2, Double_t dz, Double_t phi1, Double_t phi2)
{
   TopoDS_Solid innerCon;
   TopoDS_Solid  outerCon;
   TopoDS_Shape cons;
   gp_Trsf TT;
   gp_Trsf TR;
   if (rmin1==0) rmin1=rmin1+0.000001;
   if (rmax1==0) rmax1=rmax1+0.000001;
   if(rmin1!=rmin2)
      innerCon = BRepPrimAPI_MakeCone(rmin1,rmin2,dz*2,phi2);
   else
      innerCon = BRepPrimAPI_MakeCylinder(rmin1,dz*2,phi2);
   if(rmax1!=rmax2)
      outerCon = BRepPrimAPI_MakeCone(rmax1,rmax2,dz*2,phi2);
   else
      outerCon = BRepPrimAPI_MakeCylinder(rmax1,dz*2,phi2);
   BRepAlgoAPI_Cut cutResult(outerCon, innerCon);
   cutResult.Build();
   cons = cutResult.Shape();
   TT.SetRotation(gp_Ax1(gp_Pnt(0.,0.,0.), gp_Vec(0., 0., 1.)), phi1);
   BRepBuilderAPI_Transform theTT(TT);
   theTT.Perform(cons, Standard_True);
   cons=theTT.Shape();
   TR.SetTranslation(gp_Vec(0,0,-dz ));
   BRepBuilderAPI_Transform theTR(TR);
   theTR.Perform(cons, Standard_True);
   fOccShape=theTR.Shape();
   return Reverse(fOccShape);
}

TopoDS_Shape TGeoToOCC::OCC_Cuttub(Double_t rmin, Double_t rmax, Double_t dz,
                           Double_t, Double_t Dphi,const Double_t * Nlow,const Double_t * Nhigh)
{
   out.open("/tmp/TGeoCad.log",ios::app);
   out<<"siamo in ctube"<<rmin<<" "<<rmax<<" "<<Dphi<<" "<<dz<<" "<<Nlow[0]<<" "<<Nlow[1]<<" "<<Nlow[2]<<" "<<Nhigh[0]<<" "<<Nhigh[1]<<" "<<Nhigh[2]<<endl;
   Double_t nlow0=Nlow[0];
   Double_t nlow1=Nlow[1];
   Double_t nhigh0=Nhigh[0];
   Double_t nhigh1=Nhigh[1];
   TopoDS_Shape tubs;
   TopoDS_Shell sH, sL;
   ShapeFix_ShapeTolerance FTol;
   Double_t tolerance=1;

   gp_Trsf TT;
   gp_Trsf TR;
   if (rmin==0) rmin=rmin+0.000001;
   if (rmax==0) rmax=rmax+0.000001;
   TopoDS_Solid rminCyl= BRepPrimAPI_MakeCylinder(rmin,2*dz,Dphi);
   TopoDS_Solid rmaxCyl = BRepPrimAPI_MakeCylinder(rmax,2*dz,Dphi);
   BRepAlgoAPI_Cut cutResult(rmaxCyl, rminCyl);
   cutResult.Build();
   tubs=cutResult.Shape();
   TopExp_Explorer anExp2 (tubs, TopAbs_SOLID);
   if (anExp2.More()) {
      TopoDS_Shape aTmpShape = anExp2.Current();
      tubs = TopoDS::Solid (aTmpShape);
   }
   /*TT.SetRotation(gp_Ax1(gp_Pnt(0.,0.,0.), gp_Vec(0., 0., 1.)),phi1);
   BRepBuilderAPI_Transform theTT(TT);
   theTT.Perform(tubs, Standard_True);
   tubs=theTT.Shape();*/
   TR.SetTranslation(gp_Vec(0,0,-dz));
   BRepBuilderAPI_Transform theTR(TR);
   theTR.Perform(tubs, Standard_True);
   tubs=theTR.Shape();
   if ((Nhigh[0]>-1e-4)||(Nhigh[0]<1e-4)) nhigh0=0;
   if ((Nhigh[1]>-1e-4)||(Nhigh[1]<1e-4)) nhigh1=0;
   if ((Nlow[0]>-1e-4)||(Nlow[0]<1e-4)) nlow0=0;
   if ((Nlow[1]>-1e-4)||(Nlow[1]<1e-4)) nlow1=0;
   Handle(Geom_Plane) pH = new Geom_Plane (gp_Pnt(0,0,dz), gp_Dir(nhigh0,nhigh1,Nhigh[2]));
   Handle(Geom_Plane) pL = new Geom_Plane (gp_Pnt(0,0,-dz), gp_Dir(nlow0,nlow1,Nlow[2]));

  /* gp_Dir D(nhigh0,nhigh1,Nhigh[2]);
   gp_Pnt P(0,0,dz);
   gp_Pln Plan(P,D);
   TopoDS_Face maLame = BRepBuilderAPI_MakeFace(Plan);
   if (maLame.IsNull()) cout<<"null face"<<endl;
   TopoDS_Shape Inter = BRepAlgo_Section(tubs, maLame);
   FTol.SetTolerance(maLame, tolerance ,TopAbs_FACE);
   BRepAlgoAPI_Cut Result(tubs, Inter);*/

   /*gp_Dir D2(nlow0,nlow1,Nlow[2]);
   gp_Pnt P2(0,0,-dz);
   gp_Pln Plan2(P,D);
   TopoDS_Face maLame2 = BRepBuilderAPI_MakeFace(Plan2);
   TopoDS_Shape Inter2 = BRepAlgo_Section(tubs, maLame2);
   BRepAlgoAPI_Fuse Result(Inter, Inter2);
   Result.Build();
   */
   BRepBuilderAPI_MakeShell shell(pH);
   if (shell.IsDone())
      sH=shell.Shell();
   else
      out<<"error shell 1"<<shell.Error()<<endl;
   BRepBuilderAPI_MakeShell shell2 (pL);
   if (shell2.IsDone())
      sL=shell2.Shell();
   else
      out<<"error shell 2"<<shell2.Error()<<endl;

   FTol.SetTolerance(sH, tolerance ,TopAbs_SHELL);
   FTol.SetTolerance(sL, tolerance ,TopAbs_SHELL);

   BRepBuilderAPI_MakeSolid solid (sH, sL);
   solid.Build();
   TopoDS_Solid cut=solid.Solid();
   FTol.SetTolerance(cut, tolerance ,TopAbs_SOLID);
   BRepBuilderAPI_MakeSolid(sL, sH);
   if (!solid.IsDone())
   out<<"error solid"<<endl;

   //BRepAlgoAPI_Cut Result(tubs, cut);
   BRepAlgoAPI_Cut Result(tubs, cut);
   Result.Build();
   out<<"dopo la seconda cut"<<Result.ErrorStatus()<<endl;
  /* TopoDS_Solid newSolid;
   TopExp_Explorer anExp1 (tubs, TopAbs_SOLID);
   if (anExp1.More()) {
      TopoDS_Shape aTmpShape = anExp1.Current();
      newSolid = TopoDS::Solid (aTmpShape);
   }*/
  return Reverse(Result.Shape());
   //  return maLame;
}


TopoDS_Shape TGeoToOCC::OCC_Xtru(TGeoXtru * TG_Xtru)
{
   Int_t vert=TG_Xtru->GetNvert();
   Int_t nz=TG_Xtru->GetNz();
   Double_t x [vert];
   Double_t y [vert];
   Double_t z [nz];
   gp_Trsf TR;
   TopoDS_Wire w;
   BRepOffsetAPI_ThruSections sect(true,true);
   for (Int_t i=0;i<nz;i++) {
      for (Int_t pp=0;pp<vert;pp++) {
         x[pp]=TG_Xtru->GetXOffset(i)+(TG_Xtru->GetScale(i)*TG_Xtru->GetX(pp));
         y[pp]=TG_Xtru->GetYOffset(i)+(TG_Xtru->GetScale(i)*TG_Xtru->GetY(pp));
      }
      z[i]=TG_Xtru->GetZ(i);
      w=TGeoToOCC::Polygon(x,y,z[i],vert);
      sect.AddWire(w);
   }
   sect.Build();
   if (sect.IsDone()) fOccShape = sect.Shape();
   return fOccShape;
}


TopoDS_Shape TGeoToOCC::OCC_Hype(Double_t rmin, Double_t  rmax,Double_t  stin, Double_t stout, Double_t  dz )
{
   gp_Pnt p(0, 0, 0);
   gp_Dir d(0, 0, 1);
   TopoDS_Vertex vIn,vOut;
   TopoDS_Vertex vIn1,vOut1;
   TopoDS_Edge hyEO;
   TopoDS_Edge hyEI;
   TopoDS_Edge eT;
   TopoDS_Edge eB;
   TopoDS_Edge eT1;
   TopoDS_Wire hyW;
   TopoDS_Face hyF;
   gp_Trsf t;
   BRepBuilderAPI_MakeEdge makeHyEO;
   BRepBuilderAPI_MakeEdge makeHyEI;
   Double_t xO,xI;
   if(stout>0) {
      Double_t aO = rmax;
      Double_t bO = (rmax/(tan(stout)));
      xO = aO*sqrt(1+(dz*dz)/(bO*bO));
      gp_Hypr hyO( gp_Ax2 (p, d ), aO, bO);
      vOut = BRepBuilderAPI_MakeVertex(gp_Pnt(xO,dz,0));
      vOut1 = BRepBuilderAPI_MakeVertex(gp_Pnt(xO,-dz,0));
      makeHyEO=BRepBuilderAPI_MakeEdge(hyO,vOut,vOut1);
   }
   else
      makeHyEO=BRepBuilderAPI_MakeEdge(gp_Pnt(rmax,-dz,0), gp_Pnt(rmax,dz,0));
   if(stin>0) {
      Double_t aI = rmin;
      Double_t bI = (rmin/(tan(stin)));
      xI = aI*sqrt(1+(dz*dz)/(bI*bI));
      gp_Hypr hyI( gp_Ax2 (p, d ), aI, bI);
      vIn = BRepBuilderAPI_MakeVertex(gp_Pnt(xI,dz,0));
      vIn1  = BRepBuilderAPI_MakeVertex(gp_Pnt(xI,-dz,0));
      makeHyEI=BRepBuilderAPI_MakeEdge(hyI,vIn,vIn1);
    }
    else
       makeHyEI=BRepBuilderAPI_MakeEdge(gp_Pnt(rmin,-dz,0), gp_Pnt(rmin,dz,0));
   hyEO=TopoDS::Edge(makeHyEO.Shape());
   hyEI=TopoDS::Edge(makeHyEI.Shape());
   eT= BRepBuilderAPI_MakeEdge(makeHyEO.Vertex1(), makeHyEI.Vertex1());
   eB= BRepBuilderAPI_MakeEdge(makeHyEO.Vertex2(), makeHyEI.Vertex2());
   eT1 =BRepBuilderAPI_MakeEdge(makeHyEO.Vertex1(), makeHyEO.Vertex2());
   BRepBuilderAPI_MakeWire WIRE(hyEO,eB,hyEI,eT);
   WIRE.Add(eT1);
   hyW=WIRE.Wire();
   BRepBuilderAPI_MakeFace face(hyW);
   hyF=face.Face();
   t.SetRotation(gp::OX(), M_PI/2.);
   BRepBuilderAPI_Transform TF(t);
   TF.Perform(hyF,Standard_True);
   hyF = TopoDS::Face(TF.Shape());
   fOccShape = BRepPrimAPI_MakeRevol (hyF,gp::OZ(),2*M_PI);
   return  Reverse(fOccShape);
}

TopoDS_Shape TGeoToOCC::OCC_ParaTrap (Double_t *vertex)
{
   BRepOffsetAPI_ThruSections sect(true,true);
   TopoDS_Wire w;
   TopoDS_Face ff;
   //Int_t punti=0;
   Int_t f=0;
   TopoDS_Edge e1;
   TopoDS_Edge e2;
   TopoDS_Edge e3;
   TopoDS_Edge e4;
   gp_Pnt p1;
   gp_Pnt p2;
   gp_Pnt p3;
   gp_Pnt p4;

   while (f<24) {
      p1=gp_Pnt(vertex[f],vertex[f+1],vertex[f+2]);
      p2=gp_Pnt(vertex[f+3],vertex[f+4],vertex[f+5]);
      p3=gp_Pnt(vertex[f+6],vertex[f+7],vertex[f+8]);
      p4=gp_Pnt(vertex[f+9],vertex[f+10],vertex[f+11]);
      e1=BRepBuilderAPI_MakeEdge(p1,p2 );
      e2=BRepBuilderAPI_MakeEdge(p2,p3 );
      e3=BRepBuilderAPI_MakeEdge(p3,p4 );
      e4=BRepBuilderAPI_MakeEdge(p4,p1 );
      w = BRepBuilderAPI_MakeWire(e1,e2,e3,e4);
      sect.AddWire(w);
      f += 12;
   }
   sect.Build();
   fOccShape=sect.Shape();
   return fOccShape;
}


TopoDS_Shape TGeoToOCC::OCC_Arb8(Double_t, Double_t* , Double_t *points)
{
   out.open("/tmp/TGeoCad.log",ios::app);
   TopoDS_Shell newShell;
   TopoDS_Shape sewedShape;
   TopoDS_Shape aTmpShape;
   Int_t count=0;
   ShapeFix_ShapeTolerance FTol;
   Double_t tolerance=1;
   Handle(TColgp_HArray1OfPnt) pathArray =new TColgp_HArray1OfPnt(0,8);
   BRepBuilderAPI_Sewing sew(1.0);//e-02);
   TopoDS_Wire wire1,wire2,wire3,wire4,wire5,wire6;
   TopoDS_Face ff,ff1,ff2,ff3,ff4,ff5;
   BRepBuilderAPI_MakePolygon poly1,poly2,poly3,poly4,poly5,poly6;
   Int_t x=0,y=0,z=0;
   gp_Pnt point;
   for (Int_t i=0;i<8;i++) {

      x=count++;y=count++;z=count++;
      point=gp_Pnt(points[x],points[y],points[z]);
      if (points[x]<=0.1) { tolerance=1;}
      if (points[y]<=0.1) { tolerance=1;}
      if (points[z]<=0.1) { tolerance=1;}
      pathArray->SetValue(i,point);
   }
   poly1.Add(pathArray->Value(0));
   out<<pathArray->Value(0).X()<<" "<<pathArray->Value(0).Y()<<" "<<pathArray->Value(0).Z()<<endl;
   poly1.Add(pathArray->Value(3));
   out<<pathArray->Value(3).X()<<pathArray->Value(3).Y()<<pathArray->Value(3).Z()<<endl;
   poly1.Add(pathArray->Value(2));
   out<<pathArray->Value(2).X()<<pathArray->Value(2).Y()<<pathArray->Value(2).Z()<<endl;
   poly1.Add(pathArray->Value(1));
   out<<pathArray->Value(1).X()<<" "<<pathArray->Value(1).Y()<<" "<<pathArray->Value(1).Z()<<endl;
   poly1.Close();
   wire1=poly1.Wire();

   poly2.Add(pathArray->Value(0));
   out<<pathArray->Value(0).X()<<pathArray->Value(0).Y()<<pathArray->Value(0).Z()<<endl;
   poly2.Add(pathArray->Value(1));
   out<<pathArray->Value(1).X()<<pathArray->Value(1).Y()<<pathArray->Value(1).Z()<<endl;
   poly2.Add(pathArray->Value(5));
   out<<pathArray->Value(5).X()<<pathArray->Value(5).Y()<<pathArray->Value(5).Z()<<endl;
   poly2.Add(pathArray->Value(4));
   out<<pathArray->Value(4).X()<<pathArray->Value(4).Y()<<pathArray->Value(4).Z()<<endl;
   poly2.Close();
   wire2=poly2.Wire();
   poly3.Add(pathArray->Value(0));
   out<<pathArray->Value(0).X()<<pathArray->Value(0).Y()<<pathArray->Value(0).Z()<<endl;
   poly3.Add(pathArray->Value(4));
   out<<pathArray->Value(4).X()<<pathArray->Value(4).Y()<<pathArray->Value(4).Z()<<endl;
   poly3.Add(pathArray->Value(7));
   out<<pathArray->Value(7).X()<<pathArray->Value(7).Y()<<pathArray->Value(7).Z()<<endl;
   poly3.Add(pathArray->Value(3));
   out<<pathArray->Value(3).X()<<pathArray->Value(3).Y()<<pathArray->Value(3).Z()<<endl;
   poly3.Close();
   wire3=poly3.Wire();
   poly4.Add(pathArray->Value(3));
   out<<pathArray->Value(3).X()<<pathArray->Value(3).Y()<<pathArray->Value(3).Z()<<endl;
   poly4.Add(pathArray->Value(2));
   out<<pathArray->Value(2).X()<<pathArray->Value(2).Y()<<pathArray->Value(2).Z()<<endl;
   poly4.Add(pathArray->Value(6));
   out<<pathArray->Value(6).X()<<pathArray->Value(6).Y()<<pathArray->Value(6).Z()<<endl;
   poly4.Add(pathArray->Value(7));
   out<<pathArray->Value(7).X()<<pathArray->Value(7).Y()<<pathArray->Value(7).Z()<<endl;
   poly4.Close();
   wire4=poly4.Wire();
   poly5.Add(pathArray->Value(4));
   out<<pathArray->Value(4).X()<<pathArray->Value(4).Y()<<pathArray->Value(4).Z()<<endl;
   poly5.Add(pathArray->Value(5));
   out<<pathArray->Value(5).X()<<pathArray->Value(5).Y()<<pathArray->Value(5).Z()<<endl;
   poly5.Add(pathArray->Value(6));
   out<<pathArray->Value(6).X()<<pathArray->Value(6).Y()<<pathArray->Value(6).Z()<<endl;
   poly5.Add(pathArray->Value(7));
   out<<pathArray->Value(7).X()<<pathArray->Value(7).Y()<<pathArray->Value(7).Z()<<endl;
   poly5.Close();
   wire5=poly5.Wire();
   poly6.Add(pathArray->Value(1));
   out<<pathArray->Value(1).X()<<pathArray->Value(1).Y()<<pathArray->Value(1).Z()<<endl;
   poly6.Add(pathArray->Value(2));
   out<<pathArray->Value(2).X()<<pathArray->Value(2).Y()<<pathArray->Value(2).Z()<<endl;
   poly6.Add(pathArray->Value(6));
   out<<pathArray->Value(6).X()<<pathArray->Value(6).Y()<<pathArray->Value(6).Z()<<endl;
   poly6.Add(pathArray->Value(5));
   out<<pathArray->Value(5).X()<<pathArray->Value(5).Y()<<pathArray->Value(5).Z()<<endl;

   poly6.Close();
   wire6=poly6.Wire();

   FTol.SetTolerance(wire1, tolerance ,TopAbs_WIRE);
   FTol.SetTolerance(wire2, tolerance ,TopAbs_WIRE);
   FTol.SetTolerance(wire3, tolerance ,TopAbs_WIRE);
   FTol.SetTolerance(wire4, tolerance ,TopAbs_WIRE);
   FTol.SetTolerance(wire5, tolerance ,TopAbs_WIRE);
   FTol.SetTolerance(wire6, tolerance ,TopAbs_WIRE);

   ff  = BRepBuilderAPI_MakeFace(wire1);
   if (ff.IsNull()) out<<"face1 is null"<<endl;
   ff1 = BRepBuilderAPI_MakeFace(wire2);
   if (ff1.IsNull()) out<<"face2 is null"<<endl;
   ff2 = BRepBuilderAPI_MakeFace(wire3);
   if (ff2.IsNull()) out<<"face3 is null"<<endl;
   ff3 = BRepBuilderAPI_MakeFace(wire4);
   if (ff3.IsNull()) out<<"face4 is null"<<endl;
   ff4 = BRepBuilderAPI_MakeFace(wire5);
   if (ff4.IsNull()) out<<"face5 is null"<<endl;
   ff5 = BRepBuilderAPI_MakeFace(wire6);
   if (ff5.IsNull()) out<<"face6 is null"<<endl;
   sew.Add(ff);
   sew.Add(ff1);
   sew.Add(ff2);
   sew.Add(ff3);
   sew.Add(ff4);
   sew.Add(ff5);

   sew.Perform();
   sewedShape=sew.SewedShape();

   if (sewedShape.IsNull()) out<<"Arb8 error"<<endl;

   TopExp_Explorer anExp (sewedShape, TopAbs_SHELL);
   if (anExp.More()) {
      aTmpShape = anExp.Current();
      newShell = TopoDS::Shell (aTmpShape);
   }
   BRepBuilderAPI_MakeSolid mySolid(newShell);
   out.close();
   return Reverse(mySolid.Solid());
}



TopoDS_Shape TGeoToOCC::OCC_Box(Double_t dx, Double_t dy, Double_t dz, Double_t OX, Double_t OY, Double_t OZ )
{
   TopoDS_Solid box;
   if (dz==0)dz=0.1;
   if (dy==0)dy=0.1;if (dx==0)dx=0.1;
   box = BRepPrimAPI_MakeBox( gp_Pnt(OX-dx, OY-dy, OZ-dz), dx*2, dy*2, dz*2);
   return Reverse(box);
}


TopoDS_Shape TGeoToOCC::OCC_Trd(Double_t dx1, Double_t dx2, Double_t dy1, Double_t dy2, Double_t dz)
{
   TopoDS_Wire wire;
   BRepOffsetAPI_ThruSections sect(true,true);

   TopoDS_Edge edge1;
   TopoDS_Edge edge2;
   TopoDS_Edge edge3;
   TopoDS_Edge edge4;
   gp_Pnt point1;
   gp_Pnt point2;
   gp_Pnt point3;
   gp_Pnt point4;
   for (Int_t i=0;i<2;i++) {
      BRepBuilderAPI_MakePolygon poly;
      if (i==1) {
         dx1=dx2;
         dy1=dy2;
         dz=-dz;
      }
      point1=gp_Pnt(-dx1,-dy1,-dz);
      point2=gp_Pnt(dx1,-dy1,-dz);
      point3=gp_Pnt(dx1,dy1,-dz);
      point4=gp_Pnt(-dx1,dy1,-dz);
      poly.Add(point1);
      poly.Add(point2);
      poly.Add(point3);
      poly.Add(point4);
      poly.Close();
      wire=poly.Wire();
      sect.AddWire(wire);
      poly.Delete();
   }

   sect.Build();
   fOccShape=sect.Shape();

   return fOccShape;
}

TopoDS_Wire TGeoToOCC::Polygon(Double_t *x, Double_t *y, Double_t z, Int_t num )
{
   BRepBuilderAPI_MakePolygon poly;
   TopoDS_Wire w ;
   Int_t i;
   for(i=0; i<num; i++) {
      poly.Add(gp_Pnt(x[i], y[i],z));
   }
   poly.Add(gp_Pnt(x[0], y[0], z));
   poly.Close();
   w=poly.Wire();
   return w;
}


TopoDS_Shape TGeoToOCC::OCC_Pcon(Double_t startPhi, Double_t deltaPhi,
                          Int_t zNum, Double_t *rMin, Double_t *rMax, Double_t *z)
{

   TopoDS_Shape pCone;
   TopoDS_Shape cone;
   Double_t zHalf=0.0;
   gp_Trsf Transl;
   gp_Trsf Transf;
   for(Int_t nCon=0; nCon<zNum-1; nCon++) {
      zHalf = (z[nCon+1]-z[nCon])/2.;
      if ((zHalf==0)||(zHalf<0)) zHalf=0.1;
      cone = OCC_Cones(rMin[nCon], rMax[nCon], rMin[(nCon+1)], rMax[(nCon+1)],zHalf, startPhi, deltaPhi);
      Double_t r[] = {1,0,0,0,1,0,0,0,1};
      Double_t t[] = {0,0,zHalf+z[nCon]};
      Transl.SetTranslation(gp_Vec(t[0],t[1],t[2]));
      Transf.SetValues(r[0],r[1],r[2],0,
                       r[3],r[4],r[5],0,
                       r[6],r[7],r[8],0
#if OCC_VERSION_MAJOR == 6 && OCC_VERSION_MINOR < 8
                       ,0,1
#endif
                       );
      BRepBuilderAPI_Transform Transformation(Transf);
      BRepBuilderAPI_Transform Translation(Transl);
      Transformation.Perform(cone,true);
      cone = Transformation.Shape();
      Translation.Perform(cone, Standard_True);
      cone = Translation.Shape();
      if(nCon>0) {
         BRepAlgoAPI_Fuse fuse(pCone, cone);
         pCone=fuse.Shape();
      } else
         pCone=cone;
   }
   return Reverse(pCone);
}


TopoDS_Shape TGeoToOCC::OCC_Pgon(Int_t, Int_t nz, Double_t * p, Double_t phi1, Double_t DPhi, Int_t numpoint)
{
   BRepOffsetAPI_ThruSections sectInner(true,true);
   BRepOffsetAPI_ThruSections sectOuter(true,true);
   BRepLib_MakePolygon aPoly2;
   TopoDS_Face  f;
   TopoDS_Wire w1,w2;
   TopoDS_Solid myCut;
   Int_t i=2;
   Double_t z=p[2];
   Int_t nzvert=0;
   Double_t xx=0.0,yy=0.0,zz=0.0;
   Double_t Xmax=0.0,Ymax=0.0, Zmax=0.0, max=0.0;
   Int_t aa=0,bb=1,cc=2;
   Int_t ind=0;
   Int_t check=0;
   //Int_t k=0;
   gp_Pnt point;
   gp_Trsf TR;
   gp_Trsf TT;
   while (i<numpoint){
      if (IsEqual(p[i],z))
      nzvert=nzvert+1;
      i=i+3;
   }
   nzvert=nzvert/2;
   for(Int_t c=0;c<numpoint;c++){
      if ((p[check]>-1e-4)&&(p[check]<1e-4))
         p[check]=0;
      check=check+1;
   }
   for (i=0;i<nz;i++) {
      for(Int_t j=0; j<2; j++) {
         BRepLib_MakePolygon aPoly;
         for (Int_t h=0;h<nzvert;h++){
            xx=p[ind++];yy=p[ind++];zz=p[ind++];
            point=gp_Pnt(xx,yy,zz);
            aPoly.Add(point);
         }
         aPoly.Close();
         if (j==0) {
            w1  = aPoly.Wire();
            sectInner.AddWire(w1);
         }
         if (j==1) {
            w2  = aPoly.Wire();
            sectOuter.AddWire(w2);
         }
      aPoly.Delete();
      }
   }
   sectInner.Build();
   sectOuter.Build();

   BRepAlgoAPI_Cut Result(sectOuter.Shape(),sectInner.Shape() );
   Result.Build();

   for (Int_t e=0;e<numpoint;e++){
      if (fabs(p[aa])>Xmax) {
         Xmax=fabs(p[aa]);
      }
      if (fabs(p[bb])>Ymax) {
         Ymax=fabs(p[bb]);
      }
      if (fabs(p[cc])>Zmax) {
         Zmax=fabs(p[cc]);
      }
      if(numpoint-1==cc) break;
      aa=aa+3;
      bb=bb+3;
      cc=cc+3;
   }
   if (Xmax>Ymax)
      max=Xmax;
   else
      max=Ymax;
   if ((IsEqual(DPhi,360.0))||(IsEqual(DPhi,0.))) {
      fOccShape=Result.Shape();
      return Reverse(fOccShape);
   } else {
      myCut=BRepPrimAPI_MakeCylinder (max+1,2*Zmax,(360.-DPhi)*M_PI/180.);
      TT.SetRotation(gp_Ax1(gp_Pnt(0.,0.,0.), gp_Vec(0., 0., 1.)), (-90.0+phi1)*M_PI/180.0);
      BRepBuilderAPI_Transform theTT(TT);
      theTT.Perform(myCut, Standard_True);
      fOccShape=theTT.Shape();
      TR.SetTranslation(gp_Vec(0,0,-Zmax));
      BRepBuilderAPI_Transform theTR(TR);
      theTR.Perform(fOccShape, Standard_True);
      fOccShape=theTR.Shape();
      BRepAlgoAPI_Cut Result2(Result.Shape(),fOccShape );
      Result2.Build();
      fOccShape=Result2.Shape();
      //if (fOccShape.IsNull()) cout<<"The Pgon shae is null. Cut Operation Error: "<<Result2.ErrorStatus()<<endl;
      return Reverse(fOccShape);
   }
}


TopoDS_Shape TGeoToOCC::Reverse(TopoDS_Shape Shape)
{
   BRepClass3d_SolidClassifier * setPrecision= new BRepClass3d_SolidClassifier (Shape);
   setPrecision->PerformInfinitePoint(Precision::Confusion());
   if (setPrecision->State() == TopAbs_IN) {
      //cout<<"reverse"<<endl;
      Shape.Reverse();
   }
   delete(setPrecision);
   return Shape;
}

