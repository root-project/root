// @(#)root/gl:$Name:  $:$Id: TGLScene.h,v 1.5 2005/06/01 12:38:25 brun Exp $
// Author:  Richard Maunder  25/05/2005
// Parts taken from original TGLRender by Timur Pocheptsov

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLScene
#define ROOT_TGLScene

#ifndef ROOT_TGLBoundingBox
#include "TGLBoundingBox.h"
#endif

#include <map>

class TGLCamera;
class TGLDrawable;
class TGLLogicalShape;
class TGLPhysicalShape;

/*************************************************************************
 * TGLScene - TODO
 *
 *
 *
 *************************************************************************/
class TGLScene
{
private:
   // Fields

   // Logical shapes
   typedef std::map<ULong_t, TGLLogicalShape *> LogicalShapeMap_t;
   typedef LogicalShapeMap_t::value_type LogicalShapeMapValueType_t;
   typedef LogicalShapeMap_t::iterator LogicalShapeMapIt_t;
   typedef LogicalShapeMap_t::const_iterator LogicalShapeMapCIt_t;
   LogicalShapeMap_t fLogicalShapes;

   // TODO: Physical may be better as sorted vector - can pre-allocate?
   // Then can sort by size for drawing, with partition() to sort all opaque to front.
   // Look at Meyer's STL Item 23. Logicals will mix lookup + insertions
   // so map ok

   // Physical Shapes
   typedef std::map<ULong_t, TGLPhysicalShape *> PhysicalShapeMap_t;
   typedef PhysicalShapeMap_t::value_type PhysicalShapeMapValueType_t;
   typedef PhysicalShapeMap_t::iterator PhysicalShapeMapIt_t;
   typedef PhysicalShapeMap_t::const_iterator PhysicalShapeMapCIt_t;
   PhysicalShapeMap_t fPhysicalShapes;

   mutable TGLBoundingBox fBoundingBox;      //! bounding box for scene (axis aligned) - lazy update - use BoundingBox() to access
   mutable Bool_t         fBoundingBoxValid; //! bounding box valid?
   mutable UInt_t         fLastDrawLOD;      //! last LOD for the scene draw
   mutable Bool_t         fCanCullLowLOD;    //! cull out low LOD shapes?

   // TODO: vector for multiple selection
   TGLPhysicalShape *     fSelectedPhysical; //! current selected physical shape

   // Methods
   void   DrawNumber(Double_t num, Double_t x, Double_t y, Double_t z, Double_t yorig) const;
   UInt_t CalcPhysicalLOD(const TGLPhysicalShape & shape,
                          const TGLCamera & camera,
                          UInt_t sceneLOD) const;

   // Non-copyable class
   TGLScene(const TGLScene &);
   TGLScene & operator=(const TGLScene &);

public:
   enum EDrawMode{kFill, kOutline, kWireFrame};

   void SetDrawMode(EDrawMode mode){fDrawMode = mode;}

private:
   EDrawMode      fDrawMode;  


public:
   TGLScene();
   virtual ~TGLScene(); // ClassDef introduces virtual fns

   const TGLBoundingBox & BoundingBox() const;
   void                   Draw(const TGLCamera & camera, UInt_t sceneLOD, Double_t timeout = 0.0) const;
   void                   DrawAxes() const;
   Bool_t                 Select(const TGLCamera & camera);
   TGLPhysicalShape *     GetSelected() const { return fSelectedPhysical; }
   void                   SelectedModified();

   // Logical Shape Management
   void              AdoptLogical(TGLLogicalShape & shape);
   Bool_t            DestroyLogical(ULong_t ID);
   UInt_t            DestroyAllLogicals();
   void              PurgeNextLogical() {};
   TGLLogicalShape * FindLogical(ULong_t ID)  const;

   // Physical Shape Management
   void               AdoptPhysical(TGLPhysicalShape & shape);
   Bool_t             DestroyPhysical(ULong_t ID);
   UInt_t             DestroyPhysicals(const TGLCamera & camera);
   UInt_t             DestroyAllPhysicals();
   TGLPhysicalShape * FindPhysical(ULong_t ID) const;

   // Set color on all physicals using a logical
   void               SetColorByLogical(ULong_t logicalID, const Float_t rgba[4]);

   void Dump() const;
   UInt_t SizeOf() const;

   ClassDef(TGLScene,0) // a GL scene - collection of physical and logical shapes
};

#endif // ROOT_TGLScene
