// @(#)root/gl:$Name:  $:$Id: TGLScene.h,v 1.28 2007/06/11 19:56:33 brun Exp $
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLScene_H
#define ROOT_TGLScene_H

#include "TGLSceneBase.h"
#include "TGLSceneInfo.h"

#include "Gtypes.h"

#include <map>
#include <vector>

class TGLCamera;
class TGLLogicalShape;
class TGLPhysicalShape;


class TGLScene : public TGLSceneBase
{
private:
   TGLScene(const TGLScene&);            // Not implemented
   TGLScene& operator=(const TGLScene&); // Not implemented

protected:
   // Logical shapes
   typedef std::map<TObject*, TGLLogicalShape *>   LogicalShapeMap_t;
   typedef LogicalShapeMap_t::value_type           LogicalShapeMapValueType_t;
   typedef LogicalShapeMap_t::iterator             LogicalShapeMapIt_t;
   typedef LogicalShapeMap_t::const_iterator       LogicalShapeMapCIt_t;
   LogicalShapeMap_t                               fLogicalShapes; //!

   // Physical Shapes
   typedef std::map<UInt_t, TGLPhysicalShape *>    PhysicalShapeMap_t;
   typedef PhysicalShapeMap_t::value_type          PhysicalShapeMapValueType_t;
   typedef PhysicalShapeMap_t::iterator            PhysicalShapeMapIt_t;
   typedef PhysicalShapeMap_t::const_iterator      PhysicalShapeMapCIt_t;
   PhysicalShapeMap_t                              fPhysicalShapes; //!

   virtual void DestroyPhysicalInternal(PhysicalShapeMapIt_t pit);

   // ----------------------------------------------------------------
   // List of physical shapes ordered by volume/diagonal
   typedef std::vector<const TGLPhysicalShape *>   ShapeVec_t;
   typedef ShapeVec_t::iterator                    ShapeVec_i;
   ShapeVec_t                                      fDrawList;       //!
   Bool_t                                          fDrawListValid;  //!

   // ----------------------------------------------------------------
   // Draw-list, draw-element

   void   SortDrawList();
   static Bool_t ComparePhysicalVolumes(const TGLPhysicalShape * shape1,
                                        const TGLPhysicalShape * shape2);


public:
   struct DrawElement_t
   {
      const TGLPhysicalShape* fPhysical; // Physical shape.

      Float_t    fPixelSize; // Size of largest lod-axis in pixels.
      Short_t    fPixelLOD;  // Size in LOD units.
      Short_t    fFinalLOD;  // Corrected with SceneLOD and quantized.

      DrawElement_t() : fPhysical(0), fPixelSize(0), fPixelLOD(0), fFinalLOD(0) {}
   };

   typedef std::vector<DrawElement_t>           DrawElementVec_t;
   typedef std::vector<DrawElement_t>::iterator DrawElementVec_i;

   // ----------------------------------------------------------------
   // SceneInfo ... extended scene context

   class TSceneInfo : public TGLSceneInfo
   {
   public:
      DrawElementVec_t fOpaqueElements;
      DrawElementVec_t fTranspElements;

      TSceneInfo(TGLViewerBase* view=0, TGLScene* scene=0);
      virtual ~TSceneInfo();

      Int_t                     fOpaqueCnt;
      Int_t                     fTranspCnt;
      Int_t                     fAsPixelCnt;
      std::map<TClass*, UInt_t> fByShapeCnt;

      void ResetDrawStats();
      void UpdateDrawStats(const TGLPhysicalShape & shape, Short_t lod);
      void DumpDrawStats(); // Debug
   };
   friend class TSceneInfo; // for solaris cc

protected:

   // Smart Refresh -- will go in this version
   Bool_t                                          fInSmartRefresh;    //!
   mutable LogicalShapeMap_t                       fSmartRefreshCache; //!


   // ----------------------------------------------------------------
   // ----------------------------------------------------------------

public:
   TGLScene();
   virtual ~TGLScene();

   virtual void CalcBoundingBox() const;

   virtual TSceneInfo* CreateSceneInfo(TGLViewerBase* view);
   virtual void        UpdateSceneInfo(TGLRnrCtx& ctx);
   virtual void        LodifySceneInfo(TGLRnrCtx& ctx);


   // Rendering
   virtual void PreRender (TGLRnrCtx & rnrCtx);
   virtual void Render    (TGLRnrCtx & rnrCtx);
   virtual void PostRender(TGLRnrCtx & rnrCtx);

   virtual Double_t RenderAllPasses(TGLRnrCtx           & rnrCtx,
                                    Double_t              timeout);

   virtual Double_t RenderOnePass  (TGLRnrCtx           & rnrCtx,
                                    Double_t              timeout,
                                    const TGLPlaneSet_t * clipPlanes = 0);

   virtual Double_t RenderElements (TGLRnrCtx           & rnrCtx,
                                    DrawElementVec_t    & elementVec,
                                    Double_t              timeout,
                                    const TGLPlaneSet_t * clipPlanes = 0);

   // Selection
   virtual Bool_t ResolveSelectRecord(TGLSelectRecord& rec, Int_t curIdx);

   // Basic logical shape management
   virtual void              AdoptLogical(TGLLogicalShape & shape);
   virtual Bool_t            DestroyLogical(TObject* logid);
   virtual Int_t             DestroyLogicals();
   virtual TGLLogicalShape*  FindLogical(TObject* logid)  const;

   // Basic physical shape management
   virtual void              AdoptPhysical(TGLPhysicalShape & shape);
   virtual Bool_t            DestroyPhysical(UInt_t phid);
   virtual Int_t             DestroyPhysicals(Bool_t incModified, const TGLCamera* camera=0);
   virtual TGLPhysicalShape* FindPhysical(UInt_t phid) const;

   virtual UInt_t            GetMaxPhysicalID();

   // ----------------------------------------------------------------
   // Updates / removals of logical and physical shapes

   virtual void BeginUpdate();
   virtual void EndUpdate();

   virtual void UpdateLogical(TObject* logid);

   virtual void UpdatePhysical(UInt_t phid, Double_t* trans, UChar_t* col);
   virtual void UpdatePhysical(UInt_t phid, Double_t* trans, Color_t cidx=-1, UChar_t transp=0);

   virtual void UpdatePhysioLogical(TObject* logid, Double_t* trans, UChar_t* col);
   virtual void UpdatePhysioLogical(TObject* logid, Double_t* trans, Color_t cidx=-1, UChar_t transp=0);


   // ----------------------------------------------------------------
   // SmartRefresh

   void              BeginSmartRefresh();
   void              EndSmartRefresh();
   TGLLogicalShape*  FindLogicalSmartRefresh(TObject* ID) const;


   // ----------------------------------------------------------------
   // Helpers

   UInt_t SizeOfScene() const;
   void   DumpMapSizes() const;

   static void RGBAFromColorIdx(Float_t rgba[4], Color_t ci, Char_t transp=0);

   static Bool_t IsOutside(const TGLBoundingBox & box,
                           const TGLPlaneSet_t & planes);

   ClassDef(TGLScene, 0) // Standard ROOT OpenGL scene with logial/physical shapes.
}; // endclass TGLScene


#endif
