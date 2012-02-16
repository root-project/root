// @(#)root/gl:$Id$
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

class TGLObject;
class TGLCamera;
class TGLLogicalShape;
class TGLPhysicalShape;

class TGLContextIdentity;

class TGLScene : public TGLSceneBase
{
private:
   TGLScene(const TGLScene&);            // Not implemented
   TGLScene& operator=(const TGLScene&); // Not implemented

   // Compare physical-shape volumes/diagonals -- for draw-vec sorting
   static Bool_t ComparePhysicalVolumes(const TGLPhysicalShape* shape1,
                                        const TGLPhysicalShape* shape2);
   static Bool_t ComparePhysicalDiagonals(const TGLPhysicalShape* shape1,
                                          const TGLPhysicalShape* shape2);
public:
   // Logical shapes
   typedef std::map<TObject*, TGLLogicalShape*>    LogicalShapeMap_t;
   typedef LogicalShapeMap_t::value_type           LogicalShapeMapValueType_t;
   typedef LogicalShapeMap_t::iterator             LogicalShapeMapIt_t;
   typedef LogicalShapeMap_t::const_iterator       LogicalShapeMapCIt_t;

   // Physical Shapes
   typedef std::map<UInt_t, TGLPhysicalShape*>     PhysicalShapeMap_t;
   typedef PhysicalShapeMap_t::value_type          PhysicalShapeMapValueType_t;
   typedef PhysicalShapeMap_t::iterator            PhysicalShapeMapIt_t;
   typedef PhysicalShapeMap_t::const_iterator      PhysicalShapeMapCIt_t;


   struct DrawElement_t
   {
      const TGLPhysicalShape* fPhysical; // Physical shape.

      Float_t    fPixelSize; // Size of largest lod-axis in pixels.
      Short_t    fPixelLOD;  // Size in LOD units.
      Short_t    fFinalLOD;  // Corrected with SceneLOD and quantized.

      DrawElement_t(const TGLPhysicalShape* pshp=0) :
         fPhysical(pshp), fPixelSize(0), fPixelLOD(0), fFinalLOD(0) {}
   };

   typedef std::vector<DrawElement_t>              DrawElementVec_t;
   typedef std::vector<DrawElement_t>::iterator    DrawElementVec_i;

   typedef std::vector<DrawElement_t*>             DrawElementPtrVec_t;
   typedef std::vector<DrawElement_t*>::iterator   DrawElementPtrVec_i;

   // List of physical shapes ordered by volume/diagonal
   typedef std::vector<const TGLPhysicalShape*>    ShapeVec_t;
   typedef ShapeVec_t::iterator                    ShapeVec_i;

   // ----------------------------------------------------------------
   // SceneInfo ... extended scene context

   class TSceneInfo : public TGLSceneInfo
   {
   private:
      Bool_t CmpDrawElements(const DrawElement_t& de1, const DrawElement_t& de2);

   protected:
      void ClearDrawElementVec(DrawElementVec_t& vec, Int_t maxSize);
      void ClearDrawElementPtrVec(DrawElementPtrVec_t& vec, Int_t maxSize);

   public:
      ShapeVec_t          fShapesOfInterest;

      DrawElementVec_t    fVisibleElements;

      UInt_t              fMinorStamp;
      DrawElementPtrVec_t fOpaqueElements;
      DrawElementPtrVec_t fTranspElements;
      DrawElementPtrVec_t fSelOpaqueElements;
      DrawElementPtrVec_t fSelTranspElements;

      TSceneInfo(TGLViewerBase* view=0, TGLScene* scene=0);
      virtual ~TSceneInfo();

      void ClearAfterRebuild();
      void ClearAfterUpdate();

      void Lodify(TGLRnrCtx& ctx);

      void PreDraw();
      void PostDraw();

      // ---------------
      // Draw statistics

      Int_t                     fOpaqueCnt;
      Int_t                     fTranspCnt;
      Int_t                     fAsPixelCnt;
      std::map<TClass*, UInt_t> fByShapeCnt;

      void ResetDrawStats();
      void UpdateDrawStats(const TGLPhysicalShape& shape, Short_t lod);
      void DumpDrawStats(); // Debug
   };
   friend class TSceneInfo; // for solaris cc


protected:
   LogicalShapeMap_t      fLogicalShapes;  //!
   PhysicalShapeMap_t     fPhysicalShapes; //!

   virtual void DestroyPhysicalInternal(PhysicalShapeMapIt_t pit);

   // GLcontext
   TGLContextIdentity * fGLCtxIdentity;
   void ReleaseGLCtxIdentity();

   // Smart Refresh -- will go in this version
   Bool_t                    fInSmartRefresh;    //!
   mutable LogicalShapeMap_t fSmartRefreshCache; //!

   // State that requires recreation of display-lists
   Float_t                   fLastPointSizeScale;
   Float_t                   fLastLineWidthScale;

   // ----------------------------------------------------------------
   // ----------------------------------------------------------------

public:
   TGLScene();
   virtual ~TGLScene();

   virtual void CalcBoundingBox() const;

   virtual TSceneInfo* CreateSceneInfo(TGLViewerBase* view);
   virtual void        RebuildSceneInfo(TGLRnrCtx& rnrCtx);
   virtual void        UpdateSceneInfo(TGLRnrCtx& rnrCtx);
   virtual void        LodifySceneInfo(TGLRnrCtx& rnrCtx);


   // Rendering
   virtual void PreDraw        (TGLRnrCtx& rnrCtx);
   // virtual void PreRender   (TGLRnrCtx& rnrCtx);
   // virtual void Render      (TGLRnrCtx& rnrCtx);
   virtual void RenderOpaque   (TGLRnrCtx& rnrCtx);
   virtual void RenderTransp   (TGLRnrCtx& rnrCtx);
   virtual void RenderSelOpaque(TGLRnrCtx& rnrCtx);
   virtual void RenderSelTransp(TGLRnrCtx& rnrCtx);
   virtual void RenderSelOpaqueForHighlight(TGLRnrCtx& rnrCtx);
   virtual void RenderSelTranspForHighlight(TGLRnrCtx& rnrCtx);

   virtual void RenderHighlight(TGLRnrCtx&           rnrCtx,
                                DrawElementPtrVec_t& elVec);

   // virtual void PostRender(TGLRnrCtx& rnrCtx);
   virtual void PostDraw       (TGLRnrCtx& rnrCtx);

   virtual void RenderAllPasses(TGLRnrCtx&           rnrCtx,
                                DrawElementPtrVec_t& elVec,
                                Bool_t               check_timeout);


   virtual void RenderElements (TGLRnrCtx&           rnrCtx,
                                DrawElementPtrVec_t& elVec,
                                Bool_t               check_timeout,
                                const TGLPlaneSet_t* clipPlanes = 0);

   // Selection
   virtual Bool_t ResolveSelectRecord(TGLSelectRecord& rec, Int_t curIdx);

   // Basic logical shape management
   virtual void              AdoptLogical(TGLLogicalShape& shape);
   virtual Bool_t            DestroyLogical(TObject* logid, Bool_t mustFind=kTRUE);
   virtual Int_t             DestroyLogicals();
   virtual TGLLogicalShape*  FindLogical(TObject* logid)  const;

   // Basic physical shape management
   virtual void              AdoptPhysical(TGLPhysicalShape& shape);
   virtual Bool_t            DestroyPhysical(UInt_t phid);
   virtual Int_t             DestroyPhysicals();
   virtual TGLPhysicalShape* FindPhysical(UInt_t phid) const;

   virtual UInt_t            GetMaxPhysicalID();

   // ----------------------------------------------------------------
   // Updates / removals of logical and physical shapes

   virtual Bool_t BeginUpdate();
   virtual void   EndUpdate(Bool_t minorChange=kTRUE, Bool_t sceneChanged=kTRUE, Bool_t updateViewers=kTRUE);

   virtual void UpdateLogical(TObject* logid);

   virtual void UpdatePhysical(UInt_t phid, Double_t* trans, UChar_t* col);
   virtual void UpdatePhysical(UInt_t phid, Double_t* trans, Color_t cidx=-1, UChar_t transp=0);

   virtual void UpdatePhysioLogical(TObject* logid, Double_t* trans, UChar_t* col);
   virtual void UpdatePhysioLogical(TObject* logid, Double_t* trans, Color_t cidx, UChar_t transp);

   // Temporary export for setting selected-state of physical shapes.
   LogicalShapeMap_t& RefLogicalShapes() { return fLogicalShapes; }


   // ----------------------------------------------------------------
   // SmartRefresh

   UInt_t            BeginSmartRefresh();
   void              EndSmartRefresh();
   TGLLogicalShape*  FindLogicalSmartRefresh(TObject* ID) const;


   // ----------------------------------------------------------------
   // GL-context holding display-list definitions

   TGLContextIdentity* GetGLCtxIdentity() const { return fGLCtxIdentity; }


   // ----------------------------------------------------------------
   // Helpers

   UInt_t SizeOfScene() const;
   void   DumpMapSizes() const;

   static void RGBAFromColorIdx(Float_t rgba[4], Color_t ci, Char_t transp=0);

   static Bool_t IsOutside(const TGLBoundingBox& box,
                           const TGLPlaneSet_t& planes);

   ClassDef(TGLScene, 0); // Standard ROOT OpenGL scene with logial/physical shapes.
};


#endif
