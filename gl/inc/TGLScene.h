// @(#)root/gl:$Name:  $:$Id: TGLScene.h,v 1.26 2006/08/23 14:39:40 brun Exp $
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
#ifndef ROOT_TError
#include "TError.h"
#endif
#ifndef ROOT_TGLDrawFlags
#include "TGLDrawFlags.h"
#endif
#ifndef ROOT_TGLTransManip
#include "TGLTransManip.h"
#endif
#ifndef ROOT_TGLScaleManip
#include "TGLScaleManip.h"
#endif
#ifndef ROOT_TGLRotateManip
#include "TGLRotateManip.h"
#endif

#include <map>
#include <vector>
#include <string>

class TGLCamera;
class TGLDrawable;
class TGLLogicalShape;
class TGLPhysicalShape;
class TGLClip;
class TGLClipPlane;
class TGLClipBox;
class TGLCameraMarkupStyle;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLScene                                                             //
//                                                                      //
// A GL scene is the container for all the viewable objects (shapes)    //
// loaded into the viewer. It consists of two main stl::maps containing //
// the TGLLogicalShape and TGLPhysicalShape collections, and interface  //
// functions enabling viewers to manage objects in these. The physical  //
// shapes defined the placement of copies of the logical shapes - see   //
// TGLLogicalShape/TGLPhysicalShape for more information on relationship//
//                                                                      //
// The scene can be drawn by owning viewer, passing camera, draw style  //
// & quality (LOD), clipping etc - see Draw(). The scene can also be    //
// drawn for selection in similar fashion - see Select(). The scene     //
// keeps track of a single selected physical - which can be modified by //
// viewers.                                                             //
//                                                                      //
// The scene maintains a lazy calculated bounding box for the total     //
// scene extents, axis aligned round TGLPhysicalShape shapes.           //
//                                                                      // 
// Currently a scene is owned exclusively by one viewer - however it is //
// intended that it could easily be shared by multiple viewers - for    //
// efficiency and syncronisation reasons. Hence viewer variant objects  // 
// camera, clips etc being owned by viewer and passed at draw/select    //
//////////////////////////////////////////////////////////////////////////

class TGLScene
{
public:
   enum ELock { kUnlocked,                // Unlocked 
                kDrawLock,                // Locked for draw, cannot select or modify
                kSelectLock,              // Locked for select, cannot modify (draw part of select)
                kModifyLock };            // Locked for modify, cannot draw or select
private:
   // Fields

   // Locking - can take/release via const handle
   mutable ELock                                   fLock; //!

   // Logical shapes
   typedef std::map<ULong_t, TGLLogicalShape *>    LogicalShapeMap_t;
   typedef LogicalShapeMap_t::value_type           LogicalShapeMapValueType_t;
   typedef LogicalShapeMap_t::iterator             LogicalShapeMapIt_t;
   typedef LogicalShapeMap_t::const_iterator       LogicalShapeMapCIt_t;
   LogicalShapeMap_t                               fLogicalShapes; //!

   // Physical Shapes
   typedef std::map<ULong_t, TGLPhysicalShape *>   PhysicalShapeMap_t;
   typedef PhysicalShapeMap_t::value_type          PhysicalShapeMapValueType_t;
   typedef PhysicalShapeMap_t::iterator            PhysicalShapeMapIt_t;
   typedef PhysicalShapeMap_t::const_iterator      PhysicalShapeMapCIt_t;
   PhysicalShapeMap_t                              fPhysicalShapes; //!

   // Draw list of physical shapes
   typedef std::vector<const TGLPhysicalShape *>   DrawList_t;
   typedef DrawList_t::iterator                    DrawListIt_t;
   DrawList_t                                      fDrawList;       //! 
   Bool_t                                          fDrawListValid;  //! (do we need this & fBoundingBoxValid)

   // Smart Refresh
   Bool_t                                          fInSmartRefresh;    //!
   mutable LogicalShapeMap_t                       fSmartRefreshCache; //!


   mutable TGLBoundingBox fBoundingBox;      //! bounding box for scene (axis aligned) - lazy update - use BoundingBox() to access
   mutable Bool_t         fBoundingBoxValid; //! bounding box valid?
//   UInt_t                 fLastDrawLOD;      //! last LOD for the scene draw

   // Selection
   TGLPhysicalShape                         * fSelectedPhysical; //! current selected physical shape
   TGLPhysicalShape                         * fSelectionResult;  //! physical shape found by last Select
   Int_t                                      fNPrimHits;        //! number of primary selection hits
   Int_t                                      fNSecHits;         //! number of secondary selection hits
   Bool_t                                     fTrySecSelect;     //! flag controlling if secondary selection should be tried
   Bool_t                                     fTriedSecSelect;   //! flag reporting if secondary selection was actually tried
   std::vector<UInt_t>                        fSelectBuffer;     //! buffer holding GL selection result
   std::vector<std::pair<UInt_t, UInt_t*> >   fSortedHits;       //! depth-sorted selection result

   // Clipping
   TGLClipPlane         * fClipPlane;
   TGLClipBox           * fClipBox;
   TGLClip              * fCurrentClip;  //! the current clipping shape

   // Guides
   TGLCameraMarkupStyle * fCameraMarkup; //  overlay GL text config

   // Object manipulators - physical + clipping shapes
   TGLTransManip          fTransManip;    //! translation manipulator
   TGLScaleManip          fScaleManip;    //! scaling manipulator
   TGLRotateManip         fRotateManip;   //! rotation manipulator 
   TGLManip             * fCurrentManip;  //! current manipulator
    
   // Draw stats
   struct DrawStats_t {
      TGLDrawFlags fFlags;
      UInt_t fOpaque;
      UInt_t fTrans;
      UInt_t fPixelLOD;
      std::map<std::string, UInt_t> fByShape;
   } fDrawStats;

   // Methods

   // Draw sorting
   void   SortDrawList();
   static Bool_t ComparePhysicalVolumes(const TGLPhysicalShape * shape1, const TGLPhysicalShape * shape2);

   // Internal draw passes - repeated calls for cliping
   void  DrawPass(const TGLCamera & camera, const TGLDrawFlags & sceneFlags, 
                  Double_t timeout, const std::vector<TGLPlane> * clipPlanes = 0);

   void  DrawGuides(const TGLCamera & camera, Int_t axesType, const TGLVertex3 * reference) const;

   // Misc
   void   DrawNumber(Double_t num, const TGLVertex3 & center) const;

   // Draw stats
   void ResetDrawStats(const TGLDrawFlags & flags);
   void UpdateDrawStats(const TGLPhysicalShape & shape, const TGLDrawFlags & flags);
   void DumpDrawStats(); // Debug

   // Non-copyable class
   TGLScene(const TGLScene &);
   TGLScene & operator=(const TGLScene &);

public:
   TGLScene();
   virtual ~TGLScene(); // ClassDef introduces virtual fns

   // Drawing/Selection
   const TGLBoundingBox & BoundingBox() const; 
   void                   Draw(const TGLCamera & camera, TGLDrawFlags sceneFlags, 
                               Double_t timeout, Int_t axesType, const TGLVertex3 * reference,
                               Bool_t forSelect = kFALSE);
   Bool_t                 Select(const TGLCamera & camera, const TGLDrawFlags & sceneFlags);

   // Logical Shape Management
   void                    AdoptLogical(TGLLogicalShape & shape);
   Bool_t                  DestroyLogical(ULong_t ID);
   UInt_t                  DestroyLogicals();
   void                    PurgeNextLogical() {};
   TGLLogicalShape *       FindLogical(ULong_t ID)  const;

   // Physical Shape Management
   void                     AdoptPhysical(TGLPhysicalShape & shape);
   Bool_t                   DestroyPhysical(ULong_t ID);
   UInt_t                   DestroyPhysicals(Bool_t incModified, const TGLCamera * camera = 0);
   TGLPhysicalShape *       FindPhysical(ULong_t ID) const;

   // SmartRefresh
   void                     BeginSmartRefresh();
   void                     EndSmartRefresh();
   TGLLogicalShape *        FindLogicalSmartRefresh(ULong_t ID) const;

   // Selected Object
   TGLPhysicalShape         * GetSelected()         const { return fSelectedPhysical; }
   TGLPhysicalShape         * GetSelectionResult()  const { return fSelectionResult; }
   Int_t                      GetNPrimHits()        const { return fNPrimHits; }
   Int_t                      GetNSecHits()         const { return fNSecHits; }
   Bool_t                     GetTrySecSelect()     const { return fTrySecSelect; }
   void                       ActivateSecSelect()         { fTrySecSelect = kTRUE; }
   Bool_t                     GetTriedSecSelect()   const { return fTriedSecSelect; }
   std::pair<UInt_t, UInt_t*> GetHitRecord(Int_t i) const { return fSortedHits[i]; }

   void                       ApplySelection();

   Bool_t                     SetSelectedColor(const Float_t rgba[17]);
   Bool_t                     SetColorOnSelectedFamily(const Float_t rgba[17]);
   Bool_t                     SetSelectedGeom(const TGLVertex3 & trans, const TGLVector3 & scale);

   // Clipping
   void  SetupClips();
   void  ClearClips();
   void  GetClipState(EClipType type, Double_t data[6]) const;
   void  SetClipState(EClipType type, const Double_t data[6]);
   void  GetCurrentClip(EClipType & type, Bool_t & edit) const;
   void  SetCurrentClip(EClipType type, Bool_t edit);

   // Viewport/camera screen markup
   TGLCameraMarkupStyle* GetCameraMarkup() const  { return fCameraMarkup; }
   void SetCameraMarkup(TGLCameraMarkupStyle*  m) { fCameraMarkup = m; }

   // Manipulators
   void       SetCurrentManip(EManipType type);

   // Interaction - passed on to manipulator
   Bool_t HandleButton(const Event_t & event, const TGLCamera & camera);
   Bool_t HandleMotion(const Event_t & event, const TGLCamera & camera);

   // Locking
   Bool_t TakeLock(ELock lock) const;
   Bool_t ReleaseLock(ELock lock) const;
   Bool_t IsLocked() const;
   ELock  CurrentLock() const;
   static const char * LockName(ELock lock);
   static Bool_t       LockValid(ELock lock); 
   
   // Debug
   void   Dump() const;
   UInt_t SizeOf() const;

   ClassDef(TGLScene,0) // a GL scene - collection of physical and logical shapes
};

//______________________________________________________________________________
inline Bool_t TGLScene::HandleButton(const Event_t & event, const TGLCamera & camera)
{
   return fCurrentManip->HandleButton(event, camera);
}

//______________________________________________________________________________
inline Bool_t TGLScene::HandleMotion(const Event_t & event, const TGLCamera & camera)
{
   if (fCurrentManip->HandleMotion(event, camera, BoundingBox())) {
      // If manip processed event it *may* have modified the selected physical
      // geometry and so invalidated the scene bounding box
      fBoundingBoxValid = kFALSE;
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
inline Bool_t TGLScene::TakeLock(ELock lock) const
{
   if (LockValid(lock) && fLock == kUnlocked) {
      fLock = lock;
      if (gDebug>3) {
         Info("TGLScene::TakeLock", "took %s", LockName(fLock));
      }
      return kTRUE;
   }
   Error("TGLScene::TakeLock", "Unable take %s, already %s", LockName(lock), LockName(fLock));
   return kFALSE;
}

//______________________________________________________________________________
inline Bool_t TGLScene::ReleaseLock(ELock lock) const
{
   if (LockValid(lock) && fLock == lock) {
      fLock = kUnlocked;
      if (gDebug>3) {
         Info("TGLScene::ReleaseLock", "released %s", LockName(lock));
      }
      return kTRUE;
   }
   Error("TGLScene::ReleaseLock", "Unable release %s, is %s", LockName(lock), LockName(fLock));
   return kFALSE;
}

//______________________________________________________________________________
inline Bool_t TGLScene::IsLocked() const
{
   return (fLock != kUnlocked);
}

//______________________________________________________________________________
inline TGLScene::ELock TGLScene::CurrentLock() const
{
   return fLock;
}

//______________________________________________________________________________
inline const char * TGLScene::LockName(ELock lock)
{
   static const std::string names[5] 
      = { "Unlocked",
          "DrawLock",
          "SelectLock",
          "ModifyLock",
          "UnknownLock" };

   if (lock < 4) {
      return names[lock].c_str(); 
   } else {
      return names[4].c_str();
   }
}

//______________________________________________________________________________
inline Bool_t TGLScene::LockValid(ELock lock) 
{
   // Test if lock is a valid type to take/release
   // kUnlocked is never valid in these cases
   switch(lock) {
      case kDrawLock:
      case kSelectLock:
      case kModifyLock:
         return kTRUE;
      default:
         return kFALSE;
   }
}

#endif // ROOT_TGLScene
