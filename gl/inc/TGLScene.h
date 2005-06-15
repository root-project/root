// @(#)root/gl:$Name:  $:$Id: TGLScene.h,v 1.7 2005/06/15 10:22:57 brun Exp $
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

#include <map>
#include <vector>
#include <string>

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
public:
   enum EDrawMode{kFill, kOutline, kWireFrame};

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

   mutable TGLBoundingBox fBoundingBox;      //! bounding box for scene (axis aligned) - lazy update - use BoundingBox() to access
   mutable Bool_t         fBoundingBoxValid; //! bounding box valid?
   UInt_t                 fLastDrawLOD;      //! last LOD for the scene draw
   EDrawMode              fDrawMode;         //! current draw style (Fill/Outline/WireFrame)  
   TGLPhysicalShape *     fSelectedPhysical; //! current selected physical shape

   // Methods

   // Draw sorting
   void   SortDrawList();
   static Bool_t ComparePhysicalVolumes(const TGLPhysicalShape * shape1, const TGLPhysicalShape * shape2);
   
   // Misc
   void   DrawNumber(Double_t num, Double_t x, Double_t y, Double_t z, Double_t yorig) const;
   UInt_t CalcPhysicalLOD(const TGLPhysicalShape & shape,
                          const TGLCamera & camera,
                          UInt_t sceneLOD) const;

   // Non-copyable class
   TGLScene(const TGLScene &);
   TGLScene & operator=(const TGLScene &);

public:
   TGLScene();
   virtual ~TGLScene(); // ClassDef introduces virtual fns

   // Drawing
   const TGLBoundingBox & BoundingBox() const;
   void                   SetDrawMode(EDrawMode mode){fDrawMode = mode;}
   UInt_t                 Draw(const TGLCamera & camera, UInt_t sceneLOD, Double_t timeout = 0.0);
   void                   DrawAxes() const;
   Bool_t                 Select(const TGLCamera & camera);

   // Logical Shape Management
   void                    AdoptLogical(TGLLogicalShape & shape);
   Bool_t                  DestroyLogical(ULong_t ID);
   UInt_t                  DestroyAllLogicals();
   void                    PurgeNextLogical() {};
   TGLLogicalShape *       FindLogical(ULong_t ID)  const;

   // Physical Shape Management
   void                     AdoptPhysical(TGLPhysicalShape & shape);
   Bool_t                   DestroyPhysical(ULong_t ID);
   UInt_t                   DestroyPhysicals(const TGLCamera & camera);
   UInt_t                   DestroyAllPhysicals();
   TGLPhysicalShape *       FindPhysical(ULong_t ID) const;
   void                     SetPhysicalsColorByLogical(ULong_t logicalID, const Float_t rgba[4]);
   TGLPhysicalShape *       GetSelected() const { return fSelectedPhysical; }
   void                     SelectedModified();

   // Locking
   inline Bool_t TakeLock(ELock lock) const;
   inline Bool_t ReleaseLock(ELock lock) const;
   inline Bool_t IsLocked() const;
   inline ELock  CurrentLock() const;
   static inline const char * LockName(ELock lock);
   
   // Debug
   void   Dump() const;
   UInt_t SizeOf() const;

   ClassDef(TGLScene,0) // a GL scene - collection of physical and logical shapes
};

inline Bool_t TGLScene::TakeLock(ELock lock) const
{
   if (fLock == kUnlocked) {
      fLock = lock;
      if (gDebug>3) {
         Info("TGLScene::TakeLock", "took %s", LockName(fLock));
      }
      return kTRUE;
   }
   Error("TGLScene::TakeLock", "Unable take %s, already %s", LockName(lock), LockName(fLock));
   return kFALSE;
}

inline Bool_t TGLScene::ReleaseLock(ELock lock) const
{
   if (fLock == lock) {
      fLock = kUnlocked;
      if (gDebug>3) {
         Info("TGLScene::ReleaseLock", "released %s", LockName(lock));
      }
      return kTRUE;
   }
   Error("TGLScene::ReleaseLock", "Unable release %s, is %s", LockName(lock), LockName(fLock));
   return kFALSE;
}

inline Bool_t TGLScene::IsLocked() const
{
   return (fLock != kUnlocked);
}

inline TGLScene::ELock TGLScene::CurrentLock() const
{
   return fLock;
}

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

#endif // ROOT_TGLScene
