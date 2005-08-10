// @(#)root/gl:$Name:  $:$Id: TGLScene.h,v 1.11 2005/07/08 15:39:29 brun Exp $
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

   // Drawing/Selection
   const TGLBoundingBox & BoundingBox() const;
   UInt_t                 Draw(const TGLCamera & camera, EDrawStyle style, UInt_t sceneLOD, Double_t timeout = 0.0);
   void                   DrawAxes() const;
   Bool_t                 Select(const TGLCamera & camera, EDrawStyle style);

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

   // Selected Object
   const TGLPhysicalShape * GetSelected() const { return fSelectedPhysical; }
   Bool_t                   SetSelectedColor(const Float_t rgba[4]);
   Bool_t                   SetColorOnSelectedFamily(const Float_t rgba[4]);
   Bool_t                   ShiftSelected(const TGLVector3 & shift);
   Bool_t                   SetSelectedGeom(const TGLVertex3 & trans, const TGLVector3 & scale);


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
