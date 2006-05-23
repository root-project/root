// @(#)root/gl:$Name:  $:$Id: TGLLogicalShape.h,v 1.13 2006/05/15 11:01:13 rdm Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLLogicalShape
#define ROOT_TGLLogicalShape

#ifndef ROOT_TGLDrawable
#include "TGLDrawable.h"
#endif

class TBuffer3D;
class TObject;
class TContextMenu;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLLogicalShape                                                      //
//                                                                      //
// Abstract logical shape - a GL 'drawable' - base for all shapes -     //
// faceset sphere etc. Logical shapes are a unique piece of geometry,   //
// described in it's local frame - e.g if we have three spheres in :    //
// Sphere A - Radius r1, center v1                                      //
// Sphere B - Radius r2, center v2                                      //
// Sphere C - Radius r1, center v3                                      //
//                                                                      //
// Spheres A and C can share a common logical sphere of radius r1 - and //
// place them with two physicals with translations of v1 & v2.  Sphere B//
// requires a different logical (radius r2), placed with physical with  //
// translation v2.                                                      //
//                                                                      //
// Physical shapes know about and can share logicals. Logicals do not   //
// about (aside from reference counting) physicals or share them.       //
//                                                                      //
// This sharing of logical shapes greatly reduces memory consumption and//
// scene (re)build times in typical detector geometries which have many //
// repeated objects placements.                                         //
//                                                                      //
// TGLLogicalShapes have reference counting, performed by the client    //
// physical shapes which are using it.                                  //
//                                                                      //
// See base/src/TVirtualViewer3D for description of common external 3D  //
// viewer architecture and how external viewer clients use it.          //
//////////////////////////////////////////////////////////////////////////

class TGLLogicalShape : public TGLDrawable
{
private:
   // Fields
   mutable UInt_t  fRef;         //! physical instance ref counting

protected:
   mutable Bool_t  fRefStrong;   //! Strong ref (delete on 0 ref)
   TObject        *fExternalObj;

   TGLLogicalShape(const TGLLogicalShape&);
   TGLLogicalShape& operator=(const TGLLogicalShape&);

   // TODO: Common UInt_t flags section (in TGLDrawable?) to avoid multiple bools

public:
   TGLLogicalShape(ULong_t ID);
   TGLLogicalShape(const TBuffer3D & buffer);
   virtual ~TGLLogicalShape();

   void InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const;

   // Physical shape ref counting
   void   AddRef() const                 { ++fRef; }
   Bool_t SubRef() const;
   UInt_t Ref()    const                 { return fRef; }
   void   StrongRef(Bool_t strong) const { fRefStrong = strong; }
   TObject *GetExternal()const{return fExternalObj;}

   // Override in sub-classes that do direct object rendering (e.g. TGLObject).
   virtual Bool_t KeepDuringSmartRefresh() const { return false; }

   ClassDef(TGLLogicalShape,0) // a logical (non-placed, local frame) drawable object
};

//______________________________________________________________________________
inline Bool_t TGLLogicalShape::SubRef() const
{
   if (--fRef == 0) {
      if (fRefStrong) {
         delete this;
      }
      return kTRUE;
   }
   return kFALSE;
}

#endif // ROOT_TGLLogicalShape
