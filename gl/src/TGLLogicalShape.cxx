// @(#)root/gl:$Name:  $:$Id: TGLLogicalShape.cxx,v 1.7 2005/11/22 18:05:46 brun Exp $
// Author:  Richard Maunder  25/05/2005

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

#include "TGLLogicalShape.h"
#include "TGLDisplayListCache.h"

ClassImp(TGLLogicalShape)

//______________________________________________________________________________
TGLLogicalShape::TGLLogicalShape(ULong_t ID) :
   TGLDrawable(ID, kTRUE), // Logical shapes DL cached by default
   fRef(0), fRefStrong(kFALSE)
{
   // Construct a logical shape with unique id 'ID'.
   // Logical shapes are not display list cached by default.
}

//______________________________________________________________________________
TGLLogicalShape::~TGLLogicalShape()
{
   // Destroy logical shape
   
   // Physical refs should have been cleared
   if (fRef > 0) {
      assert(kFALSE);
   }
}
