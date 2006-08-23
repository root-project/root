// @(#)root/gl:$Name:  $:$Id: TGLLogicalShape.cxx,v 1.13 2006/05/24 14:49:21 brun Exp $
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
#include "TContextMenu.h"
#include "TBuffer3D.h"
#include "TAtt3D.h"

ClassImp(TGLLogicalShape)

//______________________________________________________________________________
TGLLogicalShape::TGLLogicalShape(ULong_t ID) :
   TGLDrawable(ID, kTRUE), // Logical shapes DL cached by default
   fRef(0), fRefStrong(kFALSE), fExternalObj(0)
{
   // Construct a logical shape with unique id 'ID'.
   // Logical shapes are not display list cached by default.
}

//______________________________________________________________________________
TGLLogicalShape::TGLLogicalShape(const TBuffer3D & buffer) :
   TGLDrawable(reinterpret_cast<ULong_t>(buffer.fID), kTRUE), // Logical shapes DL cached by default
   fRef(0), fRefStrong(kFALSE), fExternalObj(buffer.fID)
{
   // Use the bounding box in buffer if valid
   if (buffer.SectionsValid(TBuffer3D::kBoundingBox)) {
      fBoundingBox.Set(buffer.fBBVertex);
   } else if (buffer.SectionsValid(TBuffer3D::kRaw)) {
   // otherwise use the raw points to generate one
      fBoundingBox.SetAligned(buffer.NbPnts(), buffer.fPnts);
   }
}

//______________________________________________________________________________
TGLLogicalShape::TGLLogicalShape(const TGLLogicalShape& gls) :
  TGLDrawable(gls),
  fRef(gls.fRef),
  fRefStrong(gls.fRefStrong),
  fExternalObj(gls.fExternalObj)
{ 
   //copy constructor
}

//______________________________________________________________________________
TGLLogicalShape& TGLLogicalShape::operator=(const TGLLogicalShape& gls)
{
   //assignement operator
   if(this!=&gls) {
      TGLDrawable::operator=(gls);
      fRef=gls.fRef;
      fRefStrong=gls.fRefStrong;
      fExternalObj=gls.fExternalObj;
   } 
   return *this;
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

//______________________________________________________________________________
void TGLLogicalShape::InvokeContextMenu(TContextMenu & menu, UInt_t x, UInt_t y) const
{
   // Invoke popup menu or our bound external TObject (if any), using passed
   // 'menu' object, at location 'x' 'y'
   if (fExternalObj) {
      menu.Popup(x, y, fExternalObj);
   }
}

//______________________________________________________________________________
void TGLLogicalShape::ProcessSelection(UInt_t* ptr, TGLViewer*, TGLScene*)
{
   // Virtual method called-back after a secondary selection hit
   // is recorded (see TGLViewer::HandleButton(), Ctrl-Button1).
   // The ptr argument holds the GL pick-record of the closest hit.
   //
   // This base-class implementation simply prints out the result.

   Int_t n = ptr[0];
   printf("TGLLogicalShape::ProcessSelection %d names on the stack (z1=%g, z2=%g).\n",
          n, Float_t(ptr[1])/0x7fffffff, Float_t(ptr[2])/0x7fffffff);
   ptr += 3;
   printf("  Names: ");
   for (Int_t j=0; j<n; ++j, ++ptr) printf ("%d ", *ptr);
   printf("\n");
}
