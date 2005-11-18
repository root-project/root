// @(#)root/gl:$Name:  $:$Id: TGLLogicalShape.cxx,v 1.5 2005/10/03 15:19:35 brun Exp $
// Author:  Richard Maunder  25/05/2005

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLLogicalShape                                                      //
//                                                                      //
// Abstract logical shape - a GL drawables - base for all shapes faceset//
// sphere etc. Logical shapes are a unique piece of geometry, described //
// in it's local frame. Object is reference counted by physical shapes  //
// which are using it - see TGLPhysicalShape description for fuller     //
// description of how logical/physical shapes are used.                 //
//////////////////////////////////////////////////////////////////////////

#include "TGLLogicalShape.h"
#include "TGLDisplayListCache.h"

ClassImp(TGLLogicalShape)

//______________________________________________________________________________
TGLLogicalShape::TGLLogicalShape(ULong_t ID) :
   TGLDrawable(ID, kFALSE), // Logical shapes not DL cached by default at present
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
