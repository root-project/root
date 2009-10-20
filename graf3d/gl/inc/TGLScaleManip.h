// @(#)root/gl:$Id$
// Author:  Richard Maunder  16/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLScaleManip
#define ROOT_TGLScaleManip

#ifndef ROOT_TGLManip
#include "TGLManip.h"
#endif


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLScaleManip                                                        //
//                                                                      //
// Scale manipulator - attaches to physical shape and draws local axes  //
// widgets with box heads. User can mouse over (turns yellow) and L     //
// click/drag to scale along this axis.                                 //
// Widgets use standard 3D package axes colours: X red, Y green, Z blue.//
//////////////////////////////////////////////////////////////////////////

class TGLScaleManip : public TGLManip
{
private:
   TGLVector3 fStartScale; //! initial scaling factors

   void LimitScale(Double_t & factor) const;

public:
   TGLScaleManip();
   TGLScaleManip(TGLPhysicalShape * shape);
   virtual ~TGLScaleManip();

   virtual void   Draw(const TGLCamera & camera) const;
   virtual Bool_t HandleButton(const Event_t & event, const TGLCamera & camera);
   virtual Bool_t HandleMotion(const Event_t & event, const TGLCamera & camera);

   ClassDef(TGLScaleManip,0) // GL scaling manipulator widget
};

#endif
