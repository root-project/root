// @(#)root/gl:$Id$
// Author:  Richard Maunder  16/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLTransManip
#define ROOT_TGLTransManip

#ifndef ROOT_TGLManip
#include "TGLManip.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLTransManip                                                        //
//                                                                      //
// Translation manipulator - attaches to physical shape and draws local //
// axes widgets with arrow heads. User can mouse over (turns yellow) and//
// L click/drag to translate along this axis.                           //
// Widgets use standard 3D package axes colours: X red, Y green, Z blue.//
//////////////////////////////////////////////////////////////////////////

class TGLTransManip : public TGLManip
{
private:

public:
   TGLTransManip();
   TGLTransManip(TGLPhysicalShape * shape);
   virtual ~TGLTransManip();

   virtual void   Draw(const TGLCamera & camera) const;
   virtual Bool_t HandleMotion(const Event_t & event, const TGLCamera & camera);

   ClassDef(TGLTransManip,0) // GL translation manipulator widget
};

#endif
