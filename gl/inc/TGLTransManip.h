// @(#)root/gl:$Name:  $:$Id: TGLTransManip.h
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

class TGLTransManip : public TGLManip
{
private:

public:
   TGLTransManip(TGLViewer & viewer);
   TGLTransManip(TGLViewer & viewer, TGLPhysicalShape * shape);
   virtual ~TGLTransManip();
   
   virtual void   Draw() const; 
   virtual Bool_t HandleMotion(Event_t * event, const TGLCamera & camera);

   ClassDef(TGLTransManip,0) // GL translation manipulator widget
};

#endif
