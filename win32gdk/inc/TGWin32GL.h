// @(#)root/win32gdk:$Name:  $:$Id: TGWin32GL.h,v 1.1 2004/08/09 15:46:53 brun Exp $
// Author: Valeriy Onuchin  05/08/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGWin32GL
#define ROOT_TGWin32GL


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32GL                                                            //
//                                                                      //
// The TGWin32GL is win32gdk implementation of TVirtualGLImp class.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualGL
#include "TVirtualGL.h"
#endif


class TGWin32GL : public TVirtualGLImp {

public:
   TGWin32GL();
   ~TGWin32GL();

   Window_t CreateGLWindow(Window_t wind);
   ULong_t  CreateContext(Window_t wind);
   void     DeleteContext(ULong_t ctx);
   void     MakeCurrent(Window_t wind, ULong_t ctx);
   void     SwapLayerBuffers(Window_t wind);

   ClassDef(TGWin32GL,0);
};

#endif
