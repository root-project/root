// @(#)root/gx11:$Name:  $:$Id: TX11GL.cxx,v 1.1 2004/08/09 15:46:53 brun Exp $
// Author: Timur Pocheptsov 09/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TX11GL                                                               //
//                                                                      //
// The TX11GL is X11 implementation of TVirtualGLImp class.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "TVirtualX.h"
#include "TX11GL.h"
#include "TError.h"

ClassImp(TX11GL)

//______________________________________________________________________________
TX11GL::TX11GL() : fDpy(0), fVisInfo(0)
{
}

//______________________________________________________________________________
Window_t TX11GL::CreateGLWindow(Window_t wind)
{
   if(!fDpy)
      fDpy = (Display *)gVirtualX->GetDisplay();

   static int dblBuf[] = {
    			  GLX_DOUBLEBUFFER,
#ifdef STEREO_GL
    			  GLX_STEREO,
#endif
			  GLX_RGBA, GLX_DEPTH_SIZE, 16,
    			  GLX_RED_SIZE, 1, GLX_GREEN_SIZE, 1,
			  GLX_BLUE_SIZE, 1,None
			 };
   static int * snglBuf = dblBuf + 1;

   if(!fVisInfo){
      fVisInfo = glXChooseVisual(fDpy, DefaultScreen(fDpy), dblBuf);

      if(!fVisInfo)
         fVisInfo = glXChooseVisual(fDpy, DefaultScreen(fDpy), snglBuf);

      if(!fVisInfo){
         ::Error("TX11GL::CreateGLWindow", "No good visual\n");
         return 0;
      }
   }

   Int_t  xval = 0, yval = 0;
   UInt_t wval = 0, hval = 0, border = 0, d = 0;
   Window root = {0};

   XGetGeometry(fDpy, wind, &root, &xval, &yval, &wval, &hval, &border, &d);
   ULong_t mask = 0;
   XSetWindowAttributes attr = {0};

   attr.colormap = XCreateColormap(fDpy, root, fVisInfo->visual, AllocNone);
   attr.event_mask = NoEventMask;
   attr.backing_store = Always;
   attr.bit_gravity = NorthWestGravity;
   mask = CWBackPixel | CWBorderPixel | CWColormap | CWEventMask |
          CWBackingStore | CWBitGravity;

   Window GLWin = XCreateWindow(fDpy, wind, xval, yval, wval, hval,
                                0, fVisInfo->depth, InputOutput,
                                fVisInfo->visual, mask, &attr);
   XMapWindow(fDpy, GLWin);
   return (Window_t)GLWin;
}

//______________________________________________________________________________
ULong_t TX11GL::CreateContext(Window_t)
{
   return (ULong_t)glXCreateContext(fDpy, fVisInfo, None, GL_TRUE);
}

//______________________________________________________________________________
void TX11GL::DeleteContext(ULong_t ctx)
{
   glXDestroyContext(fDpy, (GLXContext)ctx);
}

//______________________________________________________________________________
void TX11GL::MakeCurrent(Window_t wind, ULong_t ctx)
{
   glXMakeCurrent(fDpy, (GLXDrawable)wind, (GLXContext)ctx);
}

//______________________________________________________________________________
void TX11GL::SwapLayerBuffers(Window_t wind)
{
   glXSwapBuffers(fDpy, (GLXDrawable)wind);
}
