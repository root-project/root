// @(#)root/win32gdk:$Name:  $:$Id: TGWin32GL.cxx,v 1.1 2004/08/09 15:46:53 brun Exp $
// Author: Valeriy Onuchin  05/08/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32GL                                                            //
//                                                                      //
// The TGWin32GL is win32gdk implementation of TVirtualGLImp class.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGWin32GL.h"
#include "TGWin32VirtualGLProxy.h"
#include "TError.h"

#include "Windows4root.h"
#include "gdk/gdk.h"
#include "gdk/win32/gdkwin32.h"

#include <GL/gl.h>
#include <GL/glu.h>


//______________________________________________________________________________
TGWin32GL::TGWin32GL()
{
   // Ctor.

   gPtr2VirtualGL = &TGWin32VirtualGLProxy::ProxyObject;
}

//______________________________________________________________________________
TGWin32GL::~TGWin32GL()
{
   //

   gPtr2VirtualGL = 0;
}

//______________________________________________________________________________
Window_t TGWin32GL::CreateGLWindow(Window_t wind)
{
   // Win32gdk specific code to initialize GL window.

   GdkColormap *cmap;
   GdkWindow *GLWin;
   GdkWindowAttr xattr;
   int xval, yval;
   int wval, hval;
   ULong_t mask;
   int pixelformat;
   int depth;
   HDC hdc;
   static PIXELFORMATDESCRIPTOR pfd =
   {
      sizeof(PIXELFORMATDESCRIPTOR),  // size of this pfd
      1,                              // version number
      PFD_DRAW_TO_WINDOW |            // support window
      PFD_SUPPORT_OPENGL |            // support OpenGL
      PFD_DOUBLEBUFFER,               // double buffered
      PFD_TYPE_RGBA,                  // RGBA type
      24,                             // 24-bit color depth
      0, 0, 0, 0, 0, 0,               // color bits ignored
      0,                              // no alpha buffer
      0,                              // shift bit ignored
      0,                              // no accumulation buffer
      0, 0, 0, 0,                     // accum bits ignored
      32,                             // 32-bit z-buffer
      0,                              // no stencil buffer
      0,                              // no auxiliary buffer
      PFD_MAIN_PLANE,                 // main layer
      0,                              // reserved
      0, 0, 0                         // layer masks ignored
   };

   gdk_window_get_geometry((GdkDrawable *) wind, &xval, &yval, &wval, &hval, &depth);
   cmap = gdk_colormap_get_system();

   // window attributes
   xattr.width = wval;
   xattr.height = hval;
   xattr.x = xval;
   xattr.y = yval;
   xattr.wclass = GDK_INPUT_OUTPUT;
   xattr.event_mask = 0L; //GDK_ALL_EVENTS_MASK;
   xattr.event_mask |= GDK_EXPOSURE_MASK | GDK_STRUCTURE_MASK |
                       GDK_KEY_PRESS_MASK | GDK_KEY_RELEASE_MASK;
   xattr.colormap = cmap;
   mask = GDK_WA_X | GDK_WA_Y | GDK_WA_COLORMAP | GDK_WA_WMCLASS | GDK_WA_NOREDIR;
   xattr.window_type = GDK_WINDOW_CHILD;

   GLWin = gdk_window_new((GdkWindow *) wind, &xattr, mask);
   gdk_window_set_events(GLWin, (GdkEventMask)0);
   gdk_window_show(GLWin);
   hdc = GetWindowDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)GLWin));

   if ( (pixelformat = ChoosePixelFormat(hdc,&pfd)) == 0 ) {
       Error("InitGLWindow", "Barf! ChoosePixelFormat Failed");
   }
   if ( (SetPixelFormat(hdc, pixelformat,&pfd)) == 0 ) {
      Error("InitGLWindow", "Barf! SetPixelFormat Failed");
   }

   return (Window_t)GLWin;
}

//______________________________________________________________________________
ULong_t TGWin32GL::CreateContext(Window_t wind)
{
   //

   HDC hdc = GetWindowDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)wind));
   return (ULong_t)::wglCreateContext(hdc);
}

//______________________________________________________________________________
void TGWin32GL::DeleteContext(ULong_t ctx)
{
   //

   ::wglDeleteContext((HGLRC)ctx);
}

//______________________________________________________________________________
void TGWin32GL::MakeCurrent(Window_t wind, ULong_t ctx)
{
   //

   HDC hdc = GetWindowDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)wind));
   ::wglMakeCurrent(hdc,(HGLRC) ctx);
}

//______________________________________________________________________________
void TGWin32GL::SwapLayerBuffers(Window_t wind)
{
   //

   HDC hdc = GetWindowDC((HWND)GDK_DRAWABLE_XID((GdkWindow *)wind));
   ::wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
   ::glFinish();
}

