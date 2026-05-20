// @(#)root/gl:$Id$
// Authors:  Timur and Matevz, May 2008

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLIncludes
#define ROOT_TGLIncludes

// GL includes - include this if you are calling OpenGL functions.

#ifdef WIN32
#include "Windows4Root.h"
#endif

#include <glad/gl.h>

// This used to be included through glew.h.
#if defined(__APPLE__) && defined(__MACH__)
#  include <OpenGL/glu.h>
#else
#  include <GL/glu.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Wrapper functions fixing an issue with several glad functions not compatible with gluTessCallback.
// This is visible with the tutorial grad.C, not rendering the color gradients
[[maybe_unused]]
static void impl_glBegin(GLenum mode) {
   glBegin(mode);
}
[[maybe_unused]]
static void impl_glEnd(void) {
   glEnd();
}
[[maybe_unused]]
static void impl_glVertex3fv(const GLfloat * v) {
   glVertex3fv(v);
}
[[maybe_unused]]
static void impl_glVertex3dv(const GLdouble * v) {
   glVertex3dv(v);
}
[[maybe_unused]]
static void impl_glVertex4fv(const GLfloat * v) {
   glVertex4fv(v);
}
[[maybe_unused]]
static void impl_glVertex4dv(const GLdouble * v) {
   glVertex4dv(v);
}

#ifdef __cplusplus
}
#endif

#endif
