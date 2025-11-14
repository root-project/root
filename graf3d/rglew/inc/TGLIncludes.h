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

#ifndef _ROOT_GL_BUILDS_ITSELF
#warning "The TGLIncludes.h header is deprecated and will be removed in ROOT 6.40. Please include the required headers like <GL/gl.h> or <GL/glu.h> directly."
#endif

#ifdef WIN32
#include "Windows4Root.h"
#endif

#include <GL/glew.h>

#endif
