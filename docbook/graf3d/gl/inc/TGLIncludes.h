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

#ifndef G__DICTIONARY

#include <GL/glew.h>

#else

#include <GL/gl.h>
#include <GL/glu.h>

#endif

#endif
