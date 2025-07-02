// @(#)root/gl:$Id$
// Authors:  Timur and Matevz, May 2008

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Window-system specific GL includes.
// Inclusion should only be necessary in in low-level system files.

#ifndef ROOT_TGLWSIncludes

#include <RVersion.h> // for ROOT_VERSION

// This header is deprecated according to
// https://its.cern.ch/jira/browse/ROOT-9807
// In the 6.38 release, our users will get a warning, and then in 6.40 we
// remove this header. Conditional on the ROOT version, this header will give
// an error on inclusion to remind us to move it. Remove also the
// _ROOT_GL_BUILDS_ITSELF definition from the CMakeLists.txt then.
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 38, 00)
#error "Header should be moved outside of the public ROOT interface now"
#else
#ifndef _ROOT_GL_BUILDS_ITSELF
#warning "The TGLWSIncludes.h header is deprecated and will be removed in ROOT 6.40"
#endif
#endif


#include "RConfigure.h"
#include "TGLIncludes.h"

#if defined(WIN32)
#  include <GL/wglew.h>
#else
#  if defined(__APPLE__) && !defined(R__HAS_COCOA)
#    define GLEW_APPLE_GLX
#  endif
#  if !defined(R__HAS_COCOA)
#    include <GL/glxew.h>
#  endif
#endif

#endif
