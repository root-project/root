// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 17/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_IOSLineStyles
#define ROOT_IOSLineStyles

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Line styles.                                                         //
//                                                                      //
// Predefined line styles.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <CoreGraphics/CGPattern.h>

namespace ROOT {
namespace iOS {
namespace GraphicUtils {

/*
enum {
   kFixedLineStyles = 10
};
*/

extern const unsigned linePatternLengths[10];
extern const CGFloat dashLinePatterns[10][8];

}
}
}

#endif
