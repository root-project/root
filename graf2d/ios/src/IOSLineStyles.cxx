// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 17/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "IOSLineStyles.h"

namespace ROOT {
namespace iOS {
namespace GraphicUtils {
//For fixed line style, number of elements in a pattern is not bigger than 8.
const unsigned linePatternLengths[] = {1, 2, 2, 4, 4, 8, 2, 6, 2, 4};

//Line pattern specyfies length of painted and unpainted fragments, for example,
//{2.f, 2.f} draw 2 pixels, skip to pixels (and repeat).
const CGFloat dashLinePatterns[10][8] = {
                                       {1},                                     //Style 1:  1 element, solid line
                                       {3.f, 3.f},                              //Style 2:  2 elements (paint one, skip the second).
                                       {1.f, 2.f},                              //Style 3:  2 elements.
                                       {3.f, 4.f, 1.f, 4.f},                    //Style 4:  4 elements.
                                       {5.f, 3.f, 1.f, 3.f},                    //Style 5:  4 elements.
                                       {5.f, 3.f, 1.f, 3.f, 1.f, 3.f, 1.f, 3.f},//Style 6:  8 elements
                                       {5.f, 5.f},                              //Style 7:  2 elements.
                                       {5.f, 3.f, 1.f, 3.f, 1.f, 3.f},          //Style 8:  6 elements.
                                       {20.f, 5.f},                             //Style 9:  2 elements.
                                       {20.f, 8.f, 1.f, 8.f}                    //Style 10: 4 elements.
                                      };

}//namespace GraphicUtils
}//namespace iOS
}//namespace ROOT
