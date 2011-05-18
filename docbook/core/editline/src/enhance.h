// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "histedit.h"

void setKeywordColors(int colorTab, int colorBracket,
                      int colorBadBracket);
void highlightKeywords(EditLine_t* el);
int matchParentheses(EditLine_t* el);
void colorWord(EditLine_t* el, int first, int last, int color);
void colorBrackets(EditLine_t* el, int open, int close, int color);
