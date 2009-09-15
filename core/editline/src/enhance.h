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

void setKeywordColors(const char* colorTab, const char* colorBracket,
                      const char* colorBadBracket);
int selectColor(const char* str);
void highlightKeywords(EditLine* el);
int matchParentheses(EditLine* el);
void colorWord(EditLine* el, int first, int last, int color);
void colorBrackets(EditLine* el, int open, int close, int color);
