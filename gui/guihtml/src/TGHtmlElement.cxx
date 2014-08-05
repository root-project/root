// $Id$
// Author:  Valeriy Onuchin   03/05/2007

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**************************************************************************

    HTML widget for xclass. Based on tkhtml 1.28
    Copyright (C) 1997-2000 D. Richard Hipp <drh@acm.org>
    Copyright (C) 2002-2003 Hector Peraza.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

**************************************************************************/

#include <string.h>

#include "TGHtml.h"
#include "TImage.h"

// TODO: make these TGHtml static members

extern void HtmlTranslateEscapes(char *z);
extern void ToLower(char *z);


//______________________________________________________________________________
TGHtmlElement::TGHtmlElement(int etype)
{
   // HTML element constructor.

   fPNext = fPPrev = 0;
   fStyle.fFont = 0;
   fStyle.fColor = 0;
   fStyle.fSubscript = 0;
   fStyle.fAlign = 0;
   fStyle.fBgcolor = 0;
   fStyle.fExpbg = 0;
   fStyle.fFlags = 0;
   fType = etype;
   fFlags = 0;
   fCount = 0;
   fElId = 0;
   fOffs = 0;
}

//______________________________________________________________________________
TGHtmlTextElement::TGHtmlTextElement(int size) : TGHtmlElement(Html_Text)
{
   // HTML element constructor.

   fZText = new char[size + 1];
   fX = 0; fY = 0; fW = 0;
   fAscent = 0;
   fDescent = 0;
   fSpaceWidth = 0;
}

//______________________________________________________________________________
TGHtmlTextElement::~TGHtmlTextElement()
{
   // HTML element destructor.

   delete[] fZText;
}

//______________________________________________________________________________
TGHtmlMarkupElement::TGHtmlMarkupElement(int type2, int argc, int arglen[],
                                         char *av[]) : TGHtmlElement(type2)
{
   // HTML mrkup element constructor.

   fCount = argc - 1;

   if (argc > 1) {
      fArgv = new char*[argc+1];
      for (int i = 1; i < argc; i++) {
         if (arglen) {
            fArgv[i-1] = new char[arglen[i]+1];
            //sprintf(fArgv[i-1], "%.*s", arglen[i], av[i]);
            strncpy(fArgv[i-1], av[i], arglen[i]);
            fArgv[i-1][arglen[i]] = 0;
            HtmlTranslateEscapes(fArgv[i-1]);
            if ((i & 1) == 1) ToLower(fArgv[i-1]);
         } else {
            fArgv[i-1] = StrDup(av[i]);
            HtmlTranslateEscapes(fArgv[i-1]);
            if ((i & 1) == 1) ToLower(fArgv[i-1]);
         }
      }
      fArgv[argc-1] = 0;

      // Following is just a flag that this is unmodified
      fArgv[argc] = (char *) fArgv;

   } else {
      fArgv = 0;
   }
}

//______________________________________________________________________________
TGHtmlMarkupElement::~TGHtmlMarkupElement()
{
   // HTML markup element destructor.

   if (fArgv) {
      for (int i = 0; i < fCount; ++i) delete [] fArgv[i];
      delete [] fArgv;
   }
}

//______________________________________________________________________________
const char *TGHtmlMarkupElement::MarkupArg(const char *tag, const char *zDefault)
{
   // Lookup an argument in the given markup with the name given.
   // Return a pointer to its value, or the given default
   // value if it doesn't appear.

   int i;

   for (i = 0; i < fCount; i += 2) {
      if (strcmp(fArgv[i], tag) == 0) return fArgv[i+1];
   }
   return zDefault;
}

//______________________________________________________________________________
int TGHtmlMarkupElement::GetAlignment(int dflt)
{
   // Return an alignment or justification flag associated with the
   // given markup. The given default value is returned if no alignment is
   // specified.

   const char *z = MarkupArg("align", 0);
   int rc = dflt;

   if (z) {
      if (strcasecmp(z, "left") == 0) {
         rc = ALIGN_Left;
      } else if (strcasecmp(z, "right") == 0) {
         rc = ALIGN_Right;
      } else if (strcasecmp(z, "center") == 0) {
         rc = ALIGN_Center;
      }
   }

   return rc;
}

//______________________________________________________________________________
int TGHtmlMarkupElement::GetOrderedListType(int dflt)
{
   // The "type" argument to the given element might describe the type
   // for an ordered list. Return the corresponding LI_TYPE_* entry
   // if this is the case, or the default value if it isn't.
   // (this and the following should be defined only for TGHtmlLi)

   const char *z = MarkupArg("type", 0);
   if (z) {
      switch (*z) {
         case 'A': dflt = LI_TYPE_Enum_A; break;
         case 'a': dflt = LI_TYPE_Enum_a; break;
         case '1': dflt = LI_TYPE_Enum_1; break;
         case 'I': dflt = LI_TYPE_Enum_I; break;
         case 'i': dflt = LI_TYPE_Enum_i; break;
         default:  break;
      }
   }

   return dflt;
}

//______________________________________________________________________________
int TGHtmlMarkupElement::GetUnorderedListType(int dflt)
{
   // The "type" argument to the given element might describe a type
   // for an unordered list.  Return the corresponding LI_TYPE entry
   // if this is the case, or the default value if it isn't.

   const char *z = MarkupArg("type", 0);
   if (z) {
      if (strcasecmp(z, "disc") == 0) {
         dflt = LI_TYPE_Bullet1;
      } else if (strcasecmp(z, "circle") == 0) {
         dflt = LI_TYPE_Bullet2;
      } else if (strcasecmp(z, "square") == 0) {
         dflt = LI_TYPE_Bullet3;
      }
   }

   return dflt;
}

//int TGHtmlMarkupElement::GetVerticalAlignment(int dflt);

//______________________________________________________________________________
TGHtmlTable::TGHtmlTable(int type2, int argc, int arglen[], char *argv2[]) :
   TGHtmlMarkupElement(type2, argc, arglen, argv2)
{
   // HTML table element constructor.

   fBorderWidth = 0;
   fNCol = 0;
   fNRow = 0;
   fX = 0; fY = 0; fW = 0; fH = 0;
   fPEnd = 0;
   fBgImage = 0;
   fHasbg = 0;
   for (int i=0;i<=HTML_MAX_COLUMNS;++i) {
      fMinW[i] = fMaxW[i] = 0;
   }
}

//______________________________________________________________________________
TGHtmlTable::~TGHtmlTable()
{
   // HTML table element destructor.

   if (fBgImage) delete fBgImage;
}

//______________________________________________________________________________
TGHtmlCell::TGHtmlCell(int type2, int argc, int arglen[], char *argv2[]) :
   TGHtmlMarkupElement(type2, argc, arglen, argv2)
{
   // HTML cell element constructor.

   fRowspan = 0;
   fColspan = 0;
   fX = 0; fY = 0; fW = 0; fH = 0;
   fPTable = 0;
   fPRow = 0;
   fPEnd = 0;
   fBgImage = 0;
}

//______________________________________________________________________________
TGHtmlCell::~TGHtmlCell()
{
   // HTML cell element destructor.

   if (fBgImage) delete fBgImage;
}

//______________________________________________________________________________
TGHtmlRef::TGHtmlRef(int type2, int argc, int arglen[], char *argv2[]) :
   TGHtmlMarkupElement(type2, argc, arglen, argv2)
{
   // HTML ref element constructor.

   fPOther = 0;
   fBgImage = 0;
}

//______________________________________________________________________________
TGHtmlRef::~TGHtmlRef()
{
   // HTML ref element destructor.

   if (fBgImage) delete fBgImage;
}

//______________________________________________________________________________
TGHtmlLi::TGHtmlLi(int type2, int argc, int arglen[], char *argv2[]) :
   TGHtmlMarkupElement(type2, argc, arglen, argv2)
{
   // HTML li element constructor.

   fLtype = 0;
   fAscent = 0;
   fDescent = 0;
   fCnt = 0;
   fX = 0; fY = 0;
}

//______________________________________________________________________________
TGHtmlListStart::TGHtmlListStart(int type2, int argc, int arglen[], char *argv2[]) :
   TGHtmlMarkupElement(type2, argc, arglen, argv2)
{
   // HTML list start element constructor.

   fLtype = 0;
   fCompact = 0;
   fCnt = 0;
   fWidth = 0;
   fLPrev = 0;
}

//______________________________________________________________________________
TGHtmlImageMarkup::TGHtmlImageMarkup(int type2, int argc,
                                     int arglen[], char *argv2[]) :
   TGHtmlMarkupElement(type2, argc, arglen, argv2)
{
   // HTML image element constructor.

   fAlign = 0;
   fTextAscent = 0;
   fTextDescent = 0;
   fRedrawNeeded = 0;
   fX = 0; fY = 0; fW = 0; fH = 0;
   fAscent = 0;
   fDescent = 0;
   fZAlt = 0;
   fPImage = 0;
   fPMap = 0;
   fINext = 0;
}

//______________________________________________________________________________
TGHtmlForm::TGHtmlForm(int type2, int argc, int arglen[], char *argv2[]) :
   TGHtmlMarkupElement(type2, argc, arglen, argv2)
{
   // HTML form element constructor.

   fFormId = 0;
   fElements = 0;
   fHasctl = 0;
   fPFirst = 0;
   fPEnd = 0;
}

//______________________________________________________________________________
TGHtmlHr::TGHtmlHr(int type2, int argc, int arglen[], char *argv2[]) :
   TGHtmlMarkupElement(type2, argc, arglen, argv2)
{
   // HTML hr element constructor.

   fX = 0; fY = 0; fW = 0; fH = 0;
   fIs3D = 0;
}

//______________________________________________________________________________
TGHtmlAnchor::TGHtmlAnchor(int type2, int argc, int arglen[], char *argv2[]) :
   TGHtmlMarkupElement(type2, argc, arglen, argv2)
{
   // HTML anchor element constructor.

   fY = 0;
}

//______________________________________________________________________________
TGHtmlScript::TGHtmlScript(int type2, int argc, int arglen[], char *argv2[]) :
   TGHtmlMarkupElement(type2, argc, arglen, argv2)
{
   // HTML script element constructor.

   fNStart = -1;
   fNScript = 0;
}

//______________________________________________________________________________
TGHtmlMapArea::TGHtmlMapArea(int type2, int argc, int arglen[], char *argv2[]) :
   TGHtmlMarkupElement(type2, argc, arglen, argv2)
{
   // HTML map area constructor.

   fMType = 0;
   fCoords = 0;
   fNum = 0;
}


//----------------------------------------------------------------------

#if 0
TGHtmlBlock::TGHtmlBlock()
{
   // HTML block element constructor.
}

TGHtmlBlock::~TGHtmlBlock()
{
   // HTML block element destructor.
}
#endif

//______________________________________________________________________________
TGHtmlInput::TGHtmlInput(int type2, int argc, int arglen[], char *argv2[]) :
   TGHtmlMarkupElement(type2, argc, arglen, argv2)
{
   // HTML input element constructor.

   fPForm = 0;
   fINext = 0;
   fFrame = 0;
   fHtml = 0;
   fPEnd = 0;
   fInpId = 0; fSubId = 0;
   fX = 0; fY = 0; fW = 0; fH = 0;
   fPadLeft = 0;
   fAlign = 0;
   fTextAscent = 0;
   fTextDescent = 0;
   fItype = 0;
   fSized = 0;
   fCnt = 0;
}

//______________________________________________________________________________
void TGHtmlInput::Empty()
{
   // Mark this element as being empty. It has no widget and doesn't appear on
   // the screen.
   //
   // This is called for HIDDEN inputs or when the corresponding widget is
   // not created.

   fFrame = NULL;
   fW = 0;
   fH = 0;
   fFlags &= ~HTML_Visible;
   fStyle.fFlags |= STY_Invisible;
   fSized = 1;
}

