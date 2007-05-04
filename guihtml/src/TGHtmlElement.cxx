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
   //

   pNext = pPrev = 0;
   style.font = 0;   
   style.color = 0;  
   style.subscript = 0;
   style.align = 0;
   style.bgcolor = 0;
   style.expbg = 0;
   style.flags = 0;  
   type = etype;     
   flags = 0;        
   count = 0;
   id = 0;
   offs = 0;
}

//______________________________________________________________________________
TGHtmlTextElement::TGHtmlTextElement(int size) : TGHtmlElement(Html_Text)
{
   //

   zText = new char[size + 1];
   x = 0; y = 0; w = 0;
   ascent = 0;
   descent = 0;
   spaceWidth = 0;
}

//______________________________________________________________________________
TGHtmlTextElement::~TGHtmlTextElement()
{
   //

   delete[] zText;
}

//______________________________________________________________________________
TGHtmlMarkupElement::TGHtmlMarkupElement(int type, int argc,
                                         int arglen[], char *av[]) :
                                         TGHtmlElement(type)
{
   //

   count = argc - 1;

   if (argc > 1) {
      argv = new char*[argc+1];
      for (int i = 1; i < argc; i++) {
         if (arglen) {
            argv[i-1] = new char[arglen[i]+1];
            //sprintf(argv[i-1], "%.*s", arglen[i], av[i]);
            strncpy(argv[i-1], av[i], arglen[i]);
            argv[i-1][arglen[i]] = 0;
            HtmlTranslateEscapes(argv[i-1]);
            if ((i & 1) == 1) ToLower(argv[i-1]);
         } else {
            argv[i-1] = StrDup(av[i]);
            HtmlTranslateEscapes(argv[i-1]);
            if ((i & 1) == 1) ToLower(argv[i-1]);
         }
      }  
      argv[argc-1] = 0;

      // Following is just a flag that this is unmodified
      argv[argc] = (char *) argv;

   } else {
      argv = 0;
   }
}

//______________________________________________________________________________
TGHtmlMarkupElement::~TGHtmlMarkupElement()
{
   //

   if (argv) {
      for (int i = 0; i < count; ++i) delete [] argv[i];
      delete [] argv;
   }
}  
   
//______________________________________________________________________________
char *TGHtmlMarkupElement::MarkupArg(const char *tag, char *zDefault)
{
   // Lookup an argument in the given markup with the name given.
   // Return a pointer to its value, or the given default  
   // value if it doesn't appear.

   int i;

   for (i = 0; i < count; i += 2) {
      if (strcmp(argv[i], tag) == 0) return argv[i+1];
   }
   return zDefault; 
}

//______________________________________________________________________________
int TGHtmlMarkupElement::GetAlignment(int dflt)
{
   // Return an alignment or justification flag associated with the
   // given markup. The given default value is returned if no alignment is
   // specified.

   char *z = MarkupArg("align", 0);
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

   char *z;

   z = MarkupArg("type", 0);
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

   char *z;

   z = MarkupArg("type", 0);
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
TGHtmlTable::TGHtmlTable(int type, int argc, int arglen[], char *argv[]) :
             TGHtmlMarkupElement(type, argc, arglen, argv)
{
   //

   borderWidth = 0;
   nCol = 0;
   nRow = 0;
   x = 0; y = 0; w = 0; h = 0;
   pEnd = 0;
   bgImage = 0;
   hasbg = 0;
}

//______________________________________________________________________________
TGHtmlTable::~TGHtmlTable()
{
   //

   if (bgImage) delete bgImage;
}

//______________________________________________________________________________
TGHtmlCell::TGHtmlCell(int type, int argc, int arglen[], char *argv[]) :
            TGHtmlMarkupElement(type, argc, arglen, argv)
{
   //

   rowspan = 0;
   colspan = 0;
   x = 0; y = 0; w = 0; h = 0;
   pTable = 0;
   pRow = 0;
   pEnd = 0;
   bgImage = 0;  
}

//______________________________________________________________________________
TGHtmlCell::~TGHtmlCell()
{
   //

   if (bgImage) delete bgImage;
}

//______________________________________________________________________________
TGHtmlRef::TGHtmlRef(int type, int argc, int arglen[], char *argv[]) :
           TGHtmlMarkupElement(type, argc, arglen, argv)
{
   //

   pOther = 0;
   bgImage = 0;
}

//______________________________________________________________________________
TGHtmlRef::~TGHtmlRef()
{
   //

   if (bgImage) delete bgImage;
}

//______________________________________________________________________________
TGHtmlLi::TGHtmlLi(int type, int argc, int arglen[], char *argv[]) :
          TGHtmlMarkupElement(type, argc, arglen, argv)
{
   //

   ltype = 0;
   ascent = 0;
   descent = 0;
   cnt = 0;
   x = 0; y = 0;
}

//______________________________________________________________________________
TGHtmlListStart::TGHtmlListStart(int type, int argc, int arglen[], char *argv[]) :
                  TGHtmlMarkupElement(type, argc, arglen, argv)
{
   //

   ltype = 0;
   compact = 0;
   cnt = 0;
   width = 0;
   lPrev = 0;
}

//______________________________________________________________________________
TGHtmlImageMarkup::TGHtmlImageMarkup(int type, int argc,
                                   int arglen[], char *argv[]) :
  TGHtmlMarkupElement(type, argc, arglen, argv)
{
   //

   align = 0;
   textAscent = 0;
   textDescent = 0;
   redrawNeeded = 0;
   x = 0; y = 0; w = 0; h = 0;
   ascent = 0;
   descent = 0;
   zAlt = 0;   
   pImage = 0;
   pMap = 0; 
   iNext = 0;  
}

//______________________________________________________________________________
TGHtmlForm::TGHtmlForm(int type, int argc, int arglen[], char *argv[]) :
  TGHtmlMarkupElement(type, argc, arglen, argv)
{
   //

   formId = 0;
   elements = 0;
   hasctl = 0;
   pFirst = 0;
   pEnd = 0;
}

//______________________________________________________________________________
TGHtmlHr::TGHtmlHr(int type, int argc, int arglen[], char *argv[]) :
  TGHtmlMarkupElement(type, argc, arglen, argv)
{
   //

   x = 0; y = 0; w = 0; h = 0;
   is3D = 0;
}

//______________________________________________________________________________
TGHtmlAnchor::TGHtmlAnchor(int type, int argc, int arglen[], char *argv[]) :
  TGHtmlMarkupElement(type, argc, arglen, argv)
{
   //

   y = 0;
}

//______________________________________________________________________________
TGHtmlScript::TGHtmlScript(int type, int argc, int arglen[], char *argv[]) :
  TGHtmlMarkupElement(type, argc, arglen, argv)
{
   //

   nStart = -1;
   nScript = 0;
}

//______________________________________________________________________________
TGHtmlMapArea::TGHtmlMapArea(int type, int argc, int arglen[], char *argv[]) :
  TGHtmlMarkupElement(type, argc, arglen, argv)
{
   //

   mType = 0;
   coords = 0;
   num = 0;
}
   

//----------------------------------------------------------------------

#if 0
TGHtmlBlock::TGHtmlBlock() {
}

TGHtmlBlock::~TGHtmlBlock() {
}
#endif

//______________________________________________________________________________
TGHtmlInput::TGHtmlInput(int type, int argc, int arglen[], char *argv[]) :
  TGHtmlMarkupElement(type, argc, arglen, argv)
{
   //

   pForm = 0;
   iNext = 0;
   frame = 0;
   html = 0; 
   pEnd = 0;
   inpId = 0; subId = 0;
   x = 0; y = 0; w = 0; h = 0;
   padLeft = 0;
   align = 0;  
   textAscent = 0;
   textDescent = 0;
   itype = 0;
   sized = 0;
   cnt = 0;  
}
   
//______________________________________________________________________________
void TGHtmlInput::Empty()
{
   // Mark this element as being empty. It has no widget and doesn't appear on
   // the screen.
   //
   // This is called for HIDDEN inputs or when the corresponding widget is
   // not created.

   frame = NULL;
   w = 0;
   h = 0;
   flags &= ~HTML_Visible;
   style.flags |= STY_Invisible;
   sized = 1;
}

