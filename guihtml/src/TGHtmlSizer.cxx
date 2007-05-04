// $Id$
// Author:  Valeriy Onuchin   03/05/2007

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

// Routines used to compute the style and size of individual elements.

#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "TGHtml.h"
#include "TImage.h"

//______________________________________________________________________________
SHtmlStyle TGHtml::GetCurrentStyle()
{
   // Get the current rendering style. In other words, get the style
   // that is currently on the top of the style stack.

   SHtmlStyle style;

   if (styleStack) {
      style = styleStack->style;
   } else {
      style.font = NormalFont(2);
      style.color = COLOR_Normal;
      style.bgcolor = COLOR_Background;
      style.subscript = 0;
      style.align = ALIGN_Left;
      style.flags = 0;
      style.expbg = 0;
   }

   return style;
}

//______________________________________________________________________________
void TGHtml::PushStyleStack(int tag, SHtmlStyle style)
{
   // Push a new rendering style onto the stack.
   //
   //  tag   - Tag for this style. Normally the end-tag such as </h3> or </em>.
   //  style - The style to push

   SHtmlStyleStack *p;

   p = new SHtmlStyleStack;
   p->pNext = styleStack;
   p->type = tag;
   p->style = style;
   styleStack = p;
}

//______________________________________________________________________________
SHtmlStyle TGHtml::PopStyleStack(int tag)
{
   // Pop a rendering style off of the stack.
   //
   // The top-most style on the stack should have a tag equal to "tag".
   // If not, then we have an HTML coding error.  Perhaps something
   // like this:  "Some text <em>Enphasized</i> more text".  It is an
   // interesting problem to figure out how to respond sanely to this
   // kind of error.  Our solution is to keep popping the stack until
   // we find the correct tag, or until the stack is empty.

   int i, type;
   SHtmlStyleStack *p;
   static Html_u8 priority[Html_TypeCount+1];

   if (priority[Html_TABLE] == 0) {
      for (i = 0; i <= Html_TypeCount; i++) priority[i] = 1;
      priority[Html_TD] = 2;
      priority[Html_EndTD] = 2;
      priority[Html_TH] = 2;
      priority[Html_EndTH] = 2;
      priority[Html_TR] = 3;
      priority[Html_EndTR] = 3;
      priority[Html_TABLE] = 4;
      priority[Html_EndTABLE] = 4;
   }
   if (tag <= 0 || tag > Html_TypeCount) {
      CANT_HAPPEN;
      return GetCurrentStyle();
   }
   while ((p = styleStack) != 0) {
      type = p->type;
      if (type <= 0 || type > Html_TypeCount) {
         CANT_HAPPEN;
         return GetCurrentStyle();
      }
      if (type != tag && priority[type] > priority[tag]) {
         return GetCurrentStyle();
      }
      styleStack = p->pNext;
      delete p;
      if (type == tag) break;
   }

   return GetCurrentStyle();
}

//______________________________________________________________________________
static void ScaleFont(SHtmlStyle *pStyle, int delta)
{
   // Change the font size on the given style by the delta-amount given

   int size = FontSize(pStyle->font) + delta;

   if (size < 0) {
      delta -= size;
   } else if (size > 6) {
      delta -= size-6;
   }

   pStyle->font += delta;
}

//______________________________________________________________________________
void TGHtml::MakeInvisible(TGHtmlElement *p_first, TGHtmlElement *p_last)
{
   // Add the STY_Invisible style to every token between p_first and p_last.

   if (p_first == 0) return;
   p_first = p_first->pNext;
   while (p_first && p_first != p_last) {
      p_first->style.flags |= STY_Invisible;
      p_first = p_first->pNext;
   }
}

//______________________________________________________________________________
int TGHtml::GetLinkColor(char *zURL)
{
   // For the markup <a href=XXX>, find out if the URL has been visited
   // before or not.  Return COLOR_Visited or COLOR_Unvisited, as 
   // appropriate.

   return IsVisited(zURL) ? COLOR_Visited : COLOR_Unvisited;
}

//______________________________________________________________________________
static int *GetCoords(char *str, int *nptr)
{
   //

   char *cp = str, *ncp;
   int  *cr, i, n = 0, sz = 4;

   cr = new int[sz];
   while (cp) {
      while (*cp && (!isdigit(*cp))) cp++;
      if ((!*cp) || (!isdigit(*cp))) break;
      cr[n] = (int) strtol(cp, &ncp, 10);
      if (cp == ncp) break;
      cp = ncp;
      n++;
      if (n >= sz) {
         int *tmp = new int[sz+4];
         for (i = 0; i < sz; ++i) tmp[i] = cr[i];
         delete[] cr;
         cr = tmp;
         sz += 4;
      }
   }
   *nptr = n;

   return cr;
}

//______________________________________________________________________________
void TGHtml::AddStyle(TGHtmlElement *p)
{
   // This routine adds information to the input texts that doesn't change
   // when the display is resized or when new fonts are selected, etc.
   // Mostly this means adding style attributes.  But other constant
   // information (such as numbering on <li> and images used for <IMG>)
   // is also obtained.  The key is that this routine is only called
   // once, where the sizer and layout routines can be called many times.
   //
   // This routine is called whenever the list of elements grows.  The
   // style stack is stored as part of the HTML widget so that we can
   // always continue where we left off the last time.
   //
   // In addition to adding style, this routine will invoke methods
   // needed to acquire information about a markup. The IsVisitied()
   // method is called for each <a> and the GetImage() is called
   // for each <IMG> or for each <LI> that has a SRC= field.
   //
   // When a markup is inserted or deleted from the token list, the
   // style routine must be completely rerun from the beginning.  So
   // what we said above, that this routine is only run once, is not
   // strictly true.

   SHtmlStyle style;         // Current style
   int size;                 // A new font size
   int i;                    // Loop counter
   int paraAlign;            // Current paragraph alignment
   int rowAlign;             // Current table row alignment
   SHtmlStyle nextStyle;     // Style for next token if useNextStyle==1
   int useNextStyle = 0;     // True if nextStyle is valid
   char *z;                  // A tag parameter's value

   // The size of header fonts relative to the current font size
   static int header_sizes[] = { +2, +1, 1, 1, -1, -1 };

   // Don't allow recursion
   if (flags & STYLER_RUNNING) return;
   flags |= STYLER_RUNNING;

   // Load the style state out of the TGHtml object and into local
   // variables. This is purely a matter of convenience...

   style = GetCurrentStyle();
   paraAlign = paraAlignment;
   rowAlign = rowAlignment;

   // Loop over tokens
   while (pFirst && p) {
      switch (p->type) {
         case Html_A:
            if (anchorStart) {
               style = PopStyleStack(Html_EndA);
               anchorStart = 0;
               anchorFlags = 0;
            }
            z = p->MarkupArg("href", 0);
            if (z) {
               style.color = GetLinkColor(z);
               if (underlineLinks) style.flags |= STY_Underline;
               anchorFlags |= STY_Anchor;
               PushStyleStack(Html_EndA, style);
               anchorStart = (TGHtmlAnchor *) p;
            }
            break;

         case Html_EndA:
            if (anchorStart) {
               ((TGHtmlRef *)p)->pOther = anchorStart;
               style = PopStyleStack(Html_EndA);
               anchorStart = 0;
               anchorFlags = 0;
            }
            break;

         case Html_MAP:
         break;

         case Html_EndMAP:
         break;

         case Html_AREA: {
            TGHtmlMapArea *area = (TGHtmlMapArea *) p;
            z = p->MarkupArg("shape", 0);
            area->mType = HTML_MAP_RECT;
            if (z) {
               if (strcasecmp(z, "circle") == 0) {
                  area->mType = HTML_MAP_CIRCLE;
               } else if (strcasecmp(z,"poly") == 0) {
                  area->mType = HTML_MAP_POLY;
               }
            }
            z = p->MarkupArg("coords", 0);
            if (z) {
               area->coords = GetCoords(z, &area->num);
            }
            break;
         }

         case Html_ADDRESS:
         case Html_EndADDRESS:
         case Html_BLOCKQUOTE:
         case Html_EndBLOCKQUOTE:
            paraAlign = ALIGN_None;
            break;

         case Html_APPLET:
            if (0 /* has ProcessApplet() */) {
               nextStyle = style;
               nextStyle.flags |= STY_Invisible;
               PushStyleStack(Html_EndAPPLET, nextStyle);
               useNextStyle = 1;
            } else {
               PushStyleStack(Html_EndAPPLET, style);
            }
            break;

         case Html_B:
            style.font = BoldFont(style.font);
            PushStyleStack(Html_EndB, style);
            break;

         case Html_BODY:
            z = p->MarkupArg("text", 0);
            if (z) {
               //FreeColor(apColor[COLOR_Normal]);
               apColor[COLOR_Normal] = AllocColor(z);
            }
            z = p->MarkupArg("bgcolor", 0);
            if (z) {
               //FreeColor(apColor[COLOR_Background]);
               apColor[COLOR_Background] = AllocColor(z);
               SetBackgroundColor(apColor[COLOR_Background]->fPixel);
               SetBackgroundPixmap(0);
            }
            z = p->MarkupArg("link", 0);
            if (z) {
               //FreeColor(apColor[COLOR_Unvisited]);
               apColor[COLOR_Unvisited] = AllocColor(z);
            }
            z = p->MarkupArg("vlink", 0);
            if (z) {
               //FreeColor(apColor[COLOR_Visited]);
               apColor[COLOR_Visited] = AllocColor(z);
            }
            z = p->MarkupArg("alink", 0);
            if (z) {
            }
            z = p->MarkupArg("background", 0);
            if (z) {
               z = ResolveUri(z);
               if (z) {
                  TImage *img = LoadImage(z, 0, 0);
                  if (img) {
#if 0
              SetupBackgroundPic(img->GetPicture());
#else
                     GCValues_t gcv;
                     unsigned int mask;

                     mask = kGCTile | kGCFillStyle | kGCGraphicsExposures;
                     gcv.fTile = img->GetPixmap();
                     gcv.fFillStyle = kFillTiled;
                     gcv.fGraphicsExposures = kTRUE;
                     fCanvas->SetBackgroundPixmap(img->GetPixmap());

                     gVirtualX->ChangeGC(fWhiteGC.GetGC(), &gcv);

              //NeedRedraw(TGRectangle(fVisible, fCanvas->GetSize()));
#endif
                     bgImage = img;//delete img;
                  }
                  delete [] z;
               }
            }
            break;

         case Html_EndBODY:
            break;

         case Html_EndAPPLET:
         case Html_EndB:
         case Html_EndBIG:
         case Html_EndCENTER:
         case Html_EndCITE:
         case Html_EndCODE:
         case Html_EndCOMMENT:
         case Html_EndDFN:
         case Html_EndEM:
         case Html_EndFONT:
         case Html_EndI:
         case Html_EndKBD:
         case Html_EndMARQUEE:
         case Html_EndNOBR:
         case Html_EndNOFRAMES:
         case Html_EndNOSCRIPT:
         case Html_EndNOEMBED:
         case Html_EndS:
         case Html_EndSAMP:
         case Html_EndSMALL:
         case Html_EndSTRIKE:
         case Html_EndSTRONG:
         case Html_EndSUB:
         case Html_EndSUP:
         case Html_EndTITLE:
         case Html_EndTT:
         case Html_EndU:
         case Html_EndVAR:
            style = PopStyleStack(p->type);
            break;

         case Html_BASE:
            z = p->MarkupArg("href", 0);
            if (z) {
               z = ResolveUri(z);
               if (z != 0) {
                  if (zBaseHref) delete[] zBaseHref;
                  zBaseHref = z;
               }
            }
            break;

         case Html_EndDIV:
            paraAlign = ALIGN_None;
            style = PopStyleStack(p->type);
            break;

         case Html_EndBASEFONT:
            style = PopStyleStack(Html_EndBASEFONT);
            style.font = FontFamily(style.font) + 2;
            break;

         case Html_BIG:
            ScaleFont(&style, 1);
            PushStyleStack(Html_EndBIG, style);
            break;

         case Html_CAPTION:
            paraAlign = p->GetAlignment(paraAlign);
            break;

         case Html_EndCAPTION:
            paraAlign = ALIGN_None;
            break;

         case Html_CENTER:
            paraAlign = ALIGN_None;
            style.align = ALIGN_Center;
            PushStyleStack(Html_EndCENTER, style);
            break;

         case Html_CITE:
            style.font = ItalicFont(style.font);
            PushStyleStack(Html_EndCITE, style);
            break;

         case Html_CODE:
            style.font = CWFont(style.font);
            PushStyleStack(Html_EndCODE, style);
            break;

         case Html_COMMENT:
            style.flags |= STY_Invisible;
            PushStyleStack(Html_EndCOMMENT, style);
            break;

         case Html_DD:
            if (innerList && innerList->type == Html_DL) {
               ((TGHtmlRef *)p)->pOther = innerList;
            } else {
               ((TGHtmlRef *)p)->pOther = 0;
            }
            inDt = 0;
            break;

         case Html_DFN:
            style.font = ItalicFont(style.font);
            PushStyleStack(Html_EndDFN, style);
            break;

         case Html_DIR:
         case Html_MENU:
         case Html_UL: {
            TGHtmlListStart *list = (TGHtmlListStart *) p;
            list->lPrev = innerList;
            list->cnt = 0;
            innerList = list;
            if (list->lPrev == 0) {
               list->ltype = LI_TYPE_Bullet1;
               list->compact = (list->MarkupArg("compact", 0) != 0);
            } else if (list->lPrev->lPrev == 0) {
               list->ltype = LI_TYPE_Bullet2;
               list->compact = 1;
            } else {
               list->ltype = LI_TYPE_Bullet3;
               list->compact = 1;
            }
            list->ltype = list->GetUnorderedListType(list->ltype);
            break;
         }

         case Html_EndDL:
            inDt = 0;
            /* Fall thru into the next case */
         case Html_EndDIR:
         case Html_EndMENU:
         case Html_EndOL:
         case Html_EndUL:
            ((TGHtmlRef *)p)->pOther = innerList;
            if (innerList) innerList = innerList->lPrev;
            break;

         case Html_DIV:
            paraAlign = ALIGN_None;
            style.align = p->GetAlignment(style.align);
            PushStyleStack(Html_EndDIV, style);
            break;

         case Html_DT:
            if (innerList && innerList->type == Html_DL) {
               ((TGHtmlRef *)p)->pOther = innerList;
            } else {
               ((TGHtmlRef *)p)->pOther = 0;
            }
            inDt = STY_DT;
            break;

         case Html_EndDD:
         case Html_EndDT:
            inDt = 0;
            break;

         case Html_DL: {
            TGHtmlListStart *list = (TGHtmlListStart *) p;
            list->lPrev = innerList;
            list->cnt = 0;
            innerList = list;
            list->compact = (list->MarkupArg("compact", 0) != 0);
            inDt = 0;
            break;
         }

         case Html_EM:
            style.font = ItalicFont(style.font);
            PushStyleStack(Html_EndEM, style);
            break;

         case Html_EMBED:
            break;

         case Html_BASEFONT:
         case Html_FONT:
            z = p->MarkupArg("size", 0);
            if (z && !overrideFonts) {
               if (*z == '-') {
                  size = FontSize(style.font) - atoi(&z[1]) +1;
               } else if (*z == '+') {
                  size = FontSize(style.font) + atoi(&z[1]) +1;
               } else {
                  size = atoi(z);
               }
               if (size <= 0) size = 1;
               if (size >= N_FONT_SIZE) size = N_FONT_SIZE - 1;
               style.font = FontFamily(style.font) + size - 1;
            }
            z = p->MarkupArg("color", 0);
            if (z && *z && !overrideColors) style.color = GetColorByName(z);
            PushStyleStack(p->type == Html_FONT ?
                           Html_EndFONT : Html_EndBASEFONT, style);
            break;

         case Html_FORM: {
            TGHtmlForm *form = (TGHtmlForm *) p;

            char *zUrl;
            char *zMethod;
            TGString cmd("");
            int result;
            char zToken[50];

            formStart = 0;
            //form->formId = 0;

            zUrl = p->MarkupArg("action", 0);
            if (zUrl == 0) zUrl = zBase;
            zUrl = ResolveUri(zUrl);
            if (zUrl == 0) zUrl = StrDup("");
            zMethod = p->MarkupArg("method", "GET");
            sprintf(zToken, " %d form ", form->formId);
            cmd.Append("Form:");
            cmd.Append(zToken);
            cmd.Append(zUrl);
            cmd.Append(" ");
            cmd.Append(zMethod);
            cmd.Append(" { ");
            AppendArglist(&cmd, (TGHtmlMarkupElement *) p);
            cmd.Append("} ");
            result = FormCreate(form, zUrl, cmd.GetString());
            delete[] zUrl;

            /*if (result)*/ formStart = form;

            break;
         }

         case Html_EndFORM:
            ((TGHtmlRef *)p)->pOther = formStart;
            if (formStart) formStart->pEnd = p;
            formStart = 0;
            break;

         case Html_H1:
         case Html_H2:
         case Html_H3:
         case Html_H4:
         case Html_H5:
         case Html_H6:
            if (!inTr) paraAlign = ALIGN_None;
            i = (p->type - Html_H1) / 2 + 1;
            if (i >= 1 && i <= 6) {
               ScaleFont(&style, header_sizes[i-1]);
            }
            style.font = BoldFont(style.font);
            style.align = p->GetAlignment(style.align);
            PushStyleStack(Html_EndH1, style);
            break;

         case Html_EndH1:
         case Html_EndH2:
         case Html_EndH3:
         case Html_EndH4:
         case Html_EndH5:
         case Html_EndH6:
            paraAlign = ALIGN_None;
            style = PopStyleStack(Html_EndH1);
            break;

         case Html_HR:
            nextStyle = style;
            style.align = p->GetAlignment(ALIGN_None);
            useNextStyle = 1;
            break;

         case Html_I:
            style.font = ItalicFont(style.font);
            PushStyleStack(Html_EndI, style);
            break;

         case Html_IMG:
            if (style.flags & STY_Invisible) break;
            ((TGHtmlImageMarkup *)p)->pImage = GetImage((TGHtmlImageMarkup *) p);
            break;

         case Html_OPTION:
            break;

         case Html_INPUT:
            ((TGHtmlInput *)p)->pForm = formStart;
            ////ControlSize((TGHtmlInput *) p);
            break;

         case Html_KBD:
            style.font = CWFont(style.font);
            PushStyleStack(Html_EndKBD, style);
            break;

         case Html_LI:
            if (innerList) {
               TGHtmlLi *li = (TGHtmlLi *) p;
               li->ltype = innerList->ltype;
               if (innerList->type == Html_OL) {
                  z = li->MarkupArg("value", 0);
                  if (z) {
                     int n = atoi(z);
                     if (n > 0) {
                        li->cnt = n;
                        innerList->cnt = n+1;
                     }
                  } else {
                     li->cnt = innerList->cnt++;
                  }
                  li->ltype = li->GetOrderedListType(li->ltype);
               } else {
                  li->ltype = li->GetUnorderedListType(li->ltype);
               }
            } else {
               p->flags &= ~HTML_Visible;
            }
            break;

         case Html_MARQUEE:
            style.flags |= STY_Invisible;
            PushStyleStack(Html_EndMARQUEE, style);
            break;

         case Html_NOBR:
            style.flags |= STY_NoBreak;
            PushStyleStack(Html_EndNOBR, style);
            break;

         case Html_NOFRAMES:
            if (0 /* has ProcessFrame()*/) {
               nextStyle = style;
               nextStyle.flags |= STY_Invisible;
               PushStyleStack(Html_EndNOFRAMES, nextStyle);
               useNextStyle = 1;
            } else {
               PushStyleStack(Html_EndNOFRAMES, style);
            }
            break;

         case Html_NOEMBED:
            if (0 /* has ProcessScript() && HasScript */) {
               nextStyle = style;
               nextStyle.flags |= STY_Invisible;
               PushStyleStack(Html_EndNOEMBED, nextStyle);
               useNextStyle = 1;
            } else {
               PushStyleStack(Html_EndNOEMBED, style);
            }
            break;

         case Html_NOSCRIPT:
            if (0 /* has ProcessScript() && HasScript */) {
               nextStyle = style;
               nextStyle.flags |= STY_Invisible;
               PushStyleStack(Html_EndNOSCRIPT, nextStyle);
               useNextStyle = 1;
            } else {
               PushStyleStack(Html_EndNOSCRIPT, style);
            }
            break;

         case Html_OL: {
            TGHtmlListStart *list = (TGHtmlListStart *) p;
            list->lPrev = innerList;
            list->ltype = list->GetOrderedListType(LI_TYPE_Enum_1);
            list->cnt = 1;
            z = list->MarkupArg("start", 0);
            if (z) {
               int n = atoi(z);
               if (n > 0) list->cnt = n;
            }
            list->compact = (innerList != 0 || list->MarkupArg("compact", 0) != 0);
            innerList = list;
            break;
         }

         case Html_P:
            paraAlign = p->GetAlignment(ALIGN_None);
            break;

         case Html_EndP:
            paraAlign = ALIGN_None;
            break;

         case Html_PRE:
         case Html_LISTING:
         case Html_XMP:
         case Html_PLAINTEXT:
            paraAlign = ALIGN_None;
            style.font = CWFont(style.font);
            style.flags |= STY_Preformatted;
            PushStyleStack(Html_EndPRE, style);
            break;

         case Html_EndPRE:
         case Html_EndLISTING:
         case Html_EndXMP:
            style = PopStyleStack(Html_EndPRE);
            break;

         case Html_S:
            style.flags |= STY_StrikeThru;
            PushStyleStack(Html_EndS, style);
            break;

         case Html_SCRIPT: {
            char *result;
            result = ProcessScript((TGHtmlScript *) p);   // zText[script->nStart .. script->nScript]
            if (result) {
               TGHtmlElement *b2 = p->pNext, *b3, *e1 = p, *e2 = b2, *e3;
               if (e2) while (e2->pNext) e2 = e2->pNext;
               TokenizerAppend(result);
               if (e2 && e2 != p && ((e3 = b3 = e2->pNext))) {
                  while (e3->pNext) e3 = e3->pNext;
                  e1->pNext = b3;
                  e2->pNext = 0;   b2->pPrev = e3;
                  e3->pNext = b2;  b3->pPrev = e1;
               }
               delete[] result;
            }
            nextStyle = style;
            style.flags |= STY_Invisible;
            useNextStyle = 1;
            break;
         }

         case Html_SELECT:
            ((TGHtmlInput *)p)->pForm = formStart;
            nextStyle.flags |= STY_Invisible;
            useNextStyle = 1;
            PushStyleStack(Html_EndSELECT, style);
            formElemStart = (TGHtmlInput *) p;
            break;

         case Html_EndSELECT:
            style = PopStyleStack(Html_EndSELECT);
            if (formElemStart && formElemStart->type == Html_SELECT) {
                ((TGHtmlRef *)p)->pOther = formElemStart;
               MakeInvisible(((TGHtmlRef *)p)->pOther, p);
            } else {
               ((TGHtmlRef *)p)->pOther = 0;
            }
            formElemStart = 0;
            break;

         case Html_STRIKE:
            style.flags |= STY_StrikeThru;
            PushStyleStack(Html_EndSTRIKE, style);
            break;

         case Html_STYLE:
            // Ignore style sheets
            break;

         case Html_SAMP:
            style.font = CWFont(style.font);
            PushStyleStack(Html_EndSAMP, style);
            break;

         case Html_SMALL:
            ScaleFont(&style, -1);
            PushStyleStack(Html_EndSMALL, style);
            break;

         case Html_STRONG:
            style.font = BoldFont(style.font);
            PushStyleStack(Html_EndSTRONG, style);
            break;

         case Html_SUB:
            ScaleFont(&style, -1);
            if (style.subscript > -6 ) style.subscript--;
            PushStyleStack(Html_EndSUB, style);
            break;

         case Html_SUP:
            ScaleFont(&style, -1);
            if (style.subscript < 6) style.subscript++;
            PushStyleStack(Html_EndSUP, style);
            break;

         case Html_TABLE:
            paraAlign = ALIGN_None;
            nextStyle = style;
            if (style.flags & STY_Preformatted) {
               nextStyle.flags &= ~STY_Preformatted;
               style.flags |= STY_Preformatted;
            }
            nextStyle.align = ALIGN_Left;
            z = p->MarkupArg("bgcolor", 0);
            if (z && *z && !overrideColors) {
               style.bgcolor = nextStyle.bgcolor = GetColorByName(z);
               style.expbg = 1;
//        } else {
//          nextStyle.bgcolor = COLOR_Background;
            }
            TableBgndImage(p);
            PushStyleStack(Html_EndTABLE, nextStyle);
            useNextStyle = 1;
            inTd = 0;
            inTr = 0;
            break;

         case Html_EndTABLE:
            paraAlign = ALIGN_None;
            if (inTd) {
               style = PopStyleStack(Html_EndTD);
               inTd = 0;
            }
            if (inTr) {
               style = PopStyleStack(Html_EndTR);
               inTr = 0;
            }
            style = PopStyleStack(p->type);
            break;

         case Html_TD:
            if (inTd) style = PopStyleStack(Html_EndTD);
            inTd = 1;
            paraAlign = p->GetAlignment(rowAlign);
            z = p->MarkupArg("bgcolor", 0);
            if (z && *z && !overrideColors) {
               style.bgcolor = GetColorByName(z);
               style.expbg = 1;
            }
            TableBgndImage(p);
            PushStyleStack(Html_EndTD, style);
            break;

         case Html_TEXTAREA:
            ((TGHtmlInput *)p)->pForm = formStart;
            nextStyle = style;
            nextStyle.flags |= STY_Invisible;
            PushStyleStack(Html_EndTEXTAREA, nextStyle);
            formElemStart = (TGHtmlInput *) p;
            useNextStyle = 1;
            break;

         case Html_EndTEXTAREA:
            style = PopStyleStack(Html_EndTEXTAREA);
            if (formElemStart && formElemStart->type == Html_TEXTAREA) {
               ((TGHtmlRef *)p)->pOther = formElemStart;
            } else {
               ((TGHtmlRef *)p)->pOther = 0;
            }
            formElemStart = 0;
            break;

         case Html_TH:
            //paraAlign = p->GetAlignment(rowAlign);
            if (inTd) style = PopStyleStack(Html_EndTD);
            paraAlign = p->GetAlignment(ALIGN_Center);
            style.font = BoldFont(style.font);
            z = p->MarkupArg("bgcolor", 0);
            if (z && *z && !overrideColors) {
               style.bgcolor = GetColorByName(z);
               style.expbg = 1;
            }
            PushStyleStack(Html_EndTD, style);
            inTd = 1;
            break;

         case Html_TR:
            if (inTd) {
               style = PopStyleStack(Html_EndTD);
               inTd = 0;
            }
            if (inTr) {
               style = PopStyleStack(Html_EndTR);
            }
            rowAlign = p->GetAlignment(ALIGN_None);
            z = p->MarkupArg("bgcolor", 0);
            if (z && *z && !overrideColors) {
               style.bgcolor = GetColorByName(z);
               style.expbg = 1;
            }
            TableBgndImage(p);
            PushStyleStack(Html_EndTR, style);
            inTr = 1;
            break;

         case Html_EndTR:
            if (inTd) {
               style = PopStyleStack(Html_EndTD);
               inTd = 0;
            }
            style = PopStyleStack(Html_EndTR);
            inTr = 0;
            paraAlign = ALIGN_None;
            rowAlign = ALIGN_None;
            break;

         case Html_EndTD:
         case Html_EndTH:
            style = PopStyleStack(Html_EndTD);
            inTd = 0;
            paraAlign = ALIGN_None;
            //rowAlign = ALIGN_None;
            break;

         case Html_TITLE:
            style.flags |= STY_Invisible;
            PushStyleStack(Html_EndTITLE, style);
            break;

         case Html_TT:
            style.font = CWFont(style.font);
            PushStyleStack(Html_EndTT, style);
            break;

         case Html_U:
            style.flags |= STY_Underline;
            PushStyleStack(Html_EndU, style);
            break;

         case Html_VAR:
            style.font = ItalicFont(style.font);
            PushStyleStack(Html_EndVAR, style);
            break;

         default:
            break;
      }

      p->style = style;
      p->style.flags |= anchorFlags | inDt;
      if (paraAlign != ALIGN_None) {
         p->style.align = paraAlign;
      }
      if (useNextStyle) {
         style = nextStyle;
         style.expbg = 0;
         useNextStyle = 0;
      }

      TRACE(HtmlTrace_Style,
          ("Style of 0x%08x font=%02d color=%02d bg=%02d "
           "align=%d flags=0x%04x token=%s\n",
           (int)p, p->style.font, p->style.color, p->style.bgcolor,
           p->style.align, p->style.flags, DumpToken(p)));

      p = p->pNext;
   }

   // Copy state information back into the TGHtml object for safe keeping.

   paraAlignment = paraAlign;
   rowAlignment = rowAlign;

   flags &= ~STYLER_RUNNING;
}

//______________________________________________________________________________
void TGHtml::TableBgndImage(TGHtmlElement *p)
{
   //

   char *z;

   z = p->MarkupArg("background", 0);
   if (!z) return;

   z = ResolveUri(z);
   TImage *img = LoadImage(z, 0, 0);

   switch (p->type) {
      case Html_TABLE: {
         TGHtmlTable *table = (TGHtmlTable *) p;
         if (table->bgImage) delete table->bgImage;
         table->bgImage = img;
         break;
      }
      case Html_TR: {
         TGHtmlRef *ref = (TGHtmlRef *) p;
         if (ref->bgImage) delete ref->bgImage;
            ref->bgImage = img;
            break;
      }
      case Html_TH:     
      case Html_TD: {
         TGHtmlCell *cell = (TGHtmlCell *) p;
         if (cell->bgImage) delete cell->bgImage;
            cell->bgImage = img;
            break;
         }
      default:
         if (img) delete img;
         break;
   }
}

//______________________________________________________________________________
void TGHtml::Sizer()
{
   // Compute the size of all elements in the widget. Assume that a style has
   // already been assigned to all elements.
   //
   // Some of the elements might have already been sized. Refer to the
   // lastSized and only compute sizes for elements that follow this one. If
   // lastSized is 0, then size everything.
   //
   // This routine only computes the sizes of individual elements. The size of
   // aggregate elements (like tables) are computed separately.
   //
   // The HTML_Visible flag is also set on every element that results in ink on
   // the page.
   //
   // This routine may invoke a callback procedure which could delete the HTML
   // widget.

   TGHtmlElement *p;
   int iFont = -1;
   TGFont *font;
   int spaceWidth = 0;
   FontMetrics_t fontMetrics;
   char *z;
   int stop = 0;

   if (pFirst == 0) return;

   if (lastSized == 0) {
      p = pFirst;
   } else {
      p = lastSized->pNext;
   }

   for (; !stop && p; p = p ? p->pNext : 0) {
      if (p->style.flags & STY_Invisible) {
         p->flags &= ~HTML_Visible;
         continue;
      }
      if (iFont != (int)p->style.font) {
         iFont = p->style.font;
         font = GetFont(iFont);
         font->GetFontMetrics(&fontMetrics);
         spaceWidth = 0;
      }
      switch (p->type) {
         case Html_Text: {
            TGHtmlTextElement *text = (TGHtmlTextElement *) p;
            text->w = font->TextWidth(text->zText, p->count);
            p->flags |= HTML_Visible;
            text->descent = fontMetrics.fDescent;
            text->ascent = fontMetrics.fAscent;
            if (spaceWidth == 0) spaceWidth = font->TextWidth(" ", 1);
            text->spaceWidth = spaceWidth;
            break;
         }

         case Html_Space: {
            TGHtmlSpaceElement *space = (TGHtmlSpaceElement *) p;
            if (spaceWidth == 0) spaceWidth = font->TextWidth(" ", 1);
            space->w = spaceWidth;
            space->descent = fontMetrics.fDescent;
            space->ascent = fontMetrics.fAscent;
            p->flags &= ~HTML_Visible;
            break;
         }

         case Html_TD:
         case Html_TH: {
            TGHtmlCell *cell = (TGHtmlCell *) p;
            z = p->MarkupArg("rowspan", "1");
            cell->rowspan = atoi(z);
            z = p->MarkupArg("colspan", "1");
            cell->colspan = atoi(z);
            p->flags |= HTML_Visible;
            break;
         }

         case Html_LI: {
            TGHtmlLi *li = (TGHtmlLi *) p;
            li->descent = fontMetrics.fDescent;
            li->ascent = fontMetrics.fAscent;
            p->flags |= HTML_Visible;
            break;
         }

         case Html_IMG: {
            TGHtmlImageMarkup *image = (TGHtmlImageMarkup *) p;
            z = p->MarkupArg("usemap", 0);
            if (z && *z == '#') {
               image->pMap = GetMap(z+1);
            } else {
               image->pMap = 0;
            }
            p->flags |= HTML_Visible;
            image->redrawNeeded = 0;
            image->textAscent = fontMetrics.fAscent;
            image->textDescent = fontMetrics.fDescent;
            image->align = GetImageAlignment(p);
            if (image->pImage == 0) {
               image->ascent = fontMetrics.fAscent;
               image->descent = fontMetrics.fDescent;
               image->zAlt = p->MarkupArg("alt", "<image>");
               image->w = font->TextWidth(image->zAlt, strlen(image->zAlt));
            } else {
               int w, h;
               image->iNext = image->pImage->pList;
               image->pImage->pList = image;
               w = image->pImage->image->GetWidth();
               h = image->pImage->image->GetHeight();
               image->h = h;
               image->w = w;
               image->ascent = h / 2;
               image->descent = h - image->ascent;
            }
            if ((z = p->MarkupArg("width", 0)) != 0) {
               int w = atoi(z);
               if (z[strlen(z)-1] == '%') w = 0; //// -- HP
               if (w > 0) image->w = w;
            }
            if ((z = p->MarkupArg("height", 0)) != 0) {
               int h = atoi(z);
               if (h > 0) image->h = h;
            }

#if 1  // --HP
            if (image->pImage == 0 && !*image->zAlt) {
               image->ascent = image->h / 2;
               image->descent = image->h - image->ascent;
            }
#endif
            break;
         }

         case Html_TABLE:
            p->flags |= HTML_Visible;
            break;

         case Html_HR:
            p->flags |= HTML_Visible;
            break;

         case Html_APPLET:
         case Html_EMBED:
         case Html_INPUT: {
            TGHtmlInput *input = (TGHtmlInput *) p;
            input->textAscent = fontMetrics.fAscent;
            input->textDescent = fontMetrics.fDescent;
            stop = ControlSize(input);
            break;
         }

         case Html_SELECT:
         case Html_TEXTAREA: {
            TGHtmlInput *input = (TGHtmlInput *) p;
            input->textAscent = fontMetrics.fAscent;
            input->textDescent = fontMetrics.fDescent;
            break;
         }

         case Html_EndSELECT:
         case Html_EndTEXTAREA: {
            TGHtmlRef *ref = (TGHtmlRef *) p;
            if (ref->pOther) {
               ((TGHtmlInput *)ref->pOther)->pEnd = p;
               stop = ControlSize((TGHtmlInput *) ref->pOther);
            }
            break;
         }

         default:
            p->flags &= ~HTML_Visible;
            break;
      }
   }

   if (p) {
      lastSized = p;
   } else {
      lastSized = pLast;
   }
}
