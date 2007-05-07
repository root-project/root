// $Id: TGHtmlLayout.cxx,v 1.1 2007/05/04 17:07:01 brun Exp $
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

// This file contains the code used to position elements of the
// HTML file on the screen.

#include <stdlib.h>
#include <string.h>

#include "TGHtml.h"


//______________________________________________________________________________
TGHtmlLayoutContext::TGHtmlLayoutContext()
{
   //

   pStart = 0;
   pEnd = 0;
   leftMargin = 0;
   rightMargin = 0;
   Reset();
}

//______________________________________________________________________________
void TGHtmlLayoutContext::Reset()
{
   // Reset the layout context.

   headRoom = 0;
   top = 0;
   bottom = 0;   
   ClearMarginStack(&leftMargin);
   ClearMarginStack(&rightMargin);
}

//______________________________________________________________________________
void TGHtmlLayoutContext::PushMargin(SHtmlMargin **ppMargin,
                                    int indent, int bottom, int tag)
{
   // Push a new margin onto the given margin stack.
   //
   // If the "bottom" parameter is non-negative, then this margin will
   // automatically expire for all text that is placed below the y-coordinate
   // given by "bottom". This feature is used for <IMG ALIGN=left> and <IMG
   // ALIGN=right> kinds of markup. It allows text to flow around an image.
   //
   // If "bottom" is negative, then the margin stays in force until it is
   // explicitly canceled by a call to PopMargin().
   //
   //  ppMargin - The margin stack onto which to push
   //  indent   - The indentation for the new margin
   //  bottom   - The margin expires at this Y coordinate
   //  tag      - Markup that will cancel this margin

   SHtmlMargin *pNew = new SHtmlMargin;
   pNew->pNext = *ppMargin;
   if (pNew->pNext) {
      pNew->indent = indent + pNew->pNext->indent;
   } else {
      pNew->indent = indent;
   }
   pNew->bottom = bottom;
   pNew->tag = tag;
   *ppMargin = pNew;
}

//______________________________________________________________________________
void TGHtmlLayoutContext::PopOneMargin(SHtmlMargin **ppMargin)
{
   // Pop one margin off of the given margin stack.

   if (*ppMargin) {
      SHtmlMargin *pOld = *ppMargin;
      *ppMargin = pOld->pNext;
      delete pOld;
   }
}

//______________________________________________________________________________
void TGHtmlLayoutContext::PopMargin(SHtmlMargin **ppMargin, int tag)
{
   // Pop as many margins as necessary until the margin that was
   // created with "tag" is popped off. Update the layout context
   // to move past obstacles, if necessary.
   //
   // If there are some margins on the stack that contain non-negative
   // bottom fields, that means there are some obstacles that we have
   // not yet cleared. If these margins get popped off the stack,
   // then we have to be careful to advance the 'bottom' value so
   // that the next line of text will clear the obstacle.

   int bot = -1;
   int oldTag;
   SHtmlMargin *pM;

   for (pM = *ppMargin; pM && pM->tag != tag; pM = pM->pNext) {}
   if (pM == 0) {
      // No matching margin is found. Do nothing.
      return;
   }
   while ((pM = *ppMargin) != 0) {
      if (pM->bottom > bot) bot = pM->bottom;
      oldTag = pM->tag;
      PopOneMargin(ppMargin);
      if (oldTag == tag) break;
   }
   if (bottom < bot) {
      headRoom += bot - bottom;
      bottom = bot;
   }
}

//______________________________________________________________________________
void TGHtmlLayoutContext::PopExpiredMargins(SHtmlMargin **ppMarginStack, int y)
{
   // Pop all expired margins from the stack. 
   //
   // An expired margin is one with a non-negative bottom parameter
   // that is less than the value "y". "y" is the Y-coordinate of
   // the top edge the next line of text to by positioned. What this
   // function does is check to see if we have cleared any obstacles
   // (an obstacle is an <IMG ALIGN=left> or <IMG ALIGN=right>) and
   // expands the margins if we have.

   while (*ppMarginStack && (**ppMarginStack).bottom >= 0 &&
         (**ppMarginStack).bottom <= y) {
      PopOneMargin(ppMarginStack);
   }
}

//______________________________________________________________________________
void TGHtmlLayoutContext::ClearMarginStack(SHtmlMargin **ppMargin)
{
   // Clear a margin stack to reclaim memory. This routine just blindly
   // pops everything off the stack. Typically used when the screen is
   // cleared or the widget is deleted, etc.

  while (*ppMargin) PopOneMargin(ppMargin);
}

//______________________________________________________________________________
TGHtmlElement *TGHtmlLayoutContext::GetLine(TGHtmlElement *p_start,
                TGHtmlElement *p_end, int width, int minX, int *actualWidth)
{
   // This routine gathers as many tokens as will fit on one line.
   //
   // The candidate tokens begin with pStart and go thru the end of
   // the list or to pEnd, whichever comes first. The first token
   // at the start of the next line is returned. NULL is returned if
   // we exhaust data.
   //
   // "width" is the maximum allowed width of the line. The actual
   // width is returned in *actualWidth. The actual width does not
   // include any trailing spaces. Sometimes the actual width will
   // be greater than the maximum width. This will happen, for example,
   // for text enclosed in <pre>..</pre> that has lines longer than
   // the width of the page.
   //
   // If the list begins with text, at least one token is returned,
   // even if that one token is longer than the allowed line length.
   // But if the list begins with some kind of break markup (possibly
   // preceded by white space) then the returned list may be empty.
   //
   // The "x" coordinates of all elements are set assuming that the line
   // begins at 0. The calling routine should adjust these coordinates
   // to position the line horizontally. (The FixLine() procedure does
   // this.)  Note that the "x" coordinate of <li> elements will be negative.
   // Text within <dt>..</dt> might also have a negative "x" coordinate.
   // But in no case will the x coordinate every be less than "minX".
   //
   // p_start     - First token on new line
   // p_end       - End of line. Might be NULL
   // width       - How much space is on this line
   // minX        - The minimum value of the X coordinate
   // actualWidth - Return space actually required

   int x;                        // Current X coordinate
   int spaceWanted = 0;          // Add this much space before next token
   TGHtmlElement *p;              // For looping over tokens
   TGHtmlElement *lastBreak = 0;  // Last line-break opportunity
   int isEmpty = 1;              // True if link contains nothing
   int origin;                   // Initial value of "x"

   *actualWidth = 0;
   p = p_start;
   while (p && p != p_end && (p->style.flags & STY_Invisible) != 0) {
      p = p->pNext;
   }
   if (p && p->style.flags & STY_DT) {
      origin = -HTML_INDENT;
   } else {
      origin = 0;
   }
   x = origin;
   if (x < minX) x = minX;
   if (p && p != p_end && p->type == Html_LI) {
      TGHtmlLi *li = (TGHtmlLi *) p;
      li->x = x - HTML_INDENT / 3;
      if (li->x - (HTML_INDENT * 2) / 3 < minX) {
         x += minX - li->x + (HTML_INDENT * 2) / 3;
         li->x = minX + (HTML_INDENT * 2) / 3;
      }
      isEmpty = 0;
      *actualWidth = 1;
      p = p->pNext;
      while (p && (p->type == Html_Space || p->type == Html_P)) {
         p = p->pNext;
      }
   }
   for (; p && p != p_end; p = p ? p->pNext : 0) {
      if (p->style.flags & STY_Invisible) continue;
      switch (p->type) {
         case Html_Text: {
            TGHtmlTextElement *text = (TGHtmlTextElement *) p;
            text->x = x + spaceWanted;
            if ((p->style.flags & STY_Preformatted) == 0) {
               if (lastBreak && x + spaceWanted + text->w > width)
                  return lastBreak;
            }
//        TRACE(HtmlTrace_GetLine2, ("Place token %s at x=%d w=%d\n",
//           HtmlTokenName(p), text->x, text->w));
            x += text->w + spaceWanted;
            isEmpty = 0;
            spaceWanted = 0;
            break;
         }

         case Html_Space: {
            TGHtmlSpaceElement *space = (TGHtmlSpaceElement *) p;
            if (p->style.flags & STY_Preformatted) {
               if (p->flags & HTML_NewLine) {
                  *actualWidth = (x <= 0) ? 1 : x;
                  return p->pNext;
               }
               x += space->w * p->count;
            } else {
               int w;
               if ((p->style.flags & STY_NoBreak) == 0) {
                  lastBreak = p->pNext;
                  *actualWidth = ((x <= 0) && !isEmpty) ? 1 : x;
               }
               w = space->w;
               if (spaceWanted < w && x > origin) spaceWanted = w;
            }
            break;
         }

         case Html_IMG: {
            TGHtmlImageMarkup *image = (TGHtmlImageMarkup *) p;
            switch (image->align) {
               case IMAGE_ALIGN_Left:
               case IMAGE_ALIGN_Right:
                  *actualWidth = ((x <= 0) && !isEmpty) ? 1 : x;
                  return p;
               default:
                  break;
            }
            image->x = x + spaceWanted;
            if ((p->style.flags & STY_Preformatted) == 0) {
               if (lastBreak && x + spaceWanted + image->w > width) {
                  return lastBreak;
               }
            }
//        TRACE(HtmlTrace_GetLine2, ("Place in-line image %s at x=%d w=%d\n",
//           HtmlTokenName(p), p->image.x, p->image.w));
            x += image->w + spaceWanted;
            if ((p->style.flags & STY_NoBreak) == 0) {
               lastBreak = p->pNext;
               *actualWidth = ((x <= 0) && !isEmpty) ? 1 : x;
            }
            spaceWanted = 0;
            isEmpty = 0;
            break;
         }

         case Html_APPLET:
         case Html_EMBED:
         case Html_INPUT:
         case Html_SELECT:
         case Html_TEXTAREA: {
            TGHtmlInput *input = (TGHtmlInput *) p;
            input->x = x + spaceWanted + input->padLeft;
            if ((p->style.flags & STY_Preformatted) == 0) {
               if (lastBreak && x + spaceWanted + input->w > width) {
                  return lastBreak;
               }
            }
//        TRACE(HtmlTrace_GetLine2, ("Place token %s at x=%d w=%d\n",
//           HtmlTokenName(p), p->input.x, p->input.w));
            x = input->x + input->w;
            if ((p->style.flags & STY_NoBreak) == 0) {
               lastBreak = p->pNext;
               *actualWidth = ((x <= 0) && !isEmpty) ? 1 : x;
            }
            spaceWanted = 0;
            isEmpty = 0;
            break;
         }

         case Html_EndTEXTAREA: {
            TGHtmlRef *ref = (TGHtmlRef *) p;
            if (ref->pOther) {
               // html->ResetTextarea(ref->pOther);
            }
            break;
         }

         case Html_DD: {
            TGHtmlRef *ref = (TGHtmlRef *) p;
            if (ref->pOther == 0) break;
               if (((TGHtmlListStart *)ref->pOther)->compact == 0 ||
                  x + spaceWanted >= 0) {
                  *actualWidth = ((x <= 0) && !isEmpty) ? 1 : x;
                  return p;
               }
               x = 0;
               spaceWanted = 0;
               break;
         }

         case Html_WBR:
            *actualWidth = ((x <= 0) && !isEmpty) ? 1 : x;
            if (x + spaceWanted >= width) {
               return p->pNext;
            } else {
               lastBreak = p->pNext;
            }
            break;

         case Html_ADDRESS:
         case Html_EndADDRESS:
         case Html_BLOCKQUOTE:
         case Html_EndBLOCKQUOTE:
         case Html_BODY:
         case Html_EndBODY:
         case Html_BR:
         case Html_CAPTION:
         case Html_EndCAPTION:
         case Html_CENTER:
         case Html_EndCENTER:
         case Html_EndDD:
         case Html_DIV:
         case Html_EndDIV:
         case Html_DL:
         case Html_EndDL:
         case Html_DT:
         case Html_H1:
         case Html_EndH1:
         case Html_H2:
         case Html_EndH2:
         case Html_H3:
         case Html_EndH3:
         case Html_H4:
         case Html_EndH4:
         case Html_H5:
         case Html_EndH5:
         case Html_H6:
         case Html_EndH6:
         case Html_EndHTML:
         case Html_HR:
         case Html_LI:
         case Html_LISTING:
         case Html_EndLISTING:
         case Html_MENU:
         case Html_EndMENU:
         case Html_OL:
         case Html_EndOL:
         case Html_P:
         case Html_EndP:
         case Html_PRE:
         case Html_EndPRE:
         case Html_TABLE:
         case Html_EndTABLE:
         case Html_TD:
         case Html_EndTD:
         case Html_TH:
         case Html_EndTH:
         case Html_TR:
         case Html_EndTR:
         case Html_UL:
         case Html_EndUL:
         case Html_EndFORM:
            *actualWidth = ((x <= 0) && !isEmpty) ? 1 : x;
            return p;

         default:
            break;
      }
   }
   *actualWidth = ((x <= 0) && !isEmpty) ? 1 : x;

   return p;
}

//______________________________________________________________________________
void TGHtmlLayoutContext::FixAnchors(TGHtmlElement *p, TGHtmlElement *p_end, int y)
{
   // Set the y coordinate for every anchor in the given list

   while (p && p != p_end) {
      if (p->type == Html_A) ((TGHtmlAnchor *)p)->y = y;
      p = p->pNext;
   }
}

//______________________________________________________________________________
int TGHtmlLayoutContext::FixLine(TGHtmlElement *p_start,
               TGHtmlElement *p_end, int bottom, int width,
               int actualWidth, int leftMargin, int *max_x)
{
   // This routine computes the X and Y coordinates for all elements of
   // a line that has been gathered using GetLine() above. It also figures
   // the ascent and descent for in-line images.
   //
   // The value returned is the Y coordinate of the bottom edge of the
   // new line. The X coordinates are computed by adding the left margin
   // plus any extra space needed for centering or right-justification.
   //
   // p_start     - Start of tokens for this line
   // p_end       - First token past end of this line. Maybe NULL
   // bottom      - Put the top of this line here
   // width       - This is the space available to the line
   // actualWidth - This is the actual width needed by the line
   // leftMargin  - The current left margin
   // max_x       - Write maximum X coordinate of ink here

   int dx;                // Amount by which to increase all X coordinates
   int maxAscent;         // Maximum height above baseline
   int maxTextAscent;     // Maximum height above baseline for text
   int maxDescent;        // Maximum depth below baseline
   int ascent, descent;   // Computed ascent and descent for one element
   TGHtmlElement *p;       // For looping
   int y;                 // Y coordinate of the baseline
   int dy2center;         // Distance from baseline to text font center
   int max = 0; 

   if (actualWidth > 0) {
      for (p = p_start; p && p != p_end && p->type != Html_Text; p = p->pNext) {}
      if (p == p_end || p == 0) p = p_start;
      maxAscent = maxTextAscent = 0;
      for (p = p_start; p && p != p_end; p = p->pNext) {
         int ss;
         if (p->style.align == ALIGN_Center) {
            dx = leftMargin + (width - actualWidth) / 2;
         } else if (p->style.align == ALIGN_Right) {
            dx = leftMargin + (width - actualWidth);
         } else {
            dx = leftMargin;
         }
         if (dx < 0) dx = 0;
         if (p->style.flags & STY_Invisible) continue;
         switch (p->type) {
            case Html_Text: {
               TGHtmlTextElement *text = (TGHtmlTextElement *) p;
               text->x += dx;
               max = text->x + text->w;
               ss = p->style.subscript;
               if (ss > 0) {
                  int ascent = text->ascent;
                  int delta = (ascent + text->descent) * ss / 2;
                  ascent += delta;
                  text->y = -delta;
                  if (ascent > maxAscent) maxAscent = ascent;
                  if (ascent > maxTextAscent) maxTextAscent = ascent;
               } else if (ss < 0) {
                  int descent = text->descent;
                  int delta = (descent + text->ascent) * (-ss) / 2;
                  descent += delta;
                  text->y = delta;
               } else {
                  text->y = 0;
                  if (text->ascent > maxAscent) maxAscent = text->ascent;
                  if (text->ascent > maxTextAscent) maxTextAscent = text->ascent;
               }
               break;
            }

            case Html_Space: {
               TGHtmlSpaceElement *space = (TGHtmlSpaceElement *) p;
               if (space->ascent > maxAscent) maxAscent = space->ascent;
               break;
            }

            case Html_LI: {
               TGHtmlLi *li = (TGHtmlLi *) p;
               li->x += dx;
               if (li->x > max) max = li->x; 
               break;
            }

            case Html_IMG: {
               TGHtmlImageMarkup *image = (TGHtmlImageMarkup *) p;
               image->x += dx;
               max = image->x + image->w;
               switch (image->align) {
                  case IMAGE_ALIGN_Middle:
                     image->descent = image->h / 2;
                     image->ascent = image->h - image->descent;
                     if (image->ascent > maxAscent) maxAscent = image->ascent;
                     break;

                  case IMAGE_ALIGN_AbsMiddle:
                     dy2center = (image->textDescent - image->textAscent) / 2;
                     image->descent = image->h / 2 + dy2center;
                     image->ascent = image->h - image->descent;
                     if (image->ascent > maxAscent) maxAscent = image->ascent;
                     break;

                  case IMAGE_ALIGN_Bottom:
                     image->descent = 0;
                     image->ascent = image->h;
                     if (image->ascent > maxAscent) maxAscent = image->ascent;
                     break;

                  case IMAGE_ALIGN_AbsBottom:
                     image->descent = image->textDescent;
                     image->ascent = image->h - image->descent;
                     if (image->ascent > maxAscent) maxAscent = image->ascent;
                     break;

                  default:
                     break;
               }
               break;
            }

            case Html_TABLE:
               break;

            case Html_TEXTAREA:
            case Html_INPUT:
            case Html_SELECT:
            case Html_EMBED:
            case Html_APPLET: {
               TGHtmlInput *input = (TGHtmlInput *) p;
               input->x += dx;
               max = input->x + input->w;
               dy2center = (input->textDescent - input->textAscent) / 2;
               input->y = dy2center - input->h / 2;
               ascent = -input->y;
               if (ascent > maxAscent) maxAscent = ascent;
               break;
            }

            default:
               // Shouldn't happen
               break;
         }
      }

      *max_x = max;
      y = maxAscent + bottom;
      maxDescent = 0;

      for (p = p_start; p && p != p_end; p = p->pNext) {
         if (p->style.flags & STY_Invisible) continue;
         switch (p->type) {
            case Html_Text: {
               TGHtmlTextElement *text = (TGHtmlTextElement *) p;
               text->y += y;
               if (text->descent > maxDescent) maxDescent = text->descent;
               break;
            }

            case Html_LI: {
               TGHtmlLi *li = (TGHtmlLi *) p;
               li->y = y;
               if (li->descent > maxDescent) maxDescent = li->descent;
               break;
            }

            case Html_IMG: {
               TGHtmlImageMarkup *image = (TGHtmlImageMarkup *) p;
               image->y = y;
               switch (image->align) {
                  case IMAGE_ALIGN_Top:
                     image->ascent = maxAscent;
                     image->descent = image->h - maxAscent;
                     break;

                  case IMAGE_ALIGN_TextTop:
                     image->ascent = maxTextAscent;
                     image->descent = image->h - maxTextAscent;
                     break;

                  default:
                     break;
               }
               if (image->descent > maxDescent) maxDescent = image->descent;
               break;
            }

            case Html_TABLE:
               break;

            case Html_INPUT:
            case Html_SELECT:
            case Html_TEXTAREA:
            case Html_APPLET:
            case Html_EMBED: {
               TGHtmlInput *input = (TGHtmlInput *) p;
               descent = input->y + input->h;
               input->y += y;
               if (descent > maxDescent) maxDescent = descent;
               break;
            }

            default:
               /* Shouldn't happen */
               break;
         }
      }

//    TRACE(HtmlTrace_FixLine, 
//       ("Setting baseline to %d. bottom=%d ascent=%d descent=%d dx=%d\n",
//       y, bottom, maxAscent, maxDescent, dx));

   } else {
      maxDescent = 0;
      y = bottom;
   }

   return y + maxDescent;
}

//______________________________________________________________________________
void TGHtmlLayoutContext::Paragraph(TGHtmlElement *p)
{
   // Increase the headroom to create a paragraph break at the current token

   int headroom;

   if (p == 0) return;

   if (p->type == Html_Text) {
      TGHtmlTextElement *text = (TGHtmlTextElement *) p;
      headroom = text->ascent + text->descent;
   } else if (p->pNext && p->pNext->type == Html_Text) {
      TGHtmlTextElement *text = (TGHtmlTextElement *) p->pNext;
      headroom = text->ascent + text->descent;
   } else {
      //// headroom = 10;
      FontMetrics_t fontMetrics;
      TGFont *font;
      font = html->GetFont(p->style.font);
      if (font == 0) return;
      font->GetFontMetrics(&fontMetrics);
      headroom = fontMetrics.fDescent + fontMetrics.fAscent;
   }
   if (headRoom < headroom && bottom > top) headRoom = headroom;
}

//______________________________________________________________________________
void TGHtmlLayoutContext::ComputeMargins(int *pX, int *pY, int *pW)
{
   // Compute the current margins for layout. Three values are returned:
   //
   //    *pY       The top edge of the area in which we can put ink. This
   //              takes into account any requested headroom.
   //
   //    *pX       The left edge of the inkable area. The takes into account
   //              any margin requests active at vertical position specified
   //              in pLC->bottom.
   //
   //    *pW       The width of the inkable area. This takes into account
   //              an margin requests that are active at the vertical position
   //              pLC->bottom.
   //

   int x, y, w;

   y = bottom + headRoom;
   PopExpiredMargins(&leftMargin, bottom);
   PopExpiredMargins(&rightMargin, bottom);
   w = pageWidth - right;
   if (leftMargin) {
      x = leftMargin->indent + left;
   } else {
      x = left;
   }
   w -= x;
   if (rightMargin) w -= rightMargin->indent;

   *pX = x;
   *pY = y;
   *pW = w;
}

#define CLEAR_Left  0
#define CLEAR_Right 1
#define CLEAR_Both  2
#define CLEAR_First 3
//______________________________________________________________________________
void TGHtmlLayoutContext::ClearObstacle(int mode)
{
   // Clear a wrap-around obstacle. The second option determines the
   // precise behavior.
   //
   //    CLEAR_Left        Clear all obstacles on the left.
   //
   //    CLEAR_Right       Clear all obstacles on the right.
   //
   //    CLEAR_Both        Clear all obstacles on both sides.
   //
   //    CLEAR_First       Clear only the first obstacle on either side.

   int newBottom = bottom;

   PopExpiredMargins(&leftMargin, bottom);
   PopExpiredMargins(&rightMargin, bottom);

   switch (mode) {
      case CLEAR_Both:
         ClearObstacle(CLEAR_Left);
         ClearObstacle(CLEAR_Right);
         break;

      case CLEAR_Left:
         while (leftMargin && leftMargin->bottom >= 0) {
            if (newBottom < leftMargin->bottom) {
               newBottom = leftMargin->bottom;
            }
            PopOneMargin(&leftMargin);
         }
         if (newBottom > bottom + headRoom) {
            headRoom = 0;
         } else {
            headRoom = newBottom - bottom;
         }
         bottom = newBottom;
         PopExpiredMargins(&rightMargin, bottom);
         break;

      case CLEAR_Right:
         while (rightMargin && rightMargin->bottom >= 0) {
            if (newBottom < rightMargin->bottom) {
               newBottom = rightMargin->bottom;
            }
            PopOneMargin(&rightMargin);
         }
         if (newBottom > bottom + headRoom) {
            headRoom = 0;
         } else {
            headRoom = newBottom - bottom;
         }
         bottom = newBottom;
         PopExpiredMargins(&leftMargin, bottom);
         break;

      case CLEAR_First:
         if (leftMargin && leftMargin->bottom >= 0) {
            if (rightMargin &&
                rightMargin->bottom < leftMargin->bottom) {
               if (newBottom < rightMargin->bottom) {
                  newBottom = rightMargin->bottom;
               }
               PopOneMargin(&rightMargin);
            } else {
               if (newBottom < leftMargin->bottom) {
                  newBottom = leftMargin->bottom;
               }
               PopOneMargin(&leftMargin);
            }
         } else if (rightMargin && rightMargin->bottom >= 0) {
            newBottom = rightMargin->bottom;
            PopOneMargin(&rightMargin);
         }
         if (newBottom > bottom + headRoom) {
            headRoom = 0;
         } else {
            headRoom = newBottom - bottom;
         }
         bottom = newBottom;
         break;

      default:
         break;
   }
}

//______________________________________________________________________________
int TGHtml::NextMarkupType(TGHtmlElement *p)
{
   // Return the next markup type  [TGHtmlElement::NextMarkupType]

   while ((p = p->pNext)) {
      if (p->IsMarkup()) return p->type;
   }
   return Html_Unknown;
}

//______________________________________________________________________________
TGHtmlElement *TGHtmlLayoutContext::DoBreakMarkup(TGHtmlElement *p)
{
   // Break markup is any kind of markup that might force a line-break. This
   // routine handles a single element of break markup and returns a pointer
   // to the first element past that markup. If p doesn't point to break
   // markup, then p is returned. If p is an incomplete table (a <TABLE>
   // that lacks a </TABLE>), then NULL is returned.

   TGHtmlElement *pNext = p->pNext;
   char *z;
   int x, y, w;

   switch (p->type) {
      case Html_A:
         ((TGHtmlAnchor *)p)->y = bottom;
         break;

      case Html_BLOCKQUOTE:
         PushMargin(&leftMargin, HTML_INDENT, -1, Html_EndBLOCKQUOTE);
         PushMargin(&rightMargin, HTML_INDENT, -1, Html_EndBLOCKQUOTE);
         Paragraph(p);
         break;

      case Html_EndBLOCKQUOTE:
         PopMargin(&leftMargin, Html_EndBLOCKQUOTE);
         PopMargin(&rightMargin, Html_EndBLOCKQUOTE);
         Paragraph(p);
         break;

      case Html_IMG: {
         TGHtmlImageMarkup *image = (TGHtmlImageMarkup *) p;
         switch (image->align) {
            case IMAGE_ALIGN_Left:
               ComputeMargins(&x, &y, &w);
               image->x = x;
               image->y = y;
               image->ascent = 0;
               image->descent = image->h;
               PushMargin(&leftMargin, image->w + 2, y + image->h, 0);
               if (maxY < y + image->h) maxY = y + image->h;
               if (maxX < x + image->w) maxX = x + image->w;
               break;

            case IMAGE_ALIGN_Right:
               ComputeMargins(&x, &y, &w);
               image->x = x + w - image->w;
               image->y = y;
               image->ascent = 0;
               image->descent = image->h;
               PushMargin(&rightMargin, image->w + 2, y + image->h, 0);
               if (maxY < y + image->h) maxY = y + image->h;
               if (maxX < x + image->w) maxX = x + image->w;
               break;

            default:
               pNext = p;
               break;
         }
         break;
      }

      case Html_PRE:
         // Skip space tokens thru the next newline.
         while (pNext->type == Html_Space) {
            TGHtmlElement *pThis = pNext;
            pNext = pNext->pNext;
            if (pThis->flags & HTML_NewLine) break;
         }
         Paragraph(p);
         break;

      case Html_UL:
      case Html_MENU:
      case Html_DIR:
      case Html_OL:
         if (((TGHtmlListStart *)p)->compact == 0) Paragraph(p);
         PushMargin(&leftMargin, HTML_INDENT, -1, p->type + 1);
         break;

      case Html_EndOL:
      case Html_EndUL:
      case Html_EndMENU:
      case Html_EndDIR: {
         TGHtmlRef *ref = (TGHtmlRef *) p;
         if (ref->pOther) {
            PopMargin(&leftMargin, p->type);
            if (!((TGHtmlListStart *)ref->pOther)->compact) Paragraph(p);
         }
         break;
      }

      case Html_DL:
         Paragraph(p);
         PushMargin(&leftMargin, HTML_INDENT, -1, Html_EndDL);
         break;

      case Html_EndDL:
         PopMargin(&leftMargin, Html_EndDL);
         Paragraph(p);
         break;

      case Html_HR: {
         int zl, wd;
         TGHtmlHr *hr = (TGHtmlHr *) p;
         hr->is3D = (p->MarkupArg("noshade", 0) == 0);
         z = p->MarkupArg("size", 0);
         if (z) {
            int hrsz = atoi(z);
            hr->h = (hrsz < 0) ? 2 : hrsz;
         } else {
            hr->h = 0;
         }
         if (hr->h < 1) {
            int relief = html->GetRuleRelief();
            if (hr->is3D &&
                (relief == HTML_RELIEF_SUNKEN || relief == HTML_RELIEF_RAISED)) {
               hr->h = 3;
            } else {
               hr->h = 2;
            }
         }
         ComputeMargins(&x, &y, &w);
         hr->y = y + html->GetRulePadding();
         y += hr->h + html->GetRulePadding() * 2 + 1;
         hr->x = x;
         z = p->MarkupArg("width", "100%");
         zl = strlen(z);
         if (zl > 0 && z[zl-1] == '%') {
            wd = (atoi(z) * w) / 100;
         } else {
            wd = atoi(z);
         }
         if (wd > w) wd = w;
         hr->w = wd;
         switch (p->style.align) {
            case ALIGN_Center:
            case ALIGN_None:
               hr->x += (w - wd) / 2;
               break;

            case ALIGN_Right:
               hr->x += (w - wd);
               break;

            default:
               break;
         }
         if (maxY < y) maxY = y;
         if (maxX < wd + hr->x) maxX = wd + hr->x;
         bottom = y;
         headRoom = 0;
         break;
      }

      case Html_ADDRESS:
      case Html_EndADDRESS:
      case Html_CENTER:
      case Html_EndCENTER:
      case Html_DIV:
      case Html_EndDIV:
      case Html_H1:
      case Html_EndH1:
      case Html_H2:
      case Html_EndH2:
      case Html_H3:
      case Html_EndH3:
      case Html_H4:
      case Html_EndH4:
      case Html_H5:
      case Html_EndH5:
      case Html_H6:
      case Html_EndH6:
      case Html_P:
      case Html_EndP:
      case Html_EndPRE:
      case Html_EndFORM:
         Paragraph(p);
         break;

      case Html_TABLE:
         pNext = TableLayout((TGHtmlTable *) p);
         break;

      case Html_BR:
         z = p->MarkupArg("clear",0);
         if (z) {
            if (strcasecmp(z, "left") == 0) {
               ClearObstacle(CLEAR_Left);
            } else if (strcasecmp(z, "right") == 0) {
               ClearObstacle(CLEAR_Right);
            } else {
               ClearObstacle(CLEAR_Both);
            }
         }
         if (p->pNext && p->pNext->pNext && p->pNext->type == Html_Space &&
             p->pNext->pNext->type == Html_BR) {
            Paragraph(p);
         }
         break;

      // All of the following tags need to be handed to the GetLine() routine
      case Html_Text:
      case Html_Space:
      case Html_LI:
      case Html_INPUT:
      case Html_SELECT:
      case Html_TEXTAREA:
      case Html_APPLET:
      case Html_EMBED:
         pNext = p;
         break;

      default:
         break;
   }

   return pNext;
}

//______________________________________________________________________________
int TGHtmlLayoutContext::InWrapAround()
{
   // Return TRUE (non-zero) if we are currently wrapping text around
   // one or more images.

   if (leftMargin && leftMargin->bottom >= 0) return 1;
   if (rightMargin && rightMargin->bottom >= 0) return 1;
   return 0;
}

//______________________________________________________________________________
void TGHtmlLayoutContext::WidenLine(int reqWidth, int *pX, int *pY, int *pW)
{
   // Move past obstacles until a linewidth of reqWidth is obtained,
   // or until all obstacles are cleared.
   //
   // reqWidth   - Requested line width
   // pX, pY, pW - The margins. See ComputeMargins()

   ComputeMargins(pX, pY, pW);
   if (*pW < reqWidth && InWrapAround()) {
      ClearObstacle(CLEAR_First);
      ComputeMargins(pX, pY, pW);
   }
}


#ifdef TABLE_TRIM_BLANK
int HtmlLineWasBlank = 0;
#endif // TABLE_TRIM_BLANK

//______________________________________________________________________________
void TGHtmlLayoutContext::LayoutBlock()
{
   // Do as much layout as possible on the block of text defined by
   // the HtmlLayoutContext.

   TGHtmlElement *p, *pNext;

   for (p = pStart; p && p != pEnd; p = pNext) {
      int lineWidth;
      int actualWidth;
      int y = 0;
      int leftMargin;
      int max_x = 0;

      // Do as much break markup as we can.
      while (p && p != pEnd) {
         pNext = DoBreakMarkup(p);
         if (pNext == p) break;
         if (pNext) {
//        TRACE(HtmlTrace_BreakMarkup,
//           ("Processed token %s as break markup\n", HtmlTokenName(p)));
            pStart = p;
         }
         p = pNext;
      }

      if (p == 0 || p == pEnd) break;

#ifdef TABLE_TRIM_BLANK
    HtmlLineWasBlank = 0;
#endif // TABLE_TRIM_BLANK

      // We might try several times to layout a single line...
      while (1) {

         // Compute margins
         ComputeMargins(&leftMargin, &y, &lineWidth);

         // Layout a single line of text
         pNext = GetLine(p, pEnd, lineWidth, left-leftMargin, &actualWidth);
//      TRACE(HtmlTrace_GetLine,
//         ("GetLine page=%d left=%d right=%d available=%d used=%d\n",
//         pageWidth, left, right, lineWidth, actualWidth));
         FixAnchors(p, pNext, bottom);

         // Move down and repeat the layout if we exceeded the available
         // line length and it is possible to increase the line length by
         // moving past some obstacle.

         if (actualWidth > lineWidth && InWrapAround()) {
            ClearObstacle(CLEAR_First);
             continue;
         }

         // Lock the line into place and exit the loop
         y = FixLine(p, pNext, y, lineWidth, actualWidth, leftMargin, &max_x);
         break;
      }

#ifdef TABLE_TRIM_BLANK

      // I noticed that a newline following break markup would result
      // in a blank line being drawn. So if an "empty" line was found
      // I subtract any whitespace caused by break markup.

      if (actualWidth <= 0) HtmlLineWasBlank = 1;

#endif // TABLE_TRIM_BLANK

      // If a line was completed, advance to the next line
      if (pNext && actualWidth > 0 && y > bottom) {
         PopIndent();
         bottom = y;
         pStart = pNext;
      }
      if (y > maxY) maxY = y;
      if (max_x > maxX) maxX = max_x;
   }
}

//______________________________________________________________________________
void TGHtmlLayoutContext::PushIndent()
{
   //

   headRoom += html->GetMarginHeight();
   if (html->GetMarginWidth()) {
      PushMargin(&leftMargin, html->GetMarginWidth(), -1, Html_EndBLOCKQUOTE);
      PushMargin(&rightMargin, html->GetMarginWidth(), -1, Html_EndBLOCKQUOTE);
   }
}

//______________________________________________________________________________
void TGHtmlLayoutContext::PopIndent()
{
   //

   if (headRoom <= 0) return;
   headRoom = 0;
   PopMargin(&rightMargin, Html_EndBLOCKQUOTE);
}

//______________________________________________________________________________
void TGHtml::LayoutDoc()
{
   // Advance the layout as far as possible

   int btm;

   if (pFirst == 0) return;
   Sizer();
   layoutContext.html = this;
#if 0  // orig
  layoutContext.PushIndent();
  layoutContext.pageWidth = fCanvas->GetWidth();
  layoutContext.left = 0;
#else
   layoutContext.headRoom = HTML_INDENT/4;
   layoutContext.pageWidth = fCanvas->GetWidth() - HTML_INDENT/4;
   layoutContext.left = HTML_INDENT/4;
#endif
   layoutContext.right = 0;
   layoutContext.pStart = nextPlaced;
   if (layoutContext.pStart == 0) layoutContext.pStart = pFirst;
   if (layoutContext.pStart) {
      TGHtmlElement *p;

      layoutContext.maxX = maxX;
      layoutContext.maxY = maxY;
      btm = layoutContext.bottom;
      layoutContext.LayoutBlock();
      maxX = layoutContext.maxX;
#if 0
    maxY = layoutContext.maxY;
#else
      maxY = layoutContext.maxY + fYMargin;
#endif
      nextPlaced = layoutContext.pStart;
      flags |= HSCROLL | VSCROLL;
      if (zGoto && (p = AttrElem("name", zGoto+1))) {
         fVisible.fY = ((TGHtmlAnchor *)p)->y;
         delete[] zGoto;
         zGoto = 0;
      }
      RedrawText(btm);
   }
}
