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
   // Html Layout Context constructor.

   fPStart = 0;
   fPEnd = 0;
   fLeftMargin = 0;
   fRightMargin = 0;
   fHtml = 0;
   fLeft = 0;
   fRight = 0;
   fMaxX = 0;
   fMaxY = 0;
   fPageWidth = 0;
   Reset();
}

//______________________________________________________________________________
void TGHtmlLayoutContext::Reset()
{
   // Reset the layout context.

   fHeadRoom = 0;
   fTop = 0;
   fBottom = 0;
   ClearMarginStack(&fLeftMargin);
   ClearMarginStack(&fRightMargin);
}

//______________________________________________________________________________
void TGHtmlLayoutContext::PushMargin(SHtmlMargin_t **ppMargin,
                                    int indent, int mbottom, int tag)
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
   //  mbottom  - The margin expires at this Y coordinate
   //  tag      - Markup that will cancel this margin

   SHtmlMargin_t *pNew = new SHtmlMargin_t;
   pNew->fPNext = *ppMargin;
   if (pNew->fPNext) {
      pNew->fIndent = indent + pNew->fPNext->fIndent;
   } else {
      pNew->fIndent = indent;
   }
   pNew->fBottom = mbottom;
   pNew->fTag = tag;
   *ppMargin = pNew;
}

//______________________________________________________________________________
void TGHtmlLayoutContext::PopOneMargin(SHtmlMargin_t **ppMargin)
{
   // Pop one margin off of the given margin stack.

   if (*ppMargin) {
      SHtmlMargin_t *pOld = *ppMargin;
      *ppMargin = pOld->fPNext;
      delete pOld;
   }
}

//______________________________________________________________________________
void TGHtmlLayoutContext::PopMargin(SHtmlMargin_t **ppMargin, int tag)
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
   SHtmlMargin_t *pM;

   for (pM = *ppMargin; pM && pM->fTag != tag; pM = pM->fPNext) {}
   if (pM == 0) {
      // No matching margin is found. Do nothing.
      return;
   }
   while ((pM = *ppMargin) != 0) {
      if (pM->fBottom > bot) bot = pM->fBottom;
      oldTag = pM->fTag;
      PopOneMargin(ppMargin);
      if (oldTag == tag) break;
   }
   if (fBottom < bot) {
      fHeadRoom += bot - fBottom;
      fBottom = bot;
   }
}

//______________________________________________________________________________
void TGHtmlLayoutContext::PopExpiredMargins(SHtmlMargin_t **ppMarginStack, int y)
{
   // Pop all expired margins from the stack.
   //
   // An expired margin is one with a non-negative bottom parameter
   // that is less than the value "y". "y" is the Y-coordinate of
   // the top edge the next line of text to by positioned. What this
   // function does is check to see if we have cleared any obstacles
   // (an obstacle is an <IMG ALIGN=left> or <IMG ALIGN=right>) and
   // expands the margins if we have.

   while (*ppMarginStack && (**ppMarginStack).fBottom >= 0 &&
         (**ppMarginStack).fBottom <= y) {
      PopOneMargin(ppMarginStack);
   }
}

//______________________________________________________________________________
void TGHtmlLayoutContext::ClearMarginStack(SHtmlMargin_t **ppMargin)
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
   // The candidate tokens begin with fPStart and go thru the end of
   // the list or to fPEnd, whichever comes first. The first token
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
   while (p && p != p_end && (p->fStyle.fFlags & STY_Invisible) != 0) {
      p = p->fPNext;
   }
   if (p && p->fStyle.fFlags & STY_DT) {
      origin = -HTML_INDENT;
   } else {
      origin = 0;
   }
   x = origin;
   if (x < minX) x = minX;
   if (p && p != p_end && p->fType == Html_LI) {
      TGHtmlLi *li = (TGHtmlLi *) p;
      li->fX = x - HTML_INDENT / 3;
      if (li->fX - (HTML_INDENT * 2) / 3 < minX) {
         x += minX - li->fX + (HTML_INDENT * 2) / 3;
         li->fX = minX + (HTML_INDENT * 2) / 3;
      }
      isEmpty = 0;
      *actualWidth = 1;
      p = p->fPNext;
      while (p && (p->fType == Html_Space || p->fType == Html_P)) {
         p = p->fPNext;
      }
   }
   // coverity[dead_error_line]
   for (; p && p != p_end; p = p ? p->fPNext : 0) {
      if (p->fStyle.fFlags & STY_Invisible) continue;
      switch (p->fType) {
         case Html_Text: {
            TGHtmlTextElement *text = (TGHtmlTextElement *) p;
            text->fX = x + spaceWanted;
            if ((p->fStyle.fFlags & STY_Preformatted) == 0) {
               if (lastBreak && x + spaceWanted + text->fW > width)
                  return lastBreak;
            }
//        TRACE(HtmlTrace_GetLine2, ("Place token %s at x=%d w=%d\n",
//           HtmlTokenName(p), text->fX, text->fW));
            x += text->fW + spaceWanted;
            isEmpty = 0;
            spaceWanted = 0;
            break;
         }

         case Html_Space: {
            TGHtmlSpaceElement *space = (TGHtmlSpaceElement *) p;
            if (p->fStyle.fFlags & STY_Preformatted) {
               if (p->fFlags & HTML_NewLine) {
                  *actualWidth = (x <= 0) ? 1 : x;
                  return p->fPNext;
               }
               x += space->fW * p->fCount;
            } else {
               int w;
               if ((p->fStyle.fFlags & STY_NoBreak) == 0) {
                  lastBreak = p->fPNext;
                  *actualWidth = ((x <= 0) && !isEmpty) ? 1 : x;
               }
               w = space->fW;
               if (spaceWanted < w && x > origin) spaceWanted = w;
            }
            break;
         }

         case Html_IMG: {
            TGHtmlImageMarkup *image = (TGHtmlImageMarkup *) p;
            switch (image->fAlign) {
               case IMAGE_ALIGN_Left:
               case IMAGE_ALIGN_Right:
                  *actualWidth = ((x <= 0) && !isEmpty) ? 1 : x;
                  return p;
               default:
                  break;
            }
            image->fX = x + spaceWanted;
            if ((p->fStyle.fFlags & STY_Preformatted) == 0) {
               if (lastBreak && x + spaceWanted + image->fW > width) {
                  return lastBreak;
               }
            }
//        TRACE(HtmlTrace_GetLine2, ("Place in-line image %s at x=%d w=%d\n",
//           HtmlTokenName(p), p->image.x, p->image.w));
            x += image->fW + spaceWanted;
            if ((p->fStyle.fFlags & STY_NoBreak) == 0) {
               lastBreak = p->fPNext;
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
            input->fX = x + spaceWanted + input->fPadLeft;
            if ((p->fStyle.fFlags & STY_Preformatted) == 0) {
               if (lastBreak && x + spaceWanted + input->fW > width) {
                  return lastBreak;
               }
            }
//        TRACE(HtmlTrace_GetLine2, ("Place token %s at x=%d w=%d\n",
//           HtmlTokenName(p), p->input.x, p->input.w));
            x = input->fX + input->fW;
            if ((p->fStyle.fFlags & STY_NoBreak) == 0) {
               lastBreak = p->fPNext;
               *actualWidth = ((x <= 0) && !isEmpty) ? 1 : x;
            }
            spaceWanted = 0;
            isEmpty = 0;
            break;
         }

         case Html_EndTEXTAREA: {
            TGHtmlRef *ref = (TGHtmlRef *) p;
            if (ref->fPOther) {
               // fHtml->ResetTextarea(ref->fPOther);
            }
            break;
         }

         case Html_DD: {
            TGHtmlRef *ref = (TGHtmlRef *) p;
            if (ref->fPOther == 0) break;
               if (((TGHtmlListStart *)ref->fPOther)->fCompact == 0 ||
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
               return p->fPNext;
            } else {
               lastBreak = p->fPNext;
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
      if (p->fType == Html_A) ((TGHtmlAnchor *)p)->fY = y;
      p = p->fPNext;
   }
}

//______________________________________________________________________________
int TGHtmlLayoutContext::FixLine(TGHtmlElement *p_start,
               TGHtmlElement *p_end, int mbottom, int width,
               int actualWidth, int lMargin, int *max_x)
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
   // mbottom     - Put the top of this line here
   // width       - This is the space available to the line
   // actualWidth - This is the actual width needed by the line
   // lMargin     - The current left margin
   // max_x       - Write maximum X coordinate of ink here

   int dx;                // Amount by which to increase all X coordinates
   int maxAscent;         // Maximum height above baseline
   int maxTextAscent;     // Maximum height above baseline for text
   int maxDescent;        // Maximum depth below baseline
   int ascent, descent;   // Computed ascent and descent for one element
   TGHtmlElement *p;      // For looping
   int y;                 // Y coordinate of the baseline
   int dy2center;         // Distance from baseline to text font center
   int max = 0;

   if (actualWidth > 0) {
      for (p = p_start; p && p != p_end && p->fType != Html_Text; p = p->fPNext) {}
      if (p == p_end || p == 0) p = p_start;
      maxAscent = maxTextAscent = 0;
      for (p = p_start; p && p != p_end; p = p->fPNext) {
         int ss;
         if (p->fStyle.fAlign == ALIGN_Center) {
            dx = lMargin + (width - actualWidth) / 2;
         } else if (p->fStyle.fAlign == ALIGN_Right) {
            dx = lMargin + (width - actualWidth);
         } else {
            dx = lMargin;
         }
         if (dx < 0) dx = 0;
         if (p->fStyle.fFlags & STY_Invisible) continue;
         switch (p->fType) {
            case Html_Text: {
               TGHtmlTextElement *text = (TGHtmlTextElement *) p;
               text->fX += dx;
               max = text->fX + text->fW;
               ss = p->fStyle.fSubscript;
               if (ss > 0) {
                  int ascent2 = text->fAscent;
                  int delta = (ascent2 + text->fDescent) * ss / 2;
                  ascent2 += delta;
                  text->fY = -delta;
                  if (ascent2 > maxAscent) maxAscent = ascent2;
                  if (ascent2 > maxTextAscent) maxTextAscent = ascent2;
               } else if (ss < 0) {
                  int descent2 = text->fDescent;
                  int delta = (descent2 + text->fAscent) * (-ss) / 2;
                  descent2 += delta;
                  text->fY = delta;
               } else {
                  text->fY = 0;
                  if (text->fAscent > maxAscent) maxAscent = text->fAscent;
                  if (text->fAscent > maxTextAscent) maxTextAscent = text->fAscent;
               }
               break;
            }

            case Html_Space: {
               TGHtmlSpaceElement *space = (TGHtmlSpaceElement *) p;
               if (space->fAscent > maxAscent) maxAscent = space->fAscent;
               break;
            }

            case Html_LI: {
               TGHtmlLi *li = (TGHtmlLi *) p;
               li->fX += dx;
               if (li->fX > max) max = li->fX;
               break;
            }

            case Html_IMG: {
               TGHtmlImageMarkup *image = (TGHtmlImageMarkup *) p;
               image->fX += dx;
               max = image->fX + image->fW;
               switch (image->fAlign) {
                  case IMAGE_ALIGN_Middle:
                     image->fDescent = image->fH / 2;
                     image->fAscent = image->fH - image->fDescent;
                     if (image->fAscent > maxAscent) maxAscent = image->fAscent;
                     break;

                  case IMAGE_ALIGN_AbsMiddle:
                     dy2center = (image->fTextDescent - image->fTextAscent) / 2;
                     image->fDescent = image->fH / 2 + dy2center;
                     image->fAscent = image->fH - image->fDescent;
                     if (image->fAscent > maxAscent) maxAscent = image->fAscent;
                     break;

                  case IMAGE_ALIGN_Bottom:
                     image->fDescent = 0;
                     image->fAscent = image->fH;
                     if (image->fAscent > maxAscent) maxAscent = image->fAscent;
                     break;

                  case IMAGE_ALIGN_AbsBottom:
                     image->fDescent = image->fTextDescent;
                     image->fAscent = image->fH - image->fDescent;
                     if (image->fAscent > maxAscent) maxAscent = image->fAscent;
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
               input->fX += dx;
               max = input->fX + input->fW;
               dy2center = (input->fTextDescent - input->fTextAscent) / 2;
               input->fY = dy2center - input->fH / 2;
               ascent = -input->fY;
               if (ascent > maxAscent) maxAscent = ascent;
               break;
            }

            default:
               // Shouldn't happen
               break;
         }
      }

      *max_x = max;
      y = maxAscent + mbottom;
      maxDescent = 0;

      for (p = p_start; p && p != p_end; p = p->fPNext) {
         if (p->fStyle.fFlags & STY_Invisible) continue;
         switch (p->fType) {
            case Html_Text: {
               TGHtmlTextElement *text = (TGHtmlTextElement *) p;
               text->fY += y;
               if (text->fDescent > maxDescent) maxDescent = text->fDescent;
               break;
            }

            case Html_LI: {
               TGHtmlLi *li = (TGHtmlLi *) p;
               li->fY = y;
               if (li->fDescent > maxDescent) maxDescent = li->fDescent;
               break;
            }

            case Html_IMG: {
               TGHtmlImageMarkup *image = (TGHtmlImageMarkup *) p;
               image->fY = y;
               switch (image->fAlign) {
                  case IMAGE_ALIGN_Top:
                     image->fAscent = maxAscent;
                     image->fDescent = image->fH - maxAscent;
                     break;

                  case IMAGE_ALIGN_TextTop:
                     image->fAscent = maxTextAscent;
                     image->fDescent = image->fH - maxTextAscent;
                     break;

                  default:
                     break;
               }
               if (image->fDescent > maxDescent) maxDescent = image->fDescent;
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
               descent = input->fY + input->fH;
               input->fY += y;
               if (descent > maxDescent) maxDescent = descent;
               break;
            }

            default:
               /* Shouldn't happen */
               break;
         }
      }

//    TRACE(HtmlTrace_FixLine,
//       ("Setting baseline to %d. mbottom=%d ascent=%d descent=%d dx=%d\n",
//       y, mbottom, maxAscent, maxDescent, dx));

   } else {
      maxDescent = 0;
      y = mbottom;
   }

   return y + maxDescent;
}

//______________________________________________________________________________
void TGHtmlLayoutContext::Paragraph(TGHtmlElement *p)
{
   // Increase the headroom to create a paragraph break at the current token

   int headroom;

   if (p == 0) return;

   if (p->fType == Html_Text) {
      TGHtmlTextElement *text = (TGHtmlTextElement *) p;
      headroom = text->fAscent + text->fDescent;
   } else if (p->fPNext && p->fPNext->fType == Html_Text) {
      TGHtmlTextElement *text = (TGHtmlTextElement *) p->fPNext;
      headroom = text->fAscent + text->fDescent;
   } else {
      //// headroom = 10;
      FontMetrics_t fontMetrics;
      TGFont *font;
      font = fHtml->GetFont(p->fStyle.fFont);
      if (font == 0) return;
      font->GetFontMetrics(&fontMetrics);
      headroom = fontMetrics.fDescent + fontMetrics.fAscent;
   }
   if (fHeadRoom < headroom && fBottom > fTop) fHeadRoom = headroom;
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

   y = fBottom + fHeadRoom;
   PopExpiredMargins(&fLeftMargin, fBottom);
   PopExpiredMargins(&fRightMargin, fBottom);
   w = fPageWidth - fRight;
   if (fLeftMargin) {
      x = fLeftMargin->fIndent + fLeft;
   } else {
      x = fLeft;
   }
   w -= x;
   if (fRightMargin) w -= fRightMargin->fIndent;

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

   int newBottom = fBottom;

   PopExpiredMargins(&fLeftMargin, fBottom);
   PopExpiredMargins(&fRightMargin, fBottom);

   switch (mode) {
      case CLEAR_Both:
         ClearObstacle(CLEAR_Left);
         ClearObstacle(CLEAR_Right);
         break;

      case CLEAR_Left:
         while (fLeftMargin && fLeftMargin->fBottom >= 0) {
            if (newBottom < fLeftMargin->fBottom) {
               newBottom = fLeftMargin->fBottom;
            }
            PopOneMargin(&fLeftMargin);
         }
         if (newBottom > fBottom + fHeadRoom) {
            fHeadRoom = 0;
         } else {
            fHeadRoom = newBottom - fBottom;
         }
         fBottom = newBottom;
         PopExpiredMargins(&fRightMargin, fBottom);
         break;

      case CLEAR_Right:
         while (fRightMargin && fRightMargin->fBottom >= 0) {
            if (newBottom < fRightMargin->fBottom) {
               newBottom = fRightMargin->fBottom;
            }
            PopOneMargin(&fRightMargin);
         }
         if (newBottom > fBottom + fHeadRoom) {
            fHeadRoom = 0;
         } else {
            fHeadRoom = newBottom - fBottom;
         }
         fBottom = newBottom;
         PopExpiredMargins(&fLeftMargin, fBottom);
         break;

      case CLEAR_First:
         if (fLeftMargin && fLeftMargin->fBottom >= 0) {
            if (fRightMargin &&
                fRightMargin->fBottom < fLeftMargin->fBottom) {
               if (newBottom < fRightMargin->fBottom) {
                  newBottom = fRightMargin->fBottom;
               }
               PopOneMargin(&fRightMargin);
            } else {
               if (newBottom < fLeftMargin->fBottom) {
                  newBottom = fLeftMargin->fBottom;
               }
               PopOneMargin(&fLeftMargin);
            }
         } else if (fRightMargin && fRightMargin->fBottom >= 0) {
            newBottom = fRightMargin->fBottom;
            PopOneMargin(&fRightMargin);
         }
         if (newBottom > fBottom + fHeadRoom) {
            fHeadRoom = 0;
         } else {
            fHeadRoom = newBottom - fBottom;
         }
         fBottom = newBottom;
         break;

      default:
         break;
   }
}

//______________________________________________________________________________
int TGHtml::NextMarkupType(TGHtmlElement *p)
{
   // Return the next markup type  [TGHtmlElement::NextMarkupType]

   while ((p = p->fPNext)) {
      if (p->IsMarkup()) return p->fType;
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

   TGHtmlElement *fPNext = p->fPNext;
   const char *z;
   int x, y, w;

   switch (p->fType) {
      case Html_A:
         ((TGHtmlAnchor *)p)->fY = fBottom;
         break;

      case Html_BLOCKQUOTE:
         PushMargin(&fLeftMargin, HTML_INDENT, -1, Html_EndBLOCKQUOTE);
         PushMargin(&fRightMargin, HTML_INDENT, -1, Html_EndBLOCKQUOTE);
         Paragraph(p);
         break;

      case Html_EndBLOCKQUOTE:
         PopMargin(&fLeftMargin, Html_EndBLOCKQUOTE);
         PopMargin(&fRightMargin, Html_EndBLOCKQUOTE);
         Paragraph(p);
         break;

      case Html_IMG: {
         TGHtmlImageMarkup *image = (TGHtmlImageMarkup *) p;
         switch (image->fAlign) {
            case IMAGE_ALIGN_Left:
               ComputeMargins(&x, &y, &w);
               image->fX = x;
               image->fY = y;
               image->fAscent = 0;
               image->fDescent = image->fH;
               PushMargin(&fLeftMargin, image->fW + 2, y + image->fH, 0);
               if (fMaxY < y + image->fH) fMaxY = y + image->fH;
               if (fMaxX < x + image->fW) fMaxX = x + image->fW;
               break;

            case IMAGE_ALIGN_Right:
               ComputeMargins(&x, &y, &w);
               image->fX = x + w - image->fW;
               image->fY = y;
               image->fAscent = 0;
               image->fDescent = image->fH;
               PushMargin(&fRightMargin, image->fW + 2, y + image->fH, 0);
               if (fMaxY < y + image->fH) fMaxY = y + image->fH;
               if (fMaxX < x + image->fW) fMaxX = x + image->fW;
               break;

            default:
               fPNext = p;
               break;
         }
         break;
      }

      case Html_PRE:
         // Skip space tokens thru the next newline.
         while (fPNext->fType == Html_Space) {
            TGHtmlElement *pThis = fPNext;
            fPNext = fPNext->fPNext;
            if (pThis->fFlags & HTML_NewLine) break;
         }
         Paragraph(p);
         break;

      case Html_UL:
      case Html_MENU:
      case Html_DIR:
      case Html_OL:
         if (((TGHtmlListStart *)p)->fCompact == 0) Paragraph(p);
         PushMargin(&fLeftMargin, HTML_INDENT, -1, p->fType + 1);
         break;

      case Html_EndOL:
      case Html_EndUL:
      case Html_EndMENU:
      case Html_EndDIR: {
         TGHtmlRef *ref = (TGHtmlRef *) p;
         if (ref->fPOther) {
            PopMargin(&fLeftMargin, p->fType);
            if (!((TGHtmlListStart *)ref->fPOther)->fCompact) Paragraph(p);
         }
         break;
      }

      case Html_DL:
         Paragraph(p);
         PushMargin(&fLeftMargin, HTML_INDENT, -1, Html_EndDL);
         break;

      case Html_EndDL:
         PopMargin(&fLeftMargin, Html_EndDL);
         Paragraph(p);
         break;

      case Html_HR: {
         int zl, wd;
         TGHtmlHr *hr = (TGHtmlHr *) p;
         hr->fIs3D = (p->MarkupArg("noshade", 0) == 0);
         z = p->MarkupArg("size", 0);
         if (z) {
            int hrsz = atoi(z);
            hr->fH = (hrsz < 0) ? 2 : hrsz;
         } else {
            hr->fH = 0;
         }
         if (hr->fH < 1) {
            int relief = fHtml->GetRuleRelief();
            if (hr->fIs3D &&
                (relief == HTML_RELIEF_SUNKEN || relief == HTML_RELIEF_RAISED)) {
               hr->fH = 3;
            } else {
               hr->fH = 2;
            }
         }
         ComputeMargins(&x, &y, &w);
         hr->fY = y + fHtml->GetRulePadding();
         y += hr->fH + fHtml->GetRulePadding() * 2 + 1;
         hr->fX = x;
         z = p->MarkupArg("width", "100%");
         zl = strlen(z);
         if (zl > 0 && z[zl-1] == '%') {
            wd = (atoi(z) * w) / 100;
         } else {
            wd = atoi(z);
         }
         if (wd > w) wd = w;
         hr->fW = wd;
         switch (p->fStyle.fAlign) {
            case ALIGN_Center:
            case ALIGN_None:
               hr->fX += (w - wd) / 2;
               break;

            case ALIGN_Right:
               hr->fX += (w - wd);
               break;

            default:
               break;
         }
         if (fMaxY < y) fMaxY = y;
         if (fMaxX < wd + hr->fX) fMaxX = wd + hr->fX;
         fBottom = y;
         fHeadRoom = 0;
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
         fPNext = TableLayout((TGHtmlTable *) p);
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
         if (p->fPNext && p->fPNext->fPNext && p->fPNext->fType == Html_Space &&
             p->fPNext->fPNext->fType == Html_BR) {
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
         fPNext = p;
         break;

      default:
         break;
   }

   return fPNext;
}

//______________________________________________________________________________
int TGHtmlLayoutContext::InWrapAround()
{
   // Return TRUE (non-zero) if we are currently wrapping text around
   // one or more images.

   if (fLeftMargin && fLeftMargin->fBottom >= 0) return 1;
   if (fRightMargin && fRightMargin->fBottom >= 0) return 1;
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

   for (p = fPStart; p && p != fPEnd; p = pNext) {
      int lineWidth;
      int actualWidth;
      int y = 0;
      int lMargin;
      int max_x = 0;

      // Do as much break markup as we can.
      while (p && p != fPEnd) {
         pNext = DoBreakMarkup(p);
         if (pNext == p) break;
         if (pNext) {
//        TRACE(HtmlTrace_BreakMarkup,
//           ("Processed token %s as break markup\n", HtmlTokenName(p)));
            fPStart = p;
         }
         p = pNext;
      }

      if (p == 0 || p == fPEnd) break;

#ifdef TABLE_TRIM_BLANK
    HtmlLineWasBlank = 0;
#endif // TABLE_TRIM_BLANK

      // We might try several times to layout a single line...
      while (1) {

         // Compute margins
         ComputeMargins(&lMargin, &y, &lineWidth);

         // Layout a single line of text
         pNext = GetLine(p, fPEnd, lineWidth, fLeft-lMargin, &actualWidth);
//      TRACE(HtmlTrace_GetLine,
//         ("GetLine page=%d left=%d right=%d available=%d used=%d\n",
//         fPageWidth, fLeft, fRight, lineWidth, actualWidth));
         FixAnchors(p, pNext, fBottom);

         // Move down and repeat the layout if we exceeded the available
         // line length and it is possible to increase the line length by
         // moving past some obstacle.

         if (actualWidth > lineWidth && InWrapAround()) {
            ClearObstacle(CLEAR_First);
            continue;
         }

         // Lock the line into place and exit the loop
         y = FixLine(p, pNext, y, lineWidth, actualWidth, lMargin, &max_x);
         break;
      }

#ifdef TABLE_TRIM_BLANK

      // I noticed that a newline following break markup would result
      // in a blank line being drawn. So if an "empty" line was found
      // I subtract any whitespace caused by break markup.

      if (actualWidth <= 0) HtmlLineWasBlank = 1;

#endif // TABLE_TRIM_BLANK

      // If a line was completed, advance to the next line
      if (pNext && actualWidth > 0 && y > fBottom) {
         PopIndent();
         fBottom = y;
         fPStart = pNext;
      }
      if (y > fMaxY) fMaxY = y;
      if (max_x > fMaxX) fMaxX = max_x;
   }
}

//______________________________________________________________________________
void TGHtmlLayoutContext::PushIndent()
{
   // Adjust (push) ident.

   fHeadRoom += fHtml->GetMarginHeight();
   if (fHtml->GetMarginWidth()) {
      PushMargin(&fLeftMargin, fHtml->GetMarginWidth(), -1, Html_EndBLOCKQUOTE);
      PushMargin(&fRightMargin, fHtml->GetMarginWidth(), -1, Html_EndBLOCKQUOTE);
   }
}

//______________________________________________________________________________
void TGHtmlLayoutContext::PopIndent()
{
   // Adjust (pop) ident.

   if (fHeadRoom <= 0) return;
   fHeadRoom = 0;
   PopMargin(&fRightMargin, Html_EndBLOCKQUOTE);
}

//______________________________________________________________________________
void TGHtml::LayoutDoc()
{
   // Advance the layout as far as possible

   int btm;

   if (fPFirst == 0) return;
   Sizer();
   fLayoutContext.fHtml = this;
#if 0  // orig
   fLayoutContext.PushIndent();
   fLayoutContext.fPageWidth = fCanvas->GetWidth();
   fLayoutContext.fLeft = 0;
#else
   fLayoutContext.fHeadRoom = HTML_INDENT/4;
   fLayoutContext.fPageWidth = fCanvas->GetWidth() - HTML_INDENT/4;
   fLayoutContext.fLeft = HTML_INDENT/4;
#endif
   fLayoutContext.fRight = 0;
   fLayoutContext.fPStart = fNextPlaced;
   if (fLayoutContext.fPStart == 0) fLayoutContext.fPStart = fPFirst;
   if (fLayoutContext.fPStart) {
      TGHtmlElement *p;

      fLayoutContext.fMaxX = fMaxX;
      fLayoutContext.fMaxY = fMaxY;
      btm = fLayoutContext.fBottom;
      fLayoutContext.LayoutBlock();
      fMaxX = fLayoutContext.fMaxX;
#if 0
      fMaxY = fLayoutContext.fMaxY;
#else
      fMaxY = fLayoutContext.fMaxY + fYMargin;
#endif
      fNextPlaced = fLayoutContext.fPStart;
      fFlags |= HSCROLL | VSCROLL;
      if (fZGoto && (p = AttrElem("name", fZGoto+1))) {
         fVisible.fY = ((TGHtmlAnchor *)p)->fY;
         delete[] fZGoto;
         fZGoto = 0;
      }
      RedrawText(btm);
   }
}
