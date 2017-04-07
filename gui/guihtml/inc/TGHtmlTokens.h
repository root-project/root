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

#ifndef ROOT_TGHtmlTokens
#define ROOT_TGHtmlTokens

// NOTE: this list was generated automatically. If you make any
// modifications to it, you'll have to modify also the OHtmlTokenMap.cc
// file accordingly.

enum {
  Html_Text = 1,
  Html_Space,
  Html_Unknown,
  Html_Block,
  Html_A,
  Html_EndA,
  Html_ADDRESS,
  Html_EndADDRESS,
  Html_APPLET,
  Html_EndAPPLET,
  Html_AREA,
  Html_B,
  Html_EndB,
  Html_BASE,
  Html_BASEFONT,
  Html_EndBASEFONT,
  Html_BGSOUND,
  Html_BIG,
  Html_EndBIG,
  Html_BLOCKQUOTE,
  Html_EndBLOCKQUOTE,
  Html_BODY,
  Html_EndBODY,
  Html_BR,
  Html_CAPTION,
  Html_EndCAPTION,
  Html_CENTER,
  Html_EndCENTER,
  Html_CITE,
  Html_EndCITE,
  Html_CODE,
  Html_EndCODE,
  Html_COMMENT,
  Html_EndCOMMENT,
  Html_DD,
  Html_EndDD,
  Html_DFN,
  Html_EndDFN,
  Html_DIR,
  Html_EndDIR,
  Html_DIV,
  Html_EndDIV,
  Html_DL,
  Html_EndDL,
  Html_DT,
  Html_EndDT,
  Html_EM,
  Html_EndEM,
  Html_EMBED,
  Html_FONT,
  Html_EndFONT,
  Html_FORM,
  Html_EndFORM,
  Html_FRAME,
  Html_EndFRAME,
  Html_FRAMESET,
  Html_EndFRAMESET,
  Html_H1,
  Html_EndH1,
  Html_H2,
  Html_EndH2,
  Html_H3,
  Html_EndH3,
  Html_H4,
  Html_EndH4,
  Html_H5,
  Html_EndH5,
  Html_H6,
  Html_EndH6,
  Html_HR,
  Html_HTML,
  Html_EndHTML,
  Html_I,
  Html_EndI,
  Html_IFRAME,
  Html_IMG,
  Html_INPUT,
  Html_ISINDEX,
  Html_KBD,
  Html_EndKBD,
  Html_LI,
  Html_EndLI,
  Html_LINK,
  Html_LISTING,
  Html_EndLISTING,
  Html_MAP,
  Html_EndMAP,
  Html_MARQUEE,
  Html_EndMARQUEE,
  Html_MENU,
  Html_EndMENU,
  Html_META,
  Html_NEXTID,
  Html_NOBR,
  Html_EndNOBR,
  Html_NOEMBED,
  Html_EndNOEMBED,
  Html_NOFRAMES,
  Html_EndNOFRAMES,
  Html_NOSCRIPT,
  Html_EndNOSCRIPT,
  Html_OL,
  Html_EndOL,
  Html_OPTION,
  Html_EndOPTION,
  Html_P,
  Html_EndP,
  Html_PARAM,
  Html_EndPARAM,
  Html_PLAINTEXT,
  Html_PRE,
  Html_EndPRE,
  Html_S,
  Html_EndS,
  Html_SAMP,
  Html_EndSAMP,
  Html_SCRIPT,
  Html_SELECT,
  Html_EndSELECT,
  Html_SMALL,
  Html_EndSMALL,
  Html_STRIKE,
  Html_EndSTRIKE,
  Html_STRONG,
  Html_EndSTRONG,
  Html_STYLE,
  Html_SUB,
  Html_EndSUB,
  Html_SUP,
  Html_EndSUP,
  Html_TABLE,
  Html_EndTABLE,
  Html_TD,
  Html_EndTD,
  Html_TEXTAREA,
  Html_EndTEXTAREA,
  Html_TH,
  Html_EndTH,
  Html_TITLE,
  Html_EndTITLE,
  Html_TR,
  Html_EndTR,
  Html_TT,
  Html_EndTT,
  Html_U,
  Html_EndU,
  Html_UL,
  Html_EndUL,
  Html_VAR,
  Html_EndVAR,
  Html_WBR,
  Html_XMP,
  Html_EndXMP,
  Html__TypeCount
};

#define Html_TypeCount         (Html__TypeCount - 1)
#define HTML_MARKUP_COUNT      (Html__TypeCount - 5)
#define HTML_MARKUP_HASH_SIZE  (Html__TypeCount + 11)

#endif  // ROOT_TGHtmlTokens
