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

#include "TGHtml.h"


// NOTE: this list was generated automatically. If you make any
// modifications to it, you'll have to modify also the TGHtmlTokens.h
// file accordingly.

SHtmlTokenMap_t HtmlMarkupMap[] = {
  { "a",            Html_A,                 O_HtmlAnchor,        0 },
  { "/a",           Html_EndA,              O_HtmlRef,           0 },
  { "address",      Html_ADDRESS,           O_HtmlMarkupElement, 0 },
  { "/address",     Html_EndADDRESS,        O_HtmlMarkupElement, 0 },
  { "applet",       Html_APPLET,            O_HtmlInput,         0 },
  { "/applet",      Html_EndAPPLET,         O_HtmlMarkupElement, 0 },
  { "area",         Html_AREA,              O_HtmlMapArea,       0 },
  { "b",            Html_B,                 O_HtmlMarkupElement, 0 },
  { "/b",           Html_EndB,              O_HtmlMarkupElement, 0 },
  { "base",         Html_BASE,              O_HtmlMarkupElement, 0 },
  { "basefont",     Html_BASEFONT,          O_HtmlMarkupElement, 0 },
  { "/basefont",    Html_EndBASEFONT,       O_HtmlMarkupElement, 0 },
  { "bgsound",      Html_BGSOUND,           O_HtmlMarkupElement, 0 },
  { "big",          Html_BIG,               O_HtmlMarkupElement, 0 },
  { "/big",         Html_EndBIG,            O_HtmlMarkupElement, 0 },
  { "blockquote",   Html_BLOCKQUOTE,        O_HtmlMarkupElement, 0 },
  { "/blockquote",  Html_EndBLOCKQUOTE,     O_HtmlMarkupElement, 0 },
  { "body",         Html_BODY,              O_HtmlMarkupElement, 0 },
  { "/body",        Html_EndBODY,           O_HtmlMarkupElement, 0 },
  { "br",           Html_BR,                O_HtmlMarkupElement, 0 },
  { "caption",      Html_CAPTION,           O_HtmlMarkupElement, 0 },
  { "/caption",     Html_EndCAPTION,        O_HtmlMarkupElement, 0 },
  { "center",       Html_CENTER,            O_HtmlMarkupElement, 0 },
  { "/center",      Html_EndCENTER,         O_HtmlMarkupElement, 0 },
  { "cite",         Html_CITE,              O_HtmlMarkupElement, 0 },
  { "/cite",        Html_EndCITE,           O_HtmlMarkupElement, 0 },
  { "code",         Html_CODE,              O_HtmlMarkupElement, 0 },
  { "/code",        Html_EndCODE,           O_HtmlMarkupElement, 0 },
  { "comment",      Html_COMMENT,           O_HtmlMarkupElement, 0 }, // Text!
  { "/comment",     Html_EndCOMMENT,        O_HtmlMarkupElement, 0 },
  { "dd",           Html_DD,                O_HtmlRef,           0 },
  { "/dd",          Html_EndDD,             O_HtmlMarkupElement, 0 },
  { "dfn",          Html_DFN,               O_HtmlMarkupElement, 0 },
  { "/dfn",         Html_EndDFN,            O_HtmlMarkupElement, 0 },
  { "dir",          Html_DIR,               O_HtmlListStart,     0 },
  { "/dir",         Html_EndDIR,            O_HtmlRef,           0 },
  { "div",          Html_DIV,               O_HtmlMarkupElement, 0 },
  { "/div",         Html_EndDIV,            O_HtmlMarkupElement, 0 },
  { "dl",           Html_DL,                O_HtmlListStart,     0 },
  { "/dl",          Html_EndDL,             O_HtmlRef,           0 },
  { "dt",           Html_DT,                O_HtmlRef,           0 },
  { "/dt",          Html_EndDT,             O_HtmlMarkupElement, 0 },
  { "em",           Html_EM,                O_HtmlMarkupElement, 0 },
  { "/em",          Html_EndEM,             O_HtmlMarkupElement, 0 },
  { "embed",        Html_EMBED,             O_HtmlInput,         0 },
  { "font",         Html_FONT,              O_HtmlMarkupElement, 0 },
  { "/font",        Html_EndFONT,           O_HtmlMarkupElement, 0 },
  { "form",         Html_FORM,              O_HtmlForm,          0 },
  { "/form",        Html_EndFORM,           O_HtmlRef,           0 },
  { "frame",        Html_FRAME,             O_HtmlMarkupElement, 0 },
  { "/frame",       Html_EndFRAME,          O_HtmlMarkupElement, 0 },
  { "frameset",     Html_FRAMESET,          O_HtmlMarkupElement, 0 },
  { "/frameset",    Html_EndFRAMESET,       O_HtmlMarkupElement, 0 },
  { "h1",           Html_H1,                O_HtmlMarkupElement, 0 },
  { "/h1",          Html_EndH1,             O_HtmlMarkupElement, 0 },
  { "h2",           Html_H2,                O_HtmlMarkupElement, 0 },
  { "/h2",          Html_EndH2,             O_HtmlMarkupElement, 0 },
  { "h3",           Html_H3,                O_HtmlMarkupElement, 0 },
  { "/h3",          Html_EndH3,             O_HtmlMarkupElement, 0 },
  { "h4",           Html_H4,                O_HtmlMarkupElement, 0 },
  { "/h4",          Html_EndH4,             O_HtmlMarkupElement, 0 },
  { "h5",           Html_H5,                O_HtmlMarkupElement, 0 },
  { "/h5",          Html_EndH5,             O_HtmlMarkupElement, 0 },
  { "h6",           Html_H6,                O_HtmlMarkupElement, 0 },
  { "/h6",          Html_EndH6,             O_HtmlMarkupElement, 0 },
  { "hr",           Html_HR,                O_HtmlHr,            0 },
  { "html",         Html_HTML,              O_HtmlMarkupElement, 0 },
  { "/html",        Html_EndHTML,           O_HtmlMarkupElement, 0 },
  { "i",            Html_I,                 O_HtmlMarkupElement, 0 },
  { "/i",           Html_EndI,              O_HtmlMarkupElement, 0 },
  { "iframe",       Html_IFRAME,            O_HtmlMarkupElement, 0 },
  { "img",          Html_IMG,               O_HtmlImageMarkup,   0 },
  { "input",        Html_INPUT,             O_HtmlInput,         0 },
  { "isindex",      Html_ISINDEX,           O_HtmlMarkupElement, 0 },
  { "kbd",          Html_KBD,               O_HtmlMarkupElement, 0 },
  { "/kbd",         Html_EndKBD,            O_HtmlMarkupElement, 0 },
  { "li",           Html_LI,                O_HtmlLi,            0 },
  { "/li",          Html_EndLI,             O_HtmlMarkupElement, 0 },
  { "link",         Html_LINK,              O_HtmlMarkupElement, 0 },
  { "listing",      Html_LISTING,           O_HtmlMarkupElement, 0 },
  { "/listing",     Html_EndLISTING,        O_HtmlMarkupElement, 0 },
  { "map",          Html_MAP,               O_HtmlMarkupElement, 0 },
  { "/map",         Html_EndMAP,            O_HtmlMarkupElement, 0 },
  { "marquee",      Html_MARQUEE,           O_HtmlMarkupElement, 0 },
  { "/marquee",     Html_EndMARQUEE,        O_HtmlMarkupElement, 0 },
  { "menu",         Html_MENU,              O_HtmlListStart,     0 },
  { "/menu",        Html_EndMENU,           O_HtmlRef,           0 },
  { "meta",         Html_META,              O_HtmlMarkupElement, 0 },
  { "nextid",       Html_NEXTID,            O_HtmlMarkupElement, 0 },
  { "nobr",         Html_NOBR,              O_HtmlMarkupElement, 0 },
  { "/nobr",        Html_EndNOBR,           O_HtmlMarkupElement, 0 },
  { "noembed",      Html_NOEMBED,           O_HtmlMarkupElement, 0 },
  { "/noembed",     Html_EndNOEMBED,        O_HtmlMarkupElement, 0 },
  { "noframe",      Html_NOFRAMES,          O_HtmlMarkupElement, 0 },
  { "/noframe",     Html_EndNOFRAMES,       O_HtmlMarkupElement, 0 },
  { "noscript",     Html_NOSCRIPT,          O_HtmlMarkupElement, 0 },
  { "/noscript",    Html_EndNOSCRIPT,       O_HtmlMarkupElement, 0 },
  { "ol",           Html_OL,                O_HtmlListStart,     0 },
  { "/ol",          Html_EndOL,             O_HtmlRef,           0 },
  { "option",       Html_OPTION,            O_HtmlMarkupElement, 0 },
  { "/option",      Html_EndOPTION,         O_HtmlMarkupElement, 0 },
  { "p",            Html_P,                 O_HtmlMarkupElement, 0 },
  { "/p",           Html_EndP,              O_HtmlMarkupElement, 0 },
  { "param",        Html_PARAM,             O_HtmlMarkupElement, 0 },
  { "/param",       Html_EndPARAM,          O_HtmlMarkupElement, 0 },
  { "plaintext",    Html_PLAINTEXT,         O_HtmlMarkupElement, 0 },
  { "pre",          Html_PRE,               O_HtmlMarkupElement, 0 },
  { "/pre",         Html_EndPRE,            O_HtmlMarkupElement, 0 },
  { "s",            Html_S,                 O_HtmlMarkupElement, 0 },
  { "/s",           Html_EndS,              O_HtmlMarkupElement, 0 },
  { "samp",         Html_SAMP,              O_HtmlMarkupElement, 0 },
  { "/samp",        Html_EndSAMP,           O_HtmlMarkupElement, 0 },
  { "script",       Html_SCRIPT,            O_HtmlScript,        0 },
  { "select",       Html_SELECT,            O_HtmlInput,         0 },
  { "/select",      Html_EndSELECT,         O_HtmlRef,           0 },
  { "small",        Html_SMALL,             O_HtmlMarkupElement, 0 },
  { "/small",       Html_EndSMALL,          O_HtmlMarkupElement, 0 },
  { "strike",       Html_STRIKE,            O_HtmlMarkupElement, 0 },
  { "/strike",      Html_EndSTRIKE,         O_HtmlMarkupElement, 0 },
  { "strong",       Html_STRONG,            O_HtmlMarkupElement, 0 },
  { "/strong",      Html_EndSTRONG,         O_HtmlMarkupElement, 0 },
  { "style",        Html_STYLE,             O_HtmlScript,        0 },
  { "sub",          Html_SUB,               O_HtmlMarkupElement, 0 },
  { "/sub",         Html_EndSUB,            O_HtmlMarkupElement, 0 },
  { "sup",          Html_SUP,               O_HtmlMarkupElement, 0 },
  { "/sup",         Html_EndSUP,            O_HtmlMarkupElement, 0 },
  { "table",        Html_TABLE,             O_HtmlTable,         0 },
  { "/table",       Html_EndTABLE,          O_HtmlRef,           0 },
  { "td",           Html_TD,                O_HtmlCell,          0 },
  { "/td",          Html_EndTD,             O_HtmlRef,           0 },
  { "textarea",     Html_TEXTAREA,          O_HtmlInput,         0 },
  { "/textarea",    Html_EndTEXTAREA,       O_HtmlRef,           0 },
  { "th",           Html_TH,                O_HtmlCell,          0 },
  { "/th",          Html_EndTH,             O_HtmlRef,           0 },
  { "title",        Html_TITLE,             O_HtmlMarkupElement, 0 },
  { "/title",       Html_EndTITLE,          O_HtmlMarkupElement, 0 },
  { "tr",           Html_TR,                O_HtmlRef,           0 },
  { "/tr",          Html_EndTR,             O_HtmlRef,           0 },
  { "tt",           Html_TT,                O_HtmlMarkupElement, 0 },
  { "/tt",          Html_EndTT,             O_HtmlMarkupElement, 0 },
  { "u",            Html_U,                 O_HtmlMarkupElement, 0 },
  { "/u",           Html_EndU,              O_HtmlMarkupElement, 0 },
  { "ul",           Html_UL,                O_HtmlListStart,     0 },
  { "/ul",          Html_EndUL,             O_HtmlRef,           0 },
  { "var",          Html_VAR,               O_HtmlMarkupElement, 0 },
  { "/var",         Html_EndVAR,            O_HtmlMarkupElement, 0 },
  { "wbr",          Html_WBR,               O_HtmlMarkupElement, 0 },
  { "xmp",          Html_XMP,               O_HtmlMarkupElement, 0 },
  { "/xmp",         Html_EndXMP,            O_HtmlMarkupElement, 0 },
  { 0,              0,                      0,                   0 }
};


