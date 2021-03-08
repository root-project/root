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
  { "a",            Html_A,                 O_HtmlAnchor,        nullptr },
  { "/a",           Html_EndA,              O_HtmlRef,           nullptr },
  { "address",      Html_ADDRESS,           O_HtmlMarkupElement, nullptr },
  { "/address",     Html_EndADDRESS,        O_HtmlMarkupElement, nullptr },
  { "applet",       Html_APPLET,            O_HtmlInput,         nullptr },
  { "/applet",      Html_EndAPPLET,         O_HtmlMarkupElement, nullptr },
  { "area",         Html_AREA,              O_HtmlMapArea,       nullptr },
  { "b",            Html_B,                 O_HtmlMarkupElement, nullptr },
  { "/b",           Html_EndB,              O_HtmlMarkupElement, nullptr },
  { "base",         Html_BASE,              O_HtmlMarkupElement, nullptr },
  { "basefont",     Html_BASEFONT,          O_HtmlMarkupElement, nullptr },
  { "/basefont",    Html_EndBASEFONT,       O_HtmlMarkupElement, nullptr },
  { "bgsound",      Html_BGSOUND,           O_HtmlMarkupElement, nullptr },
  { "big",          Html_BIG,               O_HtmlMarkupElement, nullptr },
  { "/big",         Html_EndBIG,            O_HtmlMarkupElement, nullptr },
  { "blockquote",   Html_BLOCKQUOTE,        O_HtmlMarkupElement, nullptr },
  { "/blockquote",  Html_EndBLOCKQUOTE,     O_HtmlMarkupElement, nullptr },
  { "body",         Html_BODY,              O_HtmlMarkupElement, nullptr },
  { "/body",        Html_EndBODY,           O_HtmlMarkupElement, nullptr },
  { "br",           Html_BR,                O_HtmlMarkupElement, nullptr },
  { "caption",      Html_CAPTION,           O_HtmlMarkupElement, nullptr },
  { "/caption",     Html_EndCAPTION,        O_HtmlMarkupElement, nullptr },
  { "center",       Html_CENTER,            O_HtmlMarkupElement, nullptr },
  { "/center",      Html_EndCENTER,         O_HtmlMarkupElement, nullptr },
  { "cite",         Html_CITE,              O_HtmlMarkupElement, nullptr },
  { "/cite",        Html_EndCITE,           O_HtmlMarkupElement, nullptr },
  { "code",         Html_CODE,              O_HtmlMarkupElement, nullptr },
  { "/code",        Html_EndCODE,           O_HtmlMarkupElement, nullptr },
  { "comment",      Html_COMMENT,           O_HtmlMarkupElement, nullptr }, // Text!
  { "/comment",     Html_EndCOMMENT,        O_HtmlMarkupElement, nullptr },
  { "dd",           Html_DD,                O_HtmlRef,           nullptr },
  { "/dd",          Html_EndDD,             O_HtmlMarkupElement, nullptr },
  { "dfn",          Html_DFN,               O_HtmlMarkupElement, nullptr },
  { "/dfn",         Html_EndDFN,            O_HtmlMarkupElement, nullptr },
  { "dir",          Html_DIR,               O_HtmlListStart,     nullptr },
  { "/dir",         Html_EndDIR,            O_HtmlRef,           nullptr },
  { "div",          Html_DIV,               O_HtmlMarkupElement, nullptr },
  { "/div",         Html_EndDIV,            O_HtmlMarkupElement, nullptr },
  { "dl",           Html_DL,                O_HtmlListStart,     nullptr },
  { "/dl",          Html_EndDL,             O_HtmlRef,           nullptr },
  { "dt",           Html_DT,                O_HtmlRef,           nullptr },
  { "/dt",          Html_EndDT,             O_HtmlMarkupElement, nullptr },
  { "em",           Html_EM,                O_HtmlMarkupElement, nullptr },
  { "/em",          Html_EndEM,             O_HtmlMarkupElement, nullptr },
  { "embed",        Html_EMBED,             O_HtmlInput,         nullptr },
  { "font",         Html_FONT,              O_HtmlMarkupElement, nullptr },
  { "/font",        Html_EndFONT,           O_HtmlMarkupElement, nullptr },
  { "form",         Html_FORM,              O_HtmlForm,          nullptr },
  { "/form",        Html_EndFORM,           O_HtmlRef,           nullptr },
  { "frame",        Html_FRAME,             O_HtmlMarkupElement, nullptr },
  { "/frame",       Html_EndFRAME,          O_HtmlMarkupElement, nullptr },
  { "frameset",     Html_FRAMESET,          O_HtmlMarkupElement, nullptr },
  { "/frameset",    Html_EndFRAMESET,       O_HtmlMarkupElement, nullptr },
  { "h1",           Html_H1,                O_HtmlMarkupElement, nullptr },
  { "/h1",          Html_EndH1,             O_HtmlMarkupElement, nullptr },
  { "h2",           Html_H2,                O_HtmlMarkupElement, nullptr },
  { "/h2",          Html_EndH2,             O_HtmlMarkupElement, nullptr },
  { "h3",           Html_H3,                O_HtmlMarkupElement, nullptr },
  { "/h3",          Html_EndH3,             O_HtmlMarkupElement, nullptr },
  { "h4",           Html_H4,                O_HtmlMarkupElement, nullptr },
  { "/h4",          Html_EndH4,             O_HtmlMarkupElement, nullptr },
  { "h5",           Html_H5,                O_HtmlMarkupElement, nullptr },
  { "/h5",          Html_EndH5,             O_HtmlMarkupElement, nullptr },
  { "h6",           Html_H6,                O_HtmlMarkupElement, nullptr },
  { "/h6",          Html_EndH6,             O_HtmlMarkupElement, nullptr },
  { "hr",           Html_HR,                O_HtmlHr,            nullptr },
  { "html",         Html_HTML,              O_HtmlMarkupElement, nullptr },
  { "/html",        Html_EndHTML,           O_HtmlMarkupElement, nullptr },
  { "i",            Html_I,                 O_HtmlMarkupElement, nullptr },
  { "/i",           Html_EndI,              O_HtmlMarkupElement, nullptr },
  { "iframe",       Html_IFRAME,            O_HtmlMarkupElement, nullptr },
  { "img",          Html_IMG,               O_HtmlImageMarkup,   nullptr },
  { "input",        Html_INPUT,             O_HtmlInput,         nullptr },
  { "isindex",      Html_ISINDEX,           O_HtmlMarkupElement, nullptr },
  { "kbd",          Html_KBD,               O_HtmlMarkupElement, nullptr },
  { "/kbd",         Html_EndKBD,            O_HtmlMarkupElement, nullptr },
  { "li",           Html_LI,                O_HtmlLi,            nullptr },
  { "/li",          Html_EndLI,             O_HtmlMarkupElement, nullptr },
  { "link",         Html_LINK,              O_HtmlMarkupElement, nullptr },
  { "listing",      Html_LISTING,           O_HtmlMarkupElement, nullptr },
  { "/listing",     Html_EndLISTING,        O_HtmlMarkupElement, nullptr },
  { "map",          Html_MAP,               O_HtmlMarkupElement, nullptr },
  { "/map",         Html_EndMAP,            O_HtmlMarkupElement, nullptr },
  { "marquee",      Html_MARQUEE,           O_HtmlMarkupElement, nullptr },
  { "/marquee",     Html_EndMARQUEE,        O_HtmlMarkupElement, nullptr },
  { "menu",         Html_MENU,              O_HtmlListStart,     nullptr },
  { "/menu",        Html_EndMENU,           O_HtmlRef,           nullptr },
  { "meta",         Html_META,              O_HtmlMarkupElement, nullptr },
  { "nextid",       Html_NEXTID,            O_HtmlMarkupElement, nullptr },
  { "nobr",         Html_NOBR,              O_HtmlMarkupElement, nullptr },
  { "/nobr",        Html_EndNOBR,           O_HtmlMarkupElement, nullptr },
  { "noembed",      Html_NOEMBED,           O_HtmlMarkupElement, nullptr },
  { "/noembed",     Html_EndNOEMBED,        O_HtmlMarkupElement, nullptr },
  { "noframe",      Html_NOFRAMES,          O_HtmlMarkupElement, nullptr },
  { "/noframe",     Html_EndNOFRAMES,       O_HtmlMarkupElement, nullptr },
  { "noscript",     Html_NOSCRIPT,          O_HtmlMarkupElement, nullptr },
  { "/noscript",    Html_EndNOSCRIPT,       O_HtmlMarkupElement, nullptr },
  { "ol",           Html_OL,                O_HtmlListStart,     nullptr },
  { "/ol",          Html_EndOL,             O_HtmlRef,           nullptr },
  { "option",       Html_OPTION,            O_HtmlMarkupElement, nullptr },
  { "/option",      Html_EndOPTION,         O_HtmlMarkupElement, nullptr },
  { "p",            Html_P,                 O_HtmlMarkupElement, nullptr },
  { "/p",           Html_EndP,              O_HtmlMarkupElement, nullptr },
  { "param",        Html_PARAM,             O_HtmlMarkupElement, nullptr },
  { "/param",       Html_EndPARAM,          O_HtmlMarkupElement, nullptr },
  { "plaintext",    Html_PLAINTEXT,         O_HtmlMarkupElement, nullptr },
  { "pre",          Html_PRE,               O_HtmlMarkupElement, nullptr },
  { "/pre",         Html_EndPRE,            O_HtmlMarkupElement, nullptr },
  { "s",            Html_S,                 O_HtmlMarkupElement, nullptr },
  { "/s",           Html_EndS,              O_HtmlMarkupElement, nullptr },
  { "samp",         Html_SAMP,              O_HtmlMarkupElement, nullptr },
  { "/samp",        Html_EndSAMP,           O_HtmlMarkupElement, nullptr },
  { "script",       Html_SCRIPT,            O_HtmlScript,        nullptr },
  { "select",       Html_SELECT,            O_HtmlInput,         nullptr },
  { "/select",      Html_EndSELECT,         O_HtmlRef,           nullptr },
  { "small",        Html_SMALL,             O_HtmlMarkupElement, nullptr },
  { "/small",       Html_EndSMALL,          O_HtmlMarkupElement, nullptr },
  { "strike",       Html_STRIKE,            O_HtmlMarkupElement, nullptr },
  { "/strike",      Html_EndSTRIKE,         O_HtmlMarkupElement, nullptr },
  { "strong",       Html_STRONG,            O_HtmlMarkupElement, nullptr },
  { "/strong",      Html_EndSTRONG,         O_HtmlMarkupElement, nullptr },
  { "style",        Html_STYLE,             O_HtmlScript,        nullptr },
  { "sub",          Html_SUB,               O_HtmlMarkupElement, nullptr },
  { "/sub",         Html_EndSUB,            O_HtmlMarkupElement, nullptr },
  { "sup",          Html_SUP,               O_HtmlMarkupElement, nullptr },
  { "/sup",         Html_EndSUP,            O_HtmlMarkupElement, nullptr },
  { "table",        Html_TABLE,             O_HtmlTable,         nullptr },
  { "/table",       Html_EndTABLE,          O_HtmlRef,           nullptr },
  { "td",           Html_TD,                O_HtmlCell,          nullptr },
  { "/td",          Html_EndTD,             O_HtmlRef,           nullptr },
  { "textarea",     Html_TEXTAREA,          O_HtmlInput,         nullptr },
  { "/textarea",    Html_EndTEXTAREA,       O_HtmlRef,           nullptr },
  { "th",           Html_TH,                O_HtmlCell,          nullptr },
  { "/th",          Html_EndTH,             O_HtmlRef,           nullptr },
  { "title",        Html_TITLE,             O_HtmlMarkupElement, nullptr },
  { "/title",       Html_EndTITLE,          O_HtmlMarkupElement, nullptr },
  { "tr",           Html_TR,                O_HtmlRef,           nullptr },
  { "/tr",          Html_EndTR,             O_HtmlRef,           nullptr },
  { "tt",           Html_TT,                O_HtmlMarkupElement, nullptr },
  { "/tt",          Html_EndTT,             O_HtmlMarkupElement, nullptr },
  { "u",            Html_U,                 O_HtmlMarkupElement, nullptr },
  { "/u",           Html_EndU,              O_HtmlMarkupElement, nullptr },
  { "ul",           Html_UL,                O_HtmlListStart,     nullptr },
  { "/ul",          Html_EndUL,             O_HtmlRef,           nullptr },
  { "var",          Html_VAR,               O_HtmlMarkupElement, nullptr },
  { "/var",         Html_EndVAR,            O_HtmlMarkupElement, nullptr },
  { "wbr",          Html_WBR,               O_HtmlMarkupElement, nullptr },
  { "xmp",          Html_XMP,               O_HtmlMarkupElement, nullptr },
  { "/xmp",         Html_EndXMP,            O_HtmlMarkupElement, nullptr },
  { nullptr,              0,                      0,                   nullptr }
};


