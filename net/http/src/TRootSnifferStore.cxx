// $Id$
// Author: Sergey Linev   22/12/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TRootSnifferStore.h"


/** \class TRootSnifferStore
\ingroup http

Used to store different results of objects scanning by TRootSniffer
*/

ClassImp(TRootSnifferStore);

////////////////////////////////////////////////////////////////////////////////
/// set pointer on found element, class and number of childs

void TRootSnifferStore::SetResult(void *_res, TClass *_rescl, TDataMember *_resmemb, Int_t _res_chld, Int_t _restr)
{
   fResPtr = _res;
   fResClass = _rescl;
   fResMember = _resmemb;
   fResNumChilds = _res_chld;
   fResRestrict = _restr;
}

// =================================================================================

/** \class TRootSnifferStoreXml
\ingroup http

Used to store scanned objects hierarchy in XML form
*/

ClassImp(TRootSnifferStoreXml);

////////////////////////////////////////////////////////////////////////////////
/// starts new xml node, will be closed by CloseNode

   void TRootSnifferStoreXml::CreateNode(Int_t lvl, const char *nodename)
{
   fBuf.Append(TString::Format("%*s<item _name=\"%s\"", fCompact ? 0 : (lvl + 1) * 2, "", nodename));
}

////////////////////////////////////////////////////////////////////////////////
/// set field (xml attribute) in current node

void TRootSnifferStoreXml::SetField(Int_t, const char *field, const char *value, Bool_t)
{
   if (strpbrk(value, "<>&\'\"") == 0) {
      fBuf.Append(TString::Format(" %s=\"%s\"", field, value));
   } else {
      fBuf.Append(TString::Format(" %s=\"", field));
      const char *v = value;
      while (*v != 0) {
         switch (*v) {
         case '<': fBuf.Append("&lt;"); break;
         case '>': fBuf.Append("&gt;"); break;
         case '&': fBuf.Append("&amp;"); break;
         case '\'': fBuf.Append("&apos;"); break;
         case '\"': fBuf.Append("&quot;"); break;
         default: fBuf.Append(*v); break;
         }
         v++;
      }

      fBuf.Append("\"");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// called before next child node created

void TRootSnifferStoreXml::BeforeNextChild(Int_t, Int_t nchld, Int_t)
{
   if (nchld == 0) fBuf.Append(TString::Format(">%s", (fCompact ? "" : "\n")));
}

////////////////////////////////////////////////////////////////////////////////
/// Called when node should be closed
///
/// depending from number of childs different xml format is applied

void TRootSnifferStoreXml::CloseNode(Int_t lvl, Int_t numchilds)
{
   if (numchilds > 0)
      fBuf.Append(TString::Format("%*s</item>%s", fCompact ? 0 : (lvl + 1) * 2, "", (fCompact ? "" : "\n")));
   else
      fBuf.Append(TString::Format("/>%s", (fCompact ? "" : "\n")));
}

// ============================================================================

/** \class TRootSnifferStoreJson
\ingroup http

Used to store scanned objects hierarchy in JSON form
*/

ClassImp(TRootSnifferStoreJson);

////////////////////////////////////////////////////////////////////////////////
/// starts new json object, will be closed by CloseNode

void TRootSnifferStoreJson::CreateNode(Int_t lvl, const char *nodename)
{
   fBuf.Append(TString::Format("%*s{", fCompact ? 0 : lvl * 4, ""));
   if (!fCompact) fBuf.Append("\n");
   fBuf.Append(
      TString::Format("%*s\"_name\"%s\"%s\"", fCompact ? 0 : lvl * 4 + 2, "", (fCompact ? ":" : " : "), nodename));
}

////////////////////////////////////////////////////////////////////////////////
/// set field (json field) in current node

void TRootSnifferStoreJson::SetField(Int_t lvl, const char *field, const char *value, Bool_t with_quotes)
{
   fBuf.Append(",");
   if (!fCompact) fBuf.Append("\n");
   fBuf.Append(TString::Format("%*s\"%s\"%s", fCompact ? 0 : lvl * 4 + 2, "", field, (fCompact ? ":" : " : ")));
   if (!with_quotes) {
      fBuf.Append(value);
   } else {
      fBuf.Append("\"");
      for (const char *v = value; *v != 0; v++) switch (*v) {
         case '\n': fBuf.Append("\\n"); break;
         case '\t': fBuf.Append("\\t"); break;
         case '\"': fBuf.Append("\\\""); break;
         case '\\': fBuf.Append("\\\\"); break;
         case '\b': fBuf.Append("\\b"); break;
         case '\f': fBuf.Append("\\f"); break;
         case '\r': fBuf.Append("\\r"); break;
         case '/': fBuf.Append("\\/"); break;
         default:
            if ((*v > 31) && (*v < 127))
               fBuf.Append(*v);
            else
               fBuf.Append(TString::Format("\\u%04x", (unsigned)*v));
         }
      fBuf.Append("\"");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// called before next child node created

void TRootSnifferStoreJson::BeforeNextChild(Int_t lvl, Int_t nchld, Int_t)
{
   fBuf.Append(",");
   if (!fCompact) fBuf.Append("\n");
   if (nchld == 0)
      fBuf.Append(TString::Format("%*s\"_childs\"%s", (fCompact ? 0 : lvl * 4 + 2), "", (fCompact ? ":[" : " : [\n")));
}

////////////////////////////////////////////////////////////////////////////////
/// called when node should be closed
/// depending from number of childs different xml format is applied

void TRootSnifferStoreJson::CloseNode(Int_t lvl, Int_t numchilds)
{
   if (numchilds > 0)
      fBuf.Append(TString::Format("%s%*s]", (fCompact ? "" : "\n"), fCompact ? 0 : lvl * 4 + 2, ""));
   fBuf.Append(TString::Format("%s%*s}", (fCompact ? "" : "\n"), fCompact ? 0 : lvl * 4, ""));
}
