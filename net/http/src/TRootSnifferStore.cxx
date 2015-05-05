#include "TRootSnifferStore.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootSnifferStore                                                    //
//                                                                      //
// Used to store different results of objects scanning by TRootSniffer  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


//______________________________________________________________________________
TRootSnifferStore::TRootSnifferStore() :
   TObject(),
   fResPtr(0),
   fResClass(0),
   fResMember(0),
   fResNumChilds(-1)
{
   // normal constructor
}

//______________________________________________________________________________
TRootSnifferStore::~TRootSnifferStore()
{
   // destructor
}

//______________________________________________________________________________
void TRootSnifferStore::SetResult(void *_res, TClass *_rescl,
                                  TDataMember *_resmemb, Int_t _res_chld)
{
   // set pointer on found element, class and number of childs

   fResPtr = _res;
   fResClass = _rescl;
   fResMember = _resmemb;
   fResNumChilds = _res_chld;
}

// =================================================================================

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootSnifferStoreXml                                                 //
//                                                                      //
// Used to store scanned objects hierarchy in XML form                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void TRootSnifferStoreXml::CreateNode(Int_t lvl, const char *nodename)
{
   // starts new xml node, will be closed by CloseNode

   fBuf->Append(TString::Format("%*s<item _name=\"%s\"", fCompact ? 0 : (lvl + 1) * 2, "", nodename));
}

//______________________________________________________________________________
void TRootSnifferStoreXml::SetField(Int_t, const char *field, const char *value,
                                    Bool_t)
{
   // set field (xml attribute) in current node

   if (strpbrk(value, "<>&\'\"") == 0) {
      fBuf->Append(TString::Format(" %s=\"%s\"", field, value));
   } else {
      fBuf->Append(TString::Format(" %s=\"", field));
      const char *v = value;
      while (*v != 0) {
         switch (*v) {
            case '<' :
               fBuf->Append("&lt;");
               break;
            case '>' :
               fBuf->Append("&gt;");
               break;
            case '&' :
               fBuf->Append("&amp;");
               break;
            case '\'' :
               fBuf->Append("&apos;");
               break;
            case '\"' :
               fBuf->Append("&quot;");
               break;
            default:
               fBuf->Append(*v);
               break;
         }
         v++;
      }

      fBuf->Append("\"");
   }
}

//______________________________________________________________________________
void TRootSnifferStoreXml::BeforeNextChild(Int_t, Int_t nchld, Int_t)
{
   // called before next child node created

   if (nchld == 0) fBuf->Append(TString::Format(">%s", (fCompact ? "" : "\n")));
}

//______________________________________________________________________________
void TRootSnifferStoreXml::CloseNode(Int_t lvl, Int_t numchilds)
{
   // called when node should be closed
   // depending from number of childs different xml format is applied

   if (numchilds > 0)
      fBuf->Append(TString::Format("%*s</item>%s", fCompact ? 0 : (lvl + 1) * 2, "", (fCompact ? "" : "\n")));
   else
      fBuf->Append(TString::Format("/>%s", (fCompact ? "" : "\n")));
}

// ============================================================================

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootSnifferStoreXml                                                 //
//                                                                      //
// Used to store scanned objects hierarchy in JSON form                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void TRootSnifferStoreJson::CreateNode(Int_t lvl, const char *nodename)
{
   // starts new json object, will be closed by CloseNode

   fBuf->Append(TString::Format("%*s{", fCompact ? 0 : lvl * 4, ""));
   if (!fCompact) fBuf->Append("\n");
   fBuf->Append(TString::Format("%*s\"_name\"%s\"%s\"", fCompact ? 0 : lvl * 4 + 2, "", (fCompact ? ":" : " : "), nodename));
}

//______________________________________________________________________________
void TRootSnifferStoreJson::SetField(Int_t lvl, const char *field,
                                     const char *value, Bool_t with_quotes)
{
   // set field (json field) in current node

   fBuf->Append(",");
   if (!fCompact) fBuf->Append("\n");
   fBuf->Append(TString::Format("%*s\"%s\"%s", fCompact ? 0 : lvl * 4 + 2, "", field, (fCompact ? ":" : " : ")));
   if (!with_quotes) {
      fBuf->Append(value);
   } else {
      fBuf->Append("\"");
      for (const char *v = value; *v != 0; v++)
         switch (*v) {
            case '\n':
               fBuf->Append("\\n");
               break;
            case '\t':
               fBuf->Append("\\t");
               break;
            case '\"':
               fBuf->Append("\\\"");
               break;
            case '\\':
               fBuf->Append("\\\\");
               break;
            case '\b':
               fBuf->Append("\\b");
               break;
            case '\f':
               fBuf->Append("\\f");
               break;
            case '\r':
               fBuf->Append("\\r");
               break;
            case '/':
               fBuf->Append("\\/");
               break;
            default:
               if ((*v > 31) && (*v < 127))
                  fBuf->Append(*v);
               else
                  fBuf->Append(TString::Format("\\u%04x", (unsigned) *v));
         }
      fBuf->Append("\"");
   }
}

//______________________________________________________________________________
void TRootSnifferStoreJson::BeforeNextChild(Int_t lvl, Int_t nchld, Int_t)
{
   // called before next child node created

   fBuf->Append(",");
   if (!fCompact) fBuf->Append("\n");
   if (nchld == 0)
      fBuf->Append(TString::Format("%*s\"_childs\"%s", (fCompact ? 0 : lvl * 4 + 2), "", (fCompact ? ":[" : " : [\n")));
}

//______________________________________________________________________________
void TRootSnifferStoreJson::CloseNode(Int_t lvl, Int_t numchilds)
{
   // called when node should be closed
   // depending from number of childs different xml format is applied

   if (numchilds > 0)
      fBuf->Append(TString::Format("%s%*s]", (fCompact ? "" : "\n"), fCompact ? 0 : lvl * 4 + 2, ""));
   fBuf->Append(TString::Format("%s%*s}", (fCompact ? "" : "\n"), fCompact ? 0 : lvl * 4, ""));
}

