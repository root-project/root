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

   buf->Append(TString::Format("%*s<%s", lvl * 2, "", nodename));
}

//______________________________________________________________________________
void TRootSnifferStoreXml::SetField(Int_t, const char *field, const char *value,
                                    Int_t)
{
   // set field (xml attribute) in current node

   if (strpbrk(value, "<>&\'\"") == 0) {
      buf->Append(TString::Format(" %s=\"%s\"", field, value));
   } else {
      buf->Append(TString::Format(" %s=\"", field));
      const char *v = value;
      while (*v != 0) {
         switch (*v) {
            case '<' :
               buf->Append("&lt;");
               break;
            case '>' :
               buf->Append("&gt;");
               break;
            case '&' :
               buf->Append("&amp;");
               break;
            case '\'' :
               buf->Append("&apos;");
               break;
            case '\"' :
               buf->Append("&quot;");
               break;
            default:
               buf->Append(*v);
               break;
         }
         v++;
      }

      buf->Append("\"");
   }
}

//______________________________________________________________________________
void TRootSnifferStoreXml::BeforeNextChild(Int_t, Int_t nchld, Int_t)
{
   // called before next child node created

   if (nchld == 0) buf->Append(">\n");
}

//______________________________________________________________________________
void TRootSnifferStoreXml::CloseNode(Int_t lvl, const char *nodename,
                                     Int_t numchilds)
{
   // called when node should be closed
   // depending from number of childs different xml format is applied

   if (numchilds > 0)
      buf->Append(TString::Format("%*s</%s>\n", lvl * 2, "", nodename));
   else
      buf->Append(TString::Format("/>\n"));
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

   buf->Append(TString::Format("%*s\"%s\" : {", lvl * 4, "", nodename));
}

//______________________________________________________________________________
void TRootSnifferStoreJson::SetField(Int_t lvl, const char *field,
                                     const char *value, Int_t nfld)
{
   // set field (json field) in current node

   if (nfld == 0)
      buf->Append("\n");
   else
      buf->Append(",\n");
   buf->Append(TString::Format("%*s\"%s\" : \"%s\"", lvl * 4 + 2, "", field, value));
}

//______________________________________________________________________________
void TRootSnifferStoreJson::BeforeNextChild(Int_t lvl, Int_t nchld, Int_t nfld)
{
   // called before next child node created

   if (nchld == 0) {
      if (nfld == 0)
         buf->Append("\n");
      else
         buf->Append(",\n");
      buf->Append(TString::Format("%*s\"childs\" : [\n", lvl * 4 + 2, ""));
   } else {
      buf->Append(",\n");
   }
}

//______________________________________________________________________________
void TRootSnifferStoreJson::CloseNode(Int_t lvl, const char *, Int_t numchilds)
{
   // called when node should be closed
   // depending from number of childs different xml format is applied

   if (numchilds > 0)
      buf->Append(TString::Format("\n%*s]", lvl * 4 + 2, ""));
   buf->Append(TString::Format("\n%*s}", lvl * 4, ""));
}

