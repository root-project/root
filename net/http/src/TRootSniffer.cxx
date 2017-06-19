// $Id$
// Author: Sergey Linev   22/12/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TRootSniffer.h"

#include "TH1.h"
#include "TGraph.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TKey.h"
#include "TList.h"
#include "TMemFile.h"
#include "TStreamerInfo.h"
#include "TBufferFile.h"
#include "TBufferJSON.h"
#include "TBufferXML.h"
#include "TROOT.h"
#include "TTimer.h"
#include "TFolder.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TClass.h"
#include "TMethod.h"
#include "TFunction.h"
#include "TMethodArg.h"
#include "TMethodCall.h"
#include "TRealData.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TBaseClass.h"
#include "TObjString.h"
#include "TUrl.h"
#include "TImage.h"
#include "RZip.h"
#include "RVersion.h"
#include "TVirtualMutex.h"
#include "TRootSnifferStore.h"
#include "THttpCallArg.h"

#include <stdlib.h>
#include <vector>
#include <string.h>

const char *item_prop_kind = "_kind";
const char *item_prop_more = "_more";
const char *item_prop_title = "_title";
const char *item_prop_hidden = "_hidden";
const char *item_prop_typename = "_typename";
const char *item_prop_arraydim = "_arraydim";
const char *item_prop_realname = "_realname"; // real object name
const char *item_prop_user = "_username";
const char *item_prop_autoload = "_autoload";
const char *item_prop_rootversion = "_root_version";

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootSnifferScanRec                                                  //
//                                                                      //
// Structure used to scan hierarchies of ROOT objects                   //
// Represents single level of hierarchy                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// constructor

TRootSnifferScanRec::TRootSnifferScanRec()
   : fParent(nullptr), fMask(0), fSearchPath(nullptr), fLevel(0), fItemName(), fItemsNames(), fRestriction(0),
     fStore(nullptr), fHasMore(kFALSE), fNodeStarted(kFALSE), fNumFields(0), fNumChilds(0)
{
   fItemsNames.SetOwner(kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TRootSnifferScanRec::~TRootSnifferScanRec()
{
   CloseNode();
}

////////////////////////////////////////////////////////////////////////////////
/// record field for current element

void TRootSnifferScanRec::SetField(const char *name, const char *value, Bool_t with_quotes)
{
   if (CanSetFields()) fStore->SetField(fLevel, name, value, with_quotes);
   fNumFields++;
}

////////////////////////////////////////////////////////////////////////////////
/// indicates that new child for current element will be started

void TRootSnifferScanRec::BeforeNextChild()
{
   if (CanSetFields()) fStore->BeforeNextChild(fLevel, fNumChilds, fNumFields);
   fNumChilds++;
}

////////////////////////////////////////////////////////////////////////////////
/// constructs item name from object name
/// if special symbols like '/', '#', ':', '&', '?'  are used in object name
/// they will be replaced with '_'.
/// To avoid item name duplication, additional id number can be appended

void TRootSnifferScanRec::MakeItemName(const char *objname, TString &itemname)
{
   std::string nnn = objname;

   size_t pos;

   // replace all special symbols which can make problem to navigate in hierarchy
   while ((pos = nnn.find_first_of("- []<>#:&?/\'\"\\")) != std::string::npos) nnn.replace(pos, 1, "_");

   itemname = nnn.c_str();
   Int_t cnt = 0;

   while (fItemsNames.FindObject(itemname.Data())) {
      itemname.Form("%s_%d", nnn.c_str(), cnt++);
   }

   fItemsNames.Add(new TObjString(itemname.Data()));
}

////////////////////////////////////////////////////////////////////////////////
/// Produce full name, including all parents

void TRootSnifferScanRec::BuildFullName(TString &buf, TRootSnifferScanRec *prnt)
{
   if (!prnt) prnt = fParent;

   if (prnt) {
      prnt->BuildFullName(buf);

      buf.Append("/");
      buf.Append(fItemName);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// creates new node with specified name
/// if special symbols like "[]&<>" are used, node name
/// will be replaced by default name like "extra_item_N" and
/// original node name will be recorded as "_original_name" field
/// Optionally, object name can be recorded as "_realname" field

void TRootSnifferScanRec::CreateNode(const char *_node_name)
{
   if (!CanSetFields()) return;

   fNodeStarted = kTRUE;

   if (fParent) fParent->BeforeNextChild();

   if (fStore) fStore->CreateNode(fLevel, _node_name);
}

////////////////////////////////////////////////////////////////////////////////
/// close started node

void TRootSnifferScanRec::CloseNode()
{
   if (fStore && fNodeStarted) {
      fStore->CloseNode(fLevel, fNumChilds);
      fNodeStarted = kFALSE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// set root class name as node kind
/// in addition, path to master item (streamer info) specified
/// Such master item required to correctly unstream data on JavaScript

void TRootSnifferScanRec::SetRootClass(TClass *cl)
{
   if ((cl != nullptr) && CanSetFields()) SetField(item_prop_kind, TString::Format("ROOT.%s", cl->GetName()));
}

////////////////////////////////////////////////////////////////////////////////
/// returns true if scanning is done
/// Can happen when searched element is found

Bool_t TRootSnifferScanRec::Done() const
{
   if (fStore == nullptr) return kFALSE;

   if ((fMask & kSearch) && fStore->GetResPtr()) return kTRUE;

   if ((fMask & kCheckChilds) && fStore->GetResPtr() && (fStore->GetResNumChilds() >= 0)) return kTRUE;

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Checks if result will be accepted.
/// Used to verify if sniffer should read object from the file

Bool_t TRootSnifferScanRec::IsReadyForResult() const
{
   if (Done()) return kFALSE;

   // only when doing search, result will be propagated
   if ((fMask & (kSearch | kCheckChilds)) == 0) return kFALSE;

   // only when full search path is scanned
   if (fSearchPath != nullptr) return kFALSE;

   if (fStore == nullptr) return kFALSE;

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// set results of scanning
/// when member should be specified, use SetFoundResult instead

Bool_t TRootSnifferScanRec::SetResult(void *obj, TClass *cl, TDataMember *member)
{
   if (member == nullptr) return SetFoundResult(obj, cl);

   fStore->Error("SetResult",
                 "When member specified, pointer on object (not member) should be provided; use SetFoundResult");
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// set results of scanning
/// when member specified, obj is pointer on object to which member belongs

Bool_t TRootSnifferScanRec::SetFoundResult(void *obj, TClass *cl, TDataMember *member)
{
   if (Done()) return kTRUE;

   if (!IsReadyForResult()) return kFALSE;

   fStore->SetResult(obj, cl, member, fNumChilds, fRestriction);

   return Done();
}

////////////////////////////////////////////////////////////////////////////////
/// returns current depth of scanned hierarchy

Int_t TRootSnifferScanRec::Depth() const
{
   Int_t cnt = 0;
   const TRootSnifferScanRec *rec = this;
   while (rec->fParent) {
      rec = rec->fParent;
      cnt++;
   }

   return cnt;
}

////////////////////////////////////////////////////////////////////////////////
/// returns true if current item can be expanded - means one could explore
/// objects members

Bool_t TRootSnifferScanRec::CanExpandItem()
{
   if (fMask & (kExpand | kSearch | kCheckChilds)) return kTRUE;

   if (!fHasMore) return kFALSE;

   // if parent has expand mask, allow to expand item
   if (fParent && (fParent->fMask & kExpand)) return kTRUE;

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// returns read-only flag for current item
/// Depends from default value and current restrictions

Bool_t TRootSnifferScanRec::IsReadOnly(Bool_t dflt)
{
   if (fRestriction == 0) return dflt;

   return fRestriction != 2;
}

////////////////////////////////////////////////////////////////////////////////
/// Method verifies if new level of hierarchy
/// should be started with provided object.
/// If required, all necessary nodes and fields will be created
/// Used when different collection kinds should be scanned

Bool_t TRootSnifferScanRec::GoInside(TRootSnifferScanRec &super, TObject *obj, const char *obj_name,
                                     TRootSniffer *sniffer)
{
   if (super.Done()) return kFALSE;

   if ((obj != nullptr) && (obj_name == nullptr)) obj_name = obj->GetName();

   // exclude zero names
   if ((obj_name == nullptr) || (*obj_name == 0)) return kFALSE;

   const char *full_name = nullptr;

   // remove slashes from file names
   if (obj && obj->InheritsFrom(TDirectoryFile::Class())) {
      const char *slash = strrchr(obj_name, '/');
      if (slash != nullptr) {
         full_name = obj_name;
         obj_name = slash + 1;
         if (*obj_name == 0) obj_name = "file";
      }
   }

   super.MakeItemName(obj_name, fItemName);

   if (sniffer && sniffer->HasRestriction(fItemName.Data())) {
      // check restriction more precisely
      TString fullname;
      BuildFullName(fullname, &super);
      fRestriction = sniffer->CheckRestriction(fullname.Data());
      if (fRestriction < 0) return kFALSE;
   }

   fParent = &super;
   fLevel = super.fLevel;
   fStore = super.fStore;
   fSearchPath = super.fSearchPath;
   fMask = super.fMask & kActions;
   if (fRestriction == 0) fRestriction = super.fRestriction; // get restriction from parent
   Bool_t topelement(kFALSE);

   if (fMask & kScan) {
      // if scanning only fields, ignore all childs
      if (super.ScanOnlyFields()) return kFALSE;
      // only when doing scan, increment level, used for text formatting
      fLevel++;
   } else {
      if (fSearchPath == nullptr) return kFALSE;

      if (strncmp(fSearchPath, fItemName.Data(), fItemName.Length()) != 0) return kFALSE;

      const char *separ = fSearchPath + fItemName.Length();

      Bool_t isslash = kFALSE;
      while (*separ == '/') {
         separ++;
         isslash = kTRUE;
      }

      if (*separ == 0) {
         fSearchPath = nullptr;
         if (fMask & kExpand) {
            topelement = kTRUE;
            fMask = (fMask & kOnlyFields) | kScan;
            fHasMore = (fMask & kOnlyFields) == 0;
         }
      } else {
         if (!isslash) return kFALSE;
         fSearchPath = separ;
      }
   }

   CreateNode(fItemName.Data());

   if ((obj_name != nullptr) && (fItemName != obj_name)) SetField(item_prop_realname, obj_name);

   if (full_name != nullptr) SetField("_fullname", full_name);

   if (topelement) SetField(item_prop_rootversion, TString::Format("%d", ROOT_VERSION_CODE));

   if (topelement && sniffer->GetAutoLoad()) SetField(item_prop_autoload, sniffer->GetAutoLoad());

   return kTRUE;
}

// ====================================================================

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootSniffer                                                         //
//                                                                      //
// Sniffer of ROOT objects, data provider for THttpServer               //
// Provides methods to scan different structures like folders,          //
// directories, files, trees, collections                               //
// Can locate objects (or its data member) per name                     //
// Can be extended to application-specific classes                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TRootSniffer);

   ////////////////////////////////////////////////////////////////////////////////
   /// constructor

TRootSniffer::TRootSniffer(const char *name, const char *objpath)
   : TNamed(name, "sniffer of root objects"), fObjectsPath(objpath), fMemFile(nullptr), fSinfo(nullptr),
     fReadOnly(kTRUE), fScanGlobalDir(kTRUE), fCurrentArg(nullptr), fCurrentRestrict(0), fCurrentAllowedMethods(0),
     fRestrictions(), fAutoLoad()
{
   fRestrictions.SetOwner(kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TRootSniffer::~TRootSniffer()
{
   if (fSinfo) {
      delete fSinfo;
      fSinfo = nullptr;
   }

   if (fMemFile) {
      delete fMemFile;
      fMemFile = nullptr;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// set current http arguments, which then used in different process methods
/// For instance, if user authorized with some user name,
/// depending from restrictions some objects will be invisible
/// or user get full access to the element

void TRootSniffer::SetCurrentCallArg(THttpCallArg *arg)
{
   fCurrentArg = arg;
   fCurrentRestrict = 0;
   fCurrentAllowedMethods = "";
}

////////////////////////////////////////////////////////////////////////////////
/// Restrict access to the specified location
///
/// Hides or provides read-only access to different parts of the hierarchy
/// Restriction done base on user-name specified with http requests
/// Options can be specified in URL style (separated with &)
/// Following parameters can be specified:
///    visible = [all|user(s)] - make item visible for all users or only specified user
///    hidden = [all|user(s)] - make item hidden from all users or only specified user
///    readonly = [all|user(s)] - make item read-only for all users or only specified user
///    allow = [all|user(s)] - make full access for all users or only specified user
///    allow_method = method(s)  - allow method(s) execution even when readonly flag specified for the object
/// Like make command seen by all but can be executed only by admin
///    sniff->Restrict("/CmdReset","allow=admin");
/// Or fully hide command from guest account
///    sniff->Restrict("/CmdRebin","hidden=guest");

void TRootSniffer::Restrict(const char *path, const char *options)
{
   const char *rslash = strrchr(path, '/');
   if (rslash) rslash++;
   if ((rslash == nullptr) || (*rslash == 0)) rslash = path;

   fRestrictions.Add(new TNamed(rslash, TString::Format("%s%s%s", path, "%%%", options).Data()));
}

////////////////////////////////////////////////////////////////////////////////
/// When specified, _autoload attribute will be always add
/// to top element of h.json/h.hml requests
/// Used to instruct browser automatically load special code

void TRootSniffer::SetAutoLoad(const char *scripts)
{
   fAutoLoad = scripts != nullptr ? scripts : "";
}

////////////////////////////////////////////////////////////////////////////////
/// return name of configured autoload scripts (or 0)

const char *TRootSniffer::GetAutoLoad() const
{
   return fAutoLoad.Length() > 0 ? fAutoLoad.Data() : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Made fast check if item with specified name is in restriction list
/// If returns true, requires precise check with CheckRestriction() method

Bool_t TRootSniffer::HasRestriction(const char *item_name)
{
   if ((item_name == nullptr) || (*item_name == 0) || (fCurrentArg == nullptr)) return kFALSE;

   return fRestrictions.FindObject(item_name) != nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// return 2 when option match to current user name
/// return 1 when option==all
/// return 0 when option does not match user name

Int_t TRootSniffer::WithCurrentUserName(const char *option)
{
   const char *username = fCurrentArg ? fCurrentArg->GetUserName() : nullptr;

   if ((username == nullptr) || (option == nullptr) || (*option == 0)) return 0;

   if (strcmp(option, "all") == 0) return 1;

   if (strcmp(username, option) == 0) return 2;

   if (strstr(option, username) == nullptr) return -1;

   TObjArray *arr = TString(option).Tokenize(",");

   Bool_t find = arr->FindObject(username) != nullptr;

   delete arr;

   return find ? 2 : -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Checked if restriction is applied to the item
/// full_item_name should have full path to the item
///
/// Returns -1 - object invisible, cannot be accessed or listed
///          0 -  no explicit restrictions, use default
///          1 - read-only access
///          2 - full access

Int_t TRootSniffer::CheckRestriction(const char *full_item_name)
{
   if ((full_item_name == nullptr) || (*full_item_name == 0)) return 0;

   const char *item_name = strrchr(full_item_name, '/');
   if (item_name) item_name++;
   if ((item_name == nullptr) || (*item_name == 0)) item_name = full_item_name;

   TString pattern1 = TString("*/") + item_name + "%%%";
   TString pattern2 = TString(full_item_name) + "%%%";

   const char *options = nullptr;
   TIter iter(&fRestrictions);
   TObject *obj;

   while ((obj = iter()) != nullptr) {
      const char *title = obj->GetTitle();

      if (strstr(title, pattern1.Data()) == title) {
         options = title + pattern1.Length();
         break;
      }
      if (strstr(title, pattern2.Data()) == title) {
         options = title + pattern2.Length();
         break;
      }
   }

   if (options == nullptr) return 0;

   TUrl url;
   url.SetOptions(options);
   url.ParseOptions();

   Int_t can_see =
      WithCurrentUserName(url.GetValueFromOptions("visible")) - WithCurrentUserName(url.GetValueFromOptions("hidden"));

   Int_t can_access =
      WithCurrentUserName(url.GetValueFromOptions("allow")) - WithCurrentUserName(url.GetValueFromOptions("readonly"));

   if (can_access > 0) return 2; // first of all, if access enabled, provide it
   if (can_see < 0) return -1;   // if object to be hidden, do it

   const char *methods = url.GetValueFromOptions("allow_method");
   if (methods != nullptr) fCurrentAllowedMethods = methods;

   if (can_access < 0) return 1; // read-only access

   return 0; // default behavior
}

////////////////////////////////////////////////////////////////////////////////
/// scan object data members
/// some members like enum or static members will be excluded

void TRootSniffer::ScanObjectMembers(TRootSnifferScanRec &rec, TClass *cl, char *ptr)
{
   if ((cl == nullptr) || (ptr == nullptr) || rec.Done()) return;

   // ensure that real class data (including parents) exists
   if (!(cl->Property() & kIsAbstract)) cl->BuildRealData();

   // scan only real data
   TObject *obj = nullptr;
   TIter iter(cl->GetListOfRealData());
   while ((obj = iter()) != nullptr) {
      TRealData *rdata = dynamic_cast<TRealData *>(obj);
      if ((rdata == nullptr) || strchr(rdata->GetName(), '.')) continue;

      TDataMember *member = rdata->GetDataMember();
      // exclude enum or static variables
      if ((member == nullptr) || (member->Property() & (kIsStatic | kIsEnum | kIsUnion))) continue;
      char *member_ptr = ptr + rdata->GetThisOffset();

      if (member->IsaPointer()) member_ptr = *((char **)member_ptr);

      TRootSnifferScanRec chld;

      if (chld.GoInside(rec, member, nullptr, this)) {

         TClass *mcl =
            (member->IsBasic() || member->IsSTLContainer()) ? nullptr : gROOT->GetClass(member->GetTypeName());

         Int_t coll_offset = mcl ? mcl->GetBaseClassOffset(TCollection::Class()) : -1;
         if (coll_offset >= 0) {
            chld.SetField(item_prop_more, "true", kFALSE);
            chld.fHasMore = kTRUE;
         }

         if (chld.SetFoundResult(ptr, cl, member)) break;

         const char *title = member->GetTitle();
         if ((title != nullptr) && (strlen(title) != 0)) chld.SetField(item_prop_title, title);

         if (member->GetTypeName()) chld.SetField(item_prop_typename, member->GetTypeName());

         if (member->GetArrayDim() > 0) {
            // store array dimensions in form [N1,N2,N3,...]
            TString dim("[");
            for (Int_t n = 0; n < member->GetArrayDim(); n++) {
               if (n > 0) dim.Append(",");
               dim.Append(TString::Format("%d", member->GetMaxIndex(n)));
            }
            dim.Append("]");
            chld.SetField(item_prop_arraydim, dim, kFALSE);
         } else if (member->GetArrayIndex() != nullptr) {
            TRealData *idata = cl->GetRealData(member->GetArrayIndex());
            TDataMember *imember = (idata != nullptr) ? idata->GetDataMember() : nullptr;
            if ((imember != nullptr) && (strcmp(imember->GetTrueTypeName(), "int") == 0)) {
               Int_t arraylen = *((int *)(ptr + idata->GetThisOffset()));
               chld.SetField(item_prop_arraydim, TString::Format("[%d]", arraylen), kFALSE);
            }
         }

         chld.SetRootClass(mcl);

         if (chld.CanExpandItem()) {
            if (coll_offset >= 0) {
               // chld.SetField("#members", "true", kFALSE);
               ScanCollection(chld, (TCollection *)(member_ptr + coll_offset));
            }
         }

         if (chld.SetFoundResult(ptr, cl, member)) break;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// scans object properties
/// here such fields as _autoload or _icon properties depending on class or object name could be assigned
/// By default properties, coded in the Class title are scanned. Example:
///   ClassDef(UserClassName, 1) //  class comments *SNIFF*  _field1=value _field2="string value"
/// Here *SNIFF* mark is important. After it all expressions like field=value are parsed
/// One could use double quotes to code string values with spaces.
/// Fields separated from each other with spaces

void TRootSniffer::ScanObjectProperties(TRootSnifferScanRec &rec, TObject *obj)
{
   TClass *cl = obj ? obj->IsA() : nullptr;

   if (cl && cl->InheritsFrom(TLeaf::Class())) {
      rec.SetField("_more", "false");
      rec.SetField("_can_draw", "false");
      return;
   }

   const char *pos = strstr(cl ? cl->GetTitle() : "", "*SNIFF*");
   if (pos == nullptr) return;

   pos += 7;
   while (*pos != 0) {
      if (*pos == ' ') {
         pos++;
         continue;
      }
      // first locate identifier
      const char *pos0 = pos;
      while ((*pos != 0) && (*pos != '=')) pos++;
      if (*pos == 0) return;
      TString name(pos0, pos - pos0);
      pos++;
      Bool_t quotes = (*pos == '\"');
      if (quotes) pos++;
      pos0 = pos;
      // then value with or without quotes
      while ((*pos != 0) && (*pos != (quotes ? '\"' : ' '))) pos++;
      TString value(pos0, pos - pos0);
      rec.SetField(name, value);
      if (quotes) pos++;
      pos++;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// scans object childs (if any)
/// here one scans collection, branches, trees and so on

void TRootSniffer::ScanObjectChilds(TRootSnifferScanRec &rec, TObject *obj)
{
   if (obj->InheritsFrom(TFolder::Class())) {
      ScanCollection(rec, ((TFolder *)obj)->GetListOfFolders());
   } else if (obj->InheritsFrom(TDirectory::Class())) {
      TDirectory *dir = (TDirectory *)obj;
      ScanCollection(rec, dir->GetList(), nullptr, dir->GetListOfKeys());
   } else if (obj->InheritsFrom(TTree::Class())) {
      if (!rec.IsReadOnly(fReadOnly)) {
         rec.SetField("_player", "JSROOT.drawTreePlayer");
         rec.SetField("_prereq", "jq2d");
      }
      ScanCollection(rec, ((TTree *)obj)->GetListOfLeaves());
   } else if (obj->InheritsFrom(TBranch::Class())) {
      ScanCollection(rec, ((TBranch *)obj)->GetListOfLeaves());
   } else if (rec.CanExpandItem()) {
      ScanObjectMembers(rec, obj->IsA(), (char *)obj);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// scan collection content

void TRootSniffer::ScanCollection(TRootSnifferScanRec &rec, TCollection *lst, const char *foldername,
                                  TCollection *keys_lst)
{
   if (((lst == nullptr) || (lst->GetSize() == 0)) && ((keys_lst == nullptr) || (keys_lst->GetSize() == 0))) return;

   TRootSnifferScanRec folderrec;
   if (foldername) {
      if (!folderrec.GoInside(rec, nullptr, foldername, this)) return;
   }

   TRootSnifferScanRec &master = foldername ? folderrec : rec;

   if (lst != nullptr) {
      TIter iter(lst);
      TObject *next = iter();
      Bool_t isany = kFALSE;

      while (next != nullptr) {
         if (IsItemField(next)) {
            // special case - in the beginning one could have items for master folder
            if (!isany && (next->GetName() != nullptr) && ((*(next->GetName()) == '_') || master.ScanOnlyFields()))
               master.SetField(next->GetName(), next->GetTitle());
            next = iter();
            continue;
         }

         isany = kTRUE;
         TObject *obj = next;

         TRootSnifferScanRec chld;
         if (!chld.GoInside(master, obj, nullptr, this)) {
            next = iter();
            continue;
         }

         if (chld.SetResult(obj, obj->IsA())) return;

         Bool_t has_kind(kFALSE), has_title(kFALSE);

         ScanObjectProperties(chld, obj);
         // now properties, coded as TNamed objects, placed after object in the hierarchy
         while ((next = iter()) != nullptr) {
            if (!IsItemField(next)) break;
            if ((next->GetName() != nullptr) && ((*(next->GetName()) == '_') || chld.ScanOnlyFields())) {
               // only fields starting with _ are stored
               chld.SetField(next->GetName(), next->GetTitle());
               if (strcmp(next->GetName(), item_prop_kind) == 0) has_kind = kTRUE;
               if (strcmp(next->GetName(), item_prop_title) == 0) has_title = kTRUE;
            }
         }

         if (!has_kind) chld.SetRootClass(obj->IsA());
         if (!has_title && (obj->GetTitle() != nullptr)) chld.SetField(item_prop_title, obj->GetTitle());

         ScanObjectChilds(chld, obj);

         if (chld.SetResult(obj, obj->IsA())) return;
      }
   }

   if (keys_lst != nullptr) {
      TIter iter(keys_lst);
      TObject *kobj(nullptr);

      while ((kobj = iter()) != nullptr) {
         TKey *key = dynamic_cast<TKey *>(kobj);
         if (key == nullptr) continue;
         TObject *obj = (lst == nullptr) ? nullptr : lst->FindObject(key->GetName());

         // even object with the name exists, it should also match with class name
         if ((obj != nullptr) && (strcmp(obj->ClassName(), key->GetClassName()) != 0)) obj = nullptr;

         // if object of that name and of that class already in the list, ignore appropriate key
         if ((obj != nullptr) && (master.fMask & TRootSnifferScanRec::kScan)) continue;

         Bool_t iskey = kFALSE;
         // if object not exists, provide key itself for the scan
         if (obj == nullptr) {
            obj = key;
            iskey = kTRUE;
         }

         TRootSnifferScanRec chld;
         TString fullname = TString::Format("%s;%d", key->GetName(), key->GetCycle());

         if (chld.GoInside(master, obj, fullname.Data(), this)) {

            if (!chld.IsReadOnly(fReadOnly) && iskey && chld.IsReadyForResult()) {
               TObject *keyobj = key->ReadObj();
               if (keyobj != nullptr)
                  if (chld.SetResult(keyobj, keyobj->IsA())) return;
            }

            if (chld.SetResult(obj, obj->IsA())) return;

            TClass *obj_class = obj->IsA();

            ScanObjectProperties(chld, obj);

            if (obj->GetTitle() != nullptr) chld.SetField(item_prop_title, obj->GetTitle());

            // special handling of TKey class - in non-readonly mode
            // sniffer allowed to fetch objects
            if (!chld.IsReadOnly(fReadOnly) && iskey) {
               if (strcmp(key->GetClassName(), "TDirectoryFile") == 0) {
                  if (chld.fLevel == 0) {
                     TDirectory *dir = dynamic_cast<TDirectory *>(key->ReadObj());
                     if (dir != nullptr) {
                        obj = dir;
                        obj_class = dir->IsA();
                     }
                  } else {
                     chld.SetField(item_prop_more, "true", kFALSE);
                     chld.fHasMore = kTRUE;
                  }
               } else {
                  obj_class = TClass::GetClass(key->GetClassName());
                  if (obj_class && obj_class->InheritsFrom(TTree::Class())) {
                     if (rec.CanExpandItem()) {
                        // it is requested to expand tree element - read it
                        obj = key->ReadObj();
                        if (obj) obj_class = obj->IsA();
                     } else {
                        rec.SetField("_player", "JSROOT.drawTreePlayerKey");
                        rec.SetField("_prereq", "jq2d");
                        // rec.SetField("_more", "true"); // one could allow to extend
                     }
                  }
               }
            }

            rec.SetRootClass(obj_class);

            ScanObjectChilds(chld, obj);

            // here we should know how many childs are accumulated
            if (chld.SetResult(obj, obj_class)) return;
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// scan complete ROOT objects hierarchy
/// For the moment it includes objects in gROOT directory
/// and list of canvases and files
/// Also all registered objects are included.
/// One could reimplement this method to provide alternative
/// scan methods or to extend some collection kinds

void TRootSniffer::ScanRoot(TRootSnifferScanRec &rec)
{
   rec.SetField(item_prop_kind, "ROOT.Session");
   if (fCurrentArg && fCurrentArg->GetUserName()) rec.SetField(item_prop_user, fCurrentArg->GetUserName());

   // should be on the top while //root/http folder could have properties for itself
   TFolder *topf = dynamic_cast<TFolder *>(gROOT->FindObject("//root/http"));
   if (topf) {
      rec.SetField(item_prop_title, topf->GetTitle());
      ScanCollection(rec, topf->GetListOfFolders());
   }

   {
      TRootSnifferScanRec chld;
      if (chld.GoInside(rec, nullptr, "StreamerInfo", this)) {
         chld.SetField(item_prop_kind, "ROOT.TStreamerInfoList");
         chld.SetField(item_prop_title, "List of streamer infos for binary I/O");
         chld.SetField(item_prop_hidden, "true");
         chld.SetField("_after_request", "JSROOT.MarkAsStreamerInfo");
      }
   }

   if (IsScanGlobalDir()) {
      ScanCollection(rec, gROOT->GetList());

      ScanCollection(rec, gROOT->GetListOfCanvases(), "Canvases");

      ScanCollection(rec, gROOT->GetListOfFiles(), "Files");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// return true if object can be drawn

Bool_t TRootSniffer::IsDrawableClass(TClass *cl)
{
   if (cl == nullptr) return kFALSE;
   if (cl->InheritsFrom(TH1::Class())) return kTRUE;
   if (cl->InheritsFrom(TGraph::Class())) return kTRUE;
   if (cl->InheritsFrom(TCanvas::Class())) return kTRUE;
   if (cl->InheritsFrom(TProfile::Class())) return kTRUE;
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// scan ROOT hierarchy with provided store object

void TRootSniffer::ScanHierarchy(const char *topname, const char *path, TRootSnifferStore *store, Bool_t only_fields)
{
   TRootSnifferScanRec rec;
   rec.fSearchPath = path;
   if (rec.fSearchPath) {
      while (*rec.fSearchPath == '/') rec.fSearchPath++;
      if (*rec.fSearchPath == 0) rec.fSearchPath = nullptr;
   }

   // if path non-empty, we should find item first and than start scanning
   rec.fMask = (rec.fSearchPath == nullptr) ? TRootSnifferScanRec::kScan : TRootSnifferScanRec::kExpand;
   if (only_fields) rec.fMask |= TRootSnifferScanRec::kOnlyFields;

   rec.fStore = store;

   rec.CreateNode(topname);

   if (rec.fSearchPath == nullptr) rec.SetField(item_prop_rootversion, TString::Format("%d", ROOT_VERSION_CODE));

   if ((rec.fSearchPath == nullptr) && (GetAutoLoad() != nullptr)) rec.SetField(item_prop_autoload, GetAutoLoad());

   ScanRoot(rec);

   rec.CloseNode();
}

////////////////////////////////////////////////////////////////////////////////
/// Search element with specified path
/// Returns pointer on element
/// Optionally one could obtain element class, member description
/// and number of childs. When chld!=0, not only element is searched,
/// but also number of childs are counted. When member!=0, any object
/// will be scanned for its data members (disregard of extra options)

void *TRootSniffer::FindInHierarchy(const char *path, TClass **cl, TDataMember **member, Int_t *chld)
{
   if (IsStreamerInfoItem(path)) {
      // special handling for streamer info
      CreateMemFile();
      if (cl && fSinfo) *cl = fSinfo->IsA();
      return fSinfo;
   }

   TRootSnifferStore store;

   TRootSnifferScanRec rec;
   rec.fSearchPath = path;
   rec.fMask = (chld != nullptr) ? TRootSnifferScanRec::kCheckChilds : TRootSnifferScanRec::kSearch;
   if (*rec.fSearchPath == '/') rec.fSearchPath++;
   rec.fStore = &store;

   ScanRoot(rec);

   TDataMember *res_member = store.GetResMember();
   TClass *res_cl = store.GetResClass();
   void *res = store.GetResPtr();

   if ((res_member != nullptr) && (res_cl != nullptr) && (member == nullptr)) {
      res_cl =
         (res_member->IsBasic() || res_member->IsSTLContainer()) ? nullptr : gROOT->GetClass(res_member->GetTypeName());
      TRealData *rdata = res_cl->GetRealData(res_member->GetName());
      if (rdata) {
         res = (char *)res + rdata->GetThisOffset();
         if (res_member->IsaPointer()) res = *((char **)res);
      } else {
         res = nullptr; // should never happen
      }
   }

   if (cl) *cl = res_cl;
   if (member) *member = res_member;
   if (chld) *chld = store.GetResNumChilds();

   // remember current restriction
   fCurrentRestrict = store.GetResRestrict();

   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Search element in hierarchy, derived from TObject

TObject *TRootSniffer::FindTObjectInHierarchy(const char *path)
{
   TClass *cl(nullptr);

   void *obj = FindInHierarchy(path, &cl);

   return (cl != nullptr) && (cl->GetBaseClassOffset(TObject::Class()) == 0) ? (TObject *)obj : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns hash value for streamer infos
/// At the moment - just number of items in streamer infos list.

ULong_t TRootSniffer::GetStreamerInfoHash()
{
   return fSinfo ? fSinfo->GetSize() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get hash function for specified item
/// used to detect any changes in the specified object

ULong_t TRootSniffer::GetItemHash(const char *itemname)
{
   if (IsStreamerInfoItem(itemname)) return GetStreamerInfoHash();

   TObject *obj = FindTObjectInHierarchy(itemname);

   return obj == nullptr ? 0 : TString::Hash(obj, obj->IsA()->Size());
}

////////////////////////////////////////////////////////////////////////////////
/// Method verifies if object can be drawn

Bool_t TRootSniffer::CanDrawItem(const char *path)
{
   TClass *obj_cl(nullptr);
   void *res = FindInHierarchy(path, &obj_cl);
   return (res != nullptr) && IsDrawableClass(obj_cl);
}

////////////////////////////////////////////////////////////////////////////////
/// Method returns true when object has childs or
/// one could try to expand item

Bool_t TRootSniffer::CanExploreItem(const char *path)
{
   TClass *obj_cl(nullptr);
   Int_t obj_chld(-1);
   void *res = FindInHierarchy(path, &obj_cl, nullptr, &obj_chld);
   return (res != nullptr) && (obj_chld > 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Creates TMemFile instance, which used for objects streaming
/// One could not use TBufferFile directly,
/// while one also require streamer infos list

void TRootSniffer::CreateMemFile()
{
   if (fMemFile != nullptr) return;

   TDirectory *olddir = gDirectory;
   gDirectory = nullptr;
   TFile *oldfile = gFile;
   gFile = nullptr;

   fMemFile = new TMemFile("dummy.file", "RECREATE");
   gROOT->GetListOfFiles()->Remove(fMemFile);

   TH1F *d = new TH1F("d", "d", 10, 0, 10);
   fMemFile->WriteObject(d, "h1");
   delete d;

   TGraph *gr = new TGraph(10);
   gr->SetName("abc");
   //      // gr->SetDrawOptions("AC*");
   fMemFile->WriteObject(gr, "gr1");
   delete gr;

   fMemFile->WriteStreamerInfo();

   // make primary list of streamer infos
   TList *l = new TList();

   l->Add(gROOT->GetListOfStreamerInfo()->FindObject("TGraph"));
   l->Add(gROOT->GetListOfStreamerInfo()->FindObject("TH1F"));
   l->Add(gROOT->GetListOfStreamerInfo()->FindObject("TH1"));
   l->Add(gROOT->GetListOfStreamerInfo()->FindObject("TNamed"));
   l->Add(gROOT->GetListOfStreamerInfo()->FindObject("TObject"));

   fMemFile->WriteObject(l, "ll");
   delete l;

   fMemFile->WriteStreamerInfo();

   fSinfo = fMemFile->GetStreamerInfoList();

   gDirectory = olddir;
   gFile = oldfile;
}

////////////////////////////////////////////////////////////////////////////////
/// produce JSON data for specified item
/// For object conversion TBufferJSON is used

Bool_t TRootSniffer::ProduceJson(const char *path, const char *options, TString &res)
{
   if ((path == nullptr) || (*path == 0)) return kFALSE;

   if (*path == '/') path++;

   TUrl url;
   url.SetOptions(options);
   url.ParseOptions();
   Int_t compact = -1;
   if (url.GetValueFromOptions("compact")) compact = url.GetIntValueFromOptions("compact");

   TClass *obj_cl(nullptr);
   TDataMember *member(nullptr);
   void *obj_ptr = FindInHierarchy(path, &obj_cl, &member);
   if ((obj_ptr == nullptr) || ((obj_cl == nullptr) && (member == nullptr))) return kFALSE;

   res = TBufferJSON::ConvertToJSON(obj_ptr, obj_cl, compact >= 0 ? compact : 0, member ? member->GetName() : nullptr);

   return res.Length() > 0;
}

////////////////////////////////////////////////////////////////////////////////
/// execute command marked as _kind=='Command'

Bool_t TRootSniffer::ExecuteCmd(const char *path, const char *options, TString &res)
{
   TFolder *parent(nullptr);
   TObject *obj = GetItem(path, parent, kFALSE, kFALSE);

   const char *kind = GetItemField(parent, obj, item_prop_kind);
   if ((kind == nullptr) || (strcmp(kind, "Command") != 0)) {
      if (gDebug > 0) Info("ExecuteCmd", "Entry %s is not a command", path);
      res = "false";
      return kTRUE;
   }

   const char *cmethod = GetItemField(parent, obj, "method");
   if ((cmethod == nullptr) || (strlen(cmethod) == 0)) {
      if (gDebug > 0) Info("ExecuteCmd", "Entry %s do not defines method for execution", path);
      res = "false";
      return kTRUE;
   }

   // if read-only specified for the command, it is not allowed for execution
   if (fRestrictions.GetLast() >= 0) {
      FindInHierarchy(path); // one need to call method to check access rights
      if (fCurrentRestrict == 1) {
         if (gDebug > 0) Info("ExecuteCmd", "Entry %s not allowed for specified user", path);
         res = "false";
         return kTRUE;
      }
   }

   TString method = cmethod;

   const char *cnumargs = GetItemField(parent, obj, "_numargs");
   Int_t numargs = cnumargs ? TString(cnumargs).Atoi() : 0;
   if (numargs > 0) {
      TUrl url;
      url.SetOptions(options);
      url.ParseOptions();

      for (Int_t n = 0; n < numargs; n++) {
         TString argname = TString::Format("arg%d", n + 1);
         const char *argvalue = url.GetValueFromOptions(argname);
         if (argvalue == nullptr) {
            if (gDebug > 0)
               Info("ExecuteCmd", "For command %s argument %s not specified in options %s", path, argname.Data(),
                    options);
            res = "false";
            return kTRUE;
         }

         TString svalue = DecodeUrlOptionValue(argvalue, kTRUE);
         argname = TString("%") + argname + TString("%");
         method.ReplaceAll(argname, svalue);
      }
   }

   if (gDebug > 0) Info("ExecuteCmd", "Executing command %s method:%s", path, method.Data());

   TObject *item_obj = nullptr;
   Ssiz_t separ = method.Index("/->");

   if (method.Index("this->") == 0) {
      // if command name started with this-> means method of sniffer will be executed
      item_obj = this;
      separ = 3;
   } else if (separ != kNPOS) {
      item_obj = FindTObjectInHierarchy(TString(method.Data(), separ).Data());
   }

   if (item_obj != nullptr) {
      method =
         TString::Format("((%s*)%lu)->%s", item_obj->ClassName(), (long unsigned)item_obj, method.Data() + separ + 3);
      if (gDebug > 2) Info("ExecuteCmd", "Executing %s", method.Data());
   }

   Long_t v = gROOT->ProcessLineSync(method.Data());

   res.Form("%ld", v);

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// produce JSON/XML for specified item
/// contrary to h.json request, only fields for specified item are stored

Bool_t TRootSniffer::ProduceItem(const char *path, const char *options, TString &res, Bool_t asjson)
{
   if (asjson) {
      TRootSnifferStoreJson store(res, strstr(options, "compact") != nullptr);
      ScanHierarchy("top", path, &store, kTRUE);
   } else {
      TRootSnifferStoreXml store(res, strstr(options, "compact") != nullptr);
      ScanHierarchy("top", path, &store, kTRUE);
   }
   return res.Length() > 0;
}

////////////////////////////////////////////////////////////////////////////////
/// produce XML data for specified item
/// For object conversion TBufferXML is used

Bool_t TRootSniffer::ProduceXml(const char *path, const char * /*options*/, TString &res)
{
   if ((path == nullptr) || (*path == 0)) return kFALSE;

   if (*path == '/') path++;

   TClass *obj_cl(nullptr);
   void *obj_ptr = FindInHierarchy(path, &obj_cl);
   if ((obj_ptr == nullptr) || (obj_cl == nullptr)) return kFALSE;

   res = TBufferXML::ConvertToXML(obj_ptr, obj_cl);

   return res.Length() > 0;
}

////////////////////////////////////////////////////////////////////////////////
/// method replaces all kind of special symbols, which could appear in URL options

TString TRootSniffer::DecodeUrlOptionValue(const char *value, Bool_t remove_quotes)
{
   if ((value == nullptr) || (strlen(value) == 0)) return TString();

   TString res = value;

   res.ReplaceAll("%27", "\'");
   res.ReplaceAll("%22", "\"");
   res.ReplaceAll("%3E", ">");
   res.ReplaceAll("%3C", "<");
   res.ReplaceAll("%20", " ");
   res.ReplaceAll("%5B", "[");
   res.ReplaceAll("%5D", "]");
   res.ReplaceAll("%3D", "=");

   if (remove_quotes && (res.Length() > 1) && ((res[0] == '\'') || (res[0] == '\"')) &&
       (res[0] == res[res.Length() - 1])) {
      res.Remove(res.Length() - 1);
      res.Remove(0, 1);
   }

   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// execute command for specified object
/// options include method and extra list of parameters
/// sniffer should be not-readonly to allow execution of the commands
/// reskind defines kind of result 0 - debug, 1 - json, 2 - binary

Bool_t TRootSniffer::ProduceExe(const char *path, const char *options, Int_t reskind, TString *res_str, void **res_ptr,
                                Long_t *res_length)
{
   TString *debug = (reskind == 0) ? res_str : nullptr;

   if ((path == nullptr) || (*path == 0)) {
      if (debug) debug->Append("Item name not specified\n");
      return debug != nullptr;
   }

   if (*path == '/') path++;

   TClass *obj_cl(nullptr);
   void *obj_ptr = FindInHierarchy(path, &obj_cl);
   if (debug) debug->Append(TString::Format("Item:%s found:%s\n", path, obj_ptr ? "true" : "false"));
   if ((obj_ptr == nullptr) || (obj_cl == nullptr)) return debug != nullptr;

   TUrl url;
   url.SetOptions(options);

   const char *method_name = url.GetValueFromOptions("method");
   TString prototype = DecodeUrlOptionValue(url.GetValueFromOptions("prototype"), kTRUE);
   TString funcname = DecodeUrlOptionValue(url.GetValueFromOptions("func"), kTRUE);
   TMethod *method = nullptr;
   TFunction *func = nullptr;
   if (method_name != nullptr) {
      if (prototype.Length() == 0) {
         if (debug) debug->Append(TString::Format("Search for any method with name \'%s\'\n", method_name));
         method = obj_cl->GetMethodAllAny(method_name);
      } else {
         if (debug)
            debug->Append(
               TString::Format("Search for method \'%s\' with prototype \'%s\'\n", method_name, prototype.Data()));
         method = obj_cl->GetMethodWithPrototype(method_name, prototype);
      }
   }

   if (method != nullptr) {
      if (debug) debug->Append(TString::Format("Method: %s\n", method->GetPrototype()));
   } else {
      if (funcname.Length() > 0) {
         if (prototype.Length() == 0) {
            if (debug) debug->Append(TString::Format("Search for any function with name \'%s\'\n", funcname.Data()));
            func = gROOT->GetGlobalFunction(funcname);
         } else {
            if (debug)
               debug->Append(TString::Format("Search for function \'%s\' with prototype \'%s\'\n", funcname.Data(),
                                             prototype.Data()));
            func = gROOT->GetGlobalFunctionWithPrototype(funcname, prototype);
         }
      }

      if (func != nullptr) {
         if (debug) debug->Append(TString::Format("Function: %s\n", func->GetPrototype()));
      }
   }

   if ((method == nullptr) && (func == nullptr)) {
      if (debug) debug->Append("Method not found\n");
      return debug != nullptr;
   }

   if ((fReadOnly && (fCurrentRestrict == 0)) || (fCurrentRestrict == 1)) {
      if ((method != nullptr) && (fCurrentAllowedMethods.Index(method_name) == kNPOS)) {
         if (debug) debug->Append("Server runs in read-only mode, method cannot be executed\n");
         return debug != nullptr;
      } else if ((func != nullptr) && (fCurrentAllowedMethods.Index(funcname) == kNPOS)) {
         if (debug) debug->Append("Server runs in read-only mode, function cannot be executed\n");
         return debug != nullptr;
      } else {
         if (debug) debug->Append("For that special method server allows access even read-only mode is specified\n");
      }
   }

   TList *args = method ? method->GetListOfMethodArgs() : func->GetListOfMethodArgs();

   TList garbage;
   garbage.SetOwner(kTRUE); // use as garbage collection
   TObject *post_obj = nullptr; // object reconstructed from post request
   TString call_args;

   TIter next(args);
   TMethodArg *arg = nullptr;
   while ((arg = (TMethodArg *)next()) != nullptr) {

      if ((strcmp(arg->GetName(), "rest_url_opt") == 0) && (strcmp(arg->GetFullTypeName(), "const char*") == 0) &&
          (args->GetSize() == 1)) {
         // very special case - function requires list of options after method=argument

         const char *pos = strstr(options, "method=");
         if ((pos == nullptr) || (strlen(pos) < strlen(method_name) + 8)) return debug != nullptr;
         call_args.Form("\"%s\"", pos + strlen(method_name) + 8);
         break;
      }

      TString sval;
      const char *val = url.GetValueFromOptions(arg->GetName());
      if (val) {
         sval = DecodeUrlOptionValue(val, kFALSE);
         val = sval.Data();
      }

      if ((val != nullptr) && (strcmp(val, "_this_") == 0)) {
         // special case - object itself is used as argument
         sval.Form("(%s*)0x%lx", obj_cl->GetName(), (long unsigned)obj_ptr);
         val = sval.Data();
      } else if ((val != nullptr) && (fCurrentArg != nullptr) && (fCurrentArg->GetPostData() != nullptr)) {
         // process several arguments which are specific for post requests
         if (strcmp(val, "_post_object_xml_") == 0) {
            // post data has extra 0 at the end and can be used as null-terminated string
            post_obj = TBufferXML::ConvertFromXML((const char *)fCurrentArg->GetPostData());
            if (post_obj == nullptr) {
               sval = "0";
            } else {
               sval.Form("(%s*)0x%lx", post_obj->ClassName(), (long unsigned)post_obj);
               if (url.HasOption("_destroy_post_")) garbage.Add(post_obj);
            }
            val = sval.Data();
         } else if ((strcmp(val, "_post_object_") == 0) && url.HasOption("_post_class_")) {
            TString clname = url.GetValueFromOptions("_post_class_");
            TClass *arg_cl = gROOT->GetClass(clname, kTRUE, kTRUE);
            if ((arg_cl != nullptr) && (arg_cl->GetBaseClassOffset(TObject::Class()) == 0) && (post_obj == nullptr)) {
               post_obj = (TObject *)arg_cl->New();
               if (post_obj == nullptr) {
                  if (debug) debug->Append(TString::Format("Fail to create object of class %s\n", clname.Data()));
               } else {
                  if (debug)
                     debug->Append(TString::Format("Reconstruct object of class %s from POST data\n", clname.Data()));
                  TBufferFile buf(TBuffer::kRead, fCurrentArg->GetPostDataLength(), fCurrentArg->GetPostData(), kFALSE);
                  buf.MapObject(post_obj, arg_cl);
                  post_obj->Streamer(buf);
                  if (url.HasOption("_destroy_post_")) garbage.Add(post_obj);
               }
            }
            sval.Form("(%s*)0x%lx", clname.Data(), (long unsigned)post_obj);
            val = sval.Data();
         } else if (strcmp(val, "_post_data_") == 0) {
            sval.Form("(void*)0x%lx", (long unsigned)*res_ptr);
            val = sval.Data();
         } else if (strcmp(val, "_post_length_") == 0) {
            sval.Form("%ld", (long)*res_length);
            val = sval.Data();
         }
      }

      if (val == nullptr) val = arg->GetDefault();

      if (debug)
         debug->Append(TString::Format("  Argument:%s Type:%s Value:%s \n", arg->GetName(), arg->GetFullTypeName(),
                                       val ? val : "<missed>"));
      if (val == nullptr) return debug != nullptr;

      if (call_args.Length() > 0) call_args += ", ";

      if ((strcmp(arg->GetFullTypeName(), "const char*") == 0) || (strcmp(arg->GetFullTypeName(), "Option_t*") == 0)) {
         int len = strlen(val);
         if ((strlen(val) < 2) || (*val != '\"') || (val[len - 1] != '\"'))
            call_args.Append(TString::Format("\"%s\"", val));
         else
            call_args.Append(val);
      } else {
         call_args.Append(val);
      }
   }

   TMethodCall *call = nullptr;

   if (method != nullptr) {
      call = new TMethodCall(obj_cl, method_name, call_args.Data());
      if (debug) debug->Append(TString::Format("Calling obj->%s(%s);\n", method_name, call_args.Data()));

   } else {
      call = new TMethodCall(funcname.Data(), call_args.Data());
      if (debug) debug->Append(TString::Format("Calling %s(%s);\n", funcname.Data(), call_args.Data()));
   }

   garbage.Add(call);

   if (!call->IsValid()) {
      if (debug) debug->Append("Fail: invalid TMethodCall\n");
      return debug != nullptr;
   }

   Int_t compact = 0;
   if (url.GetValueFromOptions("compact")) compact = url.GetIntValueFromOptions("compact");

   TString res = "null";
   void *ret_obj = nullptr;
   TClass *ret_cl = nullptr;
   TBufferFile *resbuf = nullptr;
   if (reskind == 2) {
      resbuf = new TBufferFile(TBuffer::kWrite, 10000);
      garbage.Add(resbuf);
   }

   switch (call->ReturnType()) {
   case TMethodCall::kLong: {
      Long_t l(0);
      if (method)
         call->Execute(obj_ptr, l);
      else
         call->Execute(l);
      if (resbuf)
         resbuf->WriteLong(l);
      else
         res.Form("%ld", l);
      break;
   }
   case TMethodCall::kDouble: {
      Double_t d(0.);
      if (method)
         call->Execute(obj_ptr, d);
      else
         call->Execute(d);
      if (resbuf)
         resbuf->WriteDouble(d);
      else
         res.Form(TBufferJSON::GetFloatFormat(), d);
      break;
   }
   case TMethodCall::kString: {
      char *txt(nullptr);
      if (method)
         call->Execute(obj_ptr, &txt);
      else
         call->Execute(nullptr, &txt); // here 0 is artificial, there is no proper signature
      if (txt != nullptr) {
         if (resbuf)
            resbuf->WriteString(txt);
         else
            res.Form("\"%s\"", txt);
      }
      break;
   }
   case TMethodCall::kOther: {
      std::string ret_kind = func ? func->GetReturnTypeNormalizedName() : method->GetReturnTypeNormalizedName();
      if ((ret_kind.length() > 0) && (ret_kind[ret_kind.length() - 1] == '*')) {
         ret_kind.resize(ret_kind.length() - 1);
         ret_cl = gROOT->GetClass(ret_kind.c_str(), kTRUE, kTRUE);
      }

      if (ret_cl != nullptr) {
         Long_t l(0);
         if (method)
            call->Execute(obj_ptr, l);
         else
            call->Execute(l);
         if (l != 0) ret_obj = (void *)l;
      } else {
         if (method)
            call->Execute(obj_ptr);
         else
            call->Execute();
      }

      break;
   }
   case TMethodCall::kNone: {
      if (method)
         call->Execute(obj_ptr);
      else
         call->Execute();
      break;
   }
   }

   const char *_ret_object_ = url.GetValueFromOptions("_ret_object_");
   if (_ret_object_ != nullptr) {
      TObject *obj = nullptr;
      if (gDirectory != nullptr) obj = gDirectory->Get(_ret_object_);
      if (debug) debug->Append(TString::Format("Return object %s found %s\n", _ret_object_, obj ? "true" : "false"));

      if (obj == nullptr) {
         res = "null";
      } else {
         ret_obj = obj;
         ret_cl = obj->IsA();
      }
   }

   if ((ret_obj != nullptr) && (ret_cl != nullptr)) {
      if ((resbuf != nullptr) && (ret_cl->GetBaseClassOffset(TObject::Class()) == 0)) {
         TObject *obj = (TObject *)ret_obj;
         resbuf->MapObject(obj);
         obj->Streamer(*resbuf);
         if (fCurrentArg) fCurrentArg->SetExtraHeader("RootClassName", ret_cl->GetName());
      } else {
         res = TBufferJSON::ConvertToJSON(ret_obj, ret_cl, compact);
      }
   }

   if ((resbuf != nullptr) && (resbuf->Length() > 0) && (res_ptr != nullptr) && (res_length != nullptr)) {
      *res_ptr = malloc(resbuf->Length());
      memcpy(*res_ptr, resbuf->Buffer(), resbuf->Length());
      *res_length = resbuf->Length();
   }

   if (debug) debug->Append(TString::Format("Result = %s\n", res.Data()));

   if ((reskind == 1) && res_str) *res_str = res;

   if (url.HasOption("_destroy_result_") && (ret_obj != nullptr) && (ret_cl != nullptr)) {
      ret_cl->Destructor(ret_obj);
      if (debug) debug->Append("Destroy result object at the end\n");
   }

   // delete all garbage objects, but should be also done with any return
   garbage.Delete();

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Process several requests, packing all results into binary or JSON buffer
/// Input parameters should be coded in the POST block and has
/// individual request relative to current path, separated with '\n' symbol like
/// item1/root.bin\n
/// item2/exe.bin?method=GetList\n
/// item3/exe.bin?method=GetTitle\n
/// Request requires 'number' URL option which contains number of requested items
///
/// In case of binary request output buffer looks like:
/// 4bytes length + payload, 4bytes length + payload, ...
/// In case of JSON request output is array with results for each item
/// multi.json request do not support binary requests for the items

Bool_t TRootSniffer::ProduceMulti(const char *path, const char *options, void *&ptr, Long_t &length, TString &str,
                                  Bool_t asjson)
{
   if ((fCurrentArg == nullptr) || (fCurrentArg->GetPostDataLength() <= 0) || (fCurrentArg->GetPostData() == nullptr))
      return kFALSE;

   const char *args = (const char *)fCurrentArg->GetPostData();
   const char *ends = args + fCurrentArg->GetPostDataLength();

   TUrl url;
   url.SetOptions(options);

   Int_t number = 0;
   if (url.GetValueFromOptions("number")) number = url.GetIntValueFromOptions("number");

   // binary buffers required only for binary requests, json output can be produced as is
   std::vector<void *> mem;
   std::vector<Long_t> memlen;

   if (asjson) str = "[";

   for (Int_t n = 0; n < number; n++) {
      const char *next = args;
      while ((next < ends) && (*next != '\n')) next++;
      if (next == ends) {
         Error("ProduceMulti", "Not enough arguments in POST block");
         break;
      }

      TString file1(args, next - args);
      args = next + 1;

      TString path1, opt1;

      // extract options
      Int_t pos = file1.First('?');
      if (pos != kNPOS) {
         opt1 = file1(pos + 1, file1.Length() - pos);
         file1.Resize(pos);
      }

      // extract extra path
      pos = file1.Last('/');
      if (pos != kNPOS) {
         path1 = file1(0, pos);
         file1.Remove(0, pos + 1);
      }

      if ((path != nullptr) && (*path != 0)) path1 = TString(path) + "/" + path1;

      void *ptr1 = nullptr;
      Long_t len1 = 0;
      TString str1;

      // produce next item request
      Produce(path1, file1, opt1, ptr1, len1, str1);

      if (asjson) {
         if (n > 0) str.Append(", ");
         if (ptr1 != nullptr) {
            str.Append("\"<non-supported binary>\"");
            free(ptr1);
         } else if (str1.Length() > 0)
            str.Append(str1);
         else
            str.Append("null");
      } else {
         if ((str1.Length() > 0) && (ptr1 == nullptr)) {
            len1 = str1.Length();
            ptr1 = malloc(len1);
            memcpy(ptr1, str1.Data(), len1);
         }
         mem.push_back(ptr1);
         memlen.push_back(len1);
      }
   }

   if (asjson) {
      str.Append("]");
   } else {
      length = 0;
      for (unsigned n = 0; n < mem.size(); n++) {
         length += 4 + memlen[n];
      }
      ptr = malloc(length);
      char *curr = (char *)ptr;
      for (unsigned n = 0; n < mem.size(); n++) {
         Long_t l = memlen[n];
         *curr++ = (char)(l & 0xff);
         l = l >> 8;
         *curr++ = (char)(l & 0xff);
         l = l >> 8;
         *curr++ = (char)(l & 0xff);
         l = l >> 8;
         *curr++ = (char)(l & 0xff);
         if ((mem[n] != nullptr) && (memlen[n] > 0)) memcpy(curr, mem[n], memlen[n]);
         curr += memlen[n];
      }
   }

   for (unsigned n = 0; n < mem.size(); n++) free(mem[n]);

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Return true if it is streamer info item name

Bool_t TRootSniffer::IsStreamerInfoItem(const char *itemname)
{
   if ((itemname == nullptr) || (*itemname == 0)) return kFALSE;

   return (strcmp(itemname, "StreamerInfo") == 0) || (strcmp(itemname, "StreamerInfo/") == 0);
}

////////////////////////////////////////////////////////////////////////////////
/// produce binary data for specified item
/// if "zipped" option specified in query, buffer will be compressed

Bool_t TRootSniffer::ProduceBinary(const char *path, const char * /*query*/, void *&ptr, Long_t &length)
{
   if ((path == nullptr) || (*path == 0)) return kFALSE;

   if (*path == '/') path++;

   TClass *obj_cl(nullptr);
   void *obj_ptr = FindInHierarchy(path, &obj_cl);
   if ((obj_ptr == nullptr) || (obj_cl == nullptr)) return kFALSE;

   if (obj_cl->GetBaseClassOffset(TObject::Class()) != 0) {
      Info("ProduceBinary", "Non-TObject class not supported");
      return kFALSE;
   }

   // ensure that memfile exists
   CreateMemFile();

   TDirectory *olddir = gDirectory;
   gDirectory = nullptr;
   TFile *oldfile = gFile;
   gFile = nullptr;

   TObject *obj = (TObject *)obj_ptr;

   TBufferFile *sbuf = new TBufferFile(TBuffer::kWrite, 100000);
   sbuf->SetParent(fMemFile);
   sbuf->MapObject(obj);
   obj->Streamer(*sbuf);
   if (fCurrentArg) fCurrentArg->SetExtraHeader("RootClassName", obj_cl->GetName());

   // produce actual version of streamer info
   delete fSinfo;
   fMemFile->WriteStreamerInfo();
   fSinfo = fMemFile->GetStreamerInfoList();

   gDirectory = olddir;
   gFile = oldfile;

   ptr = malloc(sbuf->Length());
   memcpy(ptr, sbuf->Buffer(), sbuf->Length());
   length = sbuf->Length();

   delete sbuf;

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Method to produce image from specified object
///
/// Parameters:
///    kind - image kind TImage::kPng, TImage::kJpeg, TImage::kGif
///    path - path to object
///    options - extra options
///
/// By default, image 300x200 is produced
/// In options string one could provide following parameters:
///    w - image width
///    h - image height
///    opt - draw options
///  For instance:
///     http://localhost:8080/Files/hsimple.root/hpx/get.png?w=500&h=500&opt=lego1
///
///  Return is memory with produced image
///  Memory must be released by user with free(ptr) call

Bool_t TRootSniffer::ProduceImage(Int_t kind, const char *path, const char *options, void *&ptr, Long_t &length)
{
   ptr = nullptr;
   length = 0;

   if ((path == nullptr) || (*path == 0)) return kFALSE;
   if (*path == '/') path++;

   TClass *obj_cl(nullptr);
   void *obj_ptr = FindInHierarchy(path, &obj_cl);
   if ((obj_ptr == nullptr) || (obj_cl == nullptr)) return kFALSE;

   if (obj_cl->GetBaseClassOffset(TObject::Class()) != 0) {
      Error("TRootSniffer", "Only derived from TObject classes can be drawn");
      return kFALSE;
   }

   TObject *obj = (TObject *)obj_ptr;

   TImage *img = TImage::Create();
   if (img == nullptr) return kFALSE;

   if (obj->InheritsFrom(TPad::Class())) {

      if (gDebug > 1) Info("TRootSniffer", "Crate IMAGE directly from pad");
      img->FromPad((TPad *)obj);
   } else if (IsDrawableClass(obj->IsA())) {

      if (gDebug > 1) Info("TRootSniffer", "Crate IMAGE from object %s", obj->GetName());

      Int_t width(300), height(200);
      TString drawopt = "";

      if ((options != nullptr) && (*options != 0)) {
         TUrl url;
         url.SetOptions(options);
         url.ParseOptions();
         Int_t w = url.GetIntValueFromOptions("w");
         if (w > 10) width = w;
         Int_t h = url.GetIntValueFromOptions("h");
         if (h > 10) height = h;
         const char *opt = url.GetValueFromOptions("opt");
         if (opt != nullptr) drawopt = opt;
      }

      Bool_t isbatch = gROOT->IsBatch();
      TVirtualPad *save_gPad = gPad;

      if (!isbatch) gROOT->SetBatch(kTRUE);

      TCanvas *c1 = new TCanvas("__online_draw_canvas__", "title", width, height);
      obj->Draw(drawopt.Data());
      img->FromPad(c1);
      delete c1;

      if (!isbatch) gROOT->SetBatch(kFALSE);
      gPad = save_gPad;

   } else {
      delete img;
      return kFALSE;
   }

   TImage *im = TImage::Create();
   im->Append(img);

   char *png_buffer(nullptr);
   int size(0);

   im->GetImageBuffer(&png_buffer, &size, (TImage::EImageFileTypes)kind);

   if ((png_buffer != nullptr) && (size > 0)) {
      ptr = malloc(size);
      length = size;
      memcpy(ptr, png_buffer, length);
   }

   delete[] png_buffer;
   delete im;

   return ptr != nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Method produce different kind of data out of object
/// Parameter 'path' specifies object or object member
/// Supported 'file' (case sensitive):
///   "root.bin"  - binary data
///   "root.png"  - png image
///   "root.jpeg" - jpeg image
///   "root.gif"  - gif image
///   "root.xml"  - xml representation
///   "root.json" - json representation
///   "exe.json"  - method execution with json reply
///   "exe.bin"   - method execution with binary reply
///   "exe.txt"   - method execution with debug output
///   "cmd.json"  - execution of registered commands
/// Result returned either as string or binary buffer,
/// which should be released with free() call

Bool_t TRootSniffer::Produce(const char *path, const char *file, const char *options, void *&ptr, Long_t &length,
                             TString &str)
{
   if ((file == nullptr) || (*file == 0)) return kFALSE;

   if (strcmp(file, "root.bin") == 0) return ProduceBinary(path, options, ptr, length);

   if (strcmp(file, "root.png") == 0) return ProduceImage(TImage::kPng, path, options, ptr, length);

   if (strcmp(file, "root.jpeg") == 0) return ProduceImage(TImage::kJpeg, path, options, ptr, length);

   if (strcmp(file, "root.gif") == 0) return ProduceImage(TImage::kGif, path, options, ptr, length);

   if (strcmp(file, "exe.bin") == 0) return ProduceExe(path, options, 2, nullptr, &ptr, &length);

   if (strcmp(file, "root.xml") == 0) return ProduceXml(path, options, str);

   if (strcmp(file, "root.json") == 0) return ProduceJson(path, options, str);

   // used for debugging
   if (strcmp(file, "exe.txt") == 0) return ProduceExe(path, options, 0, &str);

   if (strcmp(file, "exe.json") == 0) return ProduceExe(path, options, 1, &str);

   if (strcmp(file, "cmd.json") == 0) return ExecuteCmd(path, options, str);

   if (strcmp(file, "item.json") == 0) return ProduceItem(path, options, str, kTRUE);

   if (strcmp(file, "item.xml") == 0) return ProduceItem(path, options, str, kFALSE);

   if (strcmp(file, "multi.bin") == 0) return ProduceMulti(path, options, ptr, length, str, kFALSE);

   if (strcmp(file, "multi.json") == 0) return ProduceMulti(path, options, ptr, length, str, kTRUE);

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// return item from the subfolders structure

TObject *TRootSniffer::GetItem(const char *fullname, TFolder *&parent, Bool_t force, Bool_t within_objects)
{
   TFolder *topf = gROOT->GetRootFolder();

   if (topf == nullptr) {
      Error("RegisterObject", "Not found top ROOT folder!!!");
      return nullptr;
   }

   TFolder *httpfold = dynamic_cast<TFolder *>(topf->FindObject("http"));
   if (httpfold == nullptr) {
      if (!force) return nullptr;
      httpfold = topf->AddFolder("http", "ROOT http server");
      httpfold->SetBit(kCanDelete);
      // register top folder in list of cleanups
      R__LOCKGUARD(gROOTMutex);
      gROOT->GetListOfCleanups()->Add(httpfold);
   }

   parent = httpfold;
   TObject *obj = httpfold;

   if (fullname == nullptr) return httpfold;

   // when full path started not with slash, "Objects" subfolder is appended
   TString path = fullname;
   if (within_objects && ((path.Length() == 0) || (path[0] != '/'))) path = fObjectsPath + "/" + path;

   TString tok;
   Ssiz_t from(0);

   while (path.Tokenize(tok, from, "/")) {
      if (tok.Length() == 0) continue;

      TFolder *fold = dynamic_cast<TFolder *>(obj);
      if (fold == nullptr) return nullptr;

      TIter iter(fold->GetListOfFolders());
      while ((obj = iter()) != nullptr) {
         if (IsItemField(obj)) continue;
         if (tok.CompareTo(obj->GetName()) == 0) break;
      }

      if (obj == nullptr) {
         if (!force) return nullptr;
         obj = fold->AddFolder(tok, "sub-folder");
         obj->SetBit(kCanDelete);
      }

      parent = fold;
   }

   return obj;
}

////////////////////////////////////////////////////////////////////////////////
/// creates subfolder where objects can be registered

TFolder *TRootSniffer::GetSubFolder(const char *subfolder, Bool_t force)
{
   TFolder *parent = nullptr;

   return dynamic_cast<TFolder *>(GetItem(subfolder, parent, force));
}

////////////////////////////////////////////////////////////////////////////////
/// Register object in subfolder structure
/// subfolder parameter can have many levels like:
///
/// TRootSniffer* sniff = new TRootSniffer("sniff");
/// sniff->RegisterObject("my/sub/subfolder", h1);
///
/// Such objects can be later found in "Objects" folder of sniffer like
///
/// h1 = sniff->FindTObjectInHierarchy("/Objects/my/sub/subfolder/h1");
///
/// If subfolder name starts with '/', object will be registered starting from top folder.
///
/// One could provide additional fields for registered objects
/// For instance, setting "_more" field to true let browser
/// explore objects members. For instance:
///
/// TEvent* ev = new TEvent("ev");
/// sniff->RegisterObject("Events", ev);
/// sniff->SetItemField("Events/ev", "_more", "true");

Bool_t TRootSniffer::RegisterObject(const char *subfolder, TObject *obj)
{
   TFolder *f = GetSubFolder(subfolder, kTRUE);
   if (f == nullptr) return kFALSE;

   // If object will be destroyed, it will be removed from the folders automatically
   obj->SetBit(kMustCleanup);

   f->Add(obj);

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// unregister (remove) object from folders structures
/// folder itself will remain even when it will be empty

Bool_t TRootSniffer::UnregisterObject(TObject *obj)
{
   if (obj == nullptr) return kTRUE;

   TFolder *topf = dynamic_cast<TFolder *>(gROOT->FindObject("//root/http"));

   if (topf == nullptr) {
      Error("UnregisterObject", "Not found //root/http folder!!!");
      return kFALSE;
   }

   // TODO - probably we should remove all set properties as well
   if (topf) topf->RecursiveRemove(obj);

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// create item element

Bool_t TRootSniffer::CreateItem(const char *fullname, const char *title)
{
   TFolder *f = GetSubFolder(fullname, kTRUE);
   if (f == nullptr) return kFALSE;

   if (title) f->SetTitle(title);

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// return true when object is TNamed with kItemField bit set
/// such objects used to keep field values for item

Bool_t TRootSniffer::IsItemField(TObject *obj) const
{
   return (obj != nullptr) && (obj->IsA() == TNamed::Class()) && obj->TestBit(kItemField);
}

////////////////////////////////////////////////////////////////////////////////
/// set or get field for the child
/// each field coded as TNamed object, placed after chld in the parent hierarchy

Bool_t TRootSniffer::AccessField(TFolder *parent, TObject *chld, const char *name, const char *value, TNamed **only_get)
{
   if (parent == nullptr) return kFALSE;

   if (chld == nullptr) {
      Info("SetField", "Should be special case for top folder, support later");
      return kFALSE;
   }

   TIter iter(parent->GetListOfFolders());

   TObject *obj = nullptr;
   Bool_t find(kFALSE), last_find(kFALSE);
   // this is special case of top folder - fields are on very top
   if (parent == chld) {
      last_find = find = kTRUE;
   }
   TNamed *curr = nullptr;
   while ((obj = iter()) != nullptr) {
      if (IsItemField(obj)) {
         if (last_find && (obj->GetName() != nullptr) && !strcmp(name, obj->GetName())) curr = (TNamed *)obj;
      } else {
         last_find = (obj == chld);
         if (last_find) find = kTRUE;
         if (find && !last_find) break; // no need to continue
      }
   }

   // object must be in childs list
   if (!find) return kFALSE;

   if (only_get != nullptr) {
      *only_get = curr;
      return curr != nullptr;
   }

   if (curr != nullptr) {
      if (value != nullptr) {
         curr->SetTitle(value);
      } else {
         parent->Remove(curr);
         delete curr;
      }
      return kTRUE;
   }

   curr = new TNamed(name, value);
   curr->SetBit(kItemField);

   if (last_find) {
      // object is on last place, therefore just add property
      parent->Add(curr);
      return kTRUE;
   }

   // only here we do dynamic cast to the TList to use AddAfter
   TList *lst = dynamic_cast<TList *>(parent->GetListOfFolders());
   if (lst == nullptr) {
      Error("SetField", "Fail cast to TList");
      return kFALSE;
   }

   if (parent == chld)
      lst->AddFirst(curr);
   else
      lst->AddAfter(chld, curr);

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// set field for specified item

Bool_t TRootSniffer::SetItemField(const char *fullname, const char *name, const char *value)
{
   if ((fullname == nullptr) || (name == nullptr)) return kFALSE;

   TFolder *parent(nullptr);
   TObject *obj = GetItem(fullname, parent);

   if ((parent == nullptr) || (obj == nullptr)) return kFALSE;

   if (strcmp(name, item_prop_title) == 0) {
      TNamed *n = dynamic_cast<TNamed *>(obj);
      if (n != nullptr) {
         n->SetTitle(value);
         return kTRUE;
      }
   }

   return AccessField(parent, obj, name, value);
}

////////////////////////////////////////////////////////////////////////////////
/// return field for specified item

const char *TRootSniffer::GetItemField(TFolder *parent, TObject *obj, const char *name)
{
   if ((parent == nullptr) || (obj == nullptr) || (name == nullptr)) return nullptr;

   TNamed *field(nullptr);

   if (!AccessField(parent, obj, name, nullptr, &field)) return nullptr;

   return field ? field->GetTitle() : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// return field for specified item

const char *TRootSniffer::GetItemField(const char *fullname, const char *name)
{
   if (fullname == nullptr) return nullptr;

   TFolder *parent(nullptr);
   TObject *obj = GetItem(fullname, parent);

   return GetItemField(parent, obj, name);
}

////////////////////////////////////////////////////////////////////////////////
/// Register command which can be executed from web interface
///
/// As method one typically specifies string, which is executed with
/// gROOT->ProcessLine() method. For instance
///    serv->RegisterCommand("Invoke","InvokeFunction()");
///
/// Or one could specify any method of the object which is already registered
/// to the server. For instance:
///     serv->Register("/", hpx);
///     serv->RegisterCommand("/ResetHPX", "/hpx/->Reset()");
/// Here symbols '/->' separates item name from method to be executed
///
/// One could specify additional arguments in the command with
/// syntax like %arg1%, %arg2% and so on. For example:
///     serv->RegisterCommand("/ResetHPX", "/hpx/->SetTitle(\"%arg1%\")");
///     serv->RegisterCommand("/RebinHPXPY", "/hpxpy/->Rebin2D(%arg1%,%arg2%)");
/// Such parameter(s) will be requested when command clicked in the browser.
///
/// Once command is registered, one could specify icon which will appear in the browser:
///     serv->SetIcon("/ResetHPX", "rootsys/icons/ed_execute.png");
///
/// One also can set extra property '_fastcmd', that command appear as
/// tool button on the top of the browser tree:
///     serv->SetItemField("/ResetHPX", "_fastcmd", "true");
/// Or it is equivalent to specifying extra argument when register command:
///     serv->RegisterCommand("/ResetHPX", "/hpx/->Reset()", "button;rootsys/icons/ed_delete.png");

Bool_t TRootSniffer::RegisterCommand(const char *cmdname, const char *method, const char *icon)
{
   CreateItem(cmdname, Form("command %s", method));
   SetItemField(cmdname, "_kind", "Command");
   if (icon != nullptr) {
      if (strncmp(icon, "button;", 7) == 0) {
         SetItemField(cmdname, "_fastcmd", "true");
         icon += 7;
      }
      if (*icon != 0) SetItemField(cmdname, "_icon", icon);
   }
   SetItemField(cmdname, "method", method);
   Int_t numargs = 0;
   do {
      TString nextname = TString::Format("%sarg%d%s", "%", numargs + 1, "%");
      if (strstr(method, nextname.Data()) == nullptr) break;
      numargs++;
   } while (numargs < 100);
   if (numargs > 0) SetItemField(cmdname, "_numargs", TString::Format("%d", numargs));

   return kTRUE;
}
