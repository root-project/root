// $Id$
// Author: Sergey Linev   22/12/2013

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
#include "TMethodArg.h"
#include "TMethodCall.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TBaseClass.h"
#include "TObjString.h"
#include "TUrl.h"
#include "TImage.h"
#ifdef COMPILED_WITH_DABC
extern "C" void R__zip(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep);
#else
#include "RZip.h"
#endif
#include "TRootSnifferStore.h"

#include <stdlib.h>

const char *item_prop_kind = "_kind";
const char *item_prop_more = "_more";
const char *item_prop_title = "_title";
const char *item_prop_hidden = "_hidden";
const char *item_prop_typename = "_typename";
const char *item_prop_arraydim = "_arraydim";
const char *item_prop_realname = "_realname"; // real object name

// ============================================================================

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootSnifferScanRec                                                  //
//                                                                      //
// Structure used to scan hierarchies of ROOT objects                   //
// Represents single level of hierarchy                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TRootSnifferScanRec::TRootSnifferScanRec() :
   fParent(0),
   fMask(0),
   fSearchPath(0),
   fLevel(0),
   fItemsNames(),
   fStore(0),
   fHasMore(kFALSE),
   fNodeStarted(kFALSE),
   fNumFields(0),
   fNumChilds(0)
{
   // constructor

   fItemsNames.SetOwner(kTRUE);
}

//______________________________________________________________________________
TRootSnifferScanRec::~TRootSnifferScanRec()
{
   // destructor

   CloseNode();
}

//______________________________________________________________________________
void TRootSnifferScanRec::SetField(const char *name, const char *value, Bool_t with_quotes)
{
   // record field for current element

   if (CanSetFields()) fStore->SetField(fLevel, name, value, with_quotes);
   fNumFields++;
}

//______________________________________________________________________________
void TRootSnifferScanRec::BeforeNextChild()
{
   // indicates that new child for current element will be started

   if (CanSetFields()) fStore->BeforeNextChild(fLevel, fNumChilds, fNumFields);
   fNumChilds++;
}

//______________________________________________________________________________
void TRootSnifferScanRec::MakeItemName(const char *objname, TString &itemname)
{
   // constructs item name from object name
   // if special symbols like '/', '#', ':', '&', '?'  are used in object name
   // they will be replaced with '_'.
   // To avoid item name duplication, additional id number can be appended

   std::string nnn = objname;

   size_t pos;

   // replace all special symbols which can make problem to navigate in hierarchy
   while ((pos = nnn.find_first_of("- []<>#:&?/\'\"\\")) != std::string::npos)
      nnn.replace(pos, 1, "_");

   itemname = nnn.c_str();
   Int_t cnt = 0;

   while (fItemsNames.FindObject(itemname.Data())) {
      itemname.Form("%s_%d", nnn.c_str(), cnt++);
   }

   fItemsNames.Add(new TObjString(itemname.Data()));
}

//______________________________________________________________________________
void TRootSnifferScanRec::CreateNode(const char *_node_name)
{
   // creates new node with specified name
   // if special symbols like "[]&<>" are used, node name
   // will be replaced by default name like "extra_item_N" and
   // original node name will be recorded as "_original_name" field
   // Optionally, object name can be recorded as "_realname" field

   if (!CanSetFields()) return;

   fNodeStarted = kTRUE;

   if (fParent) fParent->BeforeNextChild();

   if (fStore) fStore->CreateNode(fLevel, _node_name);
}

//______________________________________________________________________________
void TRootSnifferScanRec::CloseNode()
{
   // close started node

   if (fStore && fNodeStarted) {
      fStore->CloseNode(fLevel, fNumChilds);
      fNodeStarted = kFALSE;
   }
}

//______________________________________________________________________________
void TRootSnifferScanRec::SetRootClass(TClass *cl)
{
   // set root class name as node kind
   // in addition, path to master item (streamer info) specified
   // Such master item required to correctly unstream data on JavaScript

   if ((cl != 0) && CanSetFields())
      SetField(item_prop_kind, TString::Format("ROOT.%s", cl->GetName()));
}

//______________________________________________________________________________
Bool_t TRootSnifferScanRec::Done() const
{
   // returns true if scanning is done
   // Can happen when searched element is found

   if (fStore == 0)
      return kFALSE;

   if ((fMask & kSearch) && fStore->GetResPtr())
      return kTRUE;

   if ((fMask & kCheckChilds) && fStore->GetResPtr() &&
         (fStore->GetResNumChilds() >= 0))
      return kTRUE;

   return kFALSE;
}


//______________________________________________________________________________
Bool_t TRootSnifferScanRec::IsReadyForResult() const
{
   // Checks if result will be accepted.
   // Used to verify if sniffer should read object from the file

   if (Done()) return kFALSE;

   // only when doing search, result will be propagated
   if ((fMask & (kSearch | kCheckChilds)) == 0) return kFALSE;

   // only when full search path is scanned
   if (fSearchPath != 0) return kFALSE;

   if (fStore == 0) return kFALSE;

   return kTRUE;
}


//______________________________________________________________________________
Bool_t TRootSnifferScanRec::SetResult(void *obj, TClass *cl, TDataMember *member)
{
   // set results of scanning

   if (Done()) return kTRUE;

   if (!IsReadyForResult()) return kFALSE;

   fStore->SetResult(obj, cl, member, fNumChilds);

   return Done();
}

//______________________________________________________________________________
Int_t TRootSnifferScanRec::Depth() const
{
   // returns current depth of scanned hierarchy

   Int_t cnt = 0;
   const TRootSnifferScanRec *rec = this;
   while (rec->fParent) {
      rec = rec->fParent;
      cnt++;
   }

   return cnt;
}

//______________________________________________________________________________
Bool_t TRootSnifferScanRec::CanExpandItem()
{
   // returns true if current item can be expanded - means one could explore
   // objects members

   if (fMask & (kExpand | kSearch | kCheckChilds)) return kTRUE;

   if (!fHasMore) return kFALSE;

   // if parent has expand mask, allow to expand item
   if (fParent && (fParent->fMask & kExpand)) return kTRUE;

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TRootSnifferScanRec::GoInside(TRootSnifferScanRec &super, TObject *obj,
                                     const char *obj_name)
{
   // Method verifies if new level of hierarchy
   // should be started with provided object.
   // If required, all necessary nodes and fields will be created
   // Used when different collection kinds should be scanned

   if (super.Done()) return kFALSE;

   if ((obj != 0) && (obj_name == 0)) obj_name = obj->GetName();

   // exclude zero names
   if ((obj_name == 0) || (*obj_name == 0)) return kFALSE;

   TString obj_item_name;

   const char *full_name = 0;

   // remove slashes from file names
   if (obj && obj->InheritsFrom(TDirectoryFile::Class())) {
      const char *slash = strrchr(obj_name, '/');
      if (slash != 0) {
         full_name = obj_name;
         obj_name = slash + 1;
         if (*obj_name == 0) obj_name = "file";
      }
   }

   super.MakeItemName(obj_name, obj_item_name);

   fLevel = super.fLevel;
   fStore = super.fStore;
   fSearchPath = super.fSearchPath;
   fMask = super.fMask & kActions;
   fParent = &super;

   if (fMask & kScan) {
      // if scanning only fields, ignore all childs
      if (super.ScanOnlyFields()) return kFALSE;
      // only when doing scan, increment level, used for text formatting
      fLevel++;
   } else {
      if (fSearchPath == 0) return kFALSE;

      if (strncmp(fSearchPath, obj_item_name.Data(), obj_item_name.Length()) != 0)
         return kFALSE;

      const char *separ = fSearchPath + obj_item_name.Length();

      Bool_t isslash = kFALSE;
      while (*separ == '/') {
         separ++;
         isslash = kTRUE;
      }

      if (*separ == 0) {
         fSearchPath = 0;
         if (fMask & kExpand) {
            fMask = (fMask & kOnlyFields) | kScan;
            fHasMore = (fMask & kOnlyFields) == 0;
         }
      } else {
         if (!isslash) return kFALSE;
         fSearchPath = separ;
      }
   }

   CreateNode(obj_item_name.Data());

   if ((obj_name != 0) && (obj_item_name != obj_name))
      SetField(item_prop_realname, obj_name);

   if (full_name != 0)
      SetField("_fullname", full_name);

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

//______________________________________________________________________________
TRootSniffer::TRootSniffer(const char *name, const char *objpath) :
   TNamed(name, "sniffer of root objects"),
   fObjectsPath(objpath),
   fMemFile(0),
   fSinfoSize(0),
   fReadOnly(kTRUE)
{
   // constructor
}

//______________________________________________________________________________
TRootSniffer::~TRootSniffer()
{
   // destructor

   if (fMemFile) {
      delete fMemFile;
      fMemFile = 0;
   }
}

//______________________________________________________________________________
void TRootSniffer::ScanObjectMemebers(TRootSnifferScanRec &rec, TClass *cl,
                                      char *ptr, unsigned long int cloffset)
{
   // scan object data members
   // some members like enum or static members will be excluded

   if ((cl == 0) || (ptr == 0) || rec.Done()) return;

   // first of all expand base classes
   TIter cliter(cl->GetListOfBases());
   TObject *obj = 0;
   while ((obj = cliter()) != 0) {
      TBaseClass *baseclass = dynamic_cast<TBaseClass *>(obj);
      if (baseclass == 0) continue;
      TClass *bclass = baseclass->GetClassPointer();
      if (bclass == 0) continue;

      // all parent classes scanned within same hierarchy level
      // this is how normal object streaming works
      ScanObjectMemebers(rec, bclass, ptr, cloffset + baseclass->GetDelta());
      if (rec.Done()) break;

//    this code was used when each base class creates its own sub level

//      TRootSnifferScanRec chld;
//      if (chld.GoInside(rec, baseclass)) {
//         ScanObjectMemebers(chld, bclass, ptr, cloffset + baseclass->GetDelta());
//         if (chld.Done()) break;
//      }
   }

   // than expand data members
   TIter iter(cl->GetListOfDataMembers());
   while ((obj = iter()) != 0) {
      TDataMember *member = dynamic_cast<TDataMember *>(obj);
      // exclude enum or static variables
      if ((member == 0) || (member->Property() & (kIsStatic | kIsEnum | kIsUnion))) continue;

      char *member_ptr = ptr + cloffset + member->GetOffset();
      if (member->IsaPointer()) member_ptr = *((char **) member_ptr);

      TRootSnifferScanRec chld;

      if (chld.GoInside(rec, member)) {
         TClass *mcl = (member->IsBasic() || member->IsSTLContainer()) ? 0 :
                       gROOT->GetClass(member->GetTypeName());

         Int_t coll_offset = mcl ? mcl->GetBaseClassOffset(TCollection::Class()) : -1;

         Bool_t iscollection = (coll_offset >= 0);
         if (iscollection) {
            chld.SetField(item_prop_more, "true", kFALSE);
            chld.fHasMore = kTRUE;
         }

         if (chld.SetResult(member_ptr, mcl, member)) break;

         const char *title = member->GetTitle();
         if ((title != 0) && (strlen(title) != 0))
            chld.SetField(item_prop_title, title);

         if (member->GetTypeName())
            chld.SetField(item_prop_typename, member->GetTypeName());

         if (member->GetArrayDim() > 0) {
            // store array dimensions in form [N1,N2,N3,...]
            TString dim("[");
            for (Int_t n = 0; n < member->GetArrayDim(); n++) {
               if (n > 0) dim.Append(",");
               dim.Append(TString::Format("%d", member->GetMaxIndex(n)));
            }
            dim.Append("]");
            chld.SetField(item_prop_arraydim, dim, kFALSE);
         }

         chld.SetRootClass(mcl);

         if (chld.CanExpandItem()) {
            if (iscollection) {
               // chld.SetField("#members", "true", kFALSE);
               ScanCollection(chld, (TCollection *)(member_ptr + coll_offset));
            }
         }

         if (chld.SetResult(member_ptr, mcl, member)) break;
      }
   }
}

//_____________________________________________________________________
void TRootSniffer::ScanObjectProperties(TRootSnifferScanRec &rec, TObject *obj)
{
   // scans basic object properties
   // here such fields as _title properties can be specified

   const char *title = obj->GetTitle();
   if ((title != 0) && (*title != 0))
      rec.SetField(item_prop_title, title);
}

//_____________________________________________________________________
void TRootSniffer::ScanObjectChilds(TRootSnifferScanRec &rec, TObject *obj)
{
   // scans object childs (if any)
   // here one scans collection, branches, trees and so on

   if (obj->InheritsFrom(TFolder::Class())) {
      ScanCollection(rec, ((TFolder *) obj)->GetListOfFolders());
   } else if (obj->InheritsFrom(TDirectory::Class())) {
      TDirectory *dir = (TDirectory *) obj;
      ScanCollection(rec, dir->GetList(), 0, dir->GetListOfKeys());
   } else if (obj->InheritsFrom(TTree::Class())) {
      if (!fReadOnly) rec.SetField("_player", "JSROOT.drawTreePlayer");
      ScanCollection(rec, ((TTree *) obj)->GetListOfLeaves());
   } else if (obj->InheritsFrom(TBranch::Class())) {
      ScanCollection(rec, ((TBranch *) obj)->GetListOfLeaves());
   } else if (rec.CanExpandItem()) {
      ScanObjectMemebers(rec, obj->IsA(), (char *) obj, 0);
   }
}

//______________________________________________________________________________
void TRootSniffer::ScanCollection(TRootSnifferScanRec &rec, TCollection *lst,
                                  const char *foldername, TCollection *keys_lst)
{
   // scan collection content

   if (((lst == 0) || (lst->GetSize() == 0)) && ((keys_lst == 0) || (keys_lst->GetSize() == 0))) return;

   TRootSnifferScanRec folderrec;
   if (foldername) {
      if (!folderrec.GoInside(rec, 0, foldername)) return;
   }

   {
      TRootSnifferScanRec &master = foldername ? folderrec : rec;

      if (lst != 0) {
         TIter iter(lst);
         TObject *next = iter();

         // special case - in the beginning one could have items for parent folder
         while (IsItemField(next)) {
            if ((next->GetName() != 0) && ((*(next->GetName()) == '_') || master.ScanOnlyFields()))
               master.SetField(next->GetName(), next->GetTitle());

            next = iter();
         }

         while (next!=0) {
            TObject* obj = next;

            TRootSnifferScanRec chld;
            if (!chld.GoInside(master, obj)) { next = iter(); continue; }

            if (chld.SetResult(obj, obj->IsA())) return;

            Bool_t has_kind = kFALSE;

            ScanObjectProperties(chld, obj);
            // now properties, coded as TNamed objects, placed after object in the hierarchy
            while ((next = iter()) != 0) {
               if (!IsItemField(next)) break;
               if ((next->GetName() != 0) && ((*(next->GetName()) == '_') || chld.ScanOnlyFields())) {
                  // only fields starting with _ are stored
                  chld.SetField(next->GetName(), next->GetTitle());
                  if (strcmp(next->GetName(), item_prop_kind)==0) has_kind = kTRUE;
               }
            }

            if (!has_kind) chld.SetRootClass(obj->IsA());

            ScanObjectChilds(chld, obj);

            if (chld.SetResult(obj, obj->IsA())) return;
         }
      }

      if (keys_lst != 0) {
         TIter iter(keys_lst);
         TObject *kobj(0);

         while ((kobj = iter()) != 0) {
            TKey *key = dynamic_cast<TKey *>(kobj);
            if (key == 0) continue;
            TObject *obj = (lst == 0) ? 0 : lst->FindObject(key->GetName());

            // even object with the name exists, it should also match with class name
            if ((obj!=0) && (strcmp(obj->ClassName(),key->GetClassName())!=0)) obj = 0;

            // if object of that name and of that class already in the list, ignore appropriate key
            if ((obj != 0) && (master.fMask & TRootSnifferScanRec::kScan)) continue;

            Bool_t iskey = kFALSE;
            // if object not exists, provide key itself for the scan
            if (obj == 0) { obj = key; iskey = kTRUE; }

            TRootSnifferScanRec chld;
            TString fullname = TString::Format("%s;%d", key->GetName(), key->GetCycle());

            if (chld.GoInside(master, obj, fullname.Data())) {

               if (!fReadOnly && iskey && chld.IsReadyForResult()) {
                  TObject *keyobj = key->ReadObj();
                  if (keyobj != 0)
                     if (chld.SetResult(keyobj, keyobj->IsA())) return;
               }

               if (chld.SetResult(obj, obj->IsA())) return;

               TClass *obj_class = obj->IsA();

               ScanObjectProperties(chld, obj);

               // special handling of TKey class - in non-readonly mode
               // sniffer allowed to fetch objects
               if (!fReadOnly && iskey) {
                  if (strcmp(key->GetClassName(), "TDirectoryFile") == 0) {
                     if (chld.fLevel == 0) {
                        TDirectory *dir = dynamic_cast<TDirectory *>(key->ReadObj());
                        if (dir != 0) {
                           obj = dir;
                           obj_class = dir->IsA();
                        }
                     } else {
                        chld.SetField(item_prop_more, "true", kFALSE);
                        chld.fHasMore = kTRUE;
                     }
                  } else {
                     obj_class = TClass::GetClass(key->GetClassName());
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
}

//______________________________________________________________________________
void TRootSniffer::ScanRoot(TRootSnifferScanRec &rec)
{
   // scan complete ROOT objects hierarchy
   // For the moment it includes objects in gROOT directory
   // and list of canvases and files
   // Also all registered objects are included.
   // One could reimplement this method to provide alternative
   // scan methods or to extend some collection kinds

   rec.SetField(item_prop_kind, "ROOT.Session");

   // should be on the top while //root/http folder could have properties for itself
   TFolder *topf = dynamic_cast<TFolder *>(gROOT->FindObject("//root/http"));
   if (topf) ScanCollection(rec, topf->GetListOfFolders());

   {
      TRootSnifferScanRec chld;
      if (chld.GoInside(rec, 0, "StreamerInfo")) {
         chld.SetField(item_prop_kind, "ROOT.TStreamerInfoList");
         chld.SetField(item_prop_title, "List of streamer infos for binary I/O");
         chld.SetField(item_prop_hidden, "true");
      }
   }

   ScanCollection(rec, gROOT->GetList());

   ScanCollection(rec, gROOT->GetListOfCanvases(), "Canvases");

   ScanCollection(rec, gROOT->GetListOfFiles(), "Files");
}

//______________________________________________________________________________
Bool_t TRootSniffer::IsDrawableClass(TClass *cl)
{
   // return true if object can be drawn

   if (cl == 0) return kFALSE;
   if (cl->InheritsFrom(TH1::Class())) return kTRUE;
   if (cl->InheritsFrom(TGraph::Class())) return kTRUE;
   if (cl->InheritsFrom(TCanvas::Class())) return kTRUE;
   if (cl->InheritsFrom(TProfile::Class())) return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
void TRootSniffer::ScanHierarchy(const char *topname, const char *path,
                                 TRootSnifferStore *store, Bool_t only_fields)
{
   // scan ROOT hierarchy with provided store object

   TRootSnifferScanRec rec;
   rec.fSearchPath = path;
   if (rec.fSearchPath) {
      while(*rec.fSearchPath == '/') rec.fSearchPath++;
      if (*rec.fSearchPath == 0) rec.fSearchPath = 0;
   }

   // if path non-empty, we should find item first and than start scanning
   rec.fMask = rec.fSearchPath == 0 ? TRootSnifferScanRec::kScan : TRootSnifferScanRec::kExpand;
   if (only_fields) rec.fMask |= TRootSnifferScanRec::kOnlyFields;

   rec.fStore = store;

   rec.CreateNode(topname);

   ScanRoot(rec);

   rec.CloseNode();
}

//______________________________________________________________________________
void *TRootSniffer::FindInHierarchy(const char *path, TClass **cl,
                                    TDataMember **member, Int_t *chld)
{
   // Search element with specified path
   // Returns pointer on element
   // Optionally one could obtain element class, member description
   // and number of childs. When chld!=0, not only element is searched,
   // but also number of childs are counted. When member!=0, any object
   // will be scanned for its data members (disregard of extra options)

   TRootSnifferStore store;

   TRootSnifferScanRec rec;
   rec.fSearchPath = path;
   rec.fMask = (chld != 0) ? TRootSnifferScanRec::kCheckChilds : TRootSnifferScanRec::kSearch;
   if (*rec.fSearchPath == '/') rec.fSearchPath++;
   rec.fStore = &store;

   ScanRoot(rec);

   if (cl) *cl = store.GetResClass();
   if (member) *member = store.GetResMember();
   if (chld) *chld = store.GetResNumChilds();

   return store.GetResPtr();
}

//______________________________________________________________________________
TObject *TRootSniffer::FindTObjectInHierarchy(const char *path)
{
   // Search element in hierarchy, derived from TObject

   TClass *cl(0);

   void *obj = FindInHierarchy(path, &cl);

   return (cl != 0) && (cl->GetBaseClassOffset(TObject::Class()) == 0) ? (TObject *) obj : 0;
}

//______________________________________________________________________________
ULong_t TRootSniffer::GetStreamerInfoHash()
{
   // Returns hash value for streamer infos
   // At the moment - just number of items in streamer infos list.

   return fSinfoSize;
}

//______________________________________________________________________________
ULong_t TRootSniffer::GetItemHash(const char *itemname)
{
   // Get hash function for specified item
   // used to detect any changes in the specified object

   if (IsStreamerInfoItem(itemname)) return GetStreamerInfoHash();

   TObject *obj = FindTObjectInHierarchy(itemname);

   return obj == 0 ? 0 : TString::Hash(obj, obj->IsA()->Size());
}

//______________________________________________________________________________
Bool_t TRootSniffer::CanDrawItem(const char *path)
{
   // Method verifies if object can be drawn

   TClass *obj_cl(0);
   void *res = FindInHierarchy(path, &obj_cl);
   return (res != 0) && IsDrawableClass(obj_cl);
}

//______________________________________________________________________________
Bool_t TRootSniffer::CanExploreItem(const char *path)
{
   // Method returns true when object has childs or
   // one could try to expand item

   TClass *obj_cl(0);
   Int_t obj_chld(-1);
   void *res = FindInHierarchy(path, &obj_cl, 0, &obj_chld);
   return (res != 0) && (obj_chld > 0);
}

//______________________________________________________________________________
void TRootSniffer::CreateMemFile()
{
   // Creates TMemFile instance, which used for objects streaming
   // One could not use TBufferFile directly,
   // while one also require streamer infos list

   if (fMemFile != 0) return;

   TDirectory *olddir = gDirectory;
   gDirectory = 0;
   TFile *oldfile = gFile;
   gFile = 0;

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

   l = fMemFile->GetStreamerInfoList();
   // l->Print("*");
   fSinfoSize = l->GetSize();
   delete l;

   gDirectory = olddir;
   gFile = oldfile;
}

//______________________________________________________________________________
Bool_t TRootSniffer::ProduceJson(const char *path, const char *options,
                                 TString &res)
{
   // produce JSON data for specified item
   // For object conversion TBufferJSON is used

   if ((path == 0) || (*path == 0)) return kFALSE;

   if (*path == '/') path++;

   TUrl url;
   url.SetOptions(options);
   url.ParseOptions();
   Int_t compact = -1;
   if (url.GetValueFromOptions("compact"))
      compact = url.GetIntValueFromOptions("compact");

   if (IsStreamerInfoItem(path)) {

      CreateMemFile();

      TDirectory *olddir = gDirectory;
      gDirectory = 0;
      TFile *oldfile = gFile;
      gFile = 0;

      fMemFile->WriteStreamerInfo();
      TList *l = fMemFile->GetStreamerInfoList();
      fSinfoSize = l->GetSize();

      res = TBufferJSON::ConvertToJSON(l, compact);

      delete l;
      gDirectory = olddir;
      gFile = oldfile;
   } else {

      TClass *obj_cl(0);
      TDataMember *member(0);
      void *obj_ptr = FindInHierarchy(path, &obj_cl, &member);
      if ((obj_ptr == 0) || ((obj_cl == 0) && (member == 0))) return kFALSE;

      if (member == 0)
         res = TBufferJSON::ConvertToJSON(obj_ptr, obj_cl, compact >= 0 ? compact : 0);
      else
         res = TBufferJSON::ConvertToJSON(obj_ptr, member, compact >= 0 ? compact : 1);
   }

   return res.Length() > 0;
}

//______________________________________________________________________________
Bool_t TRootSniffer::ExecuteCmd(const char *path, const char * /*options*/,
                                TString &res)
{
   // execute command marked as _kind=='Command'

   TFolder *parent(0);
   TObject *obj = GetItem(path, parent, kFALSE, kFALSE);

   const char *kind = GetItemField(parent, obj, item_prop_kind);
   if ((kind == 0) || (strcmp(kind, "Command") != 0)) {
      res = "false";
      return kTRUE;
   }

   const char *method = GetItemField(parent, obj, "method");
   if ((method==0) || (strlen(method)==0)) {
      res = "false";
      return kTRUE;
   }

   if (gDebug > 0) Info("ExecuteCmd", "Executing command %s method:%s", path, method);

   TString item_method;

   const char *separ = strstr(method, "/->");
   if (separ != 0) {
      TString itemname(method, separ - method);
      TObject *item_obj = FindTObjectInHierarchy(itemname.Data());
      if (item_obj != 0) {
         item_method.Form("((%s*)%lu)->%s", item_obj->ClassName(), (long unsigned) item_obj, separ + 3);
         method = item_method.Data();
         if (gDebug > 2) Info("ExecuteCmd", "Executing %s", method);
      }
   }

   Long_t v = gROOT->ProcessLineSync(method);

   res.Form("%ld", v);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootSniffer::ProduceXml(const char *path, const char * /*options*/,
                                TString &res)
{
   // produce XML data for specified item
   // For object conversion TBufferXML is used

   if ((path == 0) || (*path == 0)) return kFALSE;

   if (*path == '/') path++;

   if (IsStreamerInfoItem(path)) {

      CreateMemFile();

      TDirectory *olddir = gDirectory;
      gDirectory = 0;
      TFile *oldfile = gFile;
      gFile = 0;

      fMemFile->WriteStreamerInfo();
      TList *l = fMemFile->GetStreamerInfoList();
      fSinfoSize = l->GetSize();

      res = TBufferXML::ConvertToXML(l);

      delete l;
      gDirectory = olddir;
      gFile = oldfile;
   } else {

      TClass *obj_cl(0);
      void *obj_ptr = FindInHierarchy(path, &obj_cl);
      if ((obj_ptr == 0) || (obj_cl == 0)) return kFALSE;

      res = TBufferXML::ConvertToXML(obj_ptr, obj_cl);
   }

   return res.Length() > 0;
}

//______________________________________________________________________________
TString TRootSniffer::DecodeUrlOptionValue(const char *value, Bool_t remove_quotes)
{
   // method replaces all kind of special symbols, which could appear in URL options

   if ((value == 0) || (strlen(value) == 0)) return TString();

   TString res = value;

   res.ReplaceAll("%27", "\'");
   res.ReplaceAll("%22", "\"");
   res.ReplaceAll("%3E", ">");
   res.ReplaceAll("%3C", "<");
   res.ReplaceAll("%20", " ");
   res.ReplaceAll("%5B", "[");
   res.ReplaceAll("%5D", "]");

   if (remove_quotes && (res.Length() > 1) &&
         ((res[0] == '\'') || (res[0] == '\"')) && (res[0] == res[res.Length() - 1])) {
      res.Remove(res.Length() - 1);
      res.Remove(0, 1);
   }

   return res;
}

//______________________________________________________________________________
Bool_t TRootSniffer::ProduceExe(const char *path, const char *options, Int_t reskind, TString *res_str, void **res_ptr, Long_t *res_length)
{
   // execute command for specified object
   // options include method and extra list of parameters
   // sniffer should be not-readonly to allow execution of the commands
   // reskind defines kind of result 0 - debug, 1 - json, 2 - binary

   TString *debug = (reskind == 0) ? res_str : 0;

   if ((path == 0) || (*path == 0)) {
      if (debug) debug->Append("Item name not specified\n");
      return debug != 0;
   }

   if (fReadOnly) {
      if (debug) debug->Append("Server runs in read-only mode, methods cannot be executed\n");
      return debug != 0;
   }

   if (*path == '/') path++;

   TClass *obj_cl(0);
   void *obj_ptr = FindInHierarchy(path, &obj_cl);
   if (debug) debug->Append(TString::Format("Item:%s found:%s\n", path, obj_ptr ? "true" : "false"));
   if ((obj_ptr == 0) || (obj_cl == 0)) return debug != 0;

   TUrl url;
   url.SetOptions(options);

   const char *method_name = url.GetValueFromOptions("method");
   TString prototype = DecodeUrlOptionValue(url.GetValueFromOptions("prototype"), kTRUE);
   TMethod *method = 0;
   if (method_name != 0) {
      if (prototype.Length() == 0) {
         if (debug) debug->Append(TString::Format("Search for any method with name \'%s\'\n", method_name));
         method = obj_cl->GetMethodAllAny(method_name);
      } else {
         if (debug) debug->Append(TString::Format("Search for method \'%s\' with prototype \'%s\'\n", method_name, prototype.Data()));
         method = obj_cl->GetMethodWithPrototype(method_name, prototype);
      }
   }

   if (method == 0) {
      if (debug) debug->Append("Method not found\n");
      return debug != 0;
   }

   if (debug) debug->Append(TString::Format("Method: %s\n", method->GetPrototype()));

   TList *args = method->GetListOfMethodArgs();

   TIter next(args);
   TMethodArg *arg = 0;
   TString call_args;
   while ((arg = (TMethodArg *) next()) != 0) {

      if ((strcmp(arg->GetName(), "rest_url_opt") == 0) &&
            (strcmp(arg->GetFullTypeName(), "const char*") == 0) && (args->GetSize() == 1)) {
         // very special case - function requires list of options after method=argument

         const char *pos = strstr(options, "method=");
         if ((pos == 0) || (strlen(pos) < strlen(method_name) + 8)) return debug != 0;
         call_args.Form("\"%s\"", pos + strlen(method_name) + 8);
         break;
      }

      TString sval;
      const char *val = url.GetValueFromOptions(arg->GetName());
      if (val) {
         sval = DecodeUrlOptionValue(val, kFALSE);
         val = sval.Data();
      }
      if (val == 0) val = arg->GetDefault();

      if (debug) debug->Append(TString::Format("  Argument:%s Type:%s Value:%s \n", arg->GetName(), arg->GetFullTypeName(), val ? val : "<missed>"));
      if (val == 0) return debug != 0;

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

   if (debug) debug->Append(TString::Format("Calling obj->%s(%s);\n", method_name, call_args.Data()));

   TMethodCall call(obj_cl, method_name, call_args.Data());

   if (!call.IsValid()) {
      if (debug) debug->Append("Fail: invalid TMethodCall\n");
      return debug != 0;
   }

   Int_t compact = 0;
   if (url.GetValueFromOptions("compact"))
      compact = url.GetIntValueFromOptions("compact");

   TString res = "null";
   void *ret_obj = 0;
   TClass *ret_cl = 0;

   switch (call.ReturnType()) {
      case TMethodCall::kLong: {
            Long_t l(0);
            call.Execute(obj_ptr, l);
            res.Form("%ld", l);
            break;
         }
      case TMethodCall::kDouble : {
            Double_t d(0.);
            call.Execute(obj_ptr, d);
            res.Form(TBufferJSON::GetFloatFormat(), d);
            break;
         }
      case TMethodCall::kString : {
            char *txt(0);
            call.Execute(obj_ptr, &txt);
            if (txt != 0)
               res.Form("\"%s\"", txt);
            break;
         }
      case TMethodCall::kOther : {
            std::string ret_kind = method->GetReturnTypeNormalizedName();
            if ((ret_kind.length() > 0) && (ret_kind[ret_kind.length() - 1] == '*')) {
               ret_kind.resize(ret_kind.length() - 1);
               ret_cl = gROOT->GetClass(ret_kind.c_str(), kFALSE, kTRUE);
            }

            if (ret_cl != 0) {
               Long_t l(0);
               call.Execute(obj_ptr, l);
               if (l != 0) ret_obj = (void *) l;
            } else {
               call.Execute(obj_ptr);
            }
            break;
         }
      case TMethodCall::kNone : {
            call.Execute(obj_ptr);
            break;
         }
   }

   const char *_ret_object_ = url.GetValueFromOptions("_ret_object_");
   if (_ret_object_ != 0) {
      TObject *obj = 0;
      if (gDirectory != 0) obj = gDirectory->Get(_ret_object_);
      if (debug) debug->Append(TString::Format("Return object %s found %s\n", _ret_object_, obj ? "true" : "false"));

      if (obj == 0) {
         res = "null";
      } else {
         ret_obj = obj;
         ret_cl = obj->IsA();
      }
   }

   if ((ret_obj != 0) && (ret_cl != 0)) {
      if ((reskind == 2) && (res_ptr != 0) && (res_length != 0) && (ret_cl->GetBaseClassOffset(TObject::Class()) == 0)) {
         TObject *obj = (TObject *) ret_obj;
         TBufferFile *sbuf = new TBufferFile(TBuffer::kWrite, 100000);
         sbuf->MapObject(obj);
         obj->Streamer(*sbuf);

         *res_ptr = malloc(sbuf->Length());
         memcpy(*res_ptr, sbuf->Buffer(), sbuf->Length());
         *res_length = sbuf->Length();
         delete sbuf;
      } else {
         res = TBufferJSON::ConvertToJSON(ret_obj, ret_cl, compact);
      }
   }

   if (debug) debug->Append(TString::Format("Result = %s\n", res.Data()));

   if ((reskind == 1) && res_str) *res_str = res;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootSniffer::IsStreamerInfoItem(const char *itemname)
{
   // Return true if it is streamer info item name

   if ((itemname == 0) || (*itemname == 0)) return kFALSE;

   return (strcmp(itemname, "StreamerInfo") == 0) || (strcmp(itemname, "StreamerInfo/") == 0);
}

//______________________________________________________________________________
Bool_t TRootSniffer::ProduceBinary(const char *path, const char * /*query*/, void *&ptr,
                                   Long_t &length)
{
   // produce binary data for specified item
   // if "zipped" option specified in query, buffer will be compressed

   if ((path == 0) || (*path == 0)) return kFALSE;

   if (*path == '/') path++;

   TBufferFile *sbuf = 0;

//   Info("ProduceBinary","Request %s", path);

   Bool_t istreamerinfo = IsStreamerInfoItem(path);

   if (istreamerinfo) {

      CreateMemFile();

      TDirectory *olddir = gDirectory;
      gDirectory = 0;
      TFile *oldfile = gFile;
      gFile = 0;

      fMemFile->WriteStreamerInfo();
      TList *l = fMemFile->GetStreamerInfoList();
      //l->Print("*");

      fSinfoSize = l->GetSize();

      // TODO: one could reuse memory from dabc::MemoryPool here
      //       now keep as it is and copy data at least once
      sbuf = new TBufferFile(TBuffer::kWrite, 100000);
      sbuf->SetParent(fMemFile);
      sbuf->MapObject(l);
      l->Streamer(*sbuf);
      delete l;

      gDirectory = olddir;
      gFile = oldfile;
   } else {

      TClass *obj_cl(0);
      void *obj_ptr = FindInHierarchy(path, &obj_cl);
      if ((obj_ptr == 0) || (obj_cl == 0)) return kFALSE;

      CreateMemFile();

      TDirectory *olddir = gDirectory;
      gDirectory = 0;
      TFile *oldfile = gFile;
      gFile = 0;

      TList *l1 = fMemFile->GetStreamerInfoList();

      if (obj_cl->GetBaseClassOffset(TObject::Class()) == 0) {
         TObject *obj = (TObject *) obj_ptr;

         sbuf = new TBufferFile(TBuffer::kWrite, 100000);
         sbuf->SetParent(fMemFile);
         sbuf->MapObject(obj);
         obj->Streamer(*sbuf);
      } else {
         Info("ProduceBinary", "Non TObject class not yet supported");
         delete sbuf;
         sbuf = 0;
      }

      Bool_t believe_not_changed = kFALSE;

      if ((fMemFile->GetClassIndex() == 0) ||
            (fMemFile->GetClassIndex()->fArray[0] == 0)) {
         believe_not_changed = kTRUE;
      }

      fMemFile->WriteStreamerInfo();
      TList *l2 = fMemFile->GetStreamerInfoList();

      if (believe_not_changed && (l1->GetSize() != l2->GetSize())) {
         Error("ProduceBinary",
               "StreamerInfo changed when we were expecting no changes!!!!!!!!!");
         delete sbuf;
         sbuf = 0;
      }

      fSinfoSize = l2->GetSize();

      delete l1;
      delete l2;

      gDirectory = olddir;
      gFile = oldfile;
   }

   if (sbuf == 0) return kFALSE;

   ptr = malloc(sbuf->Length());
   memcpy(ptr, sbuf->Buffer(), sbuf->Length());
   length = sbuf->Length();

   delete sbuf;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootSniffer::ProduceImage(Int_t kind, const char *path,
                                  const char *options, void *&ptr,
                                  Long_t &length)
{
   // Method to produce image from specified object
   //
   // Parameters:
   //    kind - image kind TImage::kPng, TImage::kJpeg, TImage::kGif
   //    path - path to object
   //    options - extra options
   //
   // By default, image 300x200 is produced
   // In options string one could provide following parameters:
   //    w - image width
   //    h - image height
   //    opt - draw options
   //  For instance:
   //     http://localhost:8080/Files/hsimple.root/hpx/get.png?w=500&h=500&opt=lego1
   //
   //  Return is memory with produced image
   //  Memory must be released by user with free(ptr) call

   ptr = 0;
   length = 0;

   if ((path == 0) || (*path == 0)) return kFALSE;
   if (*path == '/') path++;

   TClass *obj_cl(0);
   void *obj_ptr = FindInHierarchy(path, &obj_cl);
   if ((obj_ptr == 0) || (obj_cl == 0)) return kFALSE;

   if (obj_cl->GetBaseClassOffset(TObject::Class()) != 0) {
      Error("TRootSniffer", "Only derived from TObject classes can be drawn");
      return kFALSE;
   }

   TObject *obj = (TObject *) obj_ptr;

   TImage *img = TImage::Create();
   if (img == 0) return kFALSE;

   if (obj->InheritsFrom(TPad::Class())) {

      if (gDebug > 1)
         Info("TRootSniffer", "Crate IMAGE directly from pad");
      img->FromPad((TPad *) obj);
   } else if (IsDrawableClass(obj->IsA())) {

      if (gDebug > 1)
         Info("TRootSniffer", "Crate IMAGE from object %s", obj->GetName());

      Int_t width(300), height(200);
      TString drawopt = "";

      if ((options != 0) && (*options != 0)) {
         TUrl url;
         url.SetOptions(options);
         url.ParseOptions();
         Int_t w = url.GetIntValueFromOptions("w");
         if (w > 10) width = w;
         Int_t h = url.GetIntValueFromOptions("h");
         if (h > 10) height = h;
         const char *opt = url.GetValueFromOptions("opt");
         if (opt != 0) drawopt = opt;
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

   char *png_buffer(0);
   int size(0);

   im->GetImageBuffer(&png_buffer, &size, (TImage::EImageFileTypes) kind);

   if ((png_buffer != 0) && (size > 0)) {
      ptr = malloc(size);
      length = size;
      memcpy(ptr, png_buffer, length);
   }

   delete [] png_buffer;
   delete im;

   return ptr != 0;
}

//______________________________________________________________________________
Bool_t TRootSniffer::Produce(const char *path, const char *file,
                             const char *options, void *&ptr, Long_t &length, TString &str)
{
   // method to produce different kind of data
   // Supported file (case sensitive):
   //   "root.bin"  - binary data
   //   "root.png"  - png image
   //   "root.jpeg" - jpeg image
   //   "root.gif"  - gif image
   //   "root.xml"  - xml representation
   //   "root.json" - json representation
   //   "exe.json"  - method execution with json reply
   //   "exe.txt"   - method execution with debug output
   //   "cmd.json"  - execution of registered commands
   // Result returned either as string or binary buffer,
   // which should be released with free() call

   if ((file == 0) || (*file == 0)) return kFALSE;

   if (strcmp(file, "root.bin") == 0)
      return ProduceBinary(path, options, ptr, length);

   if (strcmp(file, "root.png") == 0)
      return ProduceImage(TImage::kPng, path, options, ptr, length);

   if (strcmp(file, "root.jpeg") == 0)
      return ProduceImage(TImage::kJpeg, path, options, ptr, length);

   if (strcmp(file, "root.gif") == 0)
      return ProduceImage(TImage::kGif, path, options, ptr, length);

   if (strcmp(file, "exe.bin") == 0)
      return ProduceExe(path, options, 2, 0, &ptr, &length);

   if (strcmp(file, "root.xml") == 0)
      return ProduceXml(path, options, str);

   if (strcmp(file, "root.json") == 0)
      return ProduceJson(path, options, str);

   // used for debugging
   if (strcmp(file, "exe.txt") == 0)
      return ProduceExe(path, options, 0, &str);

   if (strcmp(file, "exe.json") == 0)
      return ProduceExe(path, options, 1, &str);

   if (strcmp(file, "cmd.json") == 0)
      return ExecuteCmd(path, options, str);

   return kFALSE;
}

//______________________________________________________________________________
TObject *TRootSniffer::GetItem(const char *fullname, TFolder *&parent, Bool_t force, Bool_t within_objects)
{
   // return item from the subfolders structure

   TFolder *topf = gROOT->GetRootFolder();

   if (topf == 0) {
      Error("RegisterObject", "Not found top ROOT folder!!!");
      return 0;
   }

   TFolder *httpfold = dynamic_cast<TFolder *>(topf->FindObject("http"));
   if (httpfold == 0) {
      if (!force) return 0;
      httpfold = topf->AddFolder("http", "Top folder");
      httpfold->SetBit(kCanDelete);
      // register top folder in list of cleanups
      gROOT->GetListOfCleanups()->Add(httpfold);
   }

   parent = httpfold;
   TObject *obj = httpfold;

   if (fullname==0) return httpfold;

   // when full path started not with slash, "Objects" subfolder is appended
   TString path = fullname;
   if (within_objects && ((path.Length()==0) || (path[0]!='/')))
      path = fObjectsPath + "/" + path;

   TString tok;
   Ssiz_t from(0);

   while (path.Tokenize(tok,from,"/")) {
      if (tok.Length()==0) continue;

      TFolder *fold = dynamic_cast<TFolder *> (obj);
      if (fold == 0) return 0;

      TIter iter(fold->GetListOfFolders());
      while ((obj = iter()) != 0) {
         if (IsItemField(obj)) continue;
         if (tok.CompareTo(obj->GetName())==0) break;
      }

      if (obj == 0) {
         if (!force) return 0;
         obj = fold->AddFolder(tok, "sub-folder");
         obj->SetBit(kCanDelete);
      }

      parent = fold;
   }

   return obj;
}

//______________________________________________________________________________
TFolder *TRootSniffer::GetSubFolder(const char *subfolder, Bool_t force)
{
   // creates subfolder where objects can be registered

   TFolder *parent = 0;

   return dynamic_cast<TFolder *> (GetItem(subfolder, parent, force));
}

//______________________________________________________________________________
Bool_t TRootSniffer::RegisterObject(const char *subfolder, TObject *obj)
{
   // Register object in subfolder structure
   // subfolder parameter can have many levels like:
   //
   // TRootSniffer* sniff = new TRootSniffer("sniff");
   // sniff->RegisterObject("my/sub/subfolder", h1);
   //
   // Such objects can be later found in "Objects" folder of sniffer like
   //
   // h1 = sniff->FindTObjectInHierarchy("/Objects/my/sub/subfolder/h1");
   //
   // If subfolder name starts with '/', object will be registered starting from top folder.
   //
   // One could provide additional fields for registered objects
   // For instance, setting "_more" field to true let browser
   // explore objects members. For instance:
   //
   // TEvent* ev = new TEvent("ev");
   // sniff->RegisterObject("Events", ev);
   // sniff->SetItemField("Events/ev", "_more", "true");

   TFolder *f = GetSubFolder(subfolder, kTRUE);
   if (f == 0) return kFALSE;

   // If object will be destroyed, it will be removed from the folders automatically
   obj->SetBit(kMustCleanup);

   f->Add(obj);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootSniffer::UnregisterObject(TObject *obj)
{
   // unregister (remove) object from folders structures
   // folder itself will remain even when it will be empty

   if (obj == 0) return kTRUE;

   TFolder *topf = dynamic_cast<TFolder *>(gROOT->FindObject("//root/http"));

   if (topf == 0) {
      Error("UnregisterObject", "Not found //root/http folder!!!");
      return kFALSE;
   }

   // TODO - probably we should remove all set properties as well
   if (topf) topf->RecursiveRemove(obj);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootSniffer::CreateItem(const char *fullname, const char *title)
{
   // create item element

   TFolder *f = GetSubFolder(fullname, kTRUE);
   if (f == 0) return kFALSE;

   if (title) f->SetTitle(title);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootSniffer::IsItemField(TObject* obj) const
{
   // return true when object is TNamed with kItemField bit set
   // such objects used to keep field values for item

   return (obj!=0) && (obj->IsA() == TNamed::Class()) && obj->TestBit(kItemField);
}

//______________________________________________________________________________
Bool_t TRootSniffer::AccessField(TFolder *parent, TObject *chld,
                                 const char *name, const char *value, TNamed **only_get)
{
   // set or get field for the child
   // each field coded as TNamed object, placed after chld in the parent hierarchy

   if (parent==0) return kFALSE;

   if (chld==0) {
      Info("SetField", "Should be special case for top folder, support later");
      return kFALSE;
   }

   TIter iter(parent->GetListOfFolders());

   TObject* obj = 0;
   Bool_t find(kFALSE), last_find(kFALSE);
   // this is special case of top folder - fields are on very top
   if (parent == chld) { last_find = find = kTRUE; }
   TNamed* curr = 0;
   while ((obj = iter()) != 0) {
      if (IsItemField(obj)) {
         if (last_find && (obj->GetName()!=0) && !strcmp(name, obj->GetName())) curr = (TNamed*) obj;
      } else {
         last_find = (obj == chld);
         if (last_find) find = kTRUE;
         if (find && !last_find) break; // no need to continue
      }
   }

   // object must be in childs list
   if (!find) return kFALSE;

   if (only_get!=0) {
      *only_get = curr;
      return curr!=0;
   }

   if (curr!=0) {
      if (value!=0) { curr->SetTitle(value); }
               else { parent->Remove(curr); delete curr; }
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
   TList *lst = dynamic_cast<TList *> (parent->GetListOfFolders());
   if (lst==0) {
      Error("SetField", "Fail cast to TList");
      return kFALSE;
   }

   if (parent==chld)
      lst->AddFirst(curr);
   else
      lst->AddAfter(chld, curr);

   return kTRUE;
}

//_____________________________________________________________
Bool_t TRootSniffer::SetItemField(const char *fullname, const char *name, const char *value)
{
   // set field for specified item

   if ((fullname==0) || (name==0)) return kFALSE;

   TFolder *parent(0);
   TObject *obj = GetItem(fullname, parent);

   if ((parent==0) || (obj==0)) return kFALSE;

   return AccessField(parent, obj, name, value);
}

//______________________________________________________________________________
const char *TRootSniffer::GetItemField(TFolder *parent, TObject *obj, const char *name)
{
  // return field for specified item

   if ((parent==0) || (obj==0) || (name==0)) return 0;

   TNamed *field(0);

   if (!AccessField(parent, obj, name, 0, &field)) return 0;

   return field ? field->GetTitle() : 0;
}


//______________________________________________________________________________
const char *TRootSniffer::GetItemField(const char *fullname, const char *name)
{
   // return field for specified item

   if (fullname==0) return 0;

   TFolder *parent(0);
   TObject *obj = GetItem(fullname, parent);

   return GetItemField(parent, obj, name);
}
