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
#include "RZip.h"

#include "TRootSnifferStore.h"

#include <stdlib.h>

const char *item_prop_kind = "_kind";
const char *item_prop_more = "_more";
const char *item_prop_title = "_title";
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
   parent(0),
   mask(0),
   searchpath(0),
   lvl(0),
   fItemsNames(),
   store(0),
   has_more(kFALSE),
   started_node(),
   num_fields(0),
   num_childs(0)
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

   if (CanSetFields()) store->SetField(lvl, name, value, with_quotes);
   num_fields++;
}

//______________________________________________________________________________
void TRootSnifferScanRec::BeforeNextChild()
{
   // indicates that new child for current element will be started

   if (CanSetFields()) store->BeforeNextChild(lvl, num_childs, num_fields);
   num_childs++;
}

//______________________________________________________________________________
void TRootSnifferScanRec::MakeItemName(const char *objname, TString& itemname)
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

   started_node = _node_name;

   if (parent) parent->BeforeNextChild();

   if (store) store->CreateNode(lvl, started_node.Data());
}

//______________________________________________________________________________
void TRootSnifferScanRec::CloseNode()
{
   // close started node

   if (store && !started_node.IsNull()) {
      store->CloseNode(lvl, started_node.Data(), num_childs);
      started_node = "";
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

   if (store == 0)
      return kFALSE;

   if ((mask & mask_Search) && store->GetResPtr())
      return kTRUE;

   if ((mask & mask_CheckChld) && store->GetResPtr() &&
         (store->GetResNumChilds() >= 0))
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
   if ((mask & (mask_Search | mask_CheckChld)) == 0) return kFALSE;

   // only when full search path is scanned
   if (searchpath != 0) return kFALSE;

   if (store == 0) return kFALSE;

   return kTRUE;
}


//______________________________________________________________________________
Bool_t TRootSnifferScanRec::SetResult(void *obj, TClass *cl, TDataMember *member, Int_t chlds)
{
   // set results of scanning

   if (Done()) return kTRUE;

   if (!IsReadyForResult()) return kFALSE;

   store->SetResult(obj, cl, member, chlds);

   return Done();
}

//______________________________________________________________________________
Int_t TRootSnifferScanRec::Depth() const
{
   // returns current depth of scanned hierarchy

   Int_t cnt = 0;
   const TRootSnifferScanRec *rec = this;
   while (rec->parent) {
      rec = rec->parent;
      cnt++;
   }

   return cnt;
}

//______________________________________________________________________________
Int_t TRootSnifferScanRec::ExtraFolderLevel()
{
   // return level depth till folder, marked with extra flag
   // Objects in such folder can be 'expanded' -
   // one can get access to all class members
   // If no extra folder found, -1 is returned

   TRootSnifferScanRec *rec = this;
   Int_t cnt = 0;
   while (rec) {
      if (rec->mask & mask_ExtraFolder) return cnt;
      rec = rec->parent;
      cnt++;
   }

   return -1;
}

//______________________________________________________________________________
Bool_t TRootSnifferScanRec::CanExpandItem()
{
   // returns true if current item can be expanded - means one could explore
   // objects members

   if (mask & (mask_Expand | mask_Search | mask_CheckChld)) return kTRUE;

   if (!has_more) return kFALSE;

   // if parent has expand mask, allow to expand item
   if (parent && (parent->mask & mask_Expand)) return kTRUE;

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

   const char* full_name = 0;

   // remove slashes from file names
   if (obj && obj->InheritsFrom(TDirectoryFile::Class())) {
      const char* slash = strrchr(obj_name, '/');
      if (slash!=0) {
         full_name = obj_name;
         obj_name = slash+1;
         if (*obj_name == 0) obj_name = "file";
      }
   }

   super.MakeItemName(obj_name, obj_item_name);

   lvl = super.lvl;
   store = super.store;
   searchpath = super.searchpath;
   mask = super.mask & mask_Actions;
   parent = &super;

   if (mask & mask_Scan) {
      // only when doing scan, increment lvl, used for text formatting
      lvl++;
   } else {
      if (searchpath == 0) return kFALSE;

      if (strncmp(searchpath, obj_item_name.Data(), obj_item_name.Length()) != 0)
         return kFALSE;

      const char *separ = searchpath + obj_item_name.Length();

      Bool_t isslash = kFALSE;
      while (*separ == '/') {
         separ++;
         isslash = kTRUE;
      }

      if (*separ == 0) {
         searchpath = 0;
         if (mask & mask_Expand) {
            mask = mask_Scan;
            searchpath = 0;
            has_more = true; // when found selected object, allow to scan it (and only it)
         }

      } else {
         if (!isslash) return kFALSE;
         searchpath = separ;
      }
   }

   CreateNode(obj_item_name.Data());

   if ((obj_name!=0) && (obj_item_name != obj_name))
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

   //DOUT0("SCAN CLASS %s mask %u", cl->GetName(), rec.mask);

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
            chld.has_more = kTRUE;
         }

         if (chld.SetResult(member_ptr, mcl, member)) break;

         const char* title = member->GetTitle();
         if ((title!=0) && (strlen(title)!=0))
            chld.SetField(item_prop_title, title);

         if (member->GetTypeName())
           chld.SetField(item_prop_typename, member->GetTypeName());

         if (member->GetArrayDim() > 0) {
            // store array dimensions in form [N1,N2,N3,...]
            TString dim("[");
            for (Int_t n=0;n<member->GetArrayDim();n++) {
               if (n>0) dim.Append(",");
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

         if (chld.SetResult(member_ptr, mcl, member, chld.num_childs)) break;
      }
   }
}

//______________________________________________________________________________
void TRootSniffer::ScanObject(TRootSnifferScanRec &rec, TObject *obj)
{
   // scan object content
   // if object is known collection class (like TFolder),
   // collection content will be scanned as well

   if (obj == 0) return;

   if (!fReadOnly && obj->InheritsFrom(TKey::Class()) && rec.IsReadyForResult()) {
      TObject* keyobj = ((TKey*) obj)->ReadObj();
      if (keyobj!=0)
         if (rec.SetResult(keyobj, keyobj->IsA())) return;
   }

   if (rec.SetResult(obj, obj->IsA())) return;

   TClass* obj_class = obj->IsA();

   ScanObjectProperties(rec, obj, obj_class);

   rec.SetRootClass(obj_class);

   ScanObjectChilds(rec, obj);

   // here we should know how many childs are accumulated
   rec.SetResult(obj, obj_class, 0, rec.num_childs);
}


//______________________________________________________________________________
void TRootSniffer::ScanObjectProperties(TRootSnifferScanRec &rec, TObject* &obj, TClass* &obj_class)
{
   // scans basic object properties
   // here such fields as _typename, _title, _more properties can be specified

   const char* title = obj->GetTitle();
   if ((title!=0) && (*title!=0))
      rec.SetField(item_prop_title, title);

   int isextra = rec.ExtraFolderLevel();

   if ((isextra == 1) || ((isextra > 1) && !IsDrawableClass(obj->IsA()))) {
      rec.SetField(item_prop_more, "true", kFALSE);
      rec.has_more = kTRUE;
   }

   // special handling of TKey class - in non-readonly mode
   // sniffer allowed to fetch objects

   if (!fReadOnly && obj->InheritsFrom(TKey::Class())) {
      TKey* key = (TKey *) obj;
      if (strcmp(key->GetClassName(),"TDirectoryFile")==0) {
         if (rec.lvl==0) {
            TDirectory* dir = dynamic_cast<TDirectory*> (key->ReadObj());
            if (dir!=0) {
               obj = dir;
               obj_class = dir->IsA();
            }
         } else {
            rec.SetField(item_prop_more, "true", kFALSE);
            rec.has_more = kTRUE;
         }
      } else {
         obj_class = TClass::GetClass(key->GetClassName());
      }
   }
}

void TRootSniffer::ScanObjectChilds(TRootSnifferScanRec &rec, TObject *obj)
{
   // scans object childs (if any)
   // here one scans collection, branches, trees and so on

   if (obj->InheritsFrom(TFolder::Class())) {
      // starting from special folder, we automatically scan members

      TFolder *fold = (TFolder *) obj;
      if (fold->TestBit(BIT(19))) rec.mask = rec.mask | mask_ExtraFolder;

      ScanCollection(rec, fold->GetListOfFolders());
   } else if (obj->InheritsFrom(TDirectory::Class())) {
      TDirectory* dir = (TDirectory *) obj;
      ScanCollection(rec, dir->GetList(), 0, kFALSE, dir->GetListOfKeys());
   } else if (obj->InheritsFrom(TTree::Class())) {
      ScanCollection(rec, ((TTree *) obj)->GetListOfLeaves());
   } else if (obj->InheritsFrom(TBranch::Class())) {
      ScanCollection(rec, ((TBranch *) obj)->GetListOfLeaves());
   } else if (rec.CanExpandItem()) {
      ScanObjectMemebers(rec, obj->IsA(), (char *) obj, 0);
   }
}


//______________________________________________________________________________
void TRootSniffer::ScanCollection(TRootSnifferScanRec &rec, TCollection *lst,
                                  const char *foldername, Bool_t extra, TCollection* keys_lst)
{
   // scan collection content

   if (((lst == 0) || (lst->GetSize() == 0)) && ((keys_lst==0) || (keys_lst->GetSize()==0))) return;

   TRootSnifferScanRec folderrec;
   if (foldername) {
      if (!folderrec.GoInside(rec, 0, foldername)) return;
      if (extra) folderrec.mask = folderrec.mask | mask_ExtraFolder;
   }

   {
      TRootSnifferScanRec &master = foldername ? folderrec : rec;

      if (lst!=0) {
         TIter iter(lst);
         TObject *obj(0);

         while ((obj = iter()) != 0) {
            TRootSnifferScanRec chld;
            if (chld.GoInside(master, obj)) {
               ScanObject(chld, obj);
               if (chld.Done()) break;
            }
         }
      }

      if (keys_lst!=0) {
         TIter iter(keys_lst);
         TObject *kobj(0);

         while ((kobj = iter()) != 0) {
            TKey* key = dynamic_cast<TKey*> (kobj);
            if (key == 0) continue;
            TObject* obj = (lst == 0) ? 0 : lst->FindObject(key->GetName());
            if ((obj!=0) && (master.mask & mask_Scan)) continue;

            if (obj==0) obj = key; // if object exists, provide it to for scan instead of  key

            TRootSnifferScanRec chld;
            TString fullname = TString::Format("%s;%d", key->GetName(), key->GetCycle());

            if (chld.GoInside(master, obj, fullname.Data())) {
               ScanObject(chld, obj);
               if (chld.Done()) break;
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

   {
      TRootSnifferScanRec chld;
      if (chld.GoInside(rec, 0, "StreamerInfo")) {
         chld.SetField(item_prop_kind, "ROOT.TStreamerInfoList");
         chld.SetField(item_prop_title, "List of streamer infos for binary I/O");
      }
   }

   TFolder *topf = dynamic_cast<TFolder *>(gROOT->FindObject(TString::Format("//root/%s", fObjectsPath.Data())));

   ScanCollection(rec, gROOT->GetList());

   ScanCollection(rec, gROOT->GetListOfCanvases(), "Canvases");

   ScanCollection(rec, gROOT->GetListOfFiles(), "Files");

   ScanCollection(rec, topf ? topf->GetListOfFolders() : 0, "Objects");
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
Bool_t TRootSniffer::IsBrowsableClass(TClass *cl)
{
   // return true if object can be browsed?

   if (cl == 0) return kFALSE;

   if (cl->InheritsFrom(TTree::Class())) return kTRUE;
   if (cl->InheritsFrom(TBranch::Class())) return kTRUE;
   if (cl->InheritsFrom(TLeaf::Class())) return kTRUE;
   if (cl->InheritsFrom(TFolder::Class())) return kTRUE;

   return kFALSE;
}

//______________________________________________________________________________
void TRootSniffer::ScanHierarchy(const char *topname, const char *path,
                                 TRootSnifferStore *store)
{
   // scan ROOT hierarchy with provided store object

   TRootSnifferScanRec rec;
   rec.searchpath = path;
   if (rec.searchpath) {
      if (*rec.searchpath == '/') rec.searchpath++;
      if (*rec.searchpath == 0) rec.searchpath = 0;
   }

   // if path non-empty, we should find item first and than start scanning
   rec.mask = rec.searchpath == 0 ? mask_Scan : mask_Expand;

   rec.store = store;

   rec.CreateNode(topname);

   ScanRoot(rec);

   rec.CloseNode();
}

//______________________________________________________________________________
void *TRootSniffer::FindInHierarchy(const char *path, TClass **cl,
                                    TDataMember **member, Int_t *chld)
{
   // search element with specified path
   // returns pointer on element
   // optionally one could obtain element class, member description and number
   // of childs
   // when chld!=0, not only element is searched, but also number of childs are
   // counted
   // when member!=0, any object will be scanned for its data members (disregard
   // of extra options)

   TRootSnifferStore store;

   TRootSnifferScanRec rec;
   rec.searchpath = path;
   rec.mask = (chld != 0) ? mask_CheckChld : mask_Search;
   if (*rec.searchpath == '/') rec.searchpath++;
   rec.store = &store;

   ScanRoot(rec);

   if (cl) *cl = store.GetResClass();
   if (member) *member = store.GetResMember();
   if (chld) *chld = store.GetResNumChilds();

   return store.GetResPtr();
}

//______________________________________________________________________________
TObject *TRootSniffer::FindTObjectInHierarchy(const char *path)
{
   // search element in hierarchy, derived from TObject

   TClass *cl(0);

   void *obj = FindInHierarchy(path, &cl);

   return (cl != 0) && (cl->GetBaseClassOffset(TObject::Class()) == 0) ? (TObject *) obj : 0;
}

//______________________________________________________________________________
ULong_t TRootSniffer::GetStreamerInfoHash()
{
   // returns hash value for streamer infos
   // at the moment - just number of items in streamer infos list.

   return fSinfoSize;
}

//______________________________________________________________________________
ULong_t TRootSniffer::GetItemHash(const char *itemname)
{
   // get hash function for specified item
   // used to detect any changes in the specified object

   if (IsStreamerInfoItem(itemname)) return GetStreamerInfoHash();

   TObject *obj = FindTObjectInHierarchy(itemname);

   return obj == 0 ? 0 : TString::Hash(obj, obj->IsA()->Size());
}

//______________________________________________________________________________
Bool_t TRootSniffer::CanDrawItem(const char *path)
{
   // method verifies if object can be drawn

   TClass *obj_cl(0);
   void *res = FindInHierarchy(path, &obj_cl);
   return (res != 0) && IsDrawableClass(obj_cl);
}

//______________________________________________________________________________
Bool_t TRootSniffer::CanExploreItem(const char *path)
{
   // method returns true when object has childs or
   // one could try to expand item

   TClass *obj_cl(0);
   Int_t obj_chld(-1);
   void *res = FindInHierarchy(path, &obj_cl, 0, &obj_chld);
   return (res != 0) && (obj_chld > 0);
}

//______________________________________________________________________________
void TRootSniffer::CreateMemFile()
{
   // creates TMemFile instance, which used for objects streaming
   // One could not use TBuffer directly, while one also require streamer infos
   // list

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
Bool_t TRootSniffer::ProduceExe(const char *path, const char * options, TString &res)
{
   // execute command for specified object
   // options include method and extra list of parameters
   // sniffer should be not-readonly to allow execution of the commands

   if ((path == 0) || (*path == 0) || fReadOnly) return kFALSE;

   if (*path == '/') path++;

   TClass *obj_cl(0);
   void *obj_ptr = FindInHierarchy(path, &obj_cl);
   if ((obj_ptr == 0) || (obj_cl == 0)) return kFALSE;

   TUrl url;
   url.SetOptions(options);

   const char* method_name = url.GetValueFromOptions("method");
   if (method_name==0) return kFALSE;

   TMethod* method = obj_cl->GetMethodAllAny(method_name);
   if (method==0) return kFALSE;

   TList* args = method->GetListOfMethodArgs();

   res.Form("Method: %s\n", method_name);

   TIter next(args);
   TMethodArg* arg = 0;
   TString call_args;
   while ((arg = (TMethodArg*) next()) != 0) {

      const char* val = url.GetValueFromOptions(arg->GetName());
      if (val==0) val = arg->GetDefault();

      if ((strcmp(arg->GetName(),"rest_url_opt")==0) &&
          (strcmp(arg->GetFullTypeName(),"const char*")==0) && (args->GetSize()==1)) {
         // very special case - function requires list of options after method=argument

         const char* pos = strstr(options,"method=");
         if ((pos == 0) || (strlen(pos) < strlen(method_name)+8)) return kFALSE;
         call_args.Form("\"%s\"", pos + strlen(method_name)+8);
         break;
      }

      res += TString::Format("  Argument:%s Type:%s Specified:%s \n", arg->GetName(), arg->GetFullTypeName(), val ? val : "<missed>");

      if (val==0) { res += "missing argument\n"; return kTRUE; }

      if (call_args.Length()>0) call_args+=", ";

      if (strcmp(arg->GetFullTypeName(),"const char*")==0) {
         int len = strlen(val);
         if ((strlen(val)<2) || (*val != '\"') || (val[len-1]!='\"'))
            call_args.Append(TString::Format("\"%s\"", val));
         else
            call_args.Append(val);
      } else {
         call_args.Append(val);
      }
   }

   res += TString::Format("Calling obj->%s(%s);\n", method_name, call_args.Data());

   TMethodCall call(obj_cl, method_name, call_args.Data());

   if (!call.IsValid()) { res += "Fail: invalid TMethodCall\n"; return kTRUE; }

   call.Execute(obj_ptr);

   res += "Execution done!\n";
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootSniffer::IsStreamerInfoItem(const char *itemname)
{
   if ((itemname == 0) || (*itemname == 0)) return kFALSE;

   return (strcmp(itemname, "StreamerInfo") == 0) || (strcmp(itemname, "StreamerInfo/") == 0);
}

//______________________________________________________________________________
Bool_t TRootSniffer::ProduceBinary(const char *path, const char* query, void *&ptr,
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
         believe_not_changed = true;
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

   if (sbuf==0) return kFALSE;

   if ((query!=0) && (strstr(query,"zipped")!=0)) {
      Int_t buflen = 20 + sbuf->Length() + sbuf->Length()/20; // keep safety margin
      if (buflen<512) buflen = 512;

      ptr = malloc(buflen);

      int irep(0), srcsize(sbuf->Length()), tgtsize(buflen);

      R__zip(5, &srcsize, (char*) sbuf->Buffer(), &tgtsize, (char*) ptr, &irep);

      length = irep;

   } else {
      ptr = malloc(sbuf->Length());
      memcpy(ptr, sbuf->Buffer(), sbuf->Length());
      length = sbuf->Length();
   }



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

      if (gDebug>1)
         Info("TRootSniffer", "Crate IMAGE directly from pad");
      img->FromPad((TPad *) obj);
   } else if (IsDrawableClass(obj->IsA())) {

      if (gDebug>1)
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
                             const char *options, void *&ptr, Long_t &length)
{
   // method to produce different kind of binary data
   // Supported file (case sensitive):
   //   "root.bin"  - binary data
   //   "root.png"  - png image
   //   "root.jpeg" - jpeg image
   //   "root.gif"  - gif image
   //   "root.xml"  - xml representation
   //   "root.json" - json representation

   if ((file == 0) || (*file == 0)) return kFALSE;

   if (strcmp(file, "root.bin") == 0)
      return ProduceBinary(path, options, ptr, length);

   if (strcmp(file, "root.png") == 0)
      return ProduceImage(TImage::kPng, path, options, ptr, length);

   if (strcmp(file, "root.jpeg") == 0)
      return ProduceImage(TImage::kJpeg, path, options, ptr, length);

   if (strcmp(file, "root.gif") == 0)
      return ProduceImage(TImage::kGif, path, options, ptr, length);

   if (strcmp(file, "root.xml") == 0) {
      TString res;
      if (!ProduceXml(path, options, res)) return kFALSE;
      length = res.Length();
      ptr = malloc(length);
      memcpy(ptr, res.Data(), length);
      return kTRUE;
   }

   if ((strcmp(file, "root.json") == 0) || (strcmp(file, "get.json") == 0)) {
      TString res;
      if (!ProduceJson(path, options, res)) return kFALSE;
      length = res.Length();
      ptr = malloc(length);
      memcpy(ptr, res.Data(), length);
      return kTRUE;
   }

   if (strcmp(file, "exe.txt") == 0) {
      TString res;
      if (!ProduceExe(path, options, res)) return kFALSE;
      length = res.Length();
      ptr = malloc(length);
      memcpy(ptr, res.Data(), length);
      return kTRUE;
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TRootSniffer::RegisterObject(const char *subfolder, TObject *obj)
{
   // register object in subfolder structure
   // subfolder parameter can have many levels like:
   //
   // TRootSniffer* sniff = new TRootSniffer("sniff");
   // sniff->RegisterObject("/my/sub/subfolder", h1);
   //
   // Such objects can be later found in "Objects" folder of sniffer like
   //
   // h1 = sniff->FindTObjectInHierarchy("/Objects/my/sub/subfolder/h1");
   //
   // Objects, registered in "extra" sub-folder, can be explored.
   // Typically one used "extra" sub-folder to register event structures to
   // be able expand it later in web-browser:
   //
   // TEvent* ev = new TEvent;
   // sniff->RegisterObject("extra", ev);


   if (obj == 0) return kFALSE;

   TFolder *topf = gROOT->GetRootFolder();

   if (topf == 0) {
      Error("RegisterObject", "Not found top ROOT folder!!!");
      return kFALSE;
   }

   TFolder *topdabcfold = dynamic_cast<TFolder *>(topf->FindObject(fObjectsPath.Data()));
   if (topdabcfold == 0) {
      topdabcfold = topf->AddFolder(fObjectsPath.Data(), "Top online folder");
      topdabcfold->SetOwner(kFALSE);
   }

   TFolder *dabcfold = topdabcfold;

   if ((subfolder != 0) && (strlen(subfolder) > 0)) {

      TObjArray *arr = TString(subfolder).Tokenize("/");
      for (Int_t i = 0; i <= (arr ? arr->GetLast() : -1); i++) {

         const char *subname = arr->At(i)->GetName();
         if (strlen(subname) == 0) continue;

         TFolder *fold = dynamic_cast<TFolder *>(dabcfold->FindObject(subname));
         if (fold == 0) {
            fold = dabcfold->AddFolder(subname, "sub-folder");
            fold->SetOwner(kFALSE);

            if ((dabcfold == topdabcfold) && (strcmp(subname, "extra") == 0))
               fold->SetBit(BIT(19), kTRUE);
         }
         dabcfold = fold;
      }

   }

   // If object will be destroyed, it must be removed from the folders automatically
   obj->SetBit(kMustCleanup);

   dabcfold->Add(obj);

   // register folder for cleanup
   if (!gROOT->GetListOfCleanups()->FindObject(dabcfold))
      gROOT->GetListOfCleanups()->Add(dabcfold);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootSniffer::UnregisterObject(TObject *obj)
{
   // unregister (remove) object from folders structures
   // folder itself will remain even when it will be empty

   if (obj == 0) return kTRUE;

   TFolder *topf = gROOT->GetRootFolder();

   if (topf == 0) {
      Error("UnregisterObject", "Not found top ROOT folder!!!");
      return kFALSE;
   }

   TFolder *dabcfold = dynamic_cast<TFolder *>(topf->FindObject(fObjectsPath.Data()));

   if (dabcfold) dabcfold->RecursiveRemove(obj);

   return kTRUE;
}

