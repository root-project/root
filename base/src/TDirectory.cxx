// @(#)root/base:$Name:  $:$Id: TDirectory.cxx,v 1.27 2002/06/30 13:43:16 brun Exp $
// Author: Rene Brun   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "Strlen.h"
#include "TDirectory.h"
#include "TMapFile.h"
#include "TClassTable.h"
#include "TInterpreter.h"
#include "THashList.h"
#include "TBrowser.h"
#include "TFree.h"
#include "TKey.h"
#include "TROOT.h"
#include "TError.h"
#include "Bytes.h"
#include "TStreamerInfo.h"
#include "TRegexp.h"
#include "TProcessUUID.h"

TDirectory    *gDirectory;      //Pointer to current directory in memory

//______________________________________________________________________________
//Begin_Html
/*
<img src="gif/tdirectory_classtree.gif">
*/
//End_Html

ClassImp(TDirectory)

//______________________________________________________________________________
//
// A ROOT file is structured in Directories (like a file system).
// Each Directory has a list of Keys (see TKeys) and a list of objects
// in memory. A Key is a small object that describes the type and location
// of a persistent object in a file. The persistent object may be a directory.
//Begin_Html
/*
<img src="gif/fildir.gif">
*/
//End_Html
//
//      The structure of a file is shown in TFile::TFile
//

//______________________________________________________________________________
TDirectory::TDirectory() : TNamed()
{
//*-*-*-*-*-*-*-*-*-*-*-*Directory default constructor-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    =============================
   fFile = 0;
   fList = 0;
   fKeys = 0;
   fWritable = kFALSE;
}

//______________________________________________________________________________
TDirectory::TDirectory(const char *name, const char *title, Option_t *classname)
           : TNamed(name, title)
{
//*-*-*-*-*-*-*-*-*-*-*-* Create a new Directory *-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                     ======================
//  A new directory with name,title is created in the current directory
//  The directory header information is immediatly saved on the file
//  A new key is added in the parent directory
//
//  When this constructor is called from a class directly derived
//  from TDirectory, the third argument classname MUST be specified.
//  In this case, classname must be the name of the derived class.
//
//  Note that the directory name cannot contain slashes.
//
   if (strchr(name,'/')) {
      ::Error("TDirectory::TDirectory","directory name cannot contain a slash", name);
      gDirectory = 0;
      return;
   }
   if (strlen(GetName()) == 0) {
      ::Error("TDirectory::TDirectory","directory name cannot be \"\"");
      gDirectory = 0;
      return;
   }

   Build();
   if (gDirectory == 0) return;
   if (gFile == 0) return;
   if (!gFile->IsWritable()) return; //*-* in case of a directory in memory
   if (gDirectory->GetKey(name)) {
      Error("TDirectory","Directory %s exists already",name);
      return;
   }
   TClass *cl = IsA();
   if (strlen(classname) != 0) cl = gROOT->GetClass(classname);
   if (!cl) {
      Error("TDirectory","Invalid class name: %s",classname);
      return;
   }
   fWritable    = kTRUE;
   fSeekParent  = gFile->GetSeekDir();
   Int_t nbytes = TDirectory::Sizeof();
   TKey *key    = new TKey(fName,fTitle,cl,nbytes);
   fNbytesName  = key->GetKeylen();
   fSeekDir     = key->GetSeekKey();
   if (fSeekDir == 0) return;
   char *buffer = key->GetBuffer();
   TDirectory::FillBuffer(buffer);
   Int_t cycle = gDirectory->AppendKey(key);
   key->WriteFile(cycle);
   fModified = kFALSE;
   gROOT->GetUUIDs()->AddUUID(fUUID,this);
}

//______________________________________________________________________________
TDirectory::TDirectory(const TDirectory &directory) : TNamed(directory)
{
   ((TDirectory&)directory).Copy(*this);
}

//______________________________________________________________________________
TDirectory::~TDirectory()
{
//*-*-*-*-*-*-*-*-*-*-*-*Directory destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ====================

   if (gROOT == 0) return; //when called by TROOT destructor

   TCollection::StartGarbageCollection();

   if (fList) {
      SetBit(kCloseDirectory);
      fList->Delete();
      SafeDelete(fList);
   }
   if (fKeys) {
      fKeys->Delete();
      SafeDelete(fKeys);
   }

   TCollection::EmptyGarbageCollection();

   if (gDebug)
      cerr << "TDirectory dtor called for "<< GetName() << endl;
}

//______________________________________________________________________________
void TDirectory::Append(TObject *obj)
{
   // Append object to current directory.

   if (obj == 0 || fList == 0) return;
   fList->Add(obj);
   if (!fMother) return;
   if (fMother->IsA() == TMapFile::Class()) {
      TMapFile *mfile = (TMapFile*)fMother;
      mfile->Add(obj);
   }
}

//______________________________________________________________________________
Int_t TDirectory::AppendKey(TKey *key)
{
//*-*-*-*-*-*-*Insert key in the linked list of keys of this directory*-*-*-*
//*-*          =======================================================

   fModified = kTRUE;

   // This is a fast hash lookup in case the key does not already exist
   TKey *oldkey = (TKey*)fKeys->FindObject(key->GetName());
   if (!oldkey) {
      fKeys->Add(key);
      return 1;
   }

   // If the key name already exists we have to make a scan for it
   // and insert the new key ahead of the current one
   TObjLink *lnk = fKeys->FirstLink();
   while (lnk) {
      oldkey = (TKey*)lnk->GetObject();
      if (!strcmp(oldkey->GetName(), key->GetName()))
         break;
      lnk = lnk->Next();
   }

   fKeys->AddBefore(lnk, key);
   return oldkey->GetCycle() + 1;
}

//______________________________________________________________________________
void TDirectory::Browse(TBrowser *b)
{
   Char_t name[128];

   if (b) {
      TObject *obj = 0;
      TIter nextin(fList);
      TKey *key = 0, *keyo = 0;
      TIter next(fKeys);

      cd();

      //Add objects that are only in memory
      while ((obj = nextin())) {
         if (fKeys->FindObject(obj->GetName())) continue;
         b->Add(obj, obj->GetName());
      }

      //Add keys
      while ((key = (TKey *) next())) {
         int skip = 0;
         if (!keyo || (keyo && strcmp(keyo->GetName(), key->GetName()))) {
            skip = 0;
            obj = fList->FindObject(key->GetName());

            if (obj) {
               sprintf(name, "%s", obj->GetName());
               b->Add(obj, name);
               if (obj->IsFolder()) skip = 1;
            }
         }

         if (!skip) {
            sprintf(name, "%s;%d", key->GetName(), key->GetCycle());
            b->Add(key, name);
         }

         keyo = key;
      }
   }
}

//______________________________________________________________________________
void TDirectory::Build()
{
//*-*-*-*-*-*-*-*-*-*-*-*Initialise directory to defaults*-*-*-*-*-*-*-*-*-*
//*-*                    ================================

   // If directory is created via default ctor (when dir is read from file)
   // don't add it here to the directory since its name is not yet known.
   // It will be added to the directory in TKey::ReadObj().
   if (gDirectory && strlen(GetName()) != 0) gDirectory->Append(this);
   fModified   = kTRUE;
   fWritable   = kFALSE;
   fDatimeC.Set();
   fDatimeM.Set();
   fNbytesKeys = 0;
   fSeekDir    = 0;
   fSeekParent = 0;
   fSeekKeys   = 0;
   fList       = new THashList(100,50);
   fKeys       = new THashList(100,50);
   fMother     = gDirectory;
   fFile       = gFile;
   SetBit(kCanDelete);
}

//______________________________________________________________________________
Bool_t TDirectory::cd(const char *path)
{
   // Change current directory to "this" directory . Using path one can
   // change the current directory to "path". The absolute path syntax is:
   // file.root:/dir1/dir2
   // where file.root is the file and /dir1/dir2 the desired subdirectory
   // in the file. Relative syntax is relative to "this" directory. E.g:
   // ../aa. Returns kTRUE in case of success.

   return cd1(path);
}

//______________________________________________________________________________
Bool_t TDirectory::cd1(const char *apath)
{
   // Change current directory to "this" directory . Using path one can
   // change the current directory to "path". The absolute path syntax is:
   // file.root:/dir1/dir2
   // where file.root is the file and /dir1/dir2 the desired subdirectory
   // in the file. Relative syntax is relative to "this" directory. E.g:
   // ../aa. Returns kFALSE in case path does not exist.

   Int_t nch = 0;
   if (apath) nch = strlen(apath);
   if (!nch) {
      gDirectory = this;
      gFile      = fFile;
      return kTRUE;
   }

   TDirectory *savdir = gDirectory;

   char *path = new char[nch+1]; path[0] = 0;
   if (nch) strcpy(path,apath);
   char *s = (char*)strchr(path, ':');
   if (s) {
      *s = '\0';
      TDirectory *f = (TDirectory *)gROOT->GetListOfFiles()->FindObject(path);
      if (!f && !strcmp(gROOT->GetName(), path)) f = gROOT;
      if (s) *s = ':';
      if (f) {
         f->cd();
         if (s && *(s+1))
            if (!gDirectory->cd1(s+1)) {
               gDirectory = savdir;
               delete [] path; return kFALSE;
            }
         delete [] path; return kTRUE;
      } else {
         Error("cd", "No such file %s", path);
         delete [] path; return kFALSE;
      }
   }

   // path starts with a slash (assumes current file)
   if (path[0] == '/') {
      TDirectory *td = fFile;
      if (!fFile) td = gROOT;
#ifdef cxxbug
      //this special case to circumvent one more bug in the alpha cxx compiler.
      //seems to be same bug also found in Btree.
      td->cd(path+1);
#else
      if (!td->cd1(path+1)) {
         gDirectory = savdir;
         delete [] path; return kFALSE;
      }
#endif
      delete [] path; return kTRUE;
   }

   TObject *obj;
   char *slash = (char*)strchr(path,'/');
   if (!slash) {                     // we are at the lowest level
      if (!strcmp(path, "..")) {
         if (fMother && fMother->InheritsFrom(TDirectory::Class()))
            ((TDirectory*)fMother)->cd();
         delete [] path; return kTRUE;
      }
      obj = Get(path);
      if (!obj) {
         Error("cd","Unknown directory %s", path);
         delete [] path; return kFALSE;
      }

      //Check return object is a directory
      if (!obj->InheritsFrom(TDirectory::Class())) {
         Error("cd","Object %s is not a directory", path);
         delete [] path; return kFALSE;
      }
      ((TDirectory*)obj)->cd();
      delete [] path; return kTRUE;
   }

   char subdir[128];
   strcpy(subdir,path);
   slash = (char*)strchr(subdir,'/');
   *slash = 0;
   //Get object with path from current directory/file
   if (!strcmp(subdir, "..")) {
      if (fMother && fMother->InheritsFrom(TDirectory::Class()))
         if (!((TDirectory*)fMother)->cd1(slash+1)) {
             gDirectory = savdir;
             delete [] path; return kFALSE;
         }
      delete [] path; return kTRUE;
   }
   obj = Get(subdir);
   if (!obj) {
      Error("cd","Unknown directory %s", subdir);
      delete [] path; return kFALSE;
   }

   //Check return object is a directory
   if (!obj->InheritsFrom(TDirectory::Class())) {
      Error("cd","Object %s is not a directory", subdir);
      delete [] path; return kFALSE;
   }
   if (!((TDirectory*)obj)->cd1(slash+1)) {
      gDirectory = savdir;
      delete [] path; return kFALSE;
   }
   delete [] path; return kTRUE;
}

//______________________________________________________________________________
Bool_t TDirectory::Cd(const char *path)
{
   // Change current directory to "path". The absolute path syntax is:
   // file.root:/dir1/dir2
   // where file.root is the file and /dir1/dir2 the desired subdirectory
   // in the file. Relative syntax is relative to the current directory
   // gDirectory, e.g.: ../aa. Returns kTRUE in case of success.

   return Cd1(path);
}

//______________________________________________________________________________
Bool_t TDirectory::Cd1(const char *apath)
{
   // Change current directory to "path". The path syntax is:
   // file.root:/dir1/dir2
   // where file.root is the file and /dir1/dir2 the desired subdirectory
   // in the file. Returns kFALSE in case path does not exist.

   // null path is always true (i.e. stay in the current directory)
   Int_t nch = 0;
   if (apath) nch = strlen(apath);
   if (!nch) return kTRUE;

   TDirectory *savdir = gDirectory;

   char *path = new char[nch+1]; path[0] = 0;
   if (nch) strcpy(path,apath);
   char *s = (char*)strchr(path, ':');
   if (s) {
      *s = '\0';
      TDirectory *f = (TDirectory *)gROOT->GetListOfFiles()->FindObject(path);
      if (!f && !strcmp(gROOT->GetName(), path)) f = gROOT;
      if (s) *s = ':';
      if (f) {
         f->cd();
         if (s && *(s+1))
            if (!TDirectory::Cd1(s+1)) {
               gDirectory = savdir;
               delete [] path; return kFALSE;
            }
         delete [] path; return kTRUE;
      } else {
         ::Error("TDirectory::Cd", "No such file %s", path);
         delete [] path; return kFALSE;
      }
   }

   // path starts with a slash (assumes current file)
   if (path[0] == '/') {
      if (!TDirectory::Cd1(path+1)) {
         gDirectory = savdir;
         delete [] path; return kFALSE;
      }
      delete [] path; return kTRUE;
   }

   TObject *obj;
   char *slash = (char*)strchr(path,'/');
   if (!slash) {                     // we are at the lowest level
      if (!strcmp(path, "..")) {
         if (gDirectory->fMother && gDirectory->fMother->InheritsFrom(TDirectory::Class()))
            ((TDirectory*)gDirectory->fMother)->cd();
         delete [] path; return kTRUE;
      }
      obj = gDirectory->Get(path);
      if (!obj) {
         ::Error("TDirectory::Cd","Unknown directory %s", path);
         delete [] path; return kFALSE;
      }

      //Check return object is a directory
      if (!obj->InheritsFrom(TDirectory::Class())) {
         ::Error("TDirectory::Cd","Object %s is not a directory", path);
         delete [] path; return kFALSE;
      }
      ((TDirectory*)obj)->cd();
      delete [] path; return kTRUE;
   }

   char subdir[128];
   strcpy(subdir,path);
   slash = strchr(subdir,'/');
   *slash = 0;
   //Get object with path from current directory/file
   if (!strcmp(subdir, "..")) {
      if (gDirectory->fMother && gDirectory->fMother->InheritsFrom(TDirectory::Class())) {
         ((TDirectory*)gDirectory->fMother)->cd();
         if (!TDirectory::Cd1(slash+1)) {
            gDirectory = savdir;
            delete [] path; return kFALSE;
         }
      }
      delete [] path; return kTRUE;
   }
   obj = gDirectory->Get(subdir);
   if (!obj) {
      ::Error("TDirectory::Cd","Unknown directory %s", subdir);
      delete [] path; return kFALSE;
   }

   //Check return object is a directory
   if (!obj->InheritsFrom(TDirectory::Class())) {
      ::Error("TDirectory::Cd","Object %s is not a directory", subdir);
      delete [] path; return kFALSE;
   }
   ((TDirectory*)obj)->cd();
   if (!TDirectory::Cd1(slash+1)) {
      gDirectory = savdir;
      delete [] path; return kFALSE;
   }
   delete [] path; return kTRUE;
}

//______________________________________________________________________________
void TDirectory::Clear(Option_t *)
{
//*-*-*-*Delete all objects from a Directory list-*-*-*-*-*
//*-*    =======================================

   if (fList) fList->Clear();

}

//______________________________________________________________________________
void TDirectory::Close(Option_t *)
{
//*-*-*-*Delete all objects from memory and directory structure itself-*-*-*-*
//       ============================================================

   if (!fList) return;

   TCollection::StartGarbageCollection();

   TDirectory *cursav = gDirectory;
   cd();

   if (cursav == this)
      cursav = (TDirectory*) fMother;

   // Save the directory key list and header
   //SaveSelf();
   Save();

   // Delete objects from directory list, this in turn, recursively closes all
   // sub-directories (that were allocated on the heap)
   SetBit(kCloseDirectory);
   fList->Delete();

   // Delete keys from key list (but don't delete the list header)
   if (fKeys) fKeys->Delete();

   if (cursav)
      cursav->cd();
   else {
      gFile = 0;
      if (this == gROOT)
         gDirectory = 0;
      else
         gDirectory = gROOT;
   }

   TCollection::EmptyGarbageCollection();
}

//______________________________________________________________________________
void TDirectory::DeleteAll(Option_t *)
{
//*-*-*-*-*-*-*-*-*Delete all objects from memory*-*-*-*-*-*-*-*-*-*
//                 ==============================

   fList->Delete();
}

//______________________________________________________________________________
void TDirectory::Delete(const char *namecycle)
{
//*-*-*-*-*-*-*-* Delete Objects or/and keys in a directory *-*-*-*-*-*-*-*
//*-*             =========================================
//   namecycle has the format name;cycle
//   namecycle = "" same as namecycle ="T*"
//   name  = * means all
//   cycle = * means all cycles (memory and keys)
//   cycle = "" or cycle = 9999 ==> apply to a memory object
//   When name=* use T* to delete subdirectories also
//
//   To delete one directory, you must specify the directory cycle,
//      eg.  file.Delete("dir1;1");
//
//   examples:
//     foo   : delete object named foo in memory
//     foo*  : delete all objects with a name starting with foo
//     foo;1 : delete cycle 1 of foo on file
//     foo;* : delete all cycles of foo on file and also from memory
//     *;2   : delete all objects on file having the cycle 2
//     *;*   : delete all objects from memory and file
//    T*;*   : delete all objects from memory and file and all subdirectories
//

   TDirectory *cursav = gDirectory;
   cd();

   Short_t  cycle;
   char     name[256];
   DecodeNameCycle(namecycle, name, cycle);

   Int_t deleteall    = 0;
   Int_t deletetree   = 0;
   if(strcmp(name,"*") == 0)   deleteall = 1;
   if(strcmp(name,"*T") == 0){ deleteall = 1; deletetree = 1;}
   if(strcmp(name,"T*") == 0){ deleteall = 1; deletetree = 1;}
   if(strlen(namecycle) == 0){ deleteall = 1; deletetree = 1;}
   TRegexp re(name,kTRUE);
   TString s;

//*-*---------------------Case of Object in memory---------------------
//                        ========================
   if (cycle >= 9999 ) {
      TNamed *idcur;
      TIter   next(fList);
      while ((idcur = (TNamed *) next())) {
         Int_t deleteOK = 0;
         s = idcur->GetName();
         if (deleteall || s.Index(re) != kNPOS) {
            deleteOK = 1;
            if (idcur->IsA() == TDirectory::Class()) {
               if (!deletetree && deleteall) deleteOK = 0;
            }
         }
         if (deleteOK != 0) idcur->Delete(name);
      }
   }
//*-*---------------------Case of Key---------------------
//                        ===========
   if (cycle != 9999 ) {
      if (IsWritable()) {
         TKey *key;
         TIter nextkey(GetListOfKeys());
         while ((key = (TKey *) nextkey())) {
            Int_t deleteOK = 0;
            s = key->GetName();
            if (deleteall || s.Index(re) != kNPOS) {
               if (cycle == key->GetCycle()) deleteOK = 1;
               if (cycle > 9999) deleteOK = 1;
               if (!strcmp(key->GetClassName(),"TDirectory")) {
                  if (!deletetree && deleteall) deleteOK = 0;
                  if (cycle == key->GetCycle()) deleteOK = 1;
               }
            }
            if (deleteOK) {
               key->Delete();
               fKeys->Remove(key);
               fModified = kTRUE;
            }
         }
         if (fModified) {
            WriteKeys();            //*-* Write new keys structure
            WriteDirHeader();       //*-* Write new directory header
            gFile->WriteFree();     //*-* Write new free segments list
            gFile->WriteHeader();   //*-* Write new file header
         }
      }
   }

   cursav->cd();

   return;
}

//______________________________________________________________________________
void TDirectory::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Fill Graphics Structure and Paint*-*-*-*-*-*-*-*-*-*
//*-*                    =================================
// Loop on all objects (memory or file) and all subdirectories
//

   fList->ForEach(TObject,Draw)(option);
}

//______________________________________________________________________________
void TDirectory::FillBuffer(char *&buffer)
{
//*-*-*-*-*-*-*-*-*-*-*-*Encode directory header into output buffer*-*-*-*-*-*
//*-*                    =========================================
   Version_t version = TDirectory::Class_Version();
   tobuf(buffer, version);
   fDatimeC.FillBuffer(buffer);
   fDatimeM.FillBuffer(buffer);
   tobuf(buffer, fNbytesKeys);
   tobuf(buffer, fNbytesName);
   tobuf(buffer, fSeekDir);
   tobuf(buffer, fSeekParent);
   tobuf(buffer, fSeekKeys);
   fUUID.FillBuffer(buffer);
}

//______________________________________________________________________________
TKey *TDirectory::FindKey(const char *keyname) const
{
   // Find key with name keyname in the current directory

   Short_t  cycle;
   char     name[256];

   DecodeNameCycle(keyname, name, cycle);
   return GetKey(name,cycle);
}

//______________________________________________________________________________
TKey *TDirectory::FindKeyAny(const char *keyname) const
{
   // Find key with name keyname in the current directory or
   // its subdirectories.
   // NOTE that If a key is found, the directory containing the key becomes
   // the current directory

   TDirectory *dirsav = gDirectory;
   Short_t  cycle;
   char     name[256];

   DecodeNameCycle(keyname, name, cycle);

   TIter next(GetListOfKeys());
   TKey *key;
   while ((key = (TKey *) next())) {
      if (!strcmp(name, key->GetName())) {
         if (cycle == 9999)             return key;
         if (cycle >= key->GetCycle())  return key;
      }
   }
   //try with subdirectories
   next.Reset();
   while ((key = (TKey *) next())) {
      if (!strcmp(key->GetClassName(),"TDirectory")) {
         ((TDirectory*)this)->cd(key->GetName());
         TKey *k = gDirectory->FindKeyAny(keyname);
         if (k) return k;
      }
   }
   dirsav->cd();
   return 0;
}

//______________________________________________________________________________
TObject *TDirectory::FindObject(const TObject *obj) const
{
   // Find object in the list of memory objects.

   return fList->FindObject(obj);
}

//______________________________________________________________________________
TObject *TDirectory::FindObject(const char *name) const
{
   // Find object by name in the list of memory objects.

   return fList->FindObject(name);
}

//______________________________________________________________________________
TObject *TDirectory::FindObjectAny(const char *aname) const
{
   // Find object by name in the list of memory objects of the current
   // directory or its sub-directories.
   // After this call the current directory is not changed.
   // To automatically set the current directory where the object is found,
   // use FindKeyAny(aname)->ReadObj().

   //object may be already in the list of objects in memory
   TObject *obj = fList->FindObject(aname);
   if (obj) return obj;

   TDirectory *dirsav = gDirectory;
   Short_t  cycle;
   char     name[256];

   DecodeNameCycle(aname, name, cycle);

   TIter next(GetListOfKeys());
   TKey *key;
   //may be a key in the current directory
   while ((key = (TKey *) next())) {
      if (!strcmp(name, key->GetName())) {
         if (cycle == 9999)             return key->ReadObj();
         if (cycle >= key->GetCycle())  return key->ReadObj();
      }
   }
   //try with subdirectories
   next.Reset();
   while ((key = (TKey *) next())) {
      if (!strcmp(key->GetClassName(),"TDirectory")) {
         ((TDirectory*)this)->cd(key->GetName());
         TKey *k = gDirectory->FindKeyAny(aname);
         if (k) {dirsav->cd(); return k->ReadObj();}
      }
   }
   dirsav->cd();
   return 0;
}

//______________________________________________________________________________
TObject *TDirectory::Get(const char *namecycle)
{
//*-*-*-*-*-*-*-* return pointer to object identified by namecycle*-*-*-*-*-*
//*-*             ================================================
//   namecycle has the format name;cycle
//   name  = * is illegal, cycle = * is illegal
//   cycle = "" or cycle = 9999 ==> apply to a memory object
//
//   examples:
//     foo   : get object named foo in memory
//             if object is not in memory, try with highest cycle from file
//     foo;1 : get cycle 1 of foo on file
//
// WARNING: Never use TDirectory::Get when namecycle is a directory itself.
//
//  VERY IMPORTANT NOTE:
//  In case the class of this object derives from TObject but not
//  as a first inheritance, one must cast the return value twice.
//  Example1: Normal case:
//      class MyClass : public TObject, public AnotherClass
//   then on return, one can do:
//    MyClass *obj = (MyClass*)directory->Get("some object of MyClass");
//
//  Example2: Special case:
//      class MyClass : public AnotherClass, public TObject
//   then on return, one must do:
//    MyClass *obj = (MyClass*)((void*)directory->Get("some object of MyClass");
//

   Short_t  cycle;
   char     name[256];

   TDirectory *cursav = gDirectory;
   cd();

   DecodeNameCycle(namecycle, name, cycle);
   char *namobj = name;
   Int_t nch = strlen(name);
   for (Int_t i = nch-1; i > 0; i--) {
      if (name[i] == '/') {
         name[i] = 0;
         cd(name);
         namobj = name + i + 1;
         name[i] = '/';
         break;
      }
   }
//*-*---------------------Case of Object in memory---------------------
//                        ========================
   TObject *idcur = fList->FindObject(namobj);
   if (idcur) {
      if (idcur==gDirectory && strlen(namobj)!=0) {
         // The object has the same name has the directory and
         // that's what we picked-up!  We just need to ignore
         // it ...
         idcur = 0;
      } else if (cycle == 9999) {
         cursav->cd();
         return idcur;
      } else {
         if (idcur->InheritsFrom(TCollection::Class()))
            idcur->Delete();  // delete also list elements
         delete idcur;
         idcur = 0;
      }
   }

//*-*---------------------Case of Key---------------------
//                        ===========
   TKey *key;
   TIter nextkey(gDirectory->GetListOfKeys());
   while ((key = (TKey *) nextkey())) {
     if (strcmp(namobj,key->GetName()) == 0) {
        if ((cycle == 9999) || (cycle == key->GetCycle())) {
           idcur = key->ReadObj();
           break;
        }
     }
   }
   cursav->cd();

   // If we are on a PROOF server try to get object from client
   if (!idcur && TClassTable::GetDict("TProofServ")) {
      if (gROOT->ProcessLineFast("TProofServ::IsActive()"))
         idcur = (TObject *) gROOT->ProcessLineFast(Form("TProofServ::This()->Get(\"%s\")", namecycle));
   }
   return idcur;
}

//______________________________________________________________________________
TKey *TDirectory::GetKey(const char *name, Short_t cycle) const
{
//*-*-*-*-*-*-*-*-*-*-*Return pointer to key with name,cycle*-*-*-*-*-*-*-*
//*-*                  =====================================
//  if cycle = 9999 returns highest cycle
//
   TKey *key;
   TIter next(GetListOfKeys());
   while ((key = (TKey *) next()))
      if (!strcmp(name, key->GetName())) {
         if (cycle == 9999)             return key;
         if (cycle >= key->GetCycle())  return key;
      }
   return 0;
}

//______________________________________________________________________________
const char *TDirectory::GetPath() const
{
   // Returns the full path of the directory. E.g. file:/dir1/dir2.
   // The returned path will be re-used by the next call to GetPath().

   static char *path = 0;
   const int kMAXDEPTH = 128;
   const TDirectory *d[kMAXDEPTH];
   const TDirectory *cur = this;
   int depth = 0, len = 0;

   d[depth++] = cur;
   len = strlen(cur->GetName()) + 1;  // +1 for the /

   while (cur->fMother && depth < kMAXDEPTH) {
      cur = (TDirectory *)cur->fMother;
      d[depth++] = cur;
      len += strlen(cur->GetName()) + 1;
   }

   if (path) delete [] path;
   path = new char[len+2];

   for (int i = depth-1; i >= 0; i--) {
      if (i == depth-1) {    // file or TROOT name
         strcpy(path, d[i]->GetName());
         strcat(path, ":");
         if (i == 0) strcat(path, "/");
      } else {
         strcat(path, "/");
         strcat(path, d[i]->GetName());
      }
   }

   return path;
}

//______________________________________________________________________________
TDirectory *TDirectory::mkdir(const char *name, const char *title)
{
   // Create a sub-directory and return a pointer to the created directory.
   // Returns 0 in case of error.
   // Returns 0 if a directory with the same name already exists.
   // Note that the directory name cannot contain slashes.

   if (!name || !title || !strlen(name)) return 0;
   if (!strlen(title)) title = name;
   if (strchr(name,'/')) {
      ::Error("TDirectory::mkdir","directory name cannot contain a slash", name);
      return 0;
   }
   if (GetKey(name)) {
      Error("mkdir","Directory %s exists already",name);
      return 0;
   }

   TDirectory *cursav = gDirectory;
   cd();

   TDirectory *newdir = new TDirectory(name, title);

   cursav->cd();
   return newdir;
}

//______________________________________________________________________________
void TDirectory::ls(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*List Directory contents*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    =======================
//  Indentation is used to identify the directory tree
//  Subdirectories are listed first, then objects in memory, then objects on the file
//
//  The option can has the following format:
//     [-d |-m][<regexp>]
//  Option -d means: only list objects in the file
//         -m means: only list objects in memory
//  The <regexp> will be used to match the name of the objects.
//  By default memory and disk objects are listed.
//
   TROOT::IndentLevel();
   cout <<ClassName()<<"*\t\t"<<GetName()<<"\t"<<GetTitle()<<endl;
   TROOT::IncreaseDirLevel();

   TString opta = option;
   TString opt  = opta.Strip(TString::kBoth);
   Bool_t memobj  = kTRUE;
   Bool_t diskobj = kTRUE;
   TString reg = "*";
   if (opt.BeginsWith("-m")) {
      diskobj = kFALSE;
      if (opt.Length() > 2)
         reg = opt(2,opt.Length());
   } else if (opt.BeginsWith("-d")) {
      memobj  = kFALSE;
      if (opt.Length() > 2)
         reg = opt(2,opt.Length());
   } else if (!opt.IsNull())
      reg = opt;

   TRegexp re(reg, kTRUE);

   if (memobj) {
      TObject *obj;
      TIter nextobj(fList);
      while ((obj = (TObject *) nextobj())) {
         TString s = obj->GetName();
         if (s.Index(re) == kNPOS) continue;
         obj->ls(option);            //*-* Loop on all the objects in memory
      }
   }

   if (diskobj) {
      TKey *key;
      TIter next(GetListOfKeys());
      while ((key = (TKey *) next())) {
         TString s = key->GetName();
         if (s.Index(re) == kNPOS) continue;
         key->ls();                 //*-* Loop on all the keys
      }
   }
   TROOT::DecreaseDirLevel();
}

//______________________________________________________________________________
void TDirectory::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Paint all objects in the directory *-*-*-*-*-*-*-*
//*-*                    ==================================
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   fList->ForEach(TObject,Paint)(option);
}

//______________________________________________________________________________
void TDirectory::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*Print all objects in the directory *-*-*-*-*-*-*-*
//*-*                    ==================================
//

   fList->ForEach(TObject,Print)(option);
}

//______________________________________________________________________________
void TDirectory::Purge(Short_t)
{
   // Purge lowest key cycles in a directory.
   // By default, only the highest cycle of a key is kept. Keys for which
   // the "KEEP" flag has been set are not removed. See TKey::Keep().

   if (!IsWritable()) return;

   TDirectory *cursav = gDirectory;
   cd();

   TKey  *key;
   TIter  prev(GetListOfKeys(), kIterBackward);

   while ((key = (TKey*)prev())) {      // reverse loop on keys
      TKey *keyprev = (TKey*)GetListOfKeys()->Before(key);
      if (!keyprev) break;
      if (key->GetKeep() == 0) {
         if (strcmp(key->GetName(), keyprev->GetName()) == 0) key->Delete();
      }
   }
   if (fModified) {
      WriteKeys();                   // Write new keys structure
      WriteDirHeader();              // Write new directory header
      gFile->WriteFree();            // Write new free segments list
      gFile->WriteHeader();          // Write new file header
   }
   cursav->cd();
}

//______________________________________________________________________________
void TDirectory::pwd() const
{
   // Print the path of the directory.

   Printf("%s", GetPath());
}

//______________________________________________________________________________
void TDirectory::ReadAll(Option_t *)
{
   // Read objects from a ROOT db file directory into memory.
   // If an object is already in memory, the memory copy is deleted
   // and the object is again read from the file.

   TDirectory *cursav = gDirectory;
   cd();

   TKey *key;
   TIter next(GetListOfKeys());
   while ((key = (TKey *) next())) {
      TNamed *thing = (TNamed*)gDirectory->GetList()->FindObject(key->GetName());
      if (thing) { delete thing; }
      key->ReadObj();
   }

   cursav->cd();
}

//______________________________________________________________________________
Int_t TDirectory::ReadKeys()
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Read the KEYS linked list*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      =========================
//  Every directory has a linked list (fKeys). This linked list has been
//  written on the file via WriteKeys as a single data record.
//
//  It is interesting to call this function in the following situation.
//  Assume another process1 is connecting this directory in Update mode
//    -Process1 is adding/updating objects in this directory
//    -You want to see the latest status from process1.
//  Example Process1:
//    obj1.Write();
//    obj2.Write();
//    gDirectory->SaveSelf();
//
//  Example Process2
//    gDirectory->ReadKeys();
//    obj1->Draw();
//
//  This is an efficient way (without opening/closing files) to view
//  the latest updates of a file being modified by another process
//  as it is typically the case in a data acquisition system.

   TDirectory *cursav = gDirectory;
   cd();

   fKeys->Delete();
   //In case directory was updated by another process, read new
   //position for the keys
   Int_t nbytes = fNbytesName + TDirectory::Sizeof();
   char *header       = new char[nbytes];
   char *buffer       = header;
   fFile->Seek(fSeekDir);
   fFile->ReadBuffer(buffer,nbytes);
   buffer += fNbytesName;
   Version_t versiondir;
   frombuf(buffer,&versiondir);
   fDatimeC.ReadBuffer(buffer);
   fDatimeM.ReadBuffer(buffer);
   frombuf(buffer, &fNbytesKeys);
   frombuf(buffer, &fNbytesName);
   frombuf(buffer, &fSeekDir);
   frombuf(buffer, &fSeekParent);
   frombuf(buffer, &fSeekKeys);
   delete [] header;

   Int_t nkeys = 0;
   if ( fSeekKeys >  0) {
      TKey *headerkey    = new TKey(fSeekKeys,fNbytesKeys);
      headerkey->ReadFile();
      buffer = headerkey->GetBuffer();
      headerkey->ReadBuffer(buffer);
      TKey *key;
      frombuf(buffer, &nkeys);
      for (Int_t i = 0; i < nkeys; i++) {
         key = new TKey();
         key->ReadBuffer(buffer);
         fKeys->Add(key);
      }
      delete headerkey;
   }

   cursav->cd();
   return nkeys;
}

//______________________________________________________________________________
void TDirectory::RecursiveRemove(TObject *obj)
{
//*-*-*-*-*-*-*-*Recursively remove object from a Directory*-*-*-*-*-*-*-*
//*-*            =========================================

   fList->RecursiveRemove(obj);
}

//______________________________________________________________________________
void TDirectory::Save()
{
//*-*-*-*-*-*-*-*-*-*Save recursively all directory keys and headers-*-*-*-*-*
//*-*                ===============================================

   TDirectory *cursav = gDirectory;
   cd();

   SaveSelf();

   // recursively save all sub-directories
   if (fList) {
      TObject *idcur;
      TIter    next(fList);
      while ((idcur = next())) {
         if (idcur->InheritsFrom(TDirectory::Class())) {
            TDirectory *dir = (TDirectory*)idcur;
            dir->Save();
         }
      }
   }
   cursav->cd();
}

//______________________________________________________________________________
void TDirectory::SaveSelf(Bool_t force)
{
//*-*-*-*-*-*-*-*-*-*Save Directory keys and header*-*-*-*-*-*-*-*-*-*-*-*
//*-*                ==============================
//  If the directory has been modified (fModified set), write the keys
//  and the directory header. This function assumes the cd is correctly set.
//
//  It is recommended to use this function in the following situation:
//  Assume a process1 using a directory in Update mode
//    -New objects or modified objects have been written to the directory
//    -You do not want to close the file
//    -You want your changes be visible from another process2 already connected
//     to this directory in read mode
//    -Call this function
//    -In process2, use TDirectory::ReadKeys to refresh the directory

   if (IsWritable() && (fModified || force)) {
      TFree *f1 = (TFree*) fFile->GetListOfFree()->First();
      if (f1) {
         WriteKeys();          //*-*- Write keys record
         WriteDirHeader();     //*-*- Update directory record
      }
   }
}

//______________________________________________________________________________
Int_t TDirectory::Sizeof() const
{
//*-*-*-*-*-*-*Return the size in bytes of the directory header*-*-*-*-*-*-*
//*-*          ================================================
   Int_t nbytes = sizeof(Version_t);
   nbytes     += fDatimeC.Sizeof();
   nbytes     += fDatimeM.Sizeof();
   nbytes     += sizeof fNbytesKeys;
   nbytes     += sizeof fNbytesName;
   nbytes     += sizeof fSeekDir;
   nbytes     += sizeof fSeekParent;
   nbytes     += sizeof fSeekKeys;
   nbytes     += fUUID.Sizeof();
   return nbytes;
}


//_______________________________________________________________________
void TDirectory::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================
   if (b.IsReading()) {
      Build();
      if (fFile && fFile->IsWritable()) fWritable = kTRUE;
      Version_t v = b.ReadVersion();
      fDatimeC.Streamer(b);
      fDatimeM.Streamer(b);
      b >> fNbytesKeys;
      b >> fNbytesName;
      b >> fSeekDir;
      b >> fSeekParent;
      b >> fSeekKeys;
      if (v > 1) {
         fUUID.Streamer(b);
      }
      gROOT->GetUUIDs()->AddUUID(fUUID,this);
      if (fSeekKeys) ReadKeys();
   } else {
      b.WriteVersion(TDirectory::IsA());
      fDatimeC.Streamer(b);
      fDatimeM.Streamer(b);
      b << fNbytesKeys;
      b << fNbytesName;
      b << fSeekDir;
      b << fSeekParent;
      b << fSeekKeys;
      fUUID.Streamer(b);
   }
}

//______________________________________________________________________________
Int_t TDirectory::Write(const char *, Int_t opt, Int_t bufsiz)
{
   // Write all objects in memory to disk.
   // Loop on all objects in memory (including subdirectories).
   // A new key is created in the KEYS linked list for each object.
   // For allowed options see TObject::Write().
   // The directory header info is rewritten on the directory header record

   if (!IsWritable()) return 0;
   TDirectory *cursav = gDirectory;
   cd();

   // Loop on all objects (including subdirs)
   TIter next(fList);
   TObject *obj;
   Int_t nbytes = 0;
   while ((obj=next())) {
      nbytes += obj->Write(0,opt,bufsiz);
   }
   SaveSelf(kTRUE);   // force save itself

   cursav->cd();
   return nbytes;
}

//______________________________________________________________________________
void TDirectory::WriteDirHeader()
{
//*-*-*-*-*-*-*-*-*-*-*Overwrite the Directory header record*-*-*-*-*-*-*-*-*
//*-*                  =====================================
   Int_t nbytes  = TDirectory::Sizeof();  //Warning ! TFile has a Sizeof()
   char * header = new char[nbytes];
   char * buffer = header;
   fDatimeM.Set();
   TDirectory::FillBuffer(buffer);
   Seek_t pointer= fSeekDir + fNbytesName; // do not overwrite the name/title part
   fModified     = kFALSE;
   gFile->Seek(pointer);
   gFile->WriteBuffer(header, nbytes);
   gFile->Flush();
   delete [] header;
}

//______________________________________________________________________________
void TDirectory::WriteKeys()
{
//*-*-*-*-*-*-*-*-*-*-*-*Write KEYS linked list on the file *-*-*-*-*-*-*-*
//*-*                    ==================================
//  The linked list of keys (fKeys) is written as a single data record
//

//*-* Delete the old keys structure if it exists
   if (fSeekKeys != 0) {
      gFile->MakeFree(fSeekKeys, fSeekKeys + fNbytesKeys -1);
   }
//*-* Write new keys record
   TIter next(fKeys);
   TKey *key;
   Int_t nkeys  = fKeys->GetSize();
   Int_t nbytes = sizeof nkeys;          //*-* Compute size of all keys
   while ((key = (TKey*)next())) {
      nbytes += key->Sizeof();
   }
   TKey *headerkey  = new TKey(fName,fTitle,IsA(),nbytes);
   if (headerkey->GetSeekKey() == 0) {
      delete headerkey;
      return;
   }
   char *buffer = headerkey->GetBuffer();
   next.Reset();
   tobuf(buffer, nkeys);
   while ((key = (TKey*)next())) {
      key->FillBuffer(buffer);
   }

   fSeekKeys     = headerkey->GetSeekKey();
   fNbytesKeys   = headerkey->GetNbytes();
   headerkey->WriteFile();
   delete headerkey;
}

//______________________________________________________________________________
void TDirectory::EncodeNameCycle(char *buffer, const char *name, Short_t cycle)
{
   // Encode the name and cycle into buffer like: "aap;2".

   if (cycle == 9999)
      strcpy(buffer, name);
   else
      sprintf(buffer, "%s;%d", name, cycle);
}

//______________________________________________________________________________
void TDirectory::DecodeNameCycle(const char *buffer, char *name, Short_t &cycle)
{
   // Decode a namecycle "aap;2" into name "aap" and cycle "2".

   cycle     = 9999;
   Int_t nch = strlen(buffer);
   for (Int_t i = 0; i < nch; i++) {
      if (buffer[i] != ';')
         name[i] = buffer[i];
      else {
         name[i] = 0;
         if (i < nch-1 )
            if (buffer[i+1] == '*') {
               cycle = 10000;
               return;
            }
         sscanf(buffer+i+1, "%hd", &cycle);
         return;
      }
   }
   name[nch] = 0;
}
