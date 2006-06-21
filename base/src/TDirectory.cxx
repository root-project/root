// @(#)root/base:$Name:  $:$Id: TDirectory.cxx,v 1.85 2006/06/20 18:17:34 pcanal Exp $
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
#include "TFile.h"
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
#include "TClass.h"
#include "TRegexp.h"
#include "TProcessUUID.h"
#include "TVirtualMutex.h"

TDirectory    *gDirectory;      //Pointer to current directory in memory

const UInt_t kIsBigFile = BIT(16);
const Int_t  kMaxLen = 2048;

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
TDirectory::TDirectory() : TNamed(), fMother(0)
{
//*-*-*-*-*-*-*-*-*-*-*-*Directory default constructor-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    =============================
   fFile = 0;
   fList = 0;
   fKeys = 0;
   fWritable = kFALSE;
   fBufferSize = 0;
}

//______________________________________________________________________________
TDirectory::TDirectory(const char *name, const char *title, Option_t *classname, TDirectory* initMotherDir)
           : TNamed(name, title), fMother(0)
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
   if (initMotherDir==0) initMotherDir = gDirectory;

   if (strchr(name,'/')) {
      ::Error("TDirectory::TDirectory","directory name (%s) cannot contain a slash", name);
      gDirectory = 0;
      return;
   }
   if (strlen(GetName()) == 0) {
      ::Error("TDirectory::TDirectory","directory name cannot be \"\"");
      gDirectory = 0;
      return;
   }

   Build(initMotherDir ? initMotherDir->GetFile() : 0, initMotherDir);

   TDirectory* motherdir = GetMotherDir();
   TFile* f = GetFile();

   if ((motherdir==0) || (f==0)) return;
   if (!f->IsWritable()) return; //*-* in case of a directory in memory
   if (motherdir->GetKey(name)) {
      Error("TDirectory","An object with name %s exists already", name);
      return;
   }
   TClass *cl = IsA();
   if (strlen(classname) != 0) cl = gROOT->GetClass(classname);

   if (!cl) {
      Error("TDirectory","Invalid class name: %s",classname);
      return;
   }
   fBufferSize  = 0;
   fWritable    = kTRUE;
   
   if (f->IsBinary()) {
      fSeekParent  = f->GetSeekDir();
      Int_t nbytes = TDirectory::Sizeof();
      TKey *key    = new TKey(fName,fTitle,cl,nbytes,motherdir);
      fNbytesName  = key->GetKeylen();
      fSeekDir     = key->GetSeekKey();
      if (fSeekDir == 0) return;
      char *buffer = key->GetBuffer();
      TDirectory::FillBuffer(buffer);
      Int_t cycle = motherdir->AppendKey(key);
      key->WriteFile(cycle);
   } else {
      fSeekParent  = 0;
      fNbytesName  = 0;
      fSeekDir     = f->DirCreateEntry(this);
      if (fSeekDir == 0) return;
   }
   
   fModified = kFALSE;
   R__LOCKGUARD2(gROOTMutex);
   gROOT->GetUUIDs()->AddUUID(fUUID,this);
}

//______________________________________________________________________________
TDirectory::TDirectory(const TDirectory &directory) : TNamed(directory)
{
   // Copy constructor.
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

   if (gDirectory==this) {
      if (gFile!=this) gFile->cd();
      else {
         gDirectory = gROOT;
         gFile = 0;
      }
   }

   TDirectory *mom = GetMotherDir();

   if (mom!=0) {
      if (!mom->TestBit(TDirectory::kCloseDirectory)) {
         mom->GetList()->Remove(this);
      }
   }

   if (gDebug)
      Info("~TDirectory", "dtor called for %s", GetName());
}

//______________________________________________________________________________
void TDirectory::Append(TObject *obj)
{
   // Append object to this directory.

   if (obj == 0 || fList == 0) return;
   fList->Add(obj);
   obj->SetBit(kMustCleanup);
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

   key->SetMotherDir(this);

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
   // Browse the content of the directory.

   Char_t name[kMaxLen];

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
void TDirectory::Build(TFile* motherFile, TDirectory* motherDir)
{
//*-*-*-*-*-*-*-*-*-*-*-*Initialise directory to defaults*-*-*-*-*-*-*-*-*-*
//*-*                    ================================

   // If directory is created via default ctor (when dir is read from file)
   // don't add it here to the directory since its name is not yet known.
   // It will be added to the directory in TKey::ReadObj().

   if (motherDir && strlen(GetName()) != 0) motherDir->Append(this);

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
   fMother     = motherDir;
   fFile       = motherFile ? motherFile : gFile;
   SetBit(kCanDelete);
}

//______________________________________________________________________________
TDirectory *TDirectory::GetDirectory(const char *apath,
                                     Bool_t printError, const char *funcname)
{
   // Find a directory using apath.
   // It apath is null or empty, returns "this" directory.
   // Otherwie use apath to find a directory.
   // The absolute path syntax is:
   //    file.root:/dir1/dir2
   // where file.root is the file and /dir1/dir2 the desired subdirectory
   // in the file. Relative syntax is relative to "this" directory. E.g:
   // ../aa.
   // Returns 0 in case path does not exist.
   // If printError is true, use Error with 'funcname' to issue an error message.

   Int_t nch = 0;
   if (apath) nch = strlen(apath);
   if (!nch) {
      return this;
   }

   if (funcname==0 || strlen(funcname)==0) funcname = "GetDirectory";

   TDirectory *result = this;

   char *path = new char[nch+1]; path[0] = 0;
   if (nch) strcpy(path,apath);
   char *s = (char*)strchr(path, ':');
   if (s) {
      *s = '\0';
      R__LOCKGUARD2(gROOTMutex);
      TDirectory *f = (TDirectory *)gROOT->GetListOfFiles()->FindObject(path);
      if (!f && !strcmp(gROOT->GetName(), path)) f = gROOT;
      if (s) *s = ':';
      if (f) {
         result = f;
         if (s && *(s+1)) result = f->GetDirectory(s+1,printError,funcname);
         delete [] path; return result;
      } else {
         if (printError) Error(funcname, "No such file %s", path);
         delete [] path; return 0;
      }
   }

   // path starts with a slash (assumes current file)
   if (path[0] == '/') {
      TDirectory *td = fFile;
      if (!fFile) td = gROOT;
      result = td->GetDirectory(path+1,printError,funcname);
      delete [] path; return result;
   }

   TObject *obj;
   char *slash = (char*)strchr(path,'/');
   if (!slash) {                     // we are at the lowest level
      if (!strcmp(path, "..")) {
         result = GetMotherDir();
         delete [] path; return result;
      }
      obj = Get(path);
      if (!obj) {
         if (printError) Error(funcname,"Unknown directory %s", path);
         delete [] path; return 0;
      }

      //Check return object is a directory
      if (!obj->InheritsFrom(TDirectory::Class())) {
         if (printError) Error(funcname,"Object %s is not a directory", path);
         delete [] path; return 0;
      }
      delete [] path; return (TDirectory*)obj;
   }

   char subdir[kMaxLen];
   strcpy(subdir,path);
   slash = (char*)strchr(subdir,'/');
   *slash = 0;
   //Get object with path from current directory/file
   if (!strcmp(subdir, "..")) {
      TDirectory* mom = GetMotherDir();
      if (mom)
         result = mom->GetDirectory(slash+1,printError,funcname);
      delete [] path; return result;
   }
   obj = Get(subdir);
   if (!obj) {
      if (printError) Error(funcname,"Unknown directory %s", subdir);
      delete [] path; return 0;
   }

   //Check return object is a directory
   if (!obj->InheritsFrom(TDirectory::Class())) {
      if (printError) Error(funcname,"Object %s is not a directory", subdir);
      delete [] path; return 0;
   }
   result = ((TDirectory*)obj)->GetDirectory(slash+1,printError,funcname);
   delete [] path; return result;
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
      gFile      = GetFile();
      return kTRUE;
   }

   TDirectory *where = GetDirectory(apath,kTRUE,"cd");
   if (where) {
      where->cd();
      return kTRUE;
   }
   return kFALSE;
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

   TDirectory *where = gDirectory->GetDirectory(apath,kTRUE,"Cd");
   if (where) {
      where->cd();
      return kTRUE;
   }
   return kFALSE;
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
      cursav = GetMotherDir();

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

   SetBit(kCloseDirectory);
   fList->Delete();
   ResetBit(kCloseDirectory);
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

   if (gDebug)
     Info("Delete","Call for this = %s namecycle = %s",
               GetName(), (namecycle ? namecycle : "null"));

   TDirectory::TContext ctxt(gDirectory, this);
   Short_t  cycle;
   char     name[kMaxLen];
   DecodeNameCycle(namecycle, name, cycle);

   Int_t deleteall    = 0;
   Int_t deletetree   = 0;
   if(strcmp(name,"*") == 0)   deleteall = 1;
   if(strcmp(name,"*T") == 0){ deleteall = 1; deletetree = 1;}
   if(strcmp(name,"T*") == 0){ deleteall = 1; deletetree = 1;}
   if(strlen(namecycle) == 0){ deleteall = 1; deletetree = 1;}
   TRegexp re(name,kTRUE);
   TString s;
   Int_t deleteOK = 0;

//*-*---------------------Case of Object in memory---------------------
//                        ========================
   if (cycle >= 9999 ) {
      TNamed *idcur;
      TIter   next(fList);
      while ((idcur = (TNamed *) next())) {
         deleteOK = 0;
         s = idcur->GetName();
         if (deleteall || s.Index(re) != kNPOS) {
            deleteOK = 1;
            if (idcur->IsA() == TDirectory::Class()) {
               deleteOK = 2;
               if (!deletetree && deleteall) deleteOK = 0;
            }
         }
         if (deleteOK != 0) {
            fList->Remove(idcur);
            if (deleteOK==2) {
               // read subdirectories to correctly delete them
               if (deletetree)
                  ((TDirectory*) idcur)->ReadAll("dirs");
               idcur->Delete(deletetree ? "T*;*" : "*");
               delete idcur;
            } else
               idcur->Delete(name);
         }
      }
//      if (deleteOK == 2) {
//         Info("Delete","Dir:%x %s", fList->FindObject(name), name);
//         delete fList->FindObject(name); //deleting a TDirectory
//      }
   }
//*-*---------------------Case of Key---------------------
//                        ===========
   if (cycle != 9999 ) {
      if (IsWritable()) {
         TKey *key;
         TIter nextkey(GetListOfKeys());
         while ((key = (TKey *) nextkey())) {
            deleteOK = 0;
            s = key->GetName();
            if (deleteall || s.Index(re) != kNPOS) {
               if (cycle == key->GetCycle()) deleteOK = 1;
               if (cycle > 9999) deleteOK = 1;
               if (!strcmp(key->GetClassName(),"TDirectory")) {
                  deleteOK = 2; 
                  if (!deletetree && deleteall) deleteOK = 0;
                  if (cycle == key->GetCycle()) deleteOK = 2;
               }
            }
            if (deleteOK) {
               if (deleteOK==2) {
                  // read directory with subdirectories to correctly delete and free key structure 
                  TDirectory* dir = GetDirectory(key->GetName(), kTRUE, "Delete");
                  if (dir!=0) {
                     dir->Delete("T*;*");
                     fList->Remove(dir);
                     delete dir;
                  }
               } 
                
               key->Delete();
               fKeys->Remove(key);
               fModified = kTRUE;
               delete key;
            }
         }
         TFile* f = GetFile();
         if (fModified && (f!=0)) {
            WriteKeys();            //*-* Write new keys structure
            WriteDirHeader();       //*-* Write new directory header
            f->WriteFree();     //*-* Write new free segments list
            f->WriteHeader();   //*-* Write new file header
         }
      }
   }
}

//______________________________________________________________________________
void TDirectory::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Fill Graphics Structure and Paint*-*-*-*-*-*-*-*-*-*
//*-*                    =================================
// Loop on all objects (memory or file) and all subdirectories
//

   fList->R__FOR_EACH(TObject,Draw)(option);
}

//______________________________________________________________________________
void TDirectory::FillBuffer(char *&buffer)
{
//*-*-*-*-*-*-*-*-*-*-*-*Encode directory header into output buffer*-*-*-*-*-*
//*-*                    =========================================
   Version_t version = TDirectory::Class_Version();
   if (fSeekKeys > TFile::kStartBigFile) version += 1000;
   tobuf(buffer, version);
   fDatimeC.FillBuffer(buffer);
   fDatimeM.FillBuffer(buffer);
   tobuf(buffer, fNbytesKeys);
   tobuf(buffer, fNbytesName);
   if (version > 1000) {
      tobuf(buffer, fSeekDir);
      tobuf(buffer, fSeekParent);
      tobuf(buffer, fSeekKeys);
   } else {
      tobuf(buffer, (Int_t)fSeekDir);
      tobuf(buffer, (Int_t)fSeekParent);
      tobuf(buffer, (Int_t)fSeekKeys);
   }
   fUUID.FillBuffer(buffer);
   if (fFile && fFile->GetVersion() < 40000) return;
   if (version <=1000) for (Int_t i=0;i<3;i++) tobuf(buffer,Int_t(0));
}

//______________________________________________________________________________
TKey *TDirectory::FindKey(const char *keyname) const
{
   // Find key with name keyname in the current directory

   Short_t  cycle;
   char     name[kMaxLen];

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
   char     name[kMaxLen];

   DecodeNameCycle(keyname, name, cycle);

   TIter next(GetListOfKeys());
   TKey *key;
   while ((key = (TKey *) next())) {
      if (!strcmp(name, key->GetName()))
         if ((cycle == 9999) || (cycle >= key->GetCycle()))  {
            ((TDirectory*)this)->cd(); // may be we should not make cd ???
            return key;
         }
   }
   //try with subdirectories
   next.Reset();
   while ((key = (TKey *) next())) {
      if (!strcmp(key->GetClassName(),"TDirectory")) {
         TDirectory* subdir =
           ((TDirectory*)this)->GetDirectory(key->GetName(), kTRUE, "FindKeyAny");
         TKey *k = (subdir!=0) ? subdir->FindKeyAny(keyname) : 0;
         if (k) return k;
      }
   }
   if (dirsav) dirsav->cd();
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
   char     name[kMaxLen];

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
         TDirectory* subdir =
           ((TDirectory*)this)->GetDirectory(key->GetName(), kTRUE, "FindKeyAny");
         TKey *k = subdir==0 ? 0 : subdir->FindKeyAny(aname);
         if (k) { if (dirsav) dirsav->cd(); return k->ReadObj();}
      }
   }
   if (dirsav) dirsav->cd();
   return 0;
}

//______________________________________________________________________________
TObject *TDirectory::Get(const char *namecycle)
{
//  return pointer to object identified by namecycle
//
//   namecycle has the format name;cycle
//   name  = * is illegal, cycle = * is illegal
//   cycle = "" or cycle = 9999 ==> apply to a memory object
//
//   examples:
//     foo   : get object named foo in memory
//             if object is not in memory, try with highest cycle from file
//     foo;1 : get cycle 1 of foo on file
//
//  The retrieved object should in principle derive from TObject.
//  If not, the function TDirectory::GetObject should be called.
//  However, this function will still work for a non-TObject, providing that
//  the calling application cast the return type to the correct type (which
//  is the actual type of the object).
//
//  NOTE:
//  The method GetObject offer better protection and avoid the need
//  for any cast:
//      MyClass *obj;
//      directory->GetObject("some object",obj);
//      if (obj) { ... the object exist and inherits from MyClass ... }
//
//  VERY IMPORTANT NOTE:
//  In case the class of this object derives from TObject but not
//  as a first inheritance, one must use dynamic_cast<>().
//  Example 1: Normal case:
//      class MyClass : public TObject, public AnotherClass
//   then on return, one can do:
//      MyClass *obj = (MyClass*)directory->Get("some object of MyClass");
//
//  Example 2: Special case:
//      class MyClass : public AnotherClass, public TObject
//  then on return, one must do:
//      MyClass *obj = dynamic_cast<MyClass*>(directory->Get("some object of MyClass"));
//
//  Of course, dynamic_cast<> can also be used in the example 1.
//

   Short_t  cycle;
   char     name[kMaxLen];

   DecodeNameCycle(namecycle, name, cycle);
   char *namobj = name;
   Int_t nch = strlen(name);
   for (Int_t i = nch-1; i > 0; i--) {
      if (name[i] == '/') {
         name[i] = 0;
         TDirectory* dirToSearch=GetDirectory(name);
         namobj = name + i + 1;
         name[i] = '/';
         return dirToSearch?dirToSearch->Get(namobj):0;
      }
   }

//*-*---------------------Case of Object in memory---------------------
//                        ========================
   TObject *idcur = fList->FindObject(namobj);
   if (idcur) {
      if (idcur==this && strlen(namobj)!=0) {
         // The object has the same name has the directory and
         // that's what we picked-up!  We just need to ignore
         // it ...
         idcur = 0;
      } else if (cycle == 9999) {
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
   TIter nextkey(GetListOfKeys());
   while ((key = (TKey *) nextkey())) {
      if (strcmp(namobj,key->GetName()) == 0) {
         if ((cycle == 9999) || (cycle == key->GetCycle())) {
            TDirectory::TContext ctxt(this);
            idcur = key->ReadObj();
            break;
         }
      }
   }

   return idcur;
}

//______________________________________________________________________________
void *TDirectory::GetObjectUnchecked(const char *namecycle)
{
// return pointer to object identified by namecycle.
// The returned object may or may not derive from TObject.
//
//   namecycle has the format name;cycle
//   name  = * is illegal, cycle = * is illegal
//   cycle = "" or cycle = 9999 ==> apply to a memory object
//
//  VERY IMPORTANT NOTE:
//  The calling application must cast the returned object to
//  the final type, eg
//      MyClass *obj = (MyClass*)directory->GetObject("some object of MyClass");

   return GetObjectChecked(namecycle,(TClass*)0);
}

//_________________________________________________________________________________
void *TDirectory::GetObjectChecked(const char *namecycle, const char* classname)
{
// See documentation of TDirectory::GetObjectCheck(const char *namecycle, const TClass *cl)

   return GetObjectChecked(namecycle,ROOT::GetROOT()->GetClass(classname));
}


//____________________________________________________________________________
void *TDirectory::GetObjectChecked(const char *namecycle, const TClass* expectedClass)
{
// return pointer to object identified by namecycle if and only if the actual
// object is a type suitable to be stored as a pointer to a "expectedClass"
// If expectedClass is null, no check is performed.
//
//   namecycle has the format name;cycle
//   name  = * is illegal, cycle = * is illegal
//   cycle = "" or cycle = 9999 ==> apply to a memory object
//
//  VERY IMPORTANT NOTE:
//  The calling application must cast the returned pointer to
//  the type described by the 2 arguments (i.e. cl):
//      MyClass *obj = (MyClass*)directory->GetObjectChecked("some object of MyClass","MyClass"));
//
//  Note: We recommend using the method TDirectory::GetObject:
//      MyClass *obj = 0;
//      directory->GetObject("some object inheriting from MyClass",obj);
//      if (obj) { ... we found what we are looking for ... }

   Short_t  cycle;
   char     name[kMaxLen];

   DecodeNameCycle(namecycle, name, cycle);
   char *namobj = name;
   Int_t nch = strlen(name);
   for (Int_t i = nch-1; i > 0; i--) {
      if (name[i] == '/') {
         name[i] = 0;
         TDirectory* dirToSearch=GetDirectory(name);
         namobj = name + i + 1;
         name[i] = '/';
         if (dirToSearch) {
            return dirToSearch->GetObjectChecked(namobj, expectedClass);
         } else {
            return 0;
         }
      }
   }

//*-*---------------------Case of Object in memory---------------------
//                        ========================
   if (expectedClass==0 || expectedClass->InheritsFrom(TObject::Class())) {
      TObject *objcur = fList->FindObject(namobj);
      if (objcur) {
         if (objcur==this && strlen(namobj)!=0) {
            // The object has the same name has the directory and
            // that's what we picked-up!  We just need to ignore
            // it ...
            objcur = 0;
         } else if (cycle == 9999) {
            // Check type
            if (expectedClass && objcur->IsA()->GetBaseClassOffset(expectedClass) == -1) return 0;
            else return objcur;
         } else {
            if (objcur->InheritsFrom(TCollection::Class()))
               objcur->Delete();  // delete also list elements
            delete objcur;
            objcur = 0;
         }
      }
   }

//*-*---------------------Case of Key---------------------
//                        ===========
   void *idcur = 0;
   TKey *key;
   TIter nextkey(GetListOfKeys());
   while ((key = (TKey *) nextkey())) {
      if (strcmp(namobj,key->GetName()) == 0) {
         if ((cycle == 9999) || (cycle == key->GetCycle())) {
            TDirectory::TContext ctxt(this);
            idcur = key->ReadObjectAny(expectedClass);
            break;
         }
      }
   }

   return idcur;
}

//______________________________________________________________________________
Int_t TDirectory::GetBufferSize() const
{
   // Return the buffer size to create new TKeys.
   // If the stored fBufferSize is null, the value returned is the average
   // buffer size of objects in the file so far.

   if (fBufferSize <= 0) return fFile->GetBestBuffer();
   else                  return fBufferSize;
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
const char *TDirectory::GetPathStatic() const
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
const char *TDirectory::GetPath() const
{
   // Returns the full path of the directory. E.g. file:/dir1/dir2.
   // The returned path will be re-used by the next call to GetPath().

   //
   TString* buf = &(const_cast<TDirectory*>(this)->fPathBuffer);

   FillFullPath(*buf);
   if (GetMotherDir()==0) // case of file
      buf->Append("/");

   return buf->Data();
}

//______________________________________________________________________________
void TDirectory::FillFullPath(TString& buf) const
{
   // recursive method to fill full path for directory

   TDirectory* mom = GetMotherDir();
   if (mom!=0) {
      mom->FillFullPath(buf);
      buf += "/";
      buf += GetName();
   } else {
      buf = GetName();
      buf +=":";
   }
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
      ::Error("TDirectory::mkdir","directory name (%s) cannot contain a slash", name);
      return 0;
   }
   if (GetKey(name)) {
      Error("mkdir","An object with name %s exists already",name);
      return 0;
   }

   TDirectory::TContext ctxt(this);

   TDirectory *newdir = new TDirectory(name, title, "", this);

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

   fList->R__FOR_EACH(TObject,Paint)(option);
}

//______________________________________________________________________________
void TDirectory::Print(Option_t *option) const
{
//*-*-*-*-*-*-*-*-*-*-*-*Print all objects in the directory *-*-*-*-*-*-*-*
//*-*                    ==================================
//

   fList->R__FOR_EACH(TObject,Print)(option);
}

//______________________________________________________________________________
void TDirectory::Purge(Short_t)
{
   // Purge lowest key cycles in a directory.
   // By default, only the highest cycle of a key is kept. Keys for which
   // the "KEEP" flag has been set are not removed. See TKey::Keep().

   if (!IsWritable()) return;

   TDirectory::TContext ctxt(this);

   TKey  *key;
   TIter  prev(GetListOfKeys(), kIterBackward);

   while ((key = (TKey*)prev())) {      // reverse loop on keys
      TKey *keyprev = (TKey*)GetListOfKeys()->Before(key);
      if (!keyprev) break;
      if (key->GetKeep() == 0) {
         if (strcmp(key->GetName(), keyprev->GetName()) == 0) key->Delete();
      }
   }
   TFile* f = GetFile();
   if (fModified && (f!=0)) {
      WriteKeys();                   // Write new keys structure
      WriteDirHeader();              // Write new directory header
      f->WriteFree();                // Write new free segments list
      f->WriteHeader();              // Write new file header
   }
}

//______________________________________________________________________________
void TDirectory::pwd() const
{
   // Print the path of the directory.

   Printf("%s", GetPath());
}

//______________________________________________________________________________
void TDirectory::ReadAll(Option_t* opt)
{
   // Read objects from a ROOT db file directory into memory.
   // If an object is already in memory, the memory copy is deleted
   // and the object is again read from the file.
   // If opt=="dirs", only subdirectories will be read
   // If opt=="dirs*" complete directory tree will be read  

   TDirectory::TContext ctxt(this);

   TKey *key;
   TIter next(GetListOfKeys());
   
   Bool_t readdirs = ((opt!=0) && ((strcmp(opt,"dirs")==0) || (strcmp(opt,"dirs*")==0)));
   
   if (readdirs) 
      while ((key = (TKey *) next())) {
         
         if (strcmp(key->GetClassName(),"TDirectory")!=0) continue;
         
         TDirectory *dir = GetDirectory(key->GetName(), kTRUE, "ReadAll");

         if ((dir!=0) && (strcmp(opt,"dirs*")==0)) dir->ReadAll("dirs*");
      }
   else   
      while ((key = (TKey *) next())) {
         TObject *thing = GetList()->FindObject(key->GetName());
         if (thing) { delete thing; }
         thing = key->ReadObj();
      }
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

   if (fFile==0) return 0;
   
   if (!fFile->IsBinary()) 
      return fFile->DirReadKeys(this);

   TDirectory::TContext ctxt(this);

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
   if (versiondir > 1000) {
      frombuf(buffer, &fSeekDir);
      frombuf(buffer, &fSeekParent);
      frombuf(buffer, &fSeekKeys);
   } else {
      Int_t sdir,sparent,skeys;
      frombuf(buffer, &sdir);    fSeekDir    = (Long64_t)sdir;
      frombuf(buffer, &sparent); fSeekParent = (Long64_t)sparent;
      frombuf(buffer, &skeys);   fSeekKeys   = (Long64_t)skeys;
   }
   delete [] header;

   Int_t nkeys = 0;
   Long64_t fsize = fFile->GetSize();
   if ( fSeekKeys >  0) {
      TKey *headerkey    = new TKey(fSeekKeys, fNbytesKeys, this);
      headerkey->ReadFile();
      buffer = headerkey->GetBuffer();
      headerkey->ReadKeyBuffer(buffer);

      TKey *key;
      frombuf(buffer, &nkeys);
      for (Int_t i = 0; i < nkeys; i++) {
         key = new TKey(this);
         key->ReadKeyBuffer(buffer);
         if (key->GetSeekKey() < 64 || key->GetSeekKey() > fsize) {
            Error("ReadKeys","reading illegal key, exiting after %d keys",i);
            fKeys->Remove(key);
            nkeys = i;
            break;
         }
         if (key->GetSeekPdir() < 64 || key->GetSeekPdir() > fsize) {
            Error("ReadKeys","reading illegal key, exiting after %d keys",i);
            fKeys->Remove(key);
            nkeys = i;
            break;
         }
         fKeys->Add(key);
      }
      delete headerkey;
   }

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
void TDirectory::rmdir(const char *name)
{
   // Removes subdirectory from the directory
   // When diredctory is deleted, all keys in all subdirectories will be
   // read first and deleted from file (if exists)
   // Equivalent call is Delete("name;*");
   
   if ((name==0) || (*name==0)) return;
   
   TString mask(name);
   mask+=";*";
   Delete(mask);
}

//______________________________________________________________________________
void TDirectory::Save()
{
//*-*-*-*-*-*-*-*-*-*Save recursively all directory keys and headers-*-*-*-*-*
//*-*                ===============================================

   TDirectory::TContext ctxt(this);

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

   if (IsWritable() && (fModified || force) && fFile) {
      Bool_t dowrite = kTRUE;
      if (fFile->GetListOfFree())
        dowrite = fFile->GetListOfFree()->First() != 0; 
      if (dowrite) {
         TDirectory *dirsav = gDirectory;
         if (dirsav != this) cd();
         WriteKeys();          //*-*- Write keys record
         WriteDirHeader();     //*-*- Update directory record
         if (dirsav && dirsav != this) dirsav->cd();
      }
   }
}

//______________________________________________________________________________
void TDirectory::SetBufferSize(Int_t bufsize)
{
   // set the default buffer size when creating new TKeys
   // see also TDirectory::GetBufferSize

   fBufferSize = bufsize;
}

//______________________________________________________________________________
void TDirectory::SetName(const char* newname)
{
   // Set the name for directory
   // If the directory name is changed after the directory was written once, 
   // ROOT currently would NOT change the name of correspondent key in the 
   // mother directory.
   // DO NOT use this method to 'rename a directory'.
   // Renaming a directory is currently NOT supported.
   
   TNamed::SetName(newname);
}

//______________________________________________________________________________
void TDirectory::SetWritable(Bool_t writable)
{
//  Set the new value of fWritable recursively

   TDirectory::TContext ctxt(this);

   fWritable = writable;

   // recursively set all sub-directories
   if (fList) {
      TObject *idcur;
      TIter    next(fList);
      while ((idcur = next())) {
         if (idcur->InheritsFrom(TDirectory::Class())) {
            TDirectory *dir = (TDirectory*)idcur;
            dir->SetWritable(writable);
         }
      }
   }
}


//______________________________________________________________________________
Int_t TDirectory::Sizeof() const
{
//*-*-*-*-*-*-*Return the size in bytes of the directory header*-*-*-*-*-*-*
//*-*          ================================================
   //Int_t nbytes = sizeof(Version_t);    2
   //nbytes     += fDatimeC.Sizeof();
   //nbytes     += fDatimeM.Sizeof();
   //nbytes     += sizeof fNbytesKeys;    4
   //nbytes     += sizeof fNbytesName;    4
   //nbytes     += sizeof fSeekDir;       4 or 8
   //nbytes     += sizeof fSeekParent;    4 or 8
   //nbytes     += sizeof fSeekKeys;      4 or 8
   //nbytes     += fUUID.Sizeof();
   Int_t nbytes = 22;
   nbytes     += fDatimeC.Sizeof();
   nbytes     += fDatimeM.Sizeof();
   nbytes     += fUUID.Sizeof();
    //assume that the file may be above 2 Gbytes if file version is > 4
   if (fFile && fFile->GetVersion() >= 40000) nbytes += 12;
   return nbytes;
}


//_______________________________________________________________________
void TDirectory::Streamer(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*Stream a class object*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================================

   Version_t v,version;
   if (b.IsReading()) {
      Build((TFile*)b.GetParent(), 0);
      if (fFile && fFile->IsWritable()) fWritable = kTRUE;
      
      if (fFile && !fFile->IsBinary()) {
         Version_t R__v = b.ReadVersion(0, 0);
         b.ClassBegin(TDirectory::Class(), R__v);

         TString sbuf;
         
         b.ClassMember("CreateTime","TString");
         sbuf.Streamer(b);
         TDatime timeC(sbuf.Data());
         fDatimeC = timeC;
         
         b.ClassMember("ModifyTime","TString");
         sbuf.Streamer(b);
         TDatime timeM(sbuf.Data());
         fDatimeM = timeM;
         
         b.ClassMember("UUID","TString");
         sbuf.Streamer(b);
         TUUID id(sbuf.Data());
         fUUID = id;
         
         b.ClassEnd(TDirectory::Class());
         
         fSeekKeys = 0; // read keys later in the TKeySQL class
      } else {
         b >> version;
         fDatimeC.Streamer(b);
         fDatimeM.Streamer(b);
         b >> fNbytesKeys;
         b >> fNbytesName;
         if (version > 1000) {
            SetBit(kIsBigFile);
            b >> fSeekDir;
            b >> fSeekParent;
            b >> fSeekKeys;
         } else {
            Int_t sdir,sparent,skeys;
            b >> sdir;    fSeekDir    = (Long64_t)sdir;
            b >> sparent; fSeekParent = (Long64_t)sparent;
            b >> skeys;   fSeekKeys   = (Long64_t)skeys;
         }
         v = version%1000;
         if (v == 2) {
            fUUID.StreamerV1(b);
         } else if (v > 2) {
            fUUID.Streamer(b);
         }
      }
      R__LOCKGUARD2(gROOTMutex);
      gROOT->GetUUIDs()->AddUUID(fUUID,this);
      if (fSeekKeys) ReadKeys();
   } else {
      if (fFile && !fFile->IsBinary()) {
         b.WriteVersion(TDirectory::Class());
         
         TString sbuf;
         
         b.ClassBegin(TDirectory::Class());

         b.ClassMember("CreateTime","TString");
         sbuf = fDatimeC.AsSQLString();
         sbuf.Streamer(b);
         
         b.ClassMember("ModifyTime","TString");
         fDatimeM.Set();
         sbuf = fDatimeM.AsSQLString();
         sbuf.Streamer(b);
         
         b.ClassMember("UUID","TString");
         sbuf = fUUID.AsString();
         sbuf.Streamer(b);
         
         b.ClassEnd(TDirectory::Class());
      } else {
         version = TDirectory::Class_Version();
         if (fFile && fFile->GetEND() > TFile::kStartBigFile) version += 1000;
         b << version;
         fDatimeC.Streamer(b);
         fDatimeM.Streamer(b);
         b << fNbytesKeys;
         b << fNbytesName;
         if (version > 1000) {
            b << fSeekDir;
            b << fSeekParent;
            b << fSeekKeys;
         } else {
            b << (Int_t)fSeekDir;
            b << (Int_t)fSeekParent;
            b << (Int_t)fSeekKeys;
         }
         fUUID.Streamer(b);
         if (version <=1000) for (Int_t i=0;i<3;i++) b << Int_t(0);
      }
   }
}

//______________________________________________________________________________
Int_t TDirectory::Write(const char *, Int_t opt, Int_t bufsiz)
{
   // Write all objects in memory to disk.
   // Loop on all objects in memory (including subdirectories).
   // A new key is created in the KEYS linked list for each object.
   // For allowed options see TObject::Write().
   // The directory header info is rewritten on the directory header record.

   if (!IsWritable()) return 0;
   TDirectory::TContext ctxt(this);

   // Loop on all objects (including subdirs)
   TIter next(fList);
   TObject *obj;
   Int_t nbytes = 0;
   while ((obj=next())) {
      nbytes += obj->Write(0,opt,bufsiz);
   }
   SaveSelf(kTRUE);   // force save itself

   return nbytes;
}

//______________________________________________________________________________
Int_t TDirectory::Write(const char *n, Int_t opt, Int_t bufsize) const
{
   // One can not save a const TDirectory object.

   Error("Write const","A const TDirectory object should not be saved. We try to proceed anyway.");
   return const_cast<TDirectory*>(this)->Write(n, opt, bufsize);
}

//____________________________________________________________________________________
Int_t TDirectory::WriteTObject(const TObject *obj, const char *name, Option_t *option)
{
   // Write object obj to this directory
   // The data structure corresponding to this object is serialized.
   // The corresponding buffer is written to this directory
   // with an associated key with name "name".
   //
   // Writing an object to a file involves the following steps:
   //
   //  -Creation of a support TKey object in the directory.
   //   The TKey object creates a TBuffer object.
   //
   //  -The TBuffer object is filled via the class::Streamer function.
   //
   //  -If the file is compressed (default) a second buffer is created to
   //   hold the compressed buffer.
   //
   //  -Reservation of the corresponding space in the file by looking
   //   in the TFree list of free blocks of the file.
   //
   //  -The buffer is written to the file.
   //
   //  By default, the buffersize will be taken from the average buffer size
   //  of all objects written to the current file so far.
   //  Use TDirectory::SetBufferSize to force a given buffer size.
   //
   //  If a name is specified, it will be the name of the key.
   //  If name is not given, the name of the key will be the name as returned
   //  by obj->GetName().
   //
   //  The option can be a combination of:
   //    "SingleKey", "Overwrite" or "WriteDelete"
   //  Using the "Overwrite" option a previous key with the same name is
   //  overwritten. The previous key is deleted before writing the new object.
   //  Using the "WriteDelete" option a previous key with the same name is
   //  deleted only after the new object has been written. This option
   //  is safer than kOverwrite but it is slower.
   //  The "SingleKey" option is only used by TCollection::Write() to write
   //  a container with a single key instead of each object in the container
   //  with its own key.
   //
   //  An object is read from this directory via TDirectory::Get.
   //
   //  The function returns the total number of bytes written to the directory.
   //  It returns 0 if the object cannot be written.

   TDirectory::TContext ctxt(this);

   if (fFile==0) return 0;

   if (!fFile->IsWritable()) {
      if (!fFile->TestBit(TFile::kWriteError)) {
         // Do not print the error if the file already had a SysError.
         Error("WriteTObject","Directory %s is not writable", fFile->GetName());
      }
      return 0;
   }

   if (!obj) return 0;

   TString opt = option;
   opt.ToLower();

   TKey *key=0, *oldkey=0;
   Int_t bsize = GetBufferSize();

   const char *oname;
   if (name && *name)
      oname = name;
   else
      oname = obj->GetName();

   // Remove trailing blanks in object name
   Int_t nch = strlen(oname);
   char *newName = 0;
   if (oname[nch-1] == ' ') {
      newName = new char[nch+1];
      strcpy(newName,oname);
      for (Int_t i=0;i<nch;i++) {
         if (newName[nch-i-1] != ' ') break;
         newName[nch-i-1] = 0;
      }
      oname = newName;
   }

   if (opt.Contains("overwrite")) {
      //One must use GetKey. FindObject would return the lowest cycle of the key!
      //key = (TKey*)gDirectory->GetListOfKeys()->FindObject(oname);
      key = GetKey(oname);
      if (key) {
         key->Delete();
         delete key;
      }
   }
   if (opt.Contains("writedelete")) {
      oldkey = GetKey(oname);
   }
   key = fFile->CreateKey(this, obj, oname, bsize);
   if (newName) delete [] newName;

   if (!key->GetSeekKey()) {
      fKeys->Remove(key);
      delete key;
      return 0;
   }
   fFile->SumBuffer(key->GetObjlen());
   Int_t nbytes = key->WriteFile(0);
   if (fFile->TestBit(TFile::kWriteError)) return 0;

   if (oldkey) {
      oldkey->Delete();
      delete oldkey;
   }

   return nbytes;
}

//______________________________________________________________________________
Int_t TDirectory::WriteObjectAny(const void *obj, const char *classname, const char *name, Option_t *option)
{
   // Write object from pointer of class classname in this directory
   // obj may not derive from TObject
   // see TDirectory::WriteObject for comments
   //
   // VERY IMPORTANT NOTE:
   //    The value passed as 'obj' needs to be from a pointer to the type described by classname
   //    For example with:
   //      TopClass *top;
   //      BottomClass *bottom;
   //      top = bottom;
   //    you can do:
   //      directory->WriteObjectAny(top,"top","name of object");
   //      directory->WriteObjectAny(bottom,"bottom","name of object");
   //    BUT YOU CAN NOT DO (it will fail in particular with multiple inheritance):
   //      directory->WriteObjectAny(top,"bottom","name of object");
   //
   // We STRONGLY recommend to use
   //      TopClass *top = ....;
   //      directory->WriteObject(top,"name of object")

   TClass *cl = ROOT::GetROOT()->GetClass(classname);
   if (!cl) {
      Error("WriteObjectAny","Unknown class: %s",classname);
      return 0;
   }
   return WriteObjectAny(obj,cl,name,option);
}

//______________________________________________________________________________
Int_t TDirectory::WriteObjectAny(const void *obj, const TClass *cl, const char *name, Option_t *option)
{
   // Write object of class with dictionary cl in this directory
   // obj may not derive from TObject
   // To get the TClass* cl pointer, one can use
   //    TClass *cl = gROOT->GetClass("classname");
   // An alternative is to call the function WriteObjectAny above.
   // see TDirectory::WriteObject for comments

   TDirectory::TContext ctxt(this);

   if (fFile==0) return 0;

   if (!fFile->IsWritable()) {
      if (!fFile->TestBit(TFile::kWriteError)) {
         // Do not print the error if the file already had a SysError.
         Error("WriteObject","File %s is not writable", fFile->GetName());
      }
      return 0;
   }

   if (!obj || !cl) return 0;
   TKey *key, *oldkey=0;
   Int_t bsize = GetBufferSize();

   TString opt = option;
   opt.ToLower();

   const char *oname;
   if (name && *name)
      oname = name;
   else
      oname = cl->GetName();

   // Remove trailing blanks in object name
   Int_t nch = strlen(oname);
   char *newName = 0;
   if (oname[nch-1] == ' ') {
      newName = new char[nch+1];
      strcpy(newName,oname);
      for (Int_t i=0;i<nch;i++) {
         if (newName[nch-i-1] != ' ') break;
         newName[nch-i-1] = 0;
      }
      oname = newName;
   }

   if (opt.Contains("overwrite")) {
      //One must use GetKey. FindObject would return the lowest cycle of the key!
      //key = (TKey*)gDirectory->GetListOfKeys()->FindObject(oname);
      key = GetKey(oname);
      if (key) {
         key->Delete();
         delete key;
      }
   }
   if (opt.Contains("writedelete")) {
      oldkey = GetKey(oname);
   }
   key = fFile->CreateKey(this, obj, cl, oname, bsize);
   if (newName) delete [] newName;

   if (!key->GetSeekKey()) {
      fKeys->Remove(key);
      delete key;
      return 0;
   }
   fFile->SumBuffer(key->GetObjlen());
   Int_t nbytes = key->WriteFile(0);
   if (fFile->TestBit(TFile::kWriteError)) return 0;

   if (oldkey) {
      oldkey->Delete();
      delete oldkey;
   }

   return nbytes;
}

//______________________________________________________________________________
void TDirectory::WriteDirHeader()
{
//*-*-*-*-*-*-*-*-*-*-*Overwrite the Directory header record*-*-*-*-*-*-*-*-*
//*-*                  =====================================
   TFile* f = GetFile();
   if (f==0) return;
   
   if (!f->IsBinary()) {
      fDatimeM.Set();
      f->DirWriteHeader(this);
      return;
   }

   Int_t nbytes  = TDirectory::Sizeof();  //Warning ! TFile has a Sizeof()
   char * header = new char[nbytes];
   char * buffer = header;
   fDatimeM.Set();
   TDirectory::FillBuffer(buffer);
   Long64_t pointer = fSeekDir + fNbytesName; // do not overwrite the name/title part
   fModified     = kFALSE;
   f->Seek(pointer);
   f->WriteBuffer(header, nbytes);
   if (f->MustFlush()) f->Flush();
   delete [] header;
}

//______________________________________________________________________________
void TDirectory::WriteKeys()
{
//*-*-*-*-*-*-*-*-*-*-*-*Write KEYS linked list on the file *-*-*-*-*-*-*-*
//*-*                    ==================================
//  The linked list of keys (fKeys) is written as a single data record
//

   TFile* f = GetFile();
   if (f==0) return;
   
   if (!f->IsBinary()) {
      f->DirWriteKeys(this);
      return;
   }

//*-* Delete the old keys structure if it exists
   if (fSeekKeys != 0) {
      f->MakeFree(fSeekKeys, fSeekKeys + fNbytesKeys -1);
   }
//*-* Write new keys record
   TIter next(fKeys);
   TKey *key;
   Int_t nkeys  = fKeys->GetSize();
   Int_t nbytes = sizeof nkeys;          //*-* Compute size of all keys
   if (f->GetEND() > TFile::kStartBigFile) nbytes += 8;
   while ((key = (TKey*)next())) {
      nbytes += key->Sizeof();
   }
   TKey *headerkey  = new TKey(fName,fTitle,IsA(),nbytes,this);
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
   Int_t nch = buffer ? strlen(buffer) : 0;
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


//______________________________________________________________________________
void TDirectory::TContext::CdNull()
{
   // Set the current directory to null.
   // This is called from the TContext destructor.  Since the destructor is
   // inline, we do no want to have it use directly a global variable.

   gDirectory = 0;
}

