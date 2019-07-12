// @(#)root/io:$Id$
// Author: Rene Brun   22/01/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDirectoryFile
#define ROOT_TDirectoryFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDirectoryFile                                                       //
//                                                                      //
// Describe directory structure in a ROOT file.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Compression.h"
#include "TDirectory.h"

class TList;
class TBrowser;
class TKey;
class TFile;

class TDirectoryFile : public TDirectory {

protected:
   Bool_t      fModified;        ///< True if directory has been modified
   Bool_t      fWritable;        ///< True if directory is writable
   TDatime     fDatimeC;         ///< Date and time when directory is created
   TDatime     fDatimeM;         ///< Date and time of last modification
   Int_t       fNbytesKeys;      ///< Number of bytes for the keys
   Int_t       fNbytesName;      ///< Number of bytes in TNamed at creation time
   Int_t       fBufferSize;      ///< Default buffer size to create new TKeys
   Long64_t    fSeekDir;         ///< Location of directory on file
   Long64_t    fSeekParent;      ///< Location of parent directory on file
   Long64_t    fSeekKeys;        ///< Location of Keys record on file
   TFile      *fFile;            ///< Pointer to current file in memory
   TList      *fKeys;            ///< Pointer to keys list in memory

   virtual void         CleanTargets();
   void Init(TClass *cl = 0);

private:
   TDirectoryFile(const TDirectoryFile &directory);  //Directories cannot be copied
   void operator=(const TDirectoryFile &); //Directories cannot be copied

public:
   // TDirectory status bits
   enum { kCloseDirectory = BIT(7), kCustomBrowse = BIT(9) };

   TDirectoryFile();
   TDirectoryFile(const char *name, const char *title, Option_t *option="", TDirectory* motherDir = 0);
   virtual ~TDirectoryFile();
   virtual void        Append(TObject *obj, Bool_t replace = kFALSE);
           void        Add(TObject *obj, Bool_t replace = kFALSE) { Append(obj,replace); }
           Int_t       AppendKey(TKey *key);
   virtual void        Browse(TBrowser *b);
           void        Build(TFile* motherFile = 0, TDirectory* motherDir = 0);
   virtual TObject    *CloneObject(const TObject *obj, Bool_t autoadd = kTRUE);
   virtual void        Close(Option_t *option="");
   virtual void        Copy(TObject &) const { MayNotUse("Copy(TObject &)"); }
   virtual Bool_t      cd(const char *path = 0);
   virtual void        Delete(const char *namecycle="");
   virtual void        FillBuffer(char *&buffer);
   virtual TKey       *FindKey(const char *keyname) const;
   virtual TKey       *FindKeyAny(const char *keyname) const;
   virtual TObject    *FindObjectAny(const char *name) const;
   virtual TObject    *FindObjectAnyFile(const char *name) const;
   virtual TObject    *Get(const char *namecycle);
   /// See documentation of TDirectoryFile::Get(const char *namecycle)
   template <class T> inline T* Get(const char* namecycle)
   {
      return TDirectory::Get<T>(namecycle);
   }
   virtual TDirectory *GetDirectory(const char *apath, Bool_t printError = false, const char *funcname = "GetDirectory");
   virtual void       *GetObjectChecked(const char *namecycle, const char* classname);
   virtual void       *GetObjectChecked(const char *namecycle, const TClass* cl);
   virtual void       *GetObjectUnchecked(const char *namecycle);
   virtual Int_t       GetBufferSize() const;
   const TDatime      &GetCreationDate() const { return fDatimeC; }
   virtual TFile      *GetFile() const { return fFile; }
   virtual TKey       *GetKey(const char *name, Short_t cycle=9999) const;
   virtual TList      *GetListOfKeys() const { return fKeys; }
   const TDatime      &GetModificationDate() const { return fDatimeM; }
   virtual Int_t       GetNbytesKeys() const { return fNbytesKeys; }
   virtual Int_t       GetNkeys() const { return fKeys->GetSize(); }
   virtual Long64_t    GetSeekDir() const { return fSeekDir; }
   virtual Long64_t    GetSeekParent() const { return fSeekParent; }
   virtual Long64_t    GetSeekKeys() const { return fSeekKeys; }
   Bool_t              IsModified() const { return fModified; }
   Bool_t              IsWritable() const { return fWritable; }
   virtual void        ls(Option_t *option="") const;
   virtual TDirectory *mkdir(const char *name, const char *title="");
   virtual TFile      *OpenFile(const char *name, Option_t *option= "",
                            const char *ftitle = "", Int_t compress = ROOT::RCompressionSetting::EDefaults::kUseGeneralPurpose,
                            Int_t netopt = 0);
   virtual void        Purge(Short_t nkeep=1);
   virtual void        ReadAll(Option_t *option="");
   virtual Int_t       ReadKeys(Bool_t forceRead=kTRUE);
   virtual Int_t       ReadTObject(TObject *obj, const char *keyname);
   virtual void        ResetAfterMerge(TFileMergeInfo *);
   virtual void        rmdir(const char *name);
   virtual void        Save();
   virtual void        SaveSelf(Bool_t force = kFALSE);
   virtual Int_t       SaveObjectAs(const TObject *obj, const char *filename="", Option_t *option="") const;
   virtual void        SetBufferSize(Int_t bufsize);
   void                SetModified() {fModified = kTRUE;}
   void                SetSeekDir(Long64_t v) { fSeekDir = v; }
   virtual void        SetTRefAction(TObject *ref, TObject *parent);
   void                SetWritable(Bool_t writable=kTRUE);
   virtual Int_t       Sizeof() const;
   virtual Int_t       Write(const char *name=0, Int_t opt=0, Int_t bufsize=0);
   virtual Int_t       Write(const char *name=0, Int_t opt=0, Int_t bufsize=0) const ;
   virtual Int_t       WriteTObject(const TObject *obj, const char *name=0, Option_t *option="", Int_t bufsize=0);
   virtual Int_t       WriteObjectAny(const void *obj, const char *classname, const char *name, Option_t *option="", Int_t bufsize=0);
   virtual Int_t       WriteObjectAny(const void *obj, const TClass *cl, const char *name, Option_t *option="", Int_t bufsize=0);
   virtual void        WriteDirHeader();
   virtual void        WriteKeys();

   ClassDef(TDirectoryFile,5)  //Describe directory structure in a ROOT file
};

#endif
