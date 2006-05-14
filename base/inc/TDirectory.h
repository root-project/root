// @(#)root/base:$Name:  $:$Id: TDirectory.h,v 1.33 2006/02/01 18:54:51 pcanal Exp $
// Author: Rene Brun   28/11/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDirectory
#define ROOT_TDirectory


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDirectory                                                           //
//                                                                      //
// Describe directory structure in memory.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TDatime
#include "TDatime.h"
#endif
#ifndef ROOT_TUUID
#include "TUUID.h"
#endif

class TBrowser;
class TKey;
class TFile;

R__EXTERN TDirectory *gDirectory;

class TDirectory : public TNamed {

protected:
   Bool_t      fModified;        //true if directory has been modified
   Bool_t      fWritable;        //true if directory is writable
   TDatime     fDatimeC;         //Date and time when directory is created
   TDatime     fDatimeM;         //Date and time of last modification
   Int_t       fNbytesKeys;      //Number of bytes for the keys
   Int_t       fNbytesName;      //Number of bytes in TNamed at creation time
   Int_t       fBufferSize;      //Default buffer size to create new TKeys
   Long64_t    fSeekDir;         //Location of directory on file
   Long64_t    fSeekParent;      //Location of parent directory on file
   Long64_t    fSeekKeys;        //Location of Keys record on file
   TFile      *fFile;            //pointer to current file in memory
   TObject    *fMother;          //pointer to mother of the directory
   TList      *fList;            //Pointer to objects list in memory
   TList      *fKeys;            //Pointer to keys list in memory
   TUUID       fUUID;            //Unique identifier
   TString     fPathBuffer;      //!Buffer for GetPath() function 

          Bool_t cd1(const char *path);
   static Bool_t Cd1(const char *path);
   
          void   FillFullPath(TString& buf) const;

private:
   TDirectory(const TDirectory &directory);  //Directories cannot be copied
   void operator=(const TDirectory &);

public:
   // TDirectory status bits
   enum { kCloseDirectory = BIT(7) };

   /** @class Context
     *
     *  Small helper to keep current directory context.
     *  Automatically reverts to "old" directory
     */
   class TContext  {
   private:
      TContext(TContext&);
      TContext& operator=(TContext&);
     
      TDirectory* fPrevious;   // Pointer to the previous current directory.
      void CdNull(); 
   public:
      TContext(TDirectory* previous, TDirectory* newCurrent) 
         : fPrevious(previous)
      {
         // Store the current directory so we can restore it
         // later and cd to the new directory.
         if ( newCurrent ) newCurrent->cd();
      } 
      TContext(TDirectory* newCurrent) : fPrevious(gDirectory)
      {
         // Store the current directory so we can restore it
         // later and cd to the new directory.
         if ( newCurrent ) newCurrent->cd();
      } 
      ~TContext() 
      {
         // Destructor.   Reset the current directory to its
         // previous state.
         if ( fPrevious ) fPrevious->cd();
         else CdNull();
      }
   };

   TDirectory();
   TDirectory(const char *name, const char *title, Option_t *option="", TDirectory* motherDir = 0);
   virtual ~TDirectory();
   virtual void        Append(TObject *obj);
           void        Add(TObject *obj) { Append(obj); }
           Int_t       AppendKey(TKey *key);
   virtual void        Browse(TBrowser *b);
           void        Build(TFile* motherFile = 0, TDirectory* motherDir = 0);
   virtual void        Clear(Option_t *option="");
   virtual void        Close(Option_t *option="");
   virtual void        Copy(TObject &) const { MayNotUse("Copy(TObject &)"); }
   virtual Bool_t      cd(const char *path = 0);
   virtual void        DeleteAll(Option_t *option="");
   virtual void        Delete(const char *namecycle="");
   virtual void        Draw(Option_t *option="");
   virtual void        FillBuffer(char *&buffer);
   virtual TKey       *FindKey(const char *keyname) const;
   virtual TKey       *FindKeyAny(const char *keyname) const;
   virtual TObject    *FindObject(const char *name) const;
   virtual TObject    *FindObject(const TObject *obj) const;
   virtual TObject    *FindObjectAny(const char *name) const;
   virtual TObject    *Get(const char *namecycle); 
   virtual TDirectory *GetDirectory(const char *namecycle, Bool_t printError = false, const char *funcname = "GetDirectory"); 
   template <class T> inline void GetObject(const char* namecycle, T*& ptr) // See TDirectory::Get for information
      {
         ptr = (T*)GetObjectChecked(namecycle,TBuffer::GetClass(typeid(T)));
      }
   virtual void       *GetObjectChecked(const char *namecycle, const char* classname);
   virtual void       *GetObjectChecked(const char *namecycle, const TClass* cl);
   virtual void       *GetObjectUnchecked(const char *namecycle);
   virtual Int_t       GetBufferSize() const;
   const TDatime      &GetCreationDate() const { return fDatimeC; }
   virtual TFile      *GetFile() const { return fFile; }
   virtual TKey       *GetKey(const char *name, Short_t cycle=9999) const;
   virtual TList      *GetList() const { return fList; }
   virtual TList      *GetListOfKeys() const { return fKeys; }
   const TDatime      &GetModificationDate() const { return fDatimeM; }
   TObject            *GetMother() const { return fMother; }
   TDirectory         *GetMotherDir() const { return fMother==0 ? 0 : dynamic_cast<TDirectory*>(fMother); }
   virtual Int_t       GetNbytesKeys() const { return fNbytesKeys; }
   virtual Int_t       GetNkeys() const { return fKeys->GetSize(); }
   virtual Long64_t    GetSeekDir() const { return fSeekDir; }
   virtual Long64_t    GetSeekParent() const { return fSeekParent; }
   virtual Long64_t    GetSeekKeys() const { return fSeekKeys; }
   virtual const char *GetPathStatic() const;
   virtual const char *GetPath() const;
   TUUID               GetUUID() const {return fUUID;}
   Bool_t              IsFolder() const { return kTRUE; }
   Bool_t              IsModified() const { return fModified; }
   Bool_t              IsWritable() const { return fWritable; }
   virtual void        ls(Option_t *option="") const;
   virtual TDirectory *mkdir(const char *name, const char *title="");
   virtual void        Paint(Option_t *option="");
   virtual void        Print(Option_t *option="") const;
   virtual void        Purge(Short_t nkeep=1);
   virtual void        pwd() const;
   virtual void        ReadAll(Option_t *option="");
   virtual Int_t       ReadKeys();
   virtual void        RecursiveRemove(TObject *obj);
   virtual void        Save();
   virtual void        SaveSelf(Bool_t force = kFALSE);
   virtual void        SetBufferSize(Int_t bufsize);
   void                SetModified() {fModified = kTRUE;}
   void                SetMother(const TObject *mother) {fMother = (TObject*)mother;}
   void                SetWritable(Bool_t writable=kTRUE);
   virtual Int_t       Sizeof() const;
   virtual Int_t       Write(const char *name=0, Int_t opt=0, Int_t bufsiz=0);
   virtual Int_t       Write(const char *name=0, Int_t opt=0, Int_t bufsiz=0) const ;
   virtual Int_t       WriteTObject(const TObject *obj, const char *name=0, Option_t *option="");
   template <class T> inline Int_t WriteObject(const T* obj, const char* name, Option_t *option="") // see TDirectory::WriteObject or TDirectoryWriteObjectAny for explanation
      {
         return WriteObjectAny(obj,TBuffer::GetClass(typeid(T)),name,option);
      }
   virtual Int_t       WriteObjectAny(const void *obj, const char *classname, const char *name, Option_t *option="");
   virtual Int_t       WriteObjectAny(const void *obj, const TClass *cl, const char *name, Option_t *option="");
   virtual void        WriteDirHeader();
   virtual void        WriteKeys();

   static Bool_t       Cd(const char *path);
   static void         DecodeNameCycle(const char *namecycle, char *name, Short_t &cycle);
   static void         EncodeNameCycle(char *buffer, const char *name, Short_t cycle);

   ClassDef(TDirectory,4)  //Describe directory structure in memory
};

#endif

