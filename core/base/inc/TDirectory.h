// @(#)root/base:$Id$
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

#include "TBuffer.h"
#include "TNamed.h"
#include "TList.h"
#include "TDatime.h"
#include "TUUID.h"

class TBrowser;
class TKey;
class TFile;

class TDirectory : public TNamed {
public:
   /** @class TContext
     *
     *  Small helper to keep current directory context.
     *  Automatically reverts to "old" directory
     */
   class TContext  {
   private:
      std::atomic<TDirectory*> fDirectory{nullptr}; //! Pointer to the previous current directory.
      std::atomic<bool> fActiveDestructor{false};   //! Set to true during the destructor execution
      std::atomic<bool> fDirectoryWait{false};      //! Set to true if a TDirectory might still access this object.
      TContext   *fPrevious{nullptr};               //! Pointer to the next TContext in the implied list of context pointing to fPrevious.
      TContext   *fNext{nullptr};                   //! Pointer to the next TContext in the implied list of context pointing to fPrevious.

      TContext(TContext&) = delete;
      TContext& operator=(TContext&) = delete;

      void CdNull();
      friend class TDirectory;
   public:
      TContext(TDirectory *previous, TDirectory *newCurrent) : fDirectory(previous)
      {
         // Store the current directory so we can restore it
         // later and cd to the new directory.
         if (fDirectory)
            (*fDirectory).RegisterContext(this);
         if (newCurrent)
            newCurrent->cd();
         else
            CdNull();
      }
      TContext() : fDirectory(TDirectory::CurrentDirectory())
      {
         // Store the current directory so we can restore it
         // later and cd to the new directory.
         if (fDirectory)
            (*fDirectory).RegisterContext(this);
      }
      TContext(TDirectory *newCurrent) : fDirectory(TDirectory::CurrentDirectory())
      {
         // Store the current directory so we can restore it
         // later and cd to the new directory.
         if (fDirectory)
            (*fDirectory).RegisterContext(this);
         if (newCurrent)
            newCurrent->cd();
         else
            CdNull();
      }
      ~TContext();
   };

protected:

   TObject         *fMother{nullptr};   // pointer to mother of the directory
   TList           *fList{nullptr};     // List of objects in memory
   TUUID            fUUID;              // Unique identifier
   mutable TString  fPathBuffer;        //! Buffer for GetPath() function
   TContext        *fContext{nullptr};  //! Pointer to a list of TContext object pointing to this TDirectory

   std::atomic<size_t> fContextPeg;     //!Counter delaying the TDirectory destructor from finishing.
   mutable std::atomic_flag fSpinLock;  //! MSVC doesn't support = ATOMIC_FLAG_INIT;

   static Bool_t fgAddDirectory;        //!flag to add histograms, graphs,etc to the directory

          Bool_t  cd1(const char *path);
   static Bool_t  Cd1(const char *path);

           void   CleanTargets();
           void   FillFullPath(TString& buf) const;
           void   RegisterContext(TContext *ctxt);
           void   UnregisterContext(TContext *ctxt);
           void   BuildDirectory(TFile* motherFile, TDirectory* motherDir);

   friend class TContext;

protected:
   TDirectory(const TDirectory &directory) = delete;  //Directories cannot be copied
   void operator=(const TDirectory &) = delete; //Directories cannot be copied

public:

   TDirectory();
   TDirectory(const char *name, const char *title, Option_t *option = "", TDirectory* motherDir = nullptr);
   virtual ~TDirectory();
   static  void        AddDirectory(Bool_t add=kTRUE);
   static  Bool_t      AddDirectoryStatus();
   virtual void        Append(TObject *obj, Bool_t replace = kFALSE);
   virtual void        Add(TObject *obj, Bool_t replace = kFALSE) { Append(obj,replace); }
   virtual Int_t       AppendKey(TKey *) {return 0;}
           void        Browse(TBrowser *b) override;
   virtual void        Build(TFile* motherFile = nullptr, TDirectory* motherDir = nullptr) { BuildDirectory(motherFile, motherDir); }
           void        Clear(Option_t *option="") override;
   virtual TObject    *CloneObject(const TObject *obj, Bool_t autoadd = kTRUE);
   virtual void        Close(Option_t *option="");
   static TDirectory *&CurrentDirectory();  // Return the current directory for this thread.
           void        Copy(TObject &) const override { MayNotUse("Copy(TObject &)"); }
   virtual Bool_t      cd(const char *path = nullptr);
   virtual void        DeleteAll(Option_t *option="");
           void        Delete(const char *namecycle="") override;
           void        Draw(Option_t *option="") override;
   virtual TKey       *FindKey(const char * /*keyname*/) const {return nullptr;}
   virtual TKey       *FindKeyAny(const char * /*keyname*/) const {return nullptr;}
           TObject    *FindObject(const char *name) const override;
           TObject    *FindObject(const TObject *obj) const override;
   virtual TObject    *FindObjectAny(const char *name) const;
   virtual TObject    *FindObjectAnyFile(const char * /*name*/) const {return nullptr;}
   virtual TObject    *Get(const char *namecycle);
   /// See documentation of TDirectoryFile::Get(const char *namecycle)
   template <class T> inline T* Get(const char* namecycle)
   {
      return static_cast<T*>(GetObjectChecked(namecycle, TClass::GetClass<T>()));
   }
   virtual TDirectory *GetDirectory(const char *namecycle, Bool_t printError = false, const char *funcname = "GetDirectory");
   template <class T> inline void GetObject(const char* namecycle, T*& ptr) // See TDirectory::Get for information
      {
         ptr = (T *)GetObjectChecked(namecycle, TClass::GetClass<T>());
      }
   virtual void       *GetObjectChecked(const char *namecycle, const char* classname);
   virtual void       *GetObjectChecked(const char *namecycle, const TClass* cl);
   virtual void       *GetObjectUnchecked(const char *namecycle);
   virtual Int_t       GetBufferSize() const {return 0;}
   virtual TFile      *GetFile() const { return 0; }
   virtual TKey       *GetKey(const char * /*name */, Short_t /* cycle */=9999) const {return nullptr;}
   virtual TList      *GetList() const { return fList; }
   virtual TList      *GetListOfKeys() const { return nullptr; }
           TObject    *GetMother() const { return fMother; }
           TDirectory *GetMotherDir() const { return !fMother ? nullptr : dynamic_cast<TDirectory*>(fMother); }
   virtual Int_t       GetNbytesKeys() const { return 0; }
   virtual Int_t       GetNkeys() const { return 0; }
   virtual Long64_t    GetSeekDir() const { return 0; }
   virtual Long64_t    GetSeekParent() const { return 0; }
   virtual Long64_t    GetSeekKeys() const { return 0; }
   virtual const char *GetPathStatic() const;
   virtual const char *GetPath() const;
   TUUID               GetUUID() const {return fUUID;}
           Bool_t      IsFolder() const override { return kTRUE; }
   virtual Bool_t      IsModified() const { return kFALSE; }
   virtual Bool_t      IsWritable() const { return kFALSE; }
           void        ls(Option_t *option="") const override;
   virtual TDirectory *mkdir(const char *name, const char *title="", Bool_t returnExistingDirectory = kFALSE);
   virtual TFile      *OpenFile(const char * /*name*/, Option_t * /*option*/ = "",
                            const char * /*ftitle*/ = "", Int_t /*compress*/ = 1,
                            Int_t /*netopt*/ = 0) {return nullptr;}
           void        Paint(Option_t *option="") override;
           void        Print(Option_t *option="") const override;
   virtual void        Purge(Short_t /*nkeep*/=1) {}
   virtual void        pwd() const;
   virtual void        ReadAll(Option_t * /*option*/="") {}
   virtual Int_t       ReadKeys(Bool_t /*forceRead*/=kTRUE) {return 0;}
   virtual Int_t       ReadTObject(TObject * /*obj*/, const char * /*keyname*/) {return 0;}
   virtual TObject    *Remove(TObject*);
           void        RecursiveRemove(TObject *obj) override;
   virtual void        rmdir(const char *name);
   virtual void        Save() {}
   virtual Int_t       SaveObjectAs(const TObject * /*obj*/, const char * /*filename*/="", Option_t * /*option*/="") const;
   virtual void        SaveSelf(Bool_t /*force*/ = kFALSE) {}
   virtual void        SetBufferSize(Int_t /* bufsize */) {}
   virtual void        SetModified() {}
   virtual void        SetMother(TObject *mother) {fMother = (TObject*)mother;}
           void        SetName(const char* newname) override;
   virtual void        SetTRefAction(TObject * /*ref*/, TObject * /*parent*/) {}
   virtual void        SetSeekDir(Long64_t) {}
   virtual void        SetWritable(Bool_t) {}
           Int_t       Sizeof() const override {return 0;}
   virtual Int_t       Write(const char * /*name*/=nullptr, Int_t /*opt*/=0, Int_t /*bufsize*/=0) override {return 0;}
   virtual Int_t       Write(const char * /*name*/=nullptr, Int_t /*opt*/=0, Int_t /*bufsize*/=0) const override {return 0;}
   virtual Int_t       WriteTObject(const TObject *obj, const char *name =nullptr, Option_t * /*option*/="", Int_t /*bufsize*/ =0);
private:
           Int_t       WriteObject(void *obj, const char* name, Option_t *option="", Int_t bufsize=0); // Intentionally not implemented.
public:
   template <class T> inline Int_t WriteObject(const T* obj, const char* name, Option_t *option="", Int_t bufsize=0) // see TDirectory::WriteTObject or TDirectoryWriteObjectAny for explanation
      {
         return WriteObjectAny(obj, TClass::GetClass<T>(), name, option, bufsize);
      }
   virtual Int_t       WriteObjectAny(const void *, const char * /*classname*/, const char * /*name*/, Option_t * /*option*/="", Int_t /*bufsize*/ =0) {return 0;}
   virtual Int_t       WriteObjectAny(const void *, const TClass * /*cl*/, const char * /*name*/, Option_t * /*option*/="", Int_t /*bufsize*/ =0) {return 0;}
   virtual void        WriteDirHeader() {}
   virtual void        WriteKeys() {}

   static Bool_t       Cd(const char *path);
   static void         DecodeNameCycle(const char *namecycle, char *name, Short_t &cycle, const size_t namesize = 0);
   static void         EncodeNameCycle(char *buffer, const char *name, Short_t cycle);

   ClassDefOverride(TDirectory,5)  //Describe directory structure in memory
};

#ifndef __CINT__
#define gDirectory (TDirectory::CurrentDirectory())

#elif defined(__MAKECINT__)
// To properly handle the use of gDirectory in header files (in static declarations)
R__EXTERN TDirectory *gDirectory;
#endif

#endif
