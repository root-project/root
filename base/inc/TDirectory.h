// @(#)root/base:$Name:  $:$Id: TDirectory.h,v 1.3 2000/09/05 09:21:22 brun Exp $
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
#ifndef ROOT_TDatime
#include "TDatime.h"
#endif

class TBrowser;
class TKey;
class TFile;


class TDirectory : public TNamed {

protected:
   Bool_t      fModified;        //true if directory has been modified
   Bool_t      fWritable;        //true if directory is writable
   TDatime     fDatimeC;         //Date and time when directory is created
   TDatime     fDatimeM;         //Date and time of last modification
   Int_t       fNbytesKeys;      //Number of bytes for the keys
   Int_t       fNbytesName;      //Number of bytes in TNamed at creation time
   Seek_t      fSeekDir;         //Location of directory on file
   Seek_t      fSeekParent;      //Location of parent directory on file
   Seek_t      fSeekKeys;        //Location of Keys record on file
   TFile       *fFile;           //pointer to current file in memory
   TObject     *fMother;         //pointer to mother of the directory
   TList       *fList;           //Pointer to objects list in memory
   TList       *fKeys;           //Pointer to keys list in memory

          Bool_t cd1(const char *path);
   static Bool_t Cd1(const char *path);

private:
   TDirectory(const TDirectory &directory);  //Directories cannot be copied
   void operator=(const TDirectory &);

public:
   // TDirectory status bits
   enum { kCloseDirectory = BIT(7) };

   TDirectory();
   TDirectory(const char *name, const char *title, Option_t *option="");
   virtual ~TDirectory();
   virtual void        Append(TObject *obj);
           void        Add(TObject *obj) { Append(obj); }
           Int_t       AppendKey(TKey *key);
   virtual void        Browse(TBrowser *b);
           void        Build();
   virtual void        Clear(Option_t *option="");
   virtual void        Close(Option_t *option="");
   virtual void        Copy(TObject &) { MayNotUse("Copy(TObject &)"); }
   virtual Bool_t      cd(const char *path = 0);
   virtual void        DeleteAll(Option_t *option="");
   virtual void        Delete(const char *namecycle="");
   virtual void        Draw(Option_t *option="");
   virtual void        FillBuffer(char *&buffer);
   virtual TObject    *Get(const char *namecycle);
   virtual TFile      *GetFile() {return fFile;}
   virtual TKey       *GetKey(const char *name, const Short_t cycle=9999);
   TList              *GetList() const { return fList; }
   TList              *GetListOfKeys() const { return fKeys; }
   TObject            *GetMother() const { return fMother; }
   virtual Int_t       GetNkeys() {return fKeys->GetSize();}
   virtual Seek_t      GetSeekDir() { return fSeekDir; }
   virtual Seek_t      GetSeekParent() { return fSeekParent; }
   virtual Seek_t      GetSeekKeys() { return fSeekKeys; }
   virtual const char *GetPath() const;
   Bool_t              IsFolder() const { return kTRUE; }
   Bool_t              IsModified() const { return fModified; }
   Bool_t              IsWritable() const { return fWritable; }
   virtual void        ls(Option_t *option="");
   virtual TDirectory *mkdir(const char *name, const char *title="");
   virtual void        Paint(Option_t *option="");
   virtual void        Print(Option_t *option="");
   virtual void        Purge(Short_t nkeep=1);
   virtual void        pwd() const;
   virtual void        ReadAll(Option_t *option="");
   virtual Int_t       ReadKeys();
   virtual void        RecursiveRemove(TObject *obj);
   virtual void        Save();
   virtual void        SaveSelf(Bool_t force = kFALSE);
   void                SetModified() {fModified = kTRUE;}
   void                SetMother(TObject *mother) {fMother = mother;}
   virtual Int_t       Sizeof() const;
   virtual Int_t       Write(const char *name=0, Int_t opt=0, Int_t bufsiz=0);
   virtual void        WriteDirHeader();
   virtual void        WriteKeys();

   static Bool_t       Cd(const char *path);
   static void         DecodeNameCycle(const char *namecycle, char *name, Short_t &cycle);
   static void         EncodeNameCycle(char *buffer, const char *name, Short_t cycle);

   ClassDef(TDirectory,1)  //Describe directory structure in memory
};

R__EXTERN TDirectory   *gDirectory;

#endif

