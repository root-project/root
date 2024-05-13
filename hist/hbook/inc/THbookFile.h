// @(#)root/hbook:$Id$
// Author: Rene Brun   18/02/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THbookFile
#define ROOT_THbookFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THbookFile                                                           //
//                                                                      //
// ROOT interface to Hbook/PAW files                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TFile.h"


class TTreeFormula;

class THbookFile : public TNamed {

protected:
   Int_t         fLun;     //Fortran logical unit for this file
   Int_t         fLrecl;   //Record length in Hbook machine words
   TList        *fList;    //list of objects in memory
   TList        *fKeys;    //list of Hbook keys (Ids) on disk
   TString       fCurDir;  //name of current directory

   static Bool_t fgPawInit;
   static Int_t *fgLuns;

public:

   THbookFile();
   THbookFile(const char *fname, Int_t lrecl=1024);
   ~THbookFile() override;
   void      Browse(TBrowser *b) override;
   virtual Bool_t    cd(const char *dirname="");
   virtual void      Close(Option_t *option="");
   virtual TFile    *Convert2root(const char *rootname="", Int_t lrecl=0, Option_t *option=""); // *MENU*
   virtual TObject  *ConvertCWN(Int_t id);
   virtual TObject  *ConvertRWN(Int_t id);
   virtual TObject  *ConvertProfile(Int_t id);
   virtual TObject  *Convert1D(Int_t id);
   virtual TObject  *Convert2D(Int_t id);
           void      DeleteID(Int_t id);
   TObject  *FindObject(const char *name) const override;
   TObject  *FindObject(const TObject *obj) const override;
   TObject          *Get(Int_t id);
   const char       *GetCurDir() const {return fCurDir.Data();}
   Int_t             GetEntry(Int_t entry,Int_t id, Int_t atype, Float_t *x);
   Int_t             GetEntryBranch(Int_t entry,Int_t id);
   Long64_t          GetSize() const {return 0;}
   TList            *GetList() const {return fList;}
   TList            *GetListOfKeys() const { return fKeys; }
   void              InitLeaves(Int_t id, Int_t var, TTreeFormula *formula);
   Bool_t            IsFolder() const override { return kTRUE; }
   virtual Bool_t    IsOpen() const;
   void      ls(const char *path="") const override;
   virtual void      SetBranchAddress(Int_t id, const char *bname, void *add);

   ClassDefOverride(THbookFile,1)  //ROOT interface to Hbook/PAW files
};

#endif
