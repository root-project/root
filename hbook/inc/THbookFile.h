// @(#)root/hbook:$Name:$:$Id:$
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

#ifndef ROOT_TFile
#include "TFile.h"
#endif


class THbookFile : public TFile {

protected:
   Int_t         fLun;     //Fortran logical unit for this file

   static Bool_t fgPawInit;
   static Int_t *fgLuns;

public:

   THbookFile();
   THbookFile(const char *fname, Int_t lrecl=1024);
   virtual ~THbookFile();
   virtual Bool_t    cd(const char *dirname="");
   virtual void      Close(Option_t *option="") {;}
   virtual TObject  *ConvertCWN(Int_t id);
   virtual TObject  *ConvertRWN(Int_t id);
   virtual TObject  *ConvertProfile(Int_t id);
   virtual TObject  *Convert1D(Int_t id);
   virtual TObject  *Convert2D(Int_t id);
   virtual void      Copy(TObject &) { MayNotUse("Copy(TObject &)"); }
   virtual void      Delete(const char *namecycle="") {;}
           void      DeleteID(Int_t id);
   virtual void      Flush() {;}
   TObject          *Get(Int_t id);
   Int_t             GetBestBuffer() const {return 1024;}
   TArrayC          *GetClassIndex() const { return 0; }
   Int_t             GetCompressionLevel() const { return 0; }
   Float_t           GetCompressionFactor() {return 1;}
   Int_t             GetEntry(Int_t entry,Int_t id, Int_t atype, Float_t *x);
   Int_t             GetEntryBranch(Int_t entry,Int_t id, const char *blockname, const char *branchname);
   Int_t             GetVersion() const { return 1; }
   Seek_t            GetSize() const {return 0;}
   virtual Bool_t    IsOpen() const {return kTRUE;}
   virtual void      ls(Option_t *option="") const;
   virtual void      Map() {;}
   virtual void      Print(Option_t *option="") const {;}

   ClassDef(THbookFile,1)  //ROOT interface to Hbook/PAW files
};

#endif
