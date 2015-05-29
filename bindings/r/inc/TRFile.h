// @(#)root/r:$Id$
// Author: Omar Zapata   5/06/2014


/*************************************************************************
 * Copyright (C)  2014, Omar Andres Zapata Mesa                          *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_R_TRFILE
#define ROOT_R_TRFILE

#ifndef ROOT_R_RExports
#include<RExports.h>
#endif

#ifndef ROOT_TFile
#include<TFile.h>
#endif

#ifndef ROOT_R_TRF1
#include<TRF1.h>
#endif

#ifndef ROOT_R_TRGraph
#include<TRGraph.h>
#endif

#ifndef ROOT_R_TRCanvas
#include<TRCanvas.h>
#endif

//________________________________________________________________________________________________________
/**
   This is TFile class for R


   @ingroup R
*/
namespace ROOT {
   namespace R {

      class TRFile: public TFile {
      public:
         TRFile():TFile(){}
         TRFile(TString fname):TFile(fname.Data()){}
         TRFile(TString fname, TString option):TFile(fname.Data(),option.Data()){}
         TRFile(TString fname, TString option, TString ftitle):TFile(fname.Data(),option.Data()){}
	 TRFile(TString fname, TString option, TString ftitle, Int_t compress );
	 
         template<class T> T Get(TString object) {
            return *(T *)TFile::Get(object);
         }
         inline void Close() {
            TFile::Close(0);
         }
         inline void Close(TString opt) {
            TFile::Close(opt.Data());
         }
      };
   }
}

//______________________________________________________________________________
ROOT::R::TRFile::TRFile(TString fname, TString option , TString ftitle, Int_t compress): TFile(fname.Data(), option.Data(), ftitle.Data(), compress)
{

}

ROOTR_MODULE(ROOTR_TRFile)
{

   ROOT::R::class_<ROOT::R::TRFile>("TRFile", "TFile class to manipulate ROOT's files.")
   .constructor<TString>()
   .constructor<TString , TString>()
   .constructor<TString , TString, TString>()
   .constructor<TString , TString, TString , Int_t>()
   .method("Map", (void (ROOT::R::TRFile::*)())&ROOT::R::TRFile::Map)
   .method("ls", (void (ROOT::R::TRFile::*)(TString))&ROOT::R::TRFile::ls)
   .method("Flush", (void (ROOT::R::TRFile::*)())&ROOT::R::TRFile::Flush)
   .method("Close", (void (ROOT::R::TRFile::*)(TString))&ROOT::R::TRFile::Close)
   .method("Close", (void (ROOT::R::TRFile::*)())&ROOT::R::TRFile::Close)
   .method("Get", &ROOT::R::TRFile::Get<ROOT::R::TRF1>)
   .method("Get", &ROOT::R::TRFile::Get<ROOT::R::TRGraph>)
//    .method("Get", &ROOT::R::TRFile::Get<ROOT::R::TRCanvas>)//TRCanvas no supported at the moment
   ;
}


#endif
