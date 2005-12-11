// @(#)root/monalisa:$Name:  $:$Id: TMonaLisa.h,v 1.11 2005/09/23 13:04:53  rdm Exp $
// Author: Andreas Peters   5/10/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMonaLisa
#define ROOT_TMonaLisa

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef __CINT__
#include <ApMon.h>
#else
struct ApMon;
#endif


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMonaLisa                                                            //
//                                                                      //
// Class defining interface to MonaLisa Monitoring Services in ROOT     //
// The TMonaLisa object is used to send monitoring information to a     //
// MonaLisa server using the MonaLisa ApMon package (libapmoncpp.so/UDP //
// packets). The MonaLisa ApMon library for C++ can be downloaded at    //
// http://monalisa.cacr.caltech.edu/monalisa__Download__ApMon.html,     //
// current version:                                                     //
//http://monalisa.cacr.caltech.edu/download/apmon/ApMon_cpp-2.0.6.tar.gz//
//                                                                      //
// The ROOT implementation is primary optimized for process/job         //
// monitoring, although all other generic MonaLisa ApMon functionality  //
// can be exploited through the ApMon class directly                    //
// (gMonaLisa->GetApMon()).                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMonaLisaValue : public TNamed {

private:
   Double_t fValue;  // double monitor value

public:
   TMonaLisaValue(const char *name, Double_t value)
      : TNamed(name, ""), fValue(value) { }
   virtual ~TMonaLisaValue() { }

   Double_t  GetValue() const { return fValue; }
   Double_t *GetValuePtr() { return &fValue; }

   ClassDef(TMonaLisaValue, 1)  // Interface to MonaLisa Monitoring Values
};


class TMonaLisaText : public TNamed {

private:
   TString fText;  // text monitor value

public:
   TMonaLisaText(const char *name, const char *text)
      : TNamed(name, ""), fText(text) { }
   virtual ~TMonaLisaText() { }

   const char *GetText() const { return fText; }

   ClassDef(TMonaLisaText, 1)   // Interface to MonaLisa Monitoring Text
};


class TMonaLisa : public TNamed {

private:
   ApMon   *fApmon;              //! connection to MonaLisa
   char    *fJobId;              //! job id
   TString  fHostname;           //! hostname of MonaLisa server
   Int_t    fPid;                //! process id
   Bool_t   fInitialized;        // true if initialized
   Bool_t   fVerbose;            // verbocity

public:
   TMonaLisa(const char *monid = 0, const char *montag = "ROOT_PROCESS",
             const char *monserver = 0);
   virtual ~TMonaLisa();

   ApMon *GetApMon() const { return fApmon; }

   Bool_t SendParameters(TList *valuelist);
   Bool_t SendInfoTime();
   Bool_t SendInfoUser(const char *user = 0);
   Bool_t SendInfoDescription(const char *jobtag);
   Bool_t SendInfoStatus(const char *status);

   Bool_t SendProcessingStatus(const char *status);
   Bool_t SendProcessingProgress(Double_t nevent, Double_t nbytes);

   void   SetLogLevel(const char *loglevel = "WARNING");

   void   Verbose(Bool_t onoff) { fVerbose = onoff; }

   void   Print(Option_t *option = "") const;

   ClassDef(TMonaLisa, 1)   // Interface to MonaLisa Monitoring
};

extern TMonaLisa *gMonaLisa;

#endif
