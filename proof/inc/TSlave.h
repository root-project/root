// @(#)root/proof:$Name:  $:$Id: TSlave.h,v 1.3 2000/12/13 12:07:59 rdm Exp $
// Author: Fons Rademakers   14/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSlave
#define ROOT_TSlave


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSlave                                                               //
//                                                                      //
// This class describes a PROOF slave server.                           //
// It contains information like the slaves host name, ordinal number,   //
// performance index, socket, etc. Objects of this class can only be    //
// created via TProof member functions.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TSocket;
class TFileHandler;


class TSlave : public TObject {

friend class TProof;

private:
   TString       fName;      //slave's hostname
   TString       fImage;     //slave's image name
   TString       fWorkDir;   //slave's working directory (info obtained from slave)
   TString       fUser;      //slave's user id
   Int_t         fPort;      //slave's port number
   Int_t         fOrdinal;   //slave's ordinal number
   Int_t         fPerfIdx;   //relative CPU performance index
   Int_t         fSecurity;  //authentication method (0 = standard, 1 = SRP)
   TSocket      *fSocket;    //socket to slave
   TProof       *fProof;     //proof cluster to which slave belongs
   TFileHandler *fInput;     //input handler related to this slave
   Double_t      fBytesRead; //bytes read by slave (info is obtained from slave)
   Float_t       fRealTime;  //real time spent executing commands (info obtained from slave)
   Float_t       fCpuTime;   //CPU time spent executing commands (info obtained from slave)

   TSlave() { fOrdinal = -1; fSocket = 0; fProof = 0; }
   TSlave(const TSlave &) { }
   void operator=(const TSlave &) { }

   TSlave(const char *host, Int_t port, Int_t ord, Int_t perf,
          const char *image, const char *user, Int_t security, TProof *proof);

public:
   virtual ~TSlave();

   void          Close(Option_t *opt = "");

   Int_t         Compare(const TObject *obj) const;
   Bool_t        IsSortable() const { return kTRUE; }

   const char   *GetName() const { return fName.Data(); }
   const char   *GetImage() const { return fImage.Data(); }
   const char   *GetWorkingDirectory() const { return fWorkDir.Data(); }
   const char   *GetUser() const { return fUser.Data(); }
   Int_t         GetPort() const { return fPort; }
   Int_t         GetOrdinal() const { return fOrdinal; }
   Int_t         GetPerfIdx() const { return fPerfIdx; }
   TSocket      *GetSocket() const { return fSocket; }
   TProof       *GetProof() const { return fProof; }
   Double_t      GetBytesRead() const { return fBytesRead; }
   Float_t       GetRealTime() const { return fRealTime; }
   Float_t       GetCpuTime() const { return fCpuTime; }
   TFileHandler *GetInputHandler() const { return fInput; }
   void          SetInputHandler(TFileHandler *ih);

   Bool_t        IsValid() const { return fSocket ? kTRUE : kFALSE; }

   void          Print(Option_t *option="") const;

   ClassDef(TSlave,0)  //PROOF slave server
};

#endif
