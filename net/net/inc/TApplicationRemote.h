// @(#)root/net:$Id$
// Author: G. Ganis  10/5/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TApplicationRemote
#define ROOT_TApplicationRemote

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TApplicationRemote                                                   //
//                                                                      //
// TApplicationRemote maps a remote session. It starts a remote session //
// and takes care of redirecting the commands to be processed to the    //
// remote session, to collect the graphic output objects and to display //
// them locally.                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RRemoteProtocol.h"
#include "TApplication.h"
#include "TMD5.h"
#include "TUrl.h"
#include "TNamed.h"
#include "TMessage.h"
#include "TSysEvtHandler.h"
#include "RtypesCore.h"  // for Longptr_t


class THashList;
class TMonitor;
class TSocket;
class TBrowser;
class TRemoteObject;
class TSeqCollection;

class TApplicationRemote : public TApplication {

public:
   enum ESendFileOpt {
      kAscii  = 0x0,
      kBinary = 0x1,
      kForce  = 0x2
   };
   // TApplication specific bits
   enum EStatusBits {
      kCollecting = BIT(17)   // TRUE while collecting from server
   };

private:
   class TARFileStat : public TNamed {
      public:
         TARFileStat(const char *fn, TMD5 *md5, Long_t mt) :
                     TNamed(fn,fn), fMD5(*md5), fModtime(mt) { }
         TMD5   fMD5;        //file's md5
         Long_t fModtime;    //file's modification time
   };

   TString            fName;           //Unique name identifying this instance
   Int_t              fProtocol;       //server protocol version number
   TUrl               fUrl;            //server's url
   TSocket           *fSocket;         //socket connection to server
   TMonitor          *fMonitor;        //monitor for the input socket
   Bool_t             fInterrupt;      //flag interrupt state
   TSignalHandler    *fIntHandler;     //interrupt signal handler (ctrl-c)

   TString            fLogFilePath;    //Full remote path to log file
   THashList         *fFileList;       // List of files already uploaded

   TObject           *fReceivedObject; // last received object
   TSeqCollection    *fRootFiles;      // list of (remote) root files
   TRemoteObject     *fWorkingDir;     // working (remote) directory

   static Int_t       fgPortAttempts;  // number of attempts to find a port
   static Int_t       fgPortLower;     // lower bound for ports
   static Int_t       fgPortUpper;     // upper bound for ports

   Int_t         Broadcast(const TMessage &mess);
   Int_t         Broadcast(const char *mess, Int_t kind = kMESS_STRING, Int_t type = kRRT_Undef);
   Int_t         Broadcast(Int_t kind, Int_t type = kRRT_Undef) { return Broadcast(nullptr, kind, type); }
   Int_t         BroadcastObject(const TObject *obj, Int_t kind = kMESS_OBJECT);
   Int_t         BroadcastRaw(const void *buffer, Int_t length);
   Bool_t        CheckFile(const char *file, Long_t modtime);
   Int_t         Collect(Long_t timeout = -1);
   Int_t         CollectInput();

   void          RecvLogFile(Int_t size);

public:
   TApplicationRemote(const char *url, Int_t debug = 0, const char *script = nullptr);
   virtual ~TApplicationRemote();

   void          Browse(TBrowser *b) override;
   Bool_t        IsFolder() const override { return kTRUE; }
   const char   *ApplicationName() const override { return fName; }
   Longptr_t     ProcessLine(const char *line, Bool_t /*sync*/ = kFALSE, Int_t *error = nullptr) override;

   Int_t         SendFile(const char *file, Int_t opt = kAscii,
                          const char *rfile = nullptr);
   Int_t         SendObject(const TObject *obj);

   void          Interrupt(Int_t type = kRRI_Hard);
   Bool_t        IsValid() const { return (fSocket) ? kTRUE : kFALSE; }

   void          Print(Option_t *option="") const override;

   void          Terminate(Int_t status = 0) override;

   static void   SetPortParam(Int_t lower = -1, Int_t upper = -1, Int_t attempts = -1);

   ClassDefOverride(TApplicationRemote,0)  //Remote Application Interface
};

//
// TApplicationRemote Interrupt signal handler
//
class TARInterruptHandler : public TSignalHandler {
private:
   TApplicationRemote *fApplicationRemote;
public:
   TARInterruptHandler(TApplicationRemote *r)
      : TSignalHandler(kSigInterrupt, kFALSE), fApplicationRemote(r) { }
   Bool_t Notify() override;
};

#endif
