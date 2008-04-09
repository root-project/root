// @(#)root/net:$Id$
// Author: G. Ganis  10/5/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TApplicationServer
#define ROOT_TApplicationServer

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TApplicationServer                                                   //
//                                                                      //
// TApplicationServer is the remote application run by the roots main   //
// program. The input is taken from the socket connection to the client.//
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TApplication
#include "TApplication.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TSysEvtHandler
#include "TSysEvtHandler.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif

class TList;
class TMessage;
class TSocket;
class TRemoteObject;

class TApplicationServer : public TApplication {

private:
   Int_t         fProtocol;         //user protocol version number
   TUrl          fUrl;              //user's url
   TSocket      *fSocket;           //socket connection to user
   Bool_t        fIsValid;          //flag validity
   Bool_t        fInterrupt;        //flag interrupt state

   TString       fLogFilePath;      //Path to log file
   FILE         *fLogFile;          //log file
   Int_t         fLogFileDes;       //log file descriptor
   Bool_t        fRealTimeLog;      //TRUE if log messages should be send back in real-time

   TString       fSessId;           // Identifier for this session

   TString       fWorkDir;          // Working dir

   TList        *fSentCanvases;     // List of canvases already sent
   TRemoteObject *fWorkingDir;      // Working (remote) directory

   void          ExecLogon();
   Int_t         Setup();
   Int_t         SendCanvases();    // Send back to client any created canvas

protected:
   void          HandleCheckFile(TMessage *mess);

   static void   ErrorHandler(Int_t level, Bool_t abort, const char *location,
                              const char *msg);

public:
   TApplicationServer(Int_t *argc, char **argv, FILE *flog, const char *logfile);
   virtual ~TApplicationServer();

   void           GetOptions(Int_t *argc, char **argv);
   Int_t          GetProtocol() const   { return fProtocol; }
   Int_t          GetPort() const       { return fUrl.GetPort(); }
   const char    *GetUser() const       { return fUrl.GetUser(); }
   const char    *GetHost() const       { return fUrl.GetHost(); }
   TSocket       *GetSocket() const     { return fSocket; }

   void           HandleSocketInput();
   void           HandleUrgentData();
   void           HandleSigPipe();
   void           Interrupt() { fInterrupt = kTRUE; }
   Bool_t         IsValid() const { return fIsValid; }

   Long_t         ProcessLine(const char *line, Bool_t = kFALSE, Int_t *err = 0);

   void           Reset(const char *dir);
   Int_t          ReceiveFile(const char *file, Bool_t bin, Long64_t size);
   void           Run(Bool_t retrn = kFALSE);
   void           SendLogFile(Int_t status = 0, Int_t start = -1, Int_t end = -1);
   Int_t          BrowseDirectory(const char *dirname);
   Int_t          BrowseFile(const char *fname);
   Int_t          BrowseKey(const char *keyname);

   void           Terminate(Int_t status);

   ClassDef(TApplicationServer,0)  //Remote Application Interface
};


//----- Handles output from commands executed externally via a pipe. ---------//
//----- The output is redirected one level up (i.e., to master or client). ---//
//______________________________________________________________________________
class TASLogHandler : public TFileHandler {
private:
   TSocket     *fSocket; // Socket where to redirect the message
   FILE        *fFile;   // File connected with the open pipe
   TString      fPfx;    // Prefix to be prepended to messages

   static TString fgPfx; // Default prefix to be prepended to messages
public:
   enum EStatusBits { kFileIsPipe = BIT(23) };
   TASLogHandler(const char *cmd, TSocket *s, const char *pfx = "");
   TASLogHandler(FILE *f, TSocket *s, const char *pfx = "");
   virtual ~TASLogHandler();

   Bool_t IsValid() { return ((fFile && fSocket) ? kTRUE : kFALSE); }

   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }

   static void SetDefaultPrefix(const char *pfx);
};

//--- Guard class: close pipe, deactivatethe related descriptor --------------//
//______________________________________________________________________________
class TASLogHandlerGuard {

private:
   TASLogHandler   *fExecHandler;

public:
   TASLogHandlerGuard(const char *cmd, TSocket *s,
                      const char *pfx = "", Bool_t on = kTRUE);
   TASLogHandlerGuard(FILE *f, TSocket *s,
                      const char *pfx = "", Bool_t on = kTRUE);
   virtual ~TASLogHandlerGuard();
};

#endif
