// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXNetFile
#define ROOT_TXNetFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXNetFile                                                            //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
// TXNetFile is an extension of TNetFile able to deal with new xrootd   //
// server. Its new features are:                                        //
//  - Automatic server kind recognition (xrootd load balancer, xrootd   //
//    data server, old rootd)                                           //
//  - Backward compatibility with old rootd server (acts as an old      //
//    TNetFile)                                                         //
//  - Fault tolerance for read/write operations (read/write timeouts    //
//    and retry)                                                        //
//  - Internal connection timeout (tunable indipendently from the OS    //
//    one) handled by threads                                           //
//  - handling of redirections from server                              //
//  - Single TCP physical channel for multiple TXNetFile's instances    //
//    inside the same application                                       //
//    So, each TXNetFile object client must send messages containing    //
//    its ID (streamid). The server, of course, will respond with       //
//    messages containing the client's ID, in order to make the client  //
//    able to recognize its message by matching its streamid with that  //
//    one contained in the server's response.                           //
//  - Tunable log verbosity level (0 = nothing, 3 = dump read/write     //
//    buffers too!)                                                     //
//  - Many parameters configurable via TEnv facility (see SetParm()     //
//    methods)                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNetFile
#include "TNetFile.h" 
#endif
#ifndef ROOT_TXAbsNetCommon
#include "TXAbsNetCommon.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

#define DFLT_TRYCONNECTSERVERSLIST	240

class TXNetConn;

//
// Just a container for the last parameters passed to the Open() method
//
struct Params_Open {
   Bool_t FileOpened;
   TString option;
   TString fTitle;
   Int_t compress;
   Int_t netopt;
};

class TXNetFile : public TNetFile, public TXAbsNetCommon {

private:

   Bool_t     fAlreadyStated;
   Bool_t     fAlreadyDetected;
   TXNetConn* fConnModule;
   Bool_t     fCreateMode;
   char       fHandle[4];          // The file handle returned by the server,
                                   // to use for successive requests
   Bool_t     fIsROOT;
   struct Params_Open fOpenPars;   // Just a container for the last parameters
                                   // passed to a Open method
   Bool_t     fOpenWithRefresh;
   Long64_t   fSize;

   static Bool_t fgTagAlreadyPrinted;

   void    CreateTXNf(const char *url, Option_t *option, const char* ftitle,
                         Int_t compress, Int_t netopt);
   Bool_t  LowOpen(const char* file, Option_t *option, const char* ftitle,
                   Int_t compress, Int_t netopt,
	           Bool_t DoInit, Bool_t refresh_open = kFALSE);
   Int_t   SysStat(Int_t fd, Long_t* id, Long64_t* size, Long_t* flags,
                   Long_t* modtime);
   Int_t   SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   Int_t   SysClose(Int_t fd);

public:

   TXNetFile(const char *url, Option_t *option = "", const char* fTitle = "", 
             Int_t compress = 1, Int_t netopt = -1);
   virtual ~TXNetFile();
  
   Bool_t         OpenFileWhenRedirected(char *newfhandle, Bool_t &wasopen);
   Bool_t         ProcessUnsolicitedMsg(TXUnsolicitedMsgSender *sender,
                                        TXMessage *unsolmsg);
   virtual void   Close(const Option_t *);
   virtual void   Flush();
   virtual Long_t GetRemoteFile(void**);
   virtual Bool_t IsOpen() const;
   Int_t          LastBytesRecv(void);
   Int_t          LastBytesSent(void);
   Int_t          LastDataBytesRecv(void);
   Int_t          LastDataBytesSent(void);
   Bool_t         Open(Option_t *option, const char* fTitle, Int_t compress, 
                       Int_t netopt, Bool_t DoInit);
   virtual Bool_t ReadBuffer(char *buf, Int_t len);
   virtual Int_t  ReOpen(const Option_t *mode);
   Long64_t       Size(void);
   virtual Bool_t WriteBuffer(const char *buffer, Int_t BufferLength);

   ClassDef(TXNetFile, 1) //A TNetFile extension able to deal with new xrootd server.	
};

#endif
