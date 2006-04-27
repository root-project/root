// @(#)root/netx:$Name:  $:$Id: TXNetFile.h,v 1.7 2006/04/06 23:01:45 rdm Exp $
/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXNetFile
#define ROOT_TXNetFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXNetFile                                                           //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
// Interfaced to the posix client: G. Ganis, CERN                       //
//                                                                      //
// TXNetFile is an extension of TNetFile able to deal with new xrootd  //
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
//  - Single TCP physical channel for multiple TXNetFile's instances   //
//    inside the same application                                       //
//    So, each TXNetFile object client must send messages containing   //
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
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif

class TSocket;
class XrdClient;
class XrdOucRecMutex;

class TXNetFile : public TNetFile {

friend class TXNetSystem;

private:

   // Members
   XrdClient     *fClient;       // Handle to the client object
   Long64_t       fSize;         // File size
   Bool_t         fIsRootd;      // Nature of remote file server

   // Static members
   static Bool_t  fgInitDone;    // Avoid initializing more than once
   static Bool_t  fgRootdBC;     // Control rootd backward compatibility 

   XrdOucRecMutex *fInitMtx;     // Protects fInitDone, serializes the
                                 // attempts to Init() for this object only

   // Methods
   void    CreateXClient(const char *url, Option_t *option, Int_t netopt,
                         Bool_t parallelopen);
   void    Init(Bool_t create);
   Bool_t  Open(Option_t *option, Bool_t parallelopen);
   Int_t   SysStat(Int_t fd, Long_t* id, Long64_t* size, Long_t* flags,
                   Long_t* modtime);
   Int_t   SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   Int_t   SysClose(Int_t fd);

   // Static methods
   static void FormUrl(TUrl uut, TString &uu);
   static Int_t GetRootdProtocol(TSocket *s);
   static void SetEnv();

public:
   TXNetFile() : TNetFile() { fClient = 0; fSize = 0; fIsRootd = 0;
                              fInitMtx = 0; }
   TXNetFile(const char *url, Option_t *option = "", const char* fTitle = "",
             Int_t compress = 1, Int_t netopt = -1, Bool_t parallelopen = kFALSE);
   virtual ~TXNetFile();

   virtual void   Close(const Option_t *opt ="");
   virtual void   Flush();
   TFile::EAsyncOpenStatus GetAsyncOpenStatus();
   virtual Bool_t IsOpen() const;
   virtual Bool_t ReadBuffer(char *buf, Int_t len);
   virtual Int_t  ReOpen(const Option_t *mode);
   Long64_t       Size(void);
   virtual Bool_t WriteBuffer(const char *buffer, Int_t BufferLength);

   ClassDef(TXNetFile,0) // TFile implementation to deal with new xrootd server.
};

#endif
