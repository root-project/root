// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TError.h"
#include "TEnv.h"
#include "TXNetFile.h"
#include "TXDebug.h"
#include "TXError.h"
#include "TXUrl.h"
#include "TXNetConn.h"

#include <strings.h> 
#include <string.h>
#include <unistd.h>

ClassImp(TXNetFile);

void (*evtFunc)();

Bool_t TXNetFile::fgTagAlreadyPrinted = kFALSE;

//_____________________________________________________________________________
TXNetFile::TXNetFile(const char *url, Option_t *option, const char* ftitle, 
		     Int_t compress, Int_t netopt) : 
              TNetFile(url, ftitle, compress, kFALSE)
{
  // Create a TXNetFile object. A TXNetFile object is the same as a TNetFile 
  // (from which the former derives) except that the protocol is extended to 
  // support dealing with new xrootd data server or xrootd load balancer 
  // server.
  //
  // The "url" argument must be of the form 
  //
  //   root://server1:port1[,server2:port2,...,serverN:portN]/pathfile,
  //
  // Note that this means that multiple servers (>= 1) can be specified in 
  // the url. The connection will try to connect to the first server:port
  // and if that does not succeed, it will try the second one, and so on
  // until it finds a server that will respond.
  //
  // See the TNetFile documentation for the description of the other arguments.
  //
  // The creation consists of internal variable settings (most important is 
  // the client's domain), creation of a TXUrl array containing all specified 
  // urls (a single url is serverX:portX/pathfile), trying to connect to the 
  // servers calling Connect() method, getting a valid access to the remote 
  // server the client is connected to using GetAccessToSrv() method, 
  // recognizing the remote server (if an old rootd the TNetFile's Create 
  // method will be called)

  CreateTXNf(url, option, ftitle, compress, netopt);
}

//_____________________________________________________________________________
TXNetFile::~TXNetFile()
{
   // Destructor
   if (IsOpen()) Close(0);

   SafeDelete(fConnModule);
}

//_____________________________________________________________________________
void TXNetFile::CreateTXNf(const char *url, Option_t *option, const char* ftitle, 
		       Int_t compress, Int_t netopt) {


  short locallogid;
  Bool_t validDomain = kFALSE;
  fOpenWithRefresh = kFALSE;
  fAlreadyStated = kFALSE;
  fCreateMode = kFALSE;
  fAlreadyDetected = kFALSE;
  fIsROOT = kFALSE;
  fSize = 0;
  if (gEnv->GetValue("XNet.PrintTAG",0) == 1)
     if (!fgTagAlreadyPrinted) {
        Info("CreateTXNf","(C) 2004 SLAC TXNetFile (eXtended TNetFile) %s");
        fgTagAlreadyPrinted = kTRUE;
     }

  // Setup modified Error Handler which prints time stamps. Note that this
  // check will be done once per TXNetFile creation. It probably should be
  // moved someplace where it gets done only once per job. (And probably
  // could even be a standard option of the default ErrorHandler.)
  if (gEnv->GetValue("XNet.DebugTimestamp",0) == 1) {
     SetErrorHandler(TXNErrorHandler);
  }

  // Using ROOT mechanism to IGNORE SIGPIPE signal
  gSystem->IgnoreSignal(kSigPipe);

  fOpenPars.FileOpened = kFALSE;
  // But we initialize the internal params...
  fOpenPars.option = "";
  if (option)
    fOpenPars.option = option;
  
  fOpenPars.fTitle = "";
  if (ftitle)
    fOpenPars.fTitle = ftitle;
  
  fOpenPars.compress = compress;
  fOpenPars.netopt = netopt;
  
  // Now we try to set up the first connection
  // We cycle through the list of urls given as a single TUrl parameter
  
  fConnModule = new TXNetConn();
  if (!fConnModule) {
     Error("CreateTXNf","Fatal ERROR *** Object creation with new failed !"
                  " Probable system resources exhausted.");
     gSystem->Abort();
  }
  fConnModule->SetRedirHandler(this);

  // Max number of tries
  Int_t connectMaxTry = gEnv->GetValue("XNet.TryConnectServersList",
                                       DFLT_TRYCONNECTSERVERSLIST);
  // List of regular expressions to match
  TString allowRE = gEnv->GetValue("XNet.ConnectDomainAllowRE",
                                   fConnModule->GetClientHostDomain().Data());
  TString denyRE  = gEnv->GetValue("XNet.ConnectDomainDenyRE",
                                   "<unknown>");

  TXUrl urlArray(url);
  if (!urlArray.IsValid()) {
     Error("CreateTXNf", "The URL(s) provided are incorrect."
                     " Going into zombie state.");
     goto zombie;
  }

  for (Short_t jj=0; jj <=urlArray.Size()-1; jj++) {
     TUrl *thisUrl;
     thisUrl = urlArray.GetNextUrl();
     fUrl = *thisUrl;
     if (fConnModule->CheckHostDomain(fUrl.GetHost(), allowRE, denyRE)) {
	validDomain = kTRUE;
	break;
     }
  }

  if (!validDomain) {
     Error("CreateTXNf", "All the specified servers are disallowed. "
                     "Going into zombie state.");
     goto zombie;
  }

  urlArray.Rewind();
  locallogid = -1;
  for (Int_t connectTry = 0;
      (connectTry < connectMaxTry) && (!fConnModule->IsConnected()); 
       connectTry++) {

     TUrl *thisUrl;
     
     // Get an url from the available set
     thisUrl = urlArray.GetARandomUrl();
     
     if (thisUrl) {
        fUrl = *thisUrl;
        if (fConnModule->CheckHostDomain(fUrl.GetHost(), allowRE, denyRE)) {
           if (DebugLevel() >= TXDebug::kHIDEBUG)
              Info("CreateTXNf", "Trying to connect to %s:%d. Connect try %d.",
                           fUrl.GetHost(), fUrl.GetPort(), connectTry+1);
           locallogid = fConnModule->Connect(fUrl.GetHost(), fUrl.GetPort(), netopt);
        }
     }
     
     // We are connected to a host. Let's handshake with it.
     if (fConnModule->IsConnected()) {

        // Now the have the logical Connection ID, that we can use as streamid for 
        // communications with the server
        if (DebugLevel() >= TXDebug::kHIDEBUG)
           Info("CreateTXNf", "The logical connection id is %d. This will be the"
                        " streamid for this client",fConnModule->GetLogConnID());
        fConnModule->SetUrl(fUrl);
        
        if (DebugLevel() >= TXDebug::kHIDEBUG)
           Info("CreateTXNf", "Working url is [%s]", fUrl.GetUrl());
        
        // after connection deal with server
        if (!fConnModule->GetAccessToSrv())
           Error("CreateTXNf", "Access to server failed"); 
        else {
	   if (DebugLevel() >= TXDebug::kUSERDEBUG)
	      Info("CreateTXNf", "Access to server granted.");
           break;
	}
     }
     
     // We force a physical disconnection in this special case
     if (DebugLevel() >= TXDebug::kHIDEBUG)
        Info("CreateTXNf", "Disconnecting.");
     
     fConnModule->Disconnect(kTRUE);
     
     if (DebugLevel() >= TXDebug::kUSERDEBUG)
        Info("CreateTXNf", "Connection attempt cycle failed. Sleeping %d seconds.",
                     gEnv->GetValue("XNet.ReconnectTimeout",
                                    DFLT_RECONNECTTIMEOUT));
     
     gSystem->Sleep(1000 * gEnv->GetValue("XNet.ReconnectTimeout",
                                         DFLT_RECONNECTTIMEOUT) );

  } //for connect try

  if (!fConnModule->IsConnected()) {
     Error("CreateTXNf", "Some severe error occurred while opening a connection"
                     " with the servers [%s]. Program exits",
                     urlArray.GetServers().Data());
     goto zombie;
  }

  if (locallogid != fConnModule->GetLogConnID()) {
     Error("CreateTXNf", "Internal error. The logids do not match (%d, %d)",
                     locallogid, fConnModule->GetLogConnID());
     //    abort();
     goto zombie;
  }

  
  //
  // Variable initialization
  // If the server is a new xrootd ( load balancer or data server)
  //
  if ((fConnModule->GetServerType() != TXNetConn::kSTRootd) && 
      (fConnModule->GetServerType() != TXNetConn::kSTNone)) {
     // Now we are connected to a server that didn't redirect us after the 
     // login/auth phase
     // let's continue with the openfile sequence
     // We try to recycle the TNetFile procedure as much as possible, traslating 
     // the TNetFile's open modes into the kXR_xxx modes...
     if (DebugLevel() >= TXDebug::kUSERDEBUG)
	Info("CreateTXNf", "Opening the remote file %s", fUrl.GetFile()); 

     // url is not needed because already stored in fUrl
     // Here we have to init our TFile ancestor
     if (!Open(option, fTitle, compress, netopt, kTRUE)) {
	Error("CreateTXNf", "Error opening the file %s on host %s:%d", 
                        fUrl.GetFile(), fUrl.GetHost(), fUrl.GetPort());
	  goto zombie;
     } else {
	if (DebugLevel() >= TXDebug::kUSERDEBUG)
	   Info("CreateTXNf", "File opened succesfully.");
     }
  } else {
     // the server is an old rootd
     if (fConnModule->GetServerType() == TXNetConn::kSTRootd) {
        TNetFile::Create(fUrl.GetUrl(), option, netopt);
     }
     if (fConnModule->GetServerType() == TXNetConn::kSTNone) {
        goto zombie;
     }
  }

  return;

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();

}

//_____________________________________________________________________________
Bool_t TXNetFile::ReadBuffer(char *buffer, Int_t BufferLength)
{
   // Override TNetFile::ReadBuffer to deal with the xrootd server.
   // Returns kTRUE in case of errors.

   if (IsZombie()) {
      Error("ReadBuffer", "ReadBuffer is not possible because object"
            " is in 'zombie' state");
      return kTRUE;
   }

   if (fConnModule->GetServerType() == TXNetConn::kSTRootd) {
      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("ReadBuffer","Calling TNetFile::ReadBuffer");
      return TNetFile::ReadBuffer(buffer, BufferLength);
   }

   if (!IsOpen()) {
      Error("ReadBuFfer","The remote file %s is not open", fUrl.GetFile());
      return kTRUE;
   }

   Bool_t result = kFALSE;

   if (fCache) {
      Int_t st;
      Long64_t off = fOffset;
      if ((st = fCache->ReadBuffer(fOffset, buffer, BufferLength)) < 0) {
         Error("ReadBuffer", "error reading from cache");
         return kTRUE;
      }
      if (st > 0) {
         // fOffset might have been changed via TCache::ReadBuffer(), reset it
         Seek(off + BufferLength);
         return result;
      }
   }

   // Prepare request
   ClientRequest readFileRequest;
   memset( &readFileRequest, 0, sizeof(readFileRequest) );
   fConnModule->SetSID(readFileRequest.header.streamid);
   readFileRequest.read.requestid = kXR_read;
   memcpy( readFileRequest.read.fhandle, fHandle, sizeof(fHandle) );
   readFileRequest.read.offset = fOffset;
   readFileRequest.read.rlen = BufferLength;
   readFileRequest.read.dlen = 0;

   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("ReadBuffer", "Calling TXNetConn::SendGenCommand to read %d"
           " bytes of data at offset %Ld.",
           readFileRequest.read.rlen, 
           readFileRequest.read.offset);

   // Original version, without caching
   result = !fConnModule->SendGenCommand(&readFileRequest, 0, 0, buffer,
                                         kFALSE, (char *)"TXNetFile::ReadBuffer");  
   if (!result) {
      fOffset += BufferLength;
      fBytesRead += BufferLength;
#ifdef WIN32
      SetFileBytesRead(GetFileBytesRead() + BufferLength);
#else
      fgBytesRead += BufferLength;
#endif
   }
   return result;
}

//_____________________________________________________________________________
Bool_t TXNetFile::WriteBuffer(const char *buffer, Int_t BufferLength)
{
   // Override TNetFile::WriteBuffer to deal with the xrootd server.
   // Returns kTRUE in case of errors.

   if (IsZombie()) {
      Error("WriteBuffer", "WriteBuffer is not possible because object"
            " is in 'zombie' state");
      return kTRUE;
   }

   fAlreadyStated = kFALSE;

   if (fConnModule->GetServerType() == TXNetConn::kSTRootd) {
      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("WriteBuffer","Calling TNetFile::WriteBuffer");
      return TNetFile::WriteBuffer(buffer, BufferLength );
   }
  
   if (!IsOpen()) {
      Error("WriteBuffer","The remote file %s is not open", fUrl.GetFile());
      return kTRUE;
   }

   Bool_t result = kFALSE;

   if (fCache) {
      Int_t st;
      Long64_t off = fOffset;
      if ((st = fCache->WriteBuffer(fOffset, buffer, BufferLength)) < 0) {
         SetBit(kWriteError);
         Error("WriteBuffer", "error writing to cache");
         return kTRUE;
      }
      if (st > 0) {
         // fOffset might have been changed via TCache::WriteBuffer(), reset it
         Seek(off + BufferLength);
         return result;
      }
   }

   // Prepare request
   ClientRequest writeFileRequest;
   memset( &writeFileRequest, 0, sizeof(writeFileRequest) );
   fConnModule->SetSID(writeFileRequest.header.streamid);
   writeFileRequest.write.requestid = kXR_write;
   memcpy( writeFileRequest.write.fhandle, fHandle, sizeof(fHandle) );
   writeFileRequest.write.offset = fOffset;
   writeFileRequest.write.dlen = BufferLength;
  
   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("WriteBuffer", "Calling TXNetConn::SendGenCommand...");

   result = !fConnModule->SendGenCommand(&writeFileRequest, buffer, 0, 0,
                                kFALSE, (char *)"TXNetFile::WriteBuffer");
   if (!result) {
      fOffset += BufferLength;
      fBytesWrite += BufferLength;
#ifdef WIN32
      SetFileBytesWritten(GetFileBytesWritten() + BufferLength);
#else
      fgBytesWrite += BufferLength;
#endif
   }
   return result;
}

//_____________________________________________________________________________
Bool_t TXNetFile::IsOpen() const
{
  // Return kTRUE if the file is open, kFALSE otherwise

  if (fConnModule->GetServerType() == TXNetConn::kSTRootd) {
     if (DebugLevel() >= TXDebug::kHIDEBUG)
        Info("IsOpen","Calling TNetFile::IsOpen");
     return TNetFile::IsOpen();
  }

  return fOpenPars.FileOpened;
}

//_____________________________________________________________________________
Int_t TXNetFile::ReOpen(const Option_t *Mode)
{
  // Re-open the file (see TNetFile::ReOpen() or TFile::ReOpen() 
  // for more details)

  if (fConnModule->GetServerType() == TXNetConn::kSTRootd) {
     if (DebugLevel() >= TXDebug::kHIDEBUG)
        Info("ReOpen","Calling TNetFile::ReOpen");
     return TNetFile::ReOpen(Mode);
  }
  
  fAlreadyStated = kFALSE;
  fSize = 0;
  
  return TFile::ReOpen(Mode);
}

//_____________________________________________________________________________
void TXNetFile::Close(const Option_t *opt)
{
  // Close the file (see TNetFile::Close() or TFile::Close()
  // for more details)

  if (fConnModule->GetServerType() == TXNetConn::kSTRootd) {
     if (DebugLevel() >= TXDebug::kHIDEBUG)
        Info("Close","Calling TNetFile::Close");
     TNetFile::Close(opt);
     return;
  }
  
  TFile::Close(opt);

  fSize = 0;
  fAlreadyStated = kFALSE;
  fAlreadyDetected = kFALSE;
  fIsROOT = kFALSE;
}

//_____________________________________________________________________________
void TXNetFile::Flush()
{
   // Flushes un-written data

 
   if (IsZombie()) {
      Error("Flush", "Flush is not possible because object is"
                     " in 'zombie' state");
      return;
   }

   if (fConnModule->GetServerType() == TXNetConn::kSTRootd) {
      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("Flush","Calling TNetFile::Flush");
      TNetFile::Flush();
      return;
   }

   if (!IsOpen()) {
      Error("Flush","The remote file %s is not open", fUrl.GetFile());
      return;
   }

   // Prepare request
   ClientRequest flushFileRequest;
   memset( &flushFileRequest, 0, sizeof(flushFileRequest) );
   fConnModule->SetSID(flushFileRequest.header.streamid);
   flushFileRequest.sync.requestid = kXR_sync;
   memcpy(flushFileRequest.sync.fhandle, fHandle, sizeof(fHandle));
   flushFileRequest.sync.dlen = 0;
   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("Flush", "Calling TXNetConn::SendGenCommand...");
  
   Bool_t cmdres = fConnModule->SendGenCommand(&flushFileRequest, 0, 0, 0, 
                                       kFALSE, (char *)"TXNetFile::Flush");
  
   if (!cmdres)
      Error("Flush","SendGenCommand returned false. Command failed.");
}

//_____________________________________________________________________________
Bool_t TXNetFile::Open(Option_t *option, const char* ftitle, Int_t compress,
		       Int_t netopt, Bool_t DoInit)
{
  // High level open routine; if the remote server is an old rootd the file 
  // has been already opened by the TNetFile's Create() method

  fAlreadyStated = kFALSE;
  fSize = 0;

  if (fConnModule->GetServerType() == TXNetConn::kSTRootd) {
    // Do nothing because the file is already open
    // In fact when TNetFile is first instantiated it immediately open the 
    // remote file
    return kTRUE;
  }
  
  // First attempt to open a remote file without the kXR_refresh option ON
  Bool_t lowopenRes = LowOpen(fUrl.GetFile(), option, ftitle, compress,
                              netopt, DoInit);
  if (lowopenRes) {
     // Let's remember that we succesfully opened a file without refresh
     fOpenWithRefresh = kFALSE;
     return kTRUE;
  }

  // If the open request failed for the error "file not found" proceed, 
  // otherwise return kFALSE
  if (fConnModule->GetOpenError() != kXR_NotFound)
     return kFALSE;

  // If connected to a load balancer that says "File not Found" then we
  // try again one more time with refresh before giving up
  if ((fConnModule->GetServerType() == TXNetConn::kSTBaseXrootd) &&
      (!fOpenWithRefresh)) {
     Info("Open", "Trying to re-open the file with REFRESH option...");

     if (!LowOpen(fUrl.GetFile(), option, ftitle, compress, 
                  netopt, DoInit, kTRUE)) {
	// Even if after a "resfresh-ed open" the file has not been found
	// goto in zombie state and return; and let's remember that we used 
        // the refresh option
	//	if(DebugLevel() >= TXDebug::kUSERDEBUG)
	Error("Open", "Error opening the remote file even after a refresh"
                      " of the load balance. Going into 'zombie' state");
	fOpenWithRefresh = kTRUE;
	return kFALSE;
     } else {
	// Open succeded after the refresh; 
	// Remember that we used the refresh option in order to not use it again
	fOpenWithRefresh = kTRUE;
	return kTRUE;
     }
  }

  // If we're here who reported an open error was a data server (rootd or 
  // xrootd), then we've to check if we already tried a refresh-ed open
  // request or not...
  if (!fOpenWithRefresh) {

    // if did not use refresh in the last open, and we do not come from a 
    // load balancer then goto into zombie state and return
    if (fConnModule->GetLBSUrl() == 0) {
       Error("Open","The remote data server declared 'File Not Found'"
                    " and there's not any load balancer to go back and"
                    " refresh. Going into 'zombie' state...");
       fOpenWithRefresh = kFALSE;
       return kFALSE;
    }

    TString lbsHost = fConnModule->GetLBSUrl()->GetHost();
    Int_t   lbsPort = fConnModule->GetLBSUrl()->GetPort();

    fConnModule->Disconnect(kFALSE);

    Info("Open","The current data server did not find the file. Going back"
                " to the load balancer and reopen the file in REFRESH mode");
    fConnModule->GoToAnotherServer(lbsHost, lbsPort, netopt);

    // now try to open with refresh...
    Bool_t secondTry = LowOpen(fUrl.GetFile(), option, ftitle, compress,
                               netopt, DoInit, kTRUE);
    if (!secondTry) {
       Error("Open","File not found even after open with REFRESH mode ON.");
       return kFALSE;
    } else
       return kTRUE;
  }

  // if we're here it means that we already tried to open a file in refresh
  // mode and it failed again... then give up...
  return kFALSE;
}

//_____________________________________________________________________________
Bool_t TXNetFile::LowOpen(const char* file, Option_t *option, 
                          const char* title, Int_t compress, 
			  Int_t netopt, Bool_t DoInit, Bool_t refresh_open)
{
  // Low level Open method; deals with xrootd server

  kXR_int16 openOpt; // = 0;

  memset(&openOpt, 0, sizeof(openOpt));
  
  fOption = option;
  Bool_t forceOpen = kFALSE;
  if (option[0] == '-') {
    fOption   = &option[1];
    forceOpen = kTRUE;
    openOpt |= kXR_force;
  }
  // accept 'f', like 'frecreate' still for backward compatibility
  if (option[0] == 'F' || option[0] == 'f') {
    fOption   = &option[1];
    forceOpen = kTRUE;
    openOpt |= kXR_force;
  }
  Bool_t forceRead = kFALSE;
  if (!strcasecmp(option, "+read")) {
    fOption   = &option[1];
    forceRead = kTRUE;
    openOpt |= kXR_force;
  }
  fOption.ToUpper();
  if (!fOption.CompareTo("NEW")) {
    fOption = "CREATE";
    fWritable = kTRUE;
  }
  if (!fOption.CompareTo("CREATE")) {
    fOption = "CREATE";
    fWritable = kTRUE;
  }
  if (!fOption.CompareTo("UPDATE")) {
    fOption = "UPDATE";
    fWritable = kTRUE;
  }

  Bool_t create   = (!fOption.CompareTo("CREATE"));
  Bool_t recreate = (!fOption.CompareTo("RECREATE"));
  Bool_t update   = (!fOption.CompareTo("UPDATE"));
  Bool_t read     = (!fOption.CompareTo("READ"));

  if (!create && !recreate && !update && !read) {
    read    = kTRUE;
    fOption = "READ";
  }
  
  Bool_t __recreate = kFALSE;;
  if (recreate) {
    recreate = kFALSE;
    create   = kTRUE;
    fOption  = "CREATE";
    openOpt |= kXR_delete;
    __recreate = kTRUE;
  }
  if (update)
    openOpt |= kXR_open_updt;
  if (read)
    openOpt |= kXR_open_read;
  
  fCreateMode = create;

  if (create && (!__recreate)) 
    openOpt |= kXR_new;

  // Send a kXR_open request in order to open the remote file...
  // after formatting the proper data structure...
  ClientRequest openFileRequest;

  struct ServerResponseBody_Open openresp;
  
  memset(&openFileRequest, 0, sizeof(openFileRequest));
  fConnModule->SetSID(openFileRequest.header.streamid);
  
  openFileRequest.header.requestid = kXR_open;

  // Now set the options field basing on user's requests
  memset(&openFileRequest.open.options, 0, sizeof(openFileRequest.open.options));
  memcpy(&openFileRequest.open.options, &openOpt, sizeof(openOpt));

  if( refresh_open )
    openFileRequest.open.options |= kXR_refresh;

  // Set the open mode field
  openFileRequest.open.mode = 
           kXR_or | kXR_gr | kXR_ur | kXR_uw; // open in rw-r-r mode

  // Set the length of the data (in this case data describes the path and 
  // file name)
  openFileRequest.open.dlen = strlen( file );

  // Send request to server and receive response
  bool resp = fConnModule->SendGenCommand(&openFileRequest, (const void *)file,
					 0, &openresp, kFALSE, (char *)"Open");

  if (resp) {
    // Get the file handle to use for future read/write...
    // Note that in case of heavy load the file could have been opened by an 
    // internal retry of the kxr_open
    if (!fOpenPars.FileOpened)
      memcpy( fHandle, openresp.fhandle, sizeof(fHandle) );

    fOpenPars.FileOpened = kTRUE;
    
    // The call seems successful. We copy the parameters for later use
    if (option)
      fOpenPars.option = option;
    
    if (title)
      fOpenPars.fTitle = title;
    
    fOpenPars.compress = compress;
    fOpenPars.netopt = netopt;
    
    if (DoInit) 
      Init(create);
  }

  return fOpenPars.FileOpened;
}

//_____________________________________________________________________________
Int_t TXNetFile::SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, 
                         Long_t *modtime)
{
   // Override TNetFile::SysStat (see parent's method for more details)
   
   if (IsZombie()) {
      Error("SysStat", "SysStat is not possible because object is"
                       " in 'zombie' state");
      *size = 0;
      return 0;
   }
   
   if (fConnModule->GetServerType() == TXNetConn::kSTRootd) {
      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("SysStat","Calling TNetFile::SysStat");
      return TNetFile::SysStat(fd, id, size, flags, modtime);
   }
   
   if (!IsOpen()) {
      Error("SysStat","The remote file %s is not open",fUrl.GetFile());
      *size = 0;
      return 0;
   }
   
   // Return file stat information. The interface and return value is
   // identical to TSystem::GetPathInfo().
   
   // asks the server for stat file informations
   ClientRequest statFileRequest;
   
   memset(&statFileRequest, 0, sizeof(ClientRequest));
   
   fConnModule->SetSID(statFileRequest.header.streamid);
   
   statFileRequest.stat.requestid = kXR_stat;
   memset(statFileRequest.stat.reserved, 0, 
          sizeof(statFileRequest.stat.reserved));
   statFileRequest.stat.dlen = strlen(fUrl.GetFile());
   
   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("SysStat", "Calling TXNetConn::SendGenCommand...");
   
   char fStats[2048];
   
   fConnModule->SendGenCommand(&statFileRequest, (const char*)fUrl.GetFile(),
                               0, fStats , kFALSE, (char *)"SysStat");
   
   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("SysStat", "Returned stats=[%s]",fStats);
   
   sscanf(fStats, "%ld %lld %ld %ld", id, size, flags, modtime);
   
   if (*id == -1)
      return 1;
   
   return 0;
}

//_____________________________________________________________________________
Int_t TXNetFile::SysClose(Int_t fd)
{
   // Override TNetFile::SysClose (see parent's method for more details)

   if (IsZombie()) {
      Error("SysClose", "SysClose is not possible because object is"
                        " in 'zombie' state");
      return 0;
   }

   if (fConnModule->GetServerType() == TXNetConn::kSTRootd) {
      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("SysClose","Calling TNetFile::SysClose");
      return TNetFile::SysClose(fd);
   }
  
   ClientRequest closeFileRequest;
  
   memset(&closeFileRequest, 0, sizeof(closeFileRequest) );

   fConnModule->SetSID(closeFileRequest.header.streamid);

   closeFileRequest.close.requestid = kXR_close;
   memcpy(closeFileRequest.close.fhandle, fHandle, sizeof(fHandle) );
   closeFileRequest.close.dlen = 0;
  
   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("SysClose", "Calling TXNetConn::SendGenCommand...");
  
   fConnModule->SendGenCommand(&closeFileRequest, 0,
			       0, 0, kFALSE, (char *)"Close");
  
   // No file is opened for now
   fOpenPars.FileOpened = kFALSE;

   return 0;
}

//_____________________________________________________________________________
Int_t TXNetFile::SysOpen(const char* pathname, Int_t flags, UInt_t mode)
{
  // Override TNetFile::SysOpen (see parent's method for more details)

  if (fConnModule->GetServerType() == TXNetConn::kSTRootd) {
     if (DebugLevel() >= TXDebug::kHIDEBUG)
        Info("SysOpen", "Calling TNetFile::SysOpen");
     return TNetFile::SysOpen(pathname, flags, mode);
  }
  
  // url is not needed because already stored in fUrl
  // Here we have to init our TFile ancestor
  if( !Open(fOpenPars.option.Data(), fOpenPars.fTitle.Data(), 
            fOpenPars.compress, fOpenPars.netopt, kTRUE) ) {
      Error("SysOpen", "Error opening the file %s on host %s:%d", 
                       fUrl.GetFile(),fUrl.GetHost(), fUrl.GetPort());
      return -1;
  }
  return -2;
}

//_____________________________________________________________________________
void TXNetFile::Streamer(TBuffer &R__b)
{
   // Stream an object of class TXNetFile.
   // Dummy Streamer: rootcint chokes trying to generate one, but ROOT wants
   // one to load the shared library. 

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
      R__b.CheckByteCount(R__s, R__c, TXNetFile::IsA());
   } else {
      R__c = R__b.WriteVersion(TXNetFile::IsA(), kTRUE);
      R__b.SetByteCount(R__c, kTRUE);
   }
}

//_____________________________________________________________________________
Bool_t TXNetFile::OpenFileWhenRedirected(char *newfhandle, Bool_t &wasopen)
{
   // Called by the comm module when it needs to reopen a file
   // after a redir

   wasopen = fOpenPars.FileOpened;

   if (!fOpenPars.FileOpened)
      return kTRUE;

   fOpenPars.FileOpened = kFALSE;

   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("OpenFileWhenRedirected", "Trying to reopen the same file." );

   // After a redirection we must not reinit the TFile ancestor...
   if (Open(fOpenPars.option.Data(), fOpenPars.fTitle.Data(),
            fOpenPars.compress, fOpenPars.netopt, kFALSE)) {

      fOpenPars.FileOpened = kTRUE;

      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("OpenFileWhenRedirected",
              "Open successful. (handle='%s')", fHandle );

      memcpy(newfhandle, fHandle, sizeof(fHandle));

      return kTRUE;
   } else {
      Error("OpenFileWhenRedirected", 
	    "New redir destination server refuses to open the file.");
      MakeZombie();
      return kFALSE;
   }
}

//_____________________________________________________________________________
Long_t TXNetFile::GetRemoteFile(void **bufFile)
{
   // Retrieve a remote file a store it in memory

   if (IsZombie()) {
      Error("GetRemoteFile", "GetRemoteFile is not possible because"
                             " object is in 'zombie' state");
      return -1;
   }

   if (fConnModule->GetServerType() == TXNetConn::kSTRootd) {
      Error("GetRemoteFile",
	    "Using TNetFile to talk to old rootd server, but TNetFile"
            " doesn't support GetFile method");
      return -1;
   }

   if (!IsOpen()) {
      Error("GetRemoteFile","The remote file %s is not open", fUrl.GetFile());
      return -1;
   }

   // First of all we get the file size we want to trasnfer
   Long_t id, flags, modtime;
   Long64_t size;

   this->SysStat(0/* for TNetFile compatibility */, &id, &size, &flags, &modtime);

   ClientRequest readFileRequest;
   memset(&readFileRequest, 0, sizeof(readFileRequest));
   fConnModule->SetSID(readFileRequest.header.streamid);
   readFileRequest.read.requestid = kXR_read;
   memcpy( readFileRequest.read.fhandle, fHandle, sizeof(fHandle) );
   readFileRequest.read.offset = 0;  // we want to read starting from 0 position
                                     // of the file
   readFileRequest.read.rlen = size; // we want to read all the file and put it
                                     // in memory
   readFileRequest.read.dlen = 0;
 
   // We assume the buffer has been pre-allocated to contain BufferLength
   // bytes by the caller of this function
   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("GetRemoteFile",
	   "Calling TXNetConn::SendGenCommand to read %d bytes of data at"
           " offset %Ld.", readFileRequest.read.rlen, readFileRequest.read.offset);

   Bool_t fail = fConnModule->SendGenCommand(&readFileRequest, 0, bufFile, 
				     0, kTRUE, (char *)"TXNetFile::GetRemoteFile");  

   if (!fail) {
      Error("GetRemoteFile", "SendGenCommand returned error");
      return -1;
   }
   return size;
}

//_____________________________________________________________________________
Bool_t TXNetFile::ProcessUnsolicitedMsg(TXUnsolicitedMsgSender *,
                                        TXMessage *)
{
   // We are here if an unsolicited response comes from a logical conn
   // The response comes in the form of an TXMessage *, that must NOT be
   // destroyed after processing. It is destroyed by the first sender.
   // Remember that we are in a separate thread, since unsolicited 
   // responses are asynchronous by nature.

   Info("ProcessUnsolicitedMsg", "Processing unsolicited response");

   // Local processing ....

   return kTRUE;
}

//_____________________________________________________________________________
Int_t TXNetFile::LastBytesSent(void)
{
   // Return number of bytes last sent

   if (fConnModule)
      return fConnModule->LastBytesSent();
   else 
      return 0;
}

//_____________________________________________________________________________
Int_t TXNetFile::LastBytesRecv(void)
{
   // Return number of bytes last received

   if (fConnModule)
      return fConnModule->LastBytesRecv();
   else 
      return 0;
}

//_____________________________________________________________________________
Int_t TXNetFile::LastDataBytesSent(void)
{
   // Return number of data bytes last sent

   if (fConnModule)
      return fConnModule->LastDataBytesSent();
   else
      return 0;
}

//_____________________________________________________________________________
Int_t TXNetFile::LastDataBytesRecv(void)
{ 
   // Return number of data bytes last received

   if (fConnModule)
      return fConnModule->LastDataBytesRecv();
   else
      return 0;
}

//_____________________________________________________________________________
Long64_t TXNetFile::Size(void)
{
   // Return file size

   if (fAlreadyStated)
      return fSize;

   Long64_t size;
   Long_t i, f, m;

   this->SysStat((Int_t)0, &i, &size, &f, &m);
   fAlreadyStated = kTRUE;

   memcpy((void *)&fSize, (const void*)&size, sizeof(size));
   return fSize;
}
