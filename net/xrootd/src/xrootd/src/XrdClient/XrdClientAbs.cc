//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientAbs                                                         // 
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
//  Alvise Dorigo, Fabrizio Furano                                      //
//          INFN Padova, 2003                                           //
//                                                                      //
// Base class for objects who has to handle redirections with open files//
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//         $Id$

const char *XrdClientAbsCVSID = "$Id$";

#include "XrdClient/XrdClientAbs.hh"
#include "XrdClient/XrdClientDebug.hh"
#include "XrdClient/XrdClientEnv.hh"


//_____________________________________________________________________________
void XrdClientAbs::SetParm(const char *parm, int val) 
{
   // This method configure TXNetFile's behaviour settings through the 
   // setting of special ROOT env vars via the TEnv facility.
   // A ROOT env var is not a environment variable (that you can get using 
   // getenv() syscall). It's an internal ROOT one (see TEnv documentation
   // for more details).
   // At the moment the following env vars are handled by TXNetFile
   // XNet.ConnectTimeout   - maximum time to wait before server's 
   //                                  response on a connect
   // XNet.RequestTimeout   - maximum time to wait before considering 
   //                                  a read/write failure
   // XNet.ConnectDomainAllowRE
   //                                - sequence of TRegexp regular expressions
   //                                  separated by a |.
   //                                  A domain (or w.x.y.z addr) is granted
   //                                  access to for the
   //                                  first connection if it matches one of these
   //                                  regexps. Example:
   //                                  slac.stanford.edu|pd.infn.it|fe.infn.it
   // XNet.ConnectDomainDenyRE
   //                                - sequence of TRegexp regular expressions
   //                                  separated by a |.
   //                                  A domain (or w.x.y.z addr) is denied
   //                                  access to for the
   //                                  first connection if it matches one of these
   //                                  regexps. Example:
   //                                  slac.stanford.edu|pd.infn.it|fe.infn.it
   // XNet.RedirDomainAllowRE
   //                                - sequence of TRegexp regular expressions
   //                                  separated by a |.
   //                                  A domain (or w.x.y.z addr) is granted
   //                                  access to for a
   //                                  redirection if it matches one of these
   //                                  regexps. Example:
   //                                  slac.stanford.edu|pd.infn.it|fe.infn.it
   // XNet.RedirDomainDenyRE
   //                                - sequence of TRegexp regular expressions
   //                                  separated by a |.
   //                                  A domain (or w.x.y.z addr) is denied
   //                                  access to for a
   //                                  redirection if it matches one of these
   //                                  regexps. Example:
   //                                  slac.stanford.edu|pd.infn.it|fe.infn.it
   //
   // XNet.MaxRedirectCount - maximum number of redirections from
   //                                  server
   // XNet.Debug            - log verbosity level
   //                                  (0=nothing,
   //                                   1=messages of interest to the user,
   //                                   2=messages of interest to the developers 
   //                                     (includes also user messages),
   //                                   3=dump of all sent/received data buffers
   //                                     (includes also user and developers 
   //                                      messages).
   // XNet.ReconnectTimeout - sleep-time before going back to the 
   //                                  load balancer (or rebouncing to the same
   //                                  failing host) after a read/write error
   // XNet.StartGarbageCollectorThread -
   //                                  for test/development purposes. Normally 
   //                                  nonzero (True), but as workaround for 
   //                                  external causes someone could be
   //                                  interested in not having the garbage 
   //                                  collector thread around.
   // XNet.TryConnect       - Number of tries connect to a single 
   //                                  server before giving up
   // XNet.TryConnectServersList
   //                                - Number of connect retries to the whole 
   //                                  server list given
   // XNet.PrintTAG         - Print a particular string the developers 
   //                                  can choose to quickly recognize the 
   //                                  version at run time
   // XNet.ReadCacheSize    - The size of the cache. One cache per instance!
   //                                  0 for no cache. The cache gets all the
   //                                  kxr_read positive responses received
   // XNet.ReadAheadSize    - The size of the read-ahead blocks. 
   //                                  0 for no read-ahead.

   if (DebugLevel() >= XrdClientDebug::kUSERDEBUG)
      Info(XrdClientDebug::kUSERDEBUG,
	   "AbsNetCommon::SetParm",
	   "Setting " << parm << " to " << val);

   EnvPutInt((char *)parm, val);
}

//_____________________________________________________________________________
void XrdClientAbs::SetParm(const char *parm, double val) 
{
   // Setting TXNetFile specific ROOT-env variables (see previous method
   // for details

   if (DebugLevel() >= XrdClientDebug::kUSERDEBUG)
      Info(XrdClientDebug::kUSERDEBUG,
	   "TXAbsNetCommon::SetParm",
	   "Setting " << parm << " to " << val);

   
   //EnvPutString(parm, val);
}



//_____________________________________________________________________________
// Returns query information
bool XrdClientAbs::Query(kXR_int16 ReqCode, const kXR_char *Args, kXR_char *Resp, kXR_int32 MaxResplen) {
   if (!fConnModule) return false;
   if (!fConnModule->IsConnected()) return false;
   if (!Resp) return false;

   // Set the max transaction duration
   fConnModule->SetOpTimeLimit(EnvGetLong(NAME_TRANSACTIONTIMEOUT));

   ClientRequest qryRequest;

   memset( &qryRequest, 0, sizeof(qryRequest) );

   fConnModule->SetSID(qryRequest.header.streamid);

   qryRequest.query.requestid = kXR_query;
   qryRequest.query.infotype = ReqCode;

   if (Args)
      qryRequest.query.dlen = strlen((char *)Args);

   if (ReqCode == kXR_Qvisa)
      memcpy( qryRequest.query.fhandle, fHandle, sizeof(fHandle) );

   kXR_char *rsp = 0;
   bool ret = fConnModule->SendGenCommand(&qryRequest, (const char*)Args,
					  (void **)&rsp, 0, true,
					  (char *)"Query");
  
   if (ret) {

      if (Args) {

         if (rsp) {
            Info(XrdClientDebug::kHIDEBUG,
                 "XrdClientAdmin::Query",
                 "Query(" << ReqCode << ", '" << Args << "') returned '" << rsp << "'" );
         }
         else {
            Info(XrdClientDebug::kHIDEBUG,
                 "XrdClientAdmin::Query",
                 "Query(" << ReqCode << ", '" << Args << "') returned a null string" );
         }

      }
      else {
         Info(XrdClientDebug::kHIDEBUG,
              "XrdClientAdmin::Query",
              "Query(" << ReqCode << ", NULL') returned '" << rsp << "'" );
      }
      
      if ( rsp && (LastServerResp()->status == kXR_ok) ) {
         int l = xrdmin(MaxResplen, LastServerResp()->dlen);
         strncpy((char *)Resp, (char *)rsp, l);
         if (l >= 0) Resp[l-1] = '\0';
         free(rsp);
         rsp = 0;
      }
   }

   return ret;
}

