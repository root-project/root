//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientAbs                                                     // 
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
//  Alvise Dorigo, Fabrizio Furano                                      //
//          INFN Padova, 2003                                           //
//                                                                      //
// Base class for objects handling redirections keeping open files      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id$

#ifndef XRD_ABSCLIENTBASE_H
#define XRD_ABSCLIENTBASE_H

#include "XrdClient/XrdClientUnsolMsg.hh"
#include "XrdClient/XrdClientConn.hh"

class XrdClientAbs: public XrdClientAbsUnsolMsgHandler {

   // Do NOT abuse of this
   friend class XrdClientConn;

 protected:
   XrdClientConn*           fConnModule;

   // After a redirection the file must be reopened.
   virtual bool OpenFileWhenRedirected(char *newfhandle, 
				       bool &wasopen) = 0;

   // In some error circumstances (e.g. when writing)
   // a redirection on error must be denied
   virtual bool CanRedirOnError() = 0;

 public:

   virtual bool IsOpen_wait() {
     return true;
   };

   void SetParm(const char *parm, int val);
   void SetParm(const char *parm, double val);

   // Hook to the open connection (needed by TXNetFile)
   XrdClientConn              *GetClientConn() const { return fConnModule; }

   inline XrdClientUrlInfo GetCurrentUrl() {
      if (fConnModule)
	 return fConnModule->GetCurrentUrl();
      else {
	 XrdClientUrlInfo empty;
	 return empty;
      }
   }

   // The last response got from a non-async request
   struct ServerResponseHeader *LastServerResp() {
     IsOpen_wait();
      if (fConnModule) return &fConnModule->LastServerResp;
      else return 0;
   }

   struct ServerResponseBody_Error *LastServerError() {
      if (fConnModule) return &fConnModule->LastServerError;
      else return 0;
   }

};

#endif
