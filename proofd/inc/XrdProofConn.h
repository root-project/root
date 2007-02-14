// @(#)root/proofd:$Name:  $:$Id: XrdProofConn.h,v 1.7 2006/07/26 14:28:58 rdm Exp $
// Author: G. Ganis  June 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofConn
#define ROOT_XrdProofConn

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofConn                                                         //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// Low level handler of connections to xproofd.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#define DFLT_CONNECTMAXTRY           10

#ifndef ROOT_XProofProtocol
#include "XProofProtocol.h"
#endif
#ifndef ROOT_XProofProtUtils
#include "XProofProtUtils.h"
#endif
#ifndef XRC_UNSOLMSG_H
#include "XrdClient/XrdClientUnsolMsg.hh"
#endif
#ifndef _XRC_URLINFO_H
#include "XrdClient/XrdClientUrlInfo.hh"
#endif
#ifndef __OUC_STRING_H__
#include "XrdOuc/XrdOucString.hh"
#endif

#include <list>

class XrdClientConnectionMgr;
class XrdClientMessage;
class XrdClientPhyConnection;
class XrdOucRecMutex;
class XrdSecProtocol;
class XrdOucPlugin;

class XrdProofConn  : public XrdClientAbsUnsolMsgHandler {

friend class TXSocket;
friend class TXUnixSocket;
friend class XrdProofPhyConn;

public:

   enum ESrvType { kSTError = -1, kSTNone, kSTXProofd, kSTProofd };

private:

   char                fMode;          // Type of client
   bool                fGetAsync;      // Switch ON/OFF receipt of async messages
   bool                fConnected;
   int                 fLogConnID;     // Logical connection ID of current object
   kXR_unt16           fStreamid;      // Streamid used for normal communications
   int                 fRemoteProtocol; // Protocol of remote daemon
   int                 fServerProto;   // The server protocol
   ESrvType            fServerType;    // Server type as returned by DoHandShake()
                                       // (see enum ServerType)
   short               fSessionID;     // proofsrv: remote ID of connected session
   XrdOucString        fUser;          // Username used for login
   XrdOucString        fHost;          // Remote host
   int                 fPort;          // Remote port
   XrdOucString        fLastErrMsg;    // Msg describing last error
   XErrorCode          fLastErr;       // Last error code
   char                fCapVer;        // a version number (e.g. a protocol num)

   XrdOucString        fLoginBuffer;   // Buffer to be sent over at login

   XrdOucRecMutex     *fMutex;         // Lock SendRecv actions

   XrdClientPhyConnection *fPhyConn;   // underlying physical connection

   int                 fOpenSockFD;    // Underlying socket descriptor

   XrdClientAbsUnsolMsgHandler *fUnsolMsgHandler; // Handler of unsolicited responses

   XrdClientUrlInfo    fUrl;           // Connection URL info object with

   static XrdClientConnectionMgr *fgConnMgr; //Connection Manager

   static int          fgMaxTry; //max number of connection attempts
   static int          fgTimeWait; //Wait time between an attempt and the other

   static XrdOucPlugin *fgSecPlugin;       // Sec library plugin
   static void         *fgSecGetProtocol;  // Sec protocol getter

   XrdSecProtocol     *Authenticate(char *plist, int lsiz);
   bool                CheckErrorStatus(XrdClientMessage *, int &, const char *);
   bool                CheckResp(struct ServerResponseHeader *resp,
                                 const char *met);
   virtual int         Connect();
   ESrvType            DoHandShake();
   virtual bool        GetAccessToSrv();
   virtual bool        Init(const char *url = 0);
   bool                Login();
   XReqErrorType       LowWrite(XPClientRequest *, const void *, int);
   bool                MatchStreamID(struct ServerResponseHeader *resp);
   XrdClientMessage   *SendRecv(XPClientRequest *req,
                                const void *reqData, void **answData);
   virtual void        SetAsync(XrdClientAbsUnsolMsgHandler *uh);

   void                SetInterrupt();

public:
   XrdProofConn(const char *url, char mode = 'M', int psid = -1, char ver = -1,
                XrdClientAbsUnsolMsgHandler * uh = 0, const char *logbuf = 0);
   virtual ~XrdProofConn();

   virtual void        Close(const char *opt = "");

   int                 GetLogConnID() const { return fLogConnID; }
   int                 GetLowSocket();
   int                 GetOpenError() const { return (int)fLastErr; }
   int                 GetServType() const { return (int)fServerType; }
   short               GetSessionID() const { return fSessionID; }
   const char         *GetUrl() { return (const char *) fUrl.GetUrl().c_str(); }
   const char         *GetLastErr() { return fLastErrMsg.c_str(); }

   bool                IsValid() const { return fConnected; }

   // Send, Recv interfaces
   virtual int         ReadRaw(void *buf, int len);
   virtual XrdClientMessage *ReadMsg();
   XrdClientMessage   *SendReq(XPClientRequest *req, const void *reqData,
                               void **answData, const char *CmdName);
   void                SetSID(kXR_char *sid);
   virtual int         WriteRaw(const void *buf, int len);

   static void         GetRetryParam(int &maxtry, int &timewait);
   static void         SetRetryParam(int maxtry = 5, int timewait = 2);

   virtual UnsolRespProcResult ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *s,
                                                     XrdClientMessage *m);
};

#endif
