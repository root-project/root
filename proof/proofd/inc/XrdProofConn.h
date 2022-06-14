// @(#)root/proofd:$Id$
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

#include "XrdSysToOuc.h"
#include "XProofProtocol.h"
#include "XProofProtUtils.h"
#include "XrdClient/XrdClientUnsolMsg.hh"
#include "XrdClient/XrdClientUrlInfo.hh"
#ifndef __OUC_STRING_H__
#include "XrdOuc/XrdOucString.hh"
#endif

#include <list>

class XrdClientConnectionMgr;
class XrdClientMessage;
class XrdClientPhyConnection;
class XrdSysRecMutex;
class XrdSecProtocol;
class XrdSysPlugin;

// Generic sender
typedef int (*XrdProofConnSender_t)(const char *, int, void *);

class XrdProofConn  : public XrdClientAbsUnsolMsgHandler {

friend class TXSocket;
friend class TXUnixSocket;
friend class XrdProofPhyConn;

public:

   enum ESrvType { kSTError = -1, kSTNone, kSTXProofd, kSTProofd };

private:

   char                fMode;          // Type of client
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

   XrdSysRecMutex     *fMutex;         // Lock SendRecv actions

   XrdSysRecMutex     *fConnectInterruptMtx;  // Protect access to fConnectInterrupt
   bool                fConnectInterrupt;

   XrdClientPhyConnection *fPhyConn;   // underlying physical connection

   int                 fOpenSockFD;    // Underlying socket descriptor

   XrdClientAbsUnsolMsgHandler *fUnsolMsgHandler; // Handler of unsolicited responses

   XrdProofConnSender_t fSender;      // Generic message forwarder
   void                *fSenderArg;   // Optional rgument for the message forwarder

   XrdClientUrlInfo    fUrl;           // Connection URL info object with

   static XrdClientConnectionMgr *fgConnMgr; //Connection Manager

   static int          fgMaxTry; //max number of connection attempts
   static int          fgTimeWait; //Wait time between an attempt and the other

   static XrdSysPlugin *fgSecPlugin;       // Sec library plugin
   static void         *fgSecGetProtocol;  // Sec protocol getter

   XrdSecProtocol     *Authenticate(char *plist, int lsiz);
   bool                CheckErrorStatus(XrdClientMessage *, int &, const char *, bool);
   bool                CheckResp(struct ServerResponseHeader *resp,
                                 const char *met, bool);
   virtual void        Connect(int = -1);
   void                ReConnect();
   virtual int         TryConnect(int = -1);

   ESrvType            DoHandShake(XrdClientPhyConnection *p = 0);
   virtual bool        GetAccessToSrv(XrdClientPhyConnection *p = 0);
   virtual bool        Init(const char *url = 0, int = -1);
   bool                Login();
   bool                MatchStreamID(struct ServerResponseHeader *resp);
   XrdClientMessage   *SendRecv(XPClientRequest *req,
                                const void *reqData, char **answData);

   void                SetInterrupt();

   void                SetConnectInterrupt();
   bool                ConnectInterrupt();

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

   bool                IsValid() const;

   XReqErrorType       LowWrite(XPClientRequest *, const void *, int);

   // Send, Recv interfaces
   virtual int         ReadRaw(void *buf, int len, XrdClientPhyConnection *p = 0);
   virtual XrdClientMessage *ReadMsg();
   XrdClientMessage   *SendReq(XPClientRequest *req, const void *reqData,
                               char **answData, const char *CmdName,
                               bool notifyerr = 1);
   virtual void        SetAsync(XrdClientAbsUnsolMsgHandler *uh, XrdProofConnSender_t = 0, void * = 0);
   void                SetSID(kXR_char *sid);
   virtual int         WriteRaw(const void *buf, int len, XrdClientPhyConnection *p = 0);

   static void         GetRetryParam(int &maxtry, int &timewait);
   static void         SetRetryParam(int maxtry = 5, int timewait = 2);

   virtual UnsolRespProcResult ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *s,
                                                     XrdClientMessage *m);
};

#endif
