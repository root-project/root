#ifndef XRDSECTLAYER_HH
#define XRDSECTLAYER_HH
/******************************************************************************/
/*                                                                            */
/*                       X r d S e c T L a y e r . h h                        */
/*                                                                            */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//       $Id$

#include "XrdSec/XrdSecInterface.hh"
#include "XrdSys/XrdSysPthread.hh"

/* The XrdSecTLayer class is meant to be used as a wrapper for security
   protocols that require transport-layer interactions to complete the
   authentication exchange (e.g., native ssl). This class virtualizes a
   transport-layer socket and provides the proper framing to allow stream
   socket level interactions to occur across an existing client/xrootd
   connection. To that extent, there are certain limitations in this
   virtualization:
   1) Interactions must complete within a window whose upper bound is set to
      CPU 10 seconds (i.e., Network RTT and artificial delays do not apply).
      The window has no lower bound so that an interaction may complete as fast
      as conditions allow. An interaction is whatever bytes produce a single
      request/response. These bytes need not be produced all at once but the
      last required byte of an interaction must be produced within 10 CPU
      seconds of the 1st byte. There is no limit on the number of interactions.
   2) The use of the supplied socket must use standard and common socket
      operations (e.g., read(), write(), send(), recv(), close()).
   3) The protocol must not be sensitive to the fact that the socket will 
      identify itself as a local socket with an IPV4 address of 127.0.0.1.

   For more information, see pure abstract methods secClient() and secServer()
   which must be implemented by the derived class (in addition to delete()).
   Finally, consider the parameters you may need to pass to the constructor of
   this class.
*/

class XrdOucErrInfo;

class XrdSecTLayer : public XrdSecProtocol
{
public:

// The object inheriting this class should call the initializer indicating
// the true name of the protocol (no more that 7 characters). To optimize the
// start-up, indicate who is the initiator (i.e., first one to send data). Using
// the enum below, specify isClient (the default) or isServer. If the initiator
// is not known, use the default and the class will dynamically determine it.
//
enum Initiator {isClient = 0, isServer};

               XrdSecTLayer(const char *pName, Initiator who1st=isClient);

// This is a symmetric wrapper. At the start on each end, secClient() is
// called on the client-side and secServer() is called on the server side.
// The 1st parameter is the filedescriptor to be used for the security exchange.
// It is the responsibility of each routine to close the file descriptor prior
// to returning to the caller! No return value is expected as success or failure
// is communicated via the esecond paramter, the XrdOucErrInfo object.

// Upon success, the error code must be set to zero (the initial value) and
//               for secServer() the Entity object defined in the topmost 
//               XrdSecProtocol object must contain the client's identity.

// Upon failure, the error code must be set to a positive error number (usually
//               some errno value) as well as text explaining the problem.

// Client: theFD - file descriptor to be used
//         einfo - the error object where ending status must be returned
//
virtual void   secClient(int theFD, XrdOucErrInfo *einfo)=0;

// Server: theFD - file descriptor to be used
//         einfo - the error object where ending status must be returned
//
virtual void   secServer(int theFD, XrdOucErrInfo *einfo)=0;

// You must implete the proper delete(). Normally, do a "delete this" and join
// the secTid thread: "if (secTid) {XrdSysThread::Join(secTid,NULL);secTid=0;}".
//
virtual void    Delete()=0;

// Classes that must be public are only internally used
//

virtual int                Authenticate  (XrdSecCredentials  *cred,
                                          XrdSecParameters  **parms,
                                          XrdOucErrInfo      *einfo=0);

virtual XrdSecCredentials *getCredentials(XrdSecParameters   *parm=0,
                                          XrdOucErrInfo      *einfo=0);

        void               secXeq();

protected:
pthread_t      secTid;

virtual       ~XrdSecTLayer() {if (eText)  {free(eText);eText=0;}
                               if (myFD>0) {close(myFD);myFD=-1;}
                              }

private:

int            bootUp(Initiator Who);
int            Read(int FD, char *Buff, int rdLen);
int            secDone();
void           secDrain();
const char    *secErrno(int rc, char *buff);
void           secError(const char *Msg, int rc, int iserrno=1);

XrdSysSemaphore mySem;
Initiator       Starter;
Initiator       Responder;
int             myFD;
int             urFD;
int             Tmax; // Maximum timeslices per interaction
int             Tcur; // Current timeslice
int             eCode;
char           *eText;
XrdOucErrInfo  *eDest;

struct TLayerRR
{
             char  protName[8];    // via Constructor
             char  protCode;       // One of the below
static const char  endData = 0x00;
static const char  xfrData = 0x01;
             char  protRsvd[7];    // Reserved
}                  Hdr;

static const int   buffSz = 8192;
static const int   hdrSz  = sizeof(TLayerRR);
static const int   dataSz = buffSz - hdrSz;
};
#endif
