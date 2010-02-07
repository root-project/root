#ifndef __XRDCMSREQ_H__
#define __XRDCMSREQ_H__
/******************************************************************************/
/*                                                                            */
/*                          X r d C m s R e q . h h                           */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

class  XrdCmsRRQInfo;
class  XrdCmsNode;
struct stat;

class XrdCmsReq
{
public:

// Use this to determine if the call is advisory in nature
//
inline int Advisory() {return ReqAdv;}

// Reply with an eror message to the request. An optional length may be given.
//
void       Reply_Error(const char *emsg, int emsglen = 0);

// Reply with an error code and an error message to the request. The error
// is a string corresponding to an errno.h error code symbol. Valid names are:
// ENOENT, EPERM, EACCES, EIO, ENOMEM, ENOSPC, ENAMETOOLONG, ENETUNREACH,
// ENOTBLK, EISDIR, and ENOTEMPTY. Any other strings are converted to EINVAL.
// This mechanism supports cross platform error number delivery. The second
// takes the errno.h error number directly.
//
void       Reply_Error(const char *ecode, const char *emsg, int emsglen = 0);

void       Reply_Error(int         ecode, const char *emsg, int emsglen = 0);

// Reply by telling the client everything was successfully completed. No data
// is to be sent (do not use Reply_OK() with a data length of zero).
//
void       Reply_OK();

// Reply with success and an ASCII text message to the request. An optional data
// length may be given.
//
void       Reply_OK(const char *data, int datalen = 0);

// Reply with with success along with file stat information
//
void       Reply_OK(struct stat &buf);

// Reply by redirecting the client to a different server
//
void       Reply_Redirect(const char *sname,      // DNS name:port of server
                          const char *logincgi=0, // CGI tokens for login
                          const char *opencgi=0); // CGI tokens for open

void       Reply_Redirect(const char *sname,      // DNS name of server
                                int   port,       // Server port number
                          const char *logincgi=0, // CGI tokens for login
                          const char *opencgi=0); // CGI tokens for open

// Reply by forcing the client to wait the indicated number of seconds
//
void       Reply_Wait(int sec);

// Reply by telling the client to wait up to "sec" seconds for a response. This
// method returns a new XrdCmsReq object that must be used to actually provide
// the final response. It is automatically deleted when any Reply_xxx() method 
// is called since the object is only valid for a single reply.
//
XrdCmsReq *Reply_WaitResp(int sec=0);

           XrdCmsReq(XrdCmsNode *nP, unsigned int id, char adv=0);
           XrdCmsReq(XrdCmsReq  *rP, unsigned int rn);
          ~XrdCmsReq() {}

private:

int  StatGen(struct stat &sbuf, char *xbuf);
void noReply();
void Reply(int respCode, unsigned int respVal, const char *respData=0,
           int respLen=0, struct iovec *iov=0, int iovnum=0);

XrdCmsNode    *NodeP;
unsigned int   ReqID;
int            ReqNins;  // Node instance
short          ReqNnum;  // Node number
char           ReqAdv;
};
#endif
