/******************************************************************************/
/*                                                                            */
/*                     X r d X r o o t d A d m i n . c c                      */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

const char *XrdXrootdAdminCVSID = "$Id$";

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/types.h>

#include "XrdVersion.hh"
#include "Xrd/XrdLink.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdXrootd/XrdXrootdAdmin.hh"
#include "XrdXrootd/XrdXrootdJob.hh"
#include "XrdXrootd/XrdXrootdProtocol.hh"
#include "XrdXrootd/XrdXrootdTrace.hh"
 
/******************************************************************************/
/*                     G l o b a l s   &   S t a t i c s                      */
/******************************************************************************/

extern XrdOucTrace     *XrdXrootdTrace;

       XrdSysError     *XrdXrootdAdmin::eDest;

       XrdXrootdAdmin::JobTable        *XrdXrootdAdmin::JobList = 0;
  
/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/
  
void *XrdXrootdInitAdmin(void *carg)
      {XrdXrootdAdmin Admin;
       return Admin.Start((XrdNetSocket *)carg);
      }

void *XrdXrootdLoginAdmin(void *carg)
      {XrdXrootdAdmin *Admin = new XrdXrootdAdmin();
       Admin->Login(*(int *)carg);
       delete Admin;
       return (void *)0;
      }
 
/******************************************************************************/
/*                                a d d J o b                                 */
/******************************************************************************/

void XrdXrootdAdmin::addJob(const char *jname, XrdXrootdJob *jp)
{
     JobTable *jTabp = new JobTable();

     jTabp->Jname = strdup(jname);
     jTabp->Job   = jp;
     jTabp->Next  = JobList;
     JobList      = jTabp;
}
  
/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdXrootdAdmin::Init(XrdSysError *erp, XrdNetSocket *asock)
{
   const char *epname = "Init";
   pthread_t tid;

   eDest = erp;
   if (XrdSysThread::Run(&tid, XrdXrootdInitAdmin, (void *)asock,
                         0, "Admin traffic"))
      {eDest->Emsg(epname, errno, "start admin");
       return 0;
      }
   return 1;
}

/******************************************************************************/
/*                                 L o g i n                                  */
/******************************************************************************/
  
void XrdXrootdAdmin::Login(int socknum)
{
   const char *epname = "Admin";
   char *tp;

// Attach the socket FD to a stream
//
   Stream.SetEroute(eDest);
   Stream.AttachIO(socknum, socknum);

// Get the first request
//
   if (!Stream.GetLine())
      {eDest->Emsg(epname, "No admin login specified");
       return;
      }

// The first request better be: <reqid> login <name>
//
   if (getreqID()
   || !(tp = Stream.GetToken())
   || strcmp("login", tp)
   || do_Login())
      {eDest->Emsg(epname, "Invalid admin login sequence");
       return;
      }

// Document the login and go process the stream
//
   eDest->Emsg(epname, "Admin", TraceID, "logged in");
   Xeq();
}

/******************************************************************************/
/*                                 S t a r t                                  */
/******************************************************************************/
  
void *XrdXrootdAdmin::Start(XrdNetSocket *AdminSock)
{
   const char *epname = "Start";
   int InSock;
   pthread_t tid;

// Accept connections in an endless loop
//
   while(1) if ((InSock = AdminSock->Accept()) >= 0)
               {if (XrdSysThread::Run(&tid,XrdXrootdLoginAdmin,(void *)&InSock))
                   {eDest->Emsg(epname, errno, "start admin");
                    close(InSock);
                   }
               } else eDest->Emsg(epname, errno, "accept connection");
   return (void *)0;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                              d o _ A b o r t                               */
/******************************************************************************/

int XrdXrootdAdmin::do_Abort()
{
   char *msg;
   int   mlen, rc;

// Handle: abort <target> [msg]
//
   if ((rc = getTarget("abort", &msg))) return 0;

// Get optional message
//
   msg = getMsg(msg, mlen);

// Send off the unsolicited response
//
   if (msg) return sendResp("abort", kXR_asyncab, msg, mlen);
            return sendResp("abort", kXR_asyncab);
}

/******************************************************************************/
/*                                 d o _ C j                                  */
/******************************************************************************/
  
int XrdXrootdAdmin::do_Cj()
{
   const char *fmt1 = "<resp id=\"%s\"><rc>0</rc>";
   const char *fmt2 = "<num>%d</num></resp>\n";
   char *tp, buff[1024];
   XrdXrootdJob *jobp;
   JobTable *jTabp;
   int i, rc;

// The next token needs to be job type
//
   if (!(tp = Stream.GetToken()))
      {sendErr(8, "cj", "job type not specified.");
       return -1;
      }

// Run through the list of valid job types
//
   jTabp = JobList;
   while(jTabp && strcmp(tp, jTabp->Jname)) jTabp = jTabp->Next;

// See if we have a real job list here
//
   if (jTabp) jobp = jTabp->Job;
      else if (!strcmp(tp, "*")) jobp = 0;
              else {sendErr(8, "cj", "invalid job type specified.");
                    return -1;
                   }

// Get optional key
//
   tp = Stream.GetToken();

// Send the header of the response
//
   i = sprintf(buff, fmt1, reqID);
   if (Stream.Put(buff, i)) return -1;

// Cancel the jobs
//
   if (jobp) rc = jobp->Cancel(tp);
      else {jTabp = JobList; rc = 0;
            while(jTabp) {rc += jTabp->Job->Cancel(tp); jTabp = jTabp->Next;}
           }

// Now print the end-framing
//
   i = sprintf(buff, fmt2, rc);
   return Stream.Put(buff, i);
}
 
/******************************************************************************/
/*                               d o _ C o n t                                */
/******************************************************************************/

int XrdXrootdAdmin::do_Cont()
{
   int rc;

// Handle: cont <target>
//
   if ((rc = getTarget("cont"))) return 0;

// Send off the unsolicited response
//
   return sendResp("cont", kXR_asyncgo);
}
  
/******************************************************************************/
/*                               d o _ D i s c                                */
/******************************************************************************/

int XrdXrootdAdmin::do_Disc()
{
   kXR_int32 msg[2];
   char *tp;
   int rc;

// Handle: disc <target> <wsec> <msec>
//
   if ((rc = getTarget("disc"))) return 0;

// Make sure times are specified
//
   if (!(tp = Stream.GetToken()) || !(msg[0] = strtol(tp, 0, 10)))
      return sendErr(8, "disc", " reconnect interval missing or invalid.");
   if (!(tp = Stream.GetToken()) || !(msg[1] = strtol(tp, 0, 10)))
      return sendErr(8, "disc", "reconnect timeout missing or invalid.");

// Send off the unsolicited response
//
   msg[0] = htonl(msg[0]); msg[1] = htonl(msg[1]);
   return sendResp("disc", kXR_asyncdi, (const char *)msg, sizeof(msg));
}
  
/******************************************************************************/
/*                              d o _ L o g i n                               */
/******************************************************************************/
  
int XrdXrootdAdmin::do_Login()
{
   const char *fmt="<resp id=\"%s\"><rc>0</rc><v>" XROOTD_VERSION "</v></resp>\n";
   char *tp, buff[1024];
   int blen;

// Process: login <name>
//
   if (!(tp = Stream.GetToken()))
      {eDest->Emsg("do_Login", "login name not specified");
       return 0;
      } else strlcpy(TraceID, tp, sizeof(TraceID));

// Provide good response
//
   blen = snprintf(buff, sizeof(buff)-1, fmt, reqID);
   buff[sizeof(buff)-1] = '\0';
   return Stream.Put(buff, blen);
}
 
/******************************************************************************/
/*                                d o _ L s c                                 */
/******************************************************************************/

int XrdXrootdAdmin::do_Lsc()
{
   const char *fmt1 = "<resp id=\"%s\"><rc>0</rc><conn>";
   const char *fmt2 = "</conn></resp>\n";
   static int fmt2len = strlen(fmt2);
   char buff[1024];
   const char *mdat[3] = {buff, " ", 0};
         int   mlen[3] = {0,      1, 0};
   int i, rc, curr = -1;

// Handle: list <target>
//
   if ((rc = getTarget("lsc"))) return 0;

// Send the header of the response
//
   i = sprintf(buff, fmt1, reqID);
   if (Stream.Put(buff, i)) return -1;

// Return back matching client list
//
   while((mlen[0] = XrdLink::getName(curr, buff, sizeof(buff), &Target)))
        if (Stream.Put(mdat, mlen)) return -1;
   return Stream.Put(fmt2, fmt2len);
}

/******************************************************************************/
/*                                d o _ L s d                                 */
/******************************************************************************/
  
int XrdXrootdAdmin::do_Lsd()
{
   const char *fmt1 = "<resp id=\"%s\"><rc>0</rc>";
   const char *fmt2 = "<c r=\"%c\" t=\"%lld\" v=\"%d\" m=\"%s\">";
   const char *fmt2a= "<io u=\"%d\"><nf>%d</nf><p>%lld<n>%d</n></p>"
                      "<i>%lld<n>%d</n></i><o>%lld<n>%d</n></o>"
                      "<s>%d</s><t>%d</t></io>";
   const char *fmt3 = "<auth p=\"%s\"><n>";
   const char *fmt3e= "</r></auth>";
   const char *fmt4 = "</resp>\n";
   static int fmt3elen= strlen(fmt3e);
   static int fmt4len = strlen(fmt4);
   char ctyp, monit[3], *mm, cname[1024], buff[100];
   char aprot[XrdSecPROTOIDSIZE+2], abuff[32], iobuff[256];
   const char *mdat[24]= {buff, cname, iobuff};
         int   mlen[24]= {0};
   long long conn, inBytes, outBytes;
   int i, rc, cver, inuse, stalls, tardies, curr = -1;
   XrdLink *lp;
   XrdProtocol *xp;
   XrdXrootdProtocol *pp;

// Handle: list <target>
//
   if ((rc = getTarget("lsd"))) return 0;

// Send the header of the response
//
   i = sprintf(buff, fmt1, reqID);
   if (Stream.Put(buff, i)) return -1;

// Return back matching client list
//
   while((lp = XrdLink::Find(curr, &Target)))
         if ((xp = lp->getProtocol())
         &&  (pp = dynamic_cast<XrdXrootdProtocol *>(xp)))
            {cver = int(pp->CapVer);
             ctyp = (pp->Status & XRD_ADMINUSER ? 'a' : 'u');
             conn = static_cast<long long>(lp->timeCon());
             mm = monit;
             if (pp->monFILE) *mm++ = 'f';
             if (pp->monIO  ) *mm++ = 'i';
             *mm = '\0';
             inuse = lp->getIOStats(inBytes, outBytes, stalls, tardies);
             mlen[0] = sprintf(buff, fmt2, ctyp, conn, cver, monit);
             mlen[1] = lp->Client(cname, sizeof(cname));
             mlen[2] = sprintf(iobuff, fmt2a,inuse-1,pp->numFiles,pp->totReadP,
                               (pp->cumReadP + pp->numReadP),
                               inBytes, (pp->cumWrites+ pp->numWrites),
                               outBytes,(pp->cumReads + pp->numReads),
                               stalls, tardies);
             i = 3;
             if ((pp->Client) && pp->Client != &(pp->Entity))
                {strncpy(aprot, pp->Client->prot, XrdSecPROTOIDSIZE);
                 aprot[XrdSecPROTOIDSIZE] = '\0';
                 mdat[i]  = abuff;
                 mlen[i++]= sprintf(abuff, fmt3, aprot);
                 i = 1;
                 if (pp->Client->name && (mlen[i] = strlen(pp->Client->name)))
                    mdat[i++] = pp->Client->name;
                 mdat[i] = "</n><h>"; mlen[i++] = 7;
                 if (pp->Client->host && (mlen[i] = strlen(pp->Client->host)))
                    mdat[i++] = pp->Client->host;
                 mdat[i] = "</h><o>"; mlen[i++] = 7;
                 if (pp->Client->vorg && (mlen[i] = strlen(pp->Client->vorg)))
                    mdat[i++] = pp->Client->vorg;
                 mdat[i] = "</o><r>"; mlen[i++] = 7;
                 if (pp->Client->role && (mlen[i] = strlen(pp->Client->role)))
                    mdat[i++] = pp->Client->role;
                 mdat[i] = fmt3e; mlen[i++] = fmt3elen;
               }
             mdat[i] = "</c>"; mlen[i++] = 4;
             mdat[i] = 0;      mlen[i] = 0;
             if (Stream.Put(mdat, mlen)) {lp->setRef(-1); return -1;}
            }
   return Stream.Put(fmt4, fmt4len);
}
 
/******************************************************************************/
/*                                d o _ L s j                                 */
/******************************************************************************/

int XrdXrootdAdmin::do_Lsj()
{
   const char *fmt1 = "<resp id=\"%s\"><rc>0</rc>";
   const char *fmt2 = "</resp>\n";
   static int fmt2len = strlen(fmt2);
   char *tp, buff[1024];
   XrdXrootdJob *jobp;
   JobTable *jTabp;
   int i, rc = 0;

// The next token needs to be job type
//
   if (!(tp = Stream.GetToken()))
      {sendErr(8, "lsj", "job type not specified.");
       return -1;
      }

// Run through the list of valid job types
//
   jTabp = JobList;
   while(jTabp && strcmp(tp, jTabp->Jname)) jTabp = jTabp->Next;

// See if we have a real job list here
//
   if (jTabp) jobp = jTabp->Job;
      else if (!strcmp(tp, "*")) jobp = 0;
              else {sendErr(8, "lsj", "invalid job type specified.");
                    return -1;
                   }

// Send the header of the response
//
   i = sprintf(buff, fmt1, reqID);
   if (Stream.Put(buff, i)) return -1;

// List the jobs
//
   if (jobp) rc = do_Lsj_Xeq(jobp);
      else {jTabp = JobList;
            while(jTabp && !(rc = do_Lsj_Xeq(jTabp->Job))) jTabp = jTabp->Next;
           }

// Now print the end-framing
//
   return (rc ? rc : Stream.Put(fmt2, fmt2len));
}

/******************************************************************************/
/*                            d o _ L s j _ X e q                             */
/******************************************************************************/

int XrdXrootdAdmin::do_Lsj_Xeq(XrdXrootdJob *jp)
{
    XrdOucTList *tp, *tpprev;
    int rc = 0;

    if ((tp = jp->List()))
       while(tp && !(rc = Stream.Put(tp->text, tp->val)))
            {tpprev = tp; tp = tp->next; delete tpprev;}

    while(tp) {tpprev = tp; tp = tp->next; delete tpprev;}

    return rc;
}
  
/******************************************************************************/
/*                                d o _ M s g                                 */
/******************************************************************************/
  
int XrdXrootdAdmin::do_Msg()
{
   char *msg;
   int rc, mlen;

// Handle: msg <target> [msg]
//
   if ((rc = getTarget("msg", &msg))) return 0;

// Get optional message
//
   msg = getMsg(msg, mlen);
// Send off the unsolicited response
//
   if (msg) return sendResp("msg", kXR_asyncms, msg, mlen);
            return sendResp("msg", kXR_asyncms);
}
 
/******************************************************************************/
/*                              d o _ P a u s e                               */
/******************************************************************************/

int XrdXrootdAdmin::do_Pause()
{
   kXR_int32 msg;
   char *tp;
   int rc;

// Handle: pause <target> <wsec>
//
   if ((rc = getTarget("pause"))) return 0;

// Make sure time is specified
//
   if (!(tp = Stream.GetToken()) || !(msg = strtol(tp, 0, 10)))
      return sendErr(8, "pause", "time missing or invalid.");

// Send off the unsolicited response
//
   msg = htonl(msg);
   return sendResp("pause", kXR_asyncwt, (const char *)&msg, sizeof(msg));
}

/******************************************************************************/
/*                                d o _ R e d                                 */
/******************************************************************************/
  
int XrdXrootdAdmin::do_Red()
{
   struct msg {kXR_int32 port; char buff[8192];} myMsg;
   int rc, hlen, tlen, bsz;
   char *tp, *pn, *qq;

// Handle: redirect <target> <host>:<port>[?token]
//
   if ((rc = getTarget("redirect", 0))) return 0;

// Get the redirect target
//
   if (!(tp = Stream.GetToken()) || *tp == ':')
      return sendErr(8, "redirect", "destination host not specified.");

// Get the port number
//
   if (!(pn = index(tp, ':')) || !(myMsg.port = strtol(pn+1, &qq, 10)))
      return sendErr(8, "redirect", "port missing or invalid.");
   myMsg.port = htonl(myMsg.port);

// Copy out host
//
   *pn = '\0';
   hlen = strlcpy(myMsg.buff,tp,sizeof(myMsg.buff));
   if (static_cast<size_t>(hlen) >= sizeof(myMsg.buff))
      return sendErr(8, "redirect", "destination host too long.");

// Copy out the token
//
   if (qq && *qq == '?')
      {bsz = sizeof(myMsg.buff) - hlen;
       if ((tlen = strlcpy(myMsg.buff+hlen,qq,bsz)) >= bsz)
          return sendErr(8, "redirect", "token too long.");
      } else tlen = 0;

// Send off the unsolicited response
//
   return sendResp("redirect", kXR_asyncrd, (const char *)&myMsg, hlen+tlen+4);
}

/******************************************************************************/
/*                                g e t M s g                                 */
/******************************************************************************/
  
char *XrdXrootdAdmin::getMsg(char *msg, int &mlen)
{
   if (msg) while(*msg == ' ') msg++;
   if (msg && *msg)  mlen = strlen(msg)+1;
      else {msg = 0; mlen = 0;}
   return  msg;
}

/******************************************************************************/
/*                              g e t r e q I D                               */
/******************************************************************************/
  
int XrdXrootdAdmin::getreqID()
{
   char *tp;

   if (!(tp = Stream.GetToken()))
      {reqID[0] = '?'; reqID[1] = '\0';
       return sendErr(4, "request", "id not specified.");
      }

   if (strlen(tp) >= sizeof(reqID))
      {reqID[0] = '?'; reqID[1] = '\0';
       return sendErr(4, "request", "id too long.");
      }

   strcpy(reqID, tp);
   return 0;
}

/******************************************************************************/
/*                             g e t T a r g e t                              */
/******************************************************************************/
/* Returns 0 if a target was found, otherwise -1 */
  
int XrdXrootdAdmin::getTarget(const char *act, char **rest)
{
   char *tp;

// Get the target
//
   if (!(tp = Stream.GetToken(rest))) 
      {sendErr(8, act, "target not specified.");
       return -1;
      }
   Target.Set(tp);

   return 0;
}
 
/******************************************************************************/
/*                               s e n d E r r                                */
/******************************************************************************/
  
int XrdXrootdAdmin::sendErr(int rc, const char *act, const char *msg)
{
   const char *fmt = "<resp id=\"%s\"><rc>%d</rc><msg>%s %s</msg></resp>\n";
   char buff[1024];
   int blen;

   blen = snprintf(buff, sizeof(buff)-1, fmt, reqID, rc, act, msg);
   buff[sizeof(buff)-1] = '\0';

   return Stream.Put(buff, blen);
}
 
/******************************************************************************/
/*                                s e n d O K                                 */
/******************************************************************************/
  
int XrdXrootdAdmin::sendOK(int sent)
{
   const char *fmt = "<resp id=\"%s\"><rc>0</rc><num>%d</num></resp>\n";
   char buff[1024];
   int blen;

   blen = snprintf(buff, sizeof(buff)-1, fmt, reqID, sent);
   buff[sizeof(buff)-1] = '\0';

   return Stream.Put(buff, blen);
}
 
/******************************************************************************/
/*                              s e n d R e s p                               */
/******************************************************************************/
  
int XrdXrootdAdmin::sendResp(const char *act, XActionCode anum)
{
   XrdLink *lp;
   const kXR_int32 net4 = htonl(4);
   int numsent = 0, curr = -1;

// Complete the response header
//
   usResp.act = htonl(anum);
   usResp.len = net4;

// Send off the messages
//
   while((lp = XrdLink::Find(curr, &Target)))
        {TRACE(RSP, "sending " <<lp->ID <<' ' <<act);
         if (lp->Send((const char *)&usResp, sizeof(usResp))>0) numsent++;
        }

// Now send the response to the admin guy
//
   return sendOK(numsent);
}

/******************************************************************************/
  
int XrdXrootdAdmin::sendResp(const char *act, XActionCode anum,
                             const char *msg, int msgl)
{
   struct iovec iov[2];
   XrdLink *lp;
   int numsent = 0, curr = -1, bytes = sizeof(usResp)+msgl;

// Complete the response header
//
   usResp.act = htonl(anum);
   usResp.len = htonl(msgl+4);

// Construct message vector
//
   iov[0].iov_base = (caddr_t)&usResp;
   iov[0].iov_len  = sizeof(usResp);
   iov[1].iov_base = (caddr_t)msg;
   iov[1].iov_len  = msgl;

// Send off the messages
//
   while((lp = XrdLink::Find(curr, &Target)))
        {TRACE(RSP, "sending " <<lp->ID <<' ' <<act <<' ' <<msg);
         if (lp->Send(iov, 2, bytes)>0) numsent++;
        }

// Now send the response to the admin guy
//
   return sendOK(numsent);
}
 
/******************************************************************************/
/*                                   X e q                                    */
/******************************************************************************/
  
void XrdXrootdAdmin::Xeq()
{
   const char *epname = "Xeq";
   int rc;
   char *request, *tp;

// Start receiving requests on this stream
// Format: <msgid> <cmd> <args>
//
   rc = 0;
   while((request = Stream.GetLine()) && !rc)
        {TRACE(DEBUG, "received admin request: '" <<request <<"'");
         if ((rc = getreqID())) continue;
         if ((tp = Stream.GetToken()))
            {     if (!strcmp("abort",    tp)) rc = do_Abort();
             else if (!strcmp("cj",       tp)) rc = do_Cj();
             else if (!strcmp("cont",     tp)) rc = do_Cont();
             else if (!strcmp("disc",     tp)) rc = do_Disc();
             else if (!strcmp("lsc",      tp)) rc = do_Lsc();
             else if (!strcmp("lsd",      tp)) rc = do_Lsd();
             else if (!strcmp("lsj",      tp)) rc = do_Lsj();
             else if (!strcmp("msg",      tp)) rc = do_Msg();
             else if (!strcmp("pause",    tp)) rc = do_Pause();
             else if (!strcmp("redirect", tp)) rc = do_Red();
             else {eDest->Emsg(epname, "invalid admin request,", tp);
                   rc = sendErr(4, tp, "is an invalid request.");
                  }
            }
        }

// The socket disconnected
//
   eDest->Emsg("Admin", "Admin", TraceID, "logged out");
   return;
}
