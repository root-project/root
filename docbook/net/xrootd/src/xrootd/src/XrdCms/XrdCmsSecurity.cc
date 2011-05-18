/******************************************************************************/
/*                                                                            */
/*                     X r d C m s S e c u r i t y . c c                      */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdCmsSecurityCVSID = "$Id$";

// Bypass Solaris ELF madness
//
#ifdef __solaris__
#include <sys/isa_defs.h>
#if defined(_ILP32) && (_FILE_OFFSET_BITS != 32)
#undef  _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 32
#undef  _LARGEFILE_SOURCE
#endif
#endif
  
#include <dlfcn.h>
#ifndef __macos__
#include <link.h>
#endif

#include <stdlib.h>

#include "XProtocol/YProtocol.hh"

#include "Xrd/XrdLink.hh"

#include "XrdCms/XrdCmsSecurity.hh"
#include "XrdCms/XrdCmsTalk.hh"
#include "XrdCms/XrdCmsTrace.hh"

#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPthread.hh"

using namespace XrdCms;

/******************************************************************************/
/*                        G l o b a l   S y m b o l s                         */
/******************************************************************************/
  
extern XrdSecProtocol *(*XrdXrootdSecGetProtocol)
                                          (const char             *hostname,
                                           const struct sockaddr  &netaddr,
                                           const XrdSecParameters &parms,
                                                 XrdOucErrInfo    *einfo);

/******************************************************************************/
/*                        S t a t i c   S y m b o l s                         */
/******************************************************************************/
  
namespace XrdCms
{
XrdSecProtocol        *(*secProtocol)
                                          (const char             *hostname,
                                           const struct sockaddr  &netaddr,
                                           const XrdSecParameters &parms,
                                                 XrdOucErrInfo    *einfo)=0;
}

XrdSecService *XrdCmsSecurity::DHS    = 0;

/******************************************************************************/
/*                          A u t h e n t i c a t e                           */
/******************************************************************************/
  
int XrdCmsSecurity::Authenticate(XrdLink *Link, const char *Token, int Toksz)
{
   CmsRRHdr myHdr = {0, kYR_xauth, 0, 0};
   struct sockaddr netaddr;
   XrdSecCredentials cred;
   XrdSecProtocol   *AuthProt = 0;
   XrdSecParameters *parm = 0;
   XrdOucErrInfo     eMsg;
   const char       *eText = 0;
   char *authName, authBuff[4096];
   int rc, myDlen, abLen = sizeof(authBuff);

// Send a request for authentication
//
   if ((eText = XrdCmsTalk::Request(Link, myHdr, (char *)Token, Toksz+1)))
      {Say.Emsg("Auth",Link->Host(),"authentication failed;",eText);
       return 0;
      }

// Perform standard authentication
//
do {

// Get the response header and verify the request code
//
   if ((eText = XrdCmsTalk::Attend(Link,myHdr,authBuff,abLen,myDlen))) break;
   if (myHdr.rrCode != kYR_xauth) {eText = "invalid auth response";    break;}
   cred.size = myDlen; cred.buffer = authBuff;

// If we do not yet have a protocol, get one
//
   if (!AuthProt)
      {const char *hname = Link->Name(&netaddr);
       if (!DHS
       ||  !(AuthProt=DHS->getProtocol((char *)hname,netaddr,&cred,&eMsg)))
          {eText = eMsg.getErrText(rc); break;}
      }

// Perform the authentication
//
    if (!(rc = AuthProt->Authenticate(&cred, &parm, &eMsg))) break;
    if (rc < 0) {eText = eMsg.getErrText(rc); break;}
    if (parm) 
       {eText = XrdCmsTalk::Request(Link, myHdr, parm->buffer, parm->size);
        delete parm;
        if (eText) break;
       } else {eText = "auth interface violation"; break;}

} while(1);

// Check if we succeeded
//
   if (!eText)
      {if (!(authName = AuthProt->Entity.name)) eText = "entity name missing";
          else {Link->setID(authName,0);
                Say.Emsg("Auth",Link->Host(),"authenticated as", authName);
               }
      }

// Check if we failed
//
   if (eText) Say.Emsg("Auth",Link->Host(),"authentication failed;",eText);

// Perform final steps here
//
   if (AuthProt) AuthProt->Delete();
   return (eText == 0);
}

/******************************************************************************/
/*                             C o n f i g u r e                              */
/******************************************************************************/

int XrdCmsSecurity::Configure(const char *Lib, const char *Cfn)
{
   static XrdSysMutex myMutex;
   XrdSysMutexHelper  hlpMtx(&myMutex);
   XrdSecService *(*ep)(XrdSysLogger *, const char *cfn);
   static void *libhandle = 0;

// If we aleady have a security interface, return (may happen in client)
//
   if (!Cfn)
      {if (secProtocol) return 1;
          else if (XrdXrootdSecGetProtocol)
                  {secProtocol = XrdXrootdSecGetProtocol; return 1;}
      }

// Open the security library
//
   if (!libhandle && !(libhandle = dlopen(Lib, RTLD_NOW)))
      {Say.Emsg("Config",dlerror(),"opening shared library",Lib);
       return 0;
      }

// Get the client object creator (in case we are acting as a client)
//
   if (! secProtocol
   &&  !(secProtocol = (XrdSecProtocol *(*)(const char             *,
                                            const struct sockaddr  &,
                                            const XrdSecParameters &,
                                                  XrdOucErrInfo    *))
                       dlsym(libhandle, "XrdSecGetProtocol")))
      {Say.Emsg("Config",dlerror(),"finding XrdSecGetProtocol() in",Lib);
       return 0;
      }

// If only configuring a client or we already cnfigured a server, all done
//
   if (!Cfn || DHS) return 1;

// Get the server object creator
//
   if (!(ep = (XrdSecService *(*)(XrdSysLogger *, const char *cfn))dlsym(libhandle,
              "XrdSecgetService")))
      {Say.Emsg("Config",dlerror(),"finding XrdSecgetService() in",Lib);
       return 0;
      }

// Get the server object
//
   if (!(DHS = (*ep)(Say.logger(), Cfn)))
      {Say.Emsg("Config","Unable to create security service object via",Lib);
       return 0;
      }

// All done
//
   return 1;
}

/******************************************************************************/
/*                              g e t T o k e n                               */
/******************************************************************************/
  
const char *XrdCmsSecurity::getToken(int &size, const char *hostname)
{

// If not configured, return a null to indicate no authentication required
//
   if (!DHS) {size = 0; return 0;}

// Return actual token
//
   return DHS->getParms(size, hostname);
}

/******************************************************************************/
/*                              I d e n t i f y                               */
/******************************************************************************/
  
int XrdCmsSecurity::Identify(XrdLink *Link, XrdCms::CmsRRHdr &inHdr, 
                             char *authBuff, int abLen)
{
   CmsRRHdr outHdr = {0, kYR_xauth, 0, 0};
   struct sockaddr netaddr;
   const char *hname = Link->Host(&netaddr);
   XrdSecCredentials *cred;
   XrdSecProtocol    *AuthProt = 0;
   XrdSecParameters   AuthParm, *AuthP = 0;
   XrdOucErrInfo      eMsg;
   const char        *eText = 0;
   int rc, myDlen;

// Verify that we are configured
//
   if (!secProtocol && !Configure("libXrdSec.so"))
      {Say.Emsg("Auth",Link->Host(),"authentication configuration failed.");
       return 0;
      }

// Obtain the protocol
//
   AuthParm.buffer = (char *)authBuff; AuthParm.size = strlen(authBuff);
   if (!(AuthProt = secProtocol((char *)hname, netaddr, AuthParm, &eMsg)))
      {Say.Emsg("Auth",hname,"getProtocol() failed;",eMsg.getErrText(rc));
       return 0;
      }

// Perform standard authentication
//
do {

// Get credentials
//
   if (!(cred = AuthProt->getCredentials(AuthP, &eMsg)))
      {eText = eMsg.getErrText(rc); break;}

// Send credentials to the server
//
   eText = XrdCmsTalk::Request(Link, outHdr, cred->buffer, cred->size);
   delete cred;
   if (eText) break;

// Get the response header and prepare for next iteration if need be
//
   if ((eText = XrdCmsTalk::Attend(Link,inHdr,authBuff,abLen,myDlen))) break;
   AuthParm.size = myDlen; AuthParm.buffer = authBuff; AuthP = &AuthParm;

} while(inHdr.rrCode == kYR_xauth);

// Check if we failed
//
   if (eText) Say.Emsg("Auth",Link->Host(),"authentication failed;",eText);

// Perform final steps here
//
   if (AuthProt) AuthProt->Delete();
   return (eText == 0);
}

/******************************************************************************/
/*                           s e t S y s t e m I D                            */
/******************************************************************************/
  
char *XrdCmsSecurity::setSystemID(XrdOucTList *tp, const char *iName,
                                  const char  *iHost,    char  iType)
{
   XrdOucTList *tpF;
   char sidbuff[8192], *sidend = sidbuff+sizeof(sidbuff)-32, *sp, *cP;
   char *fMan, *fp, *xp;
   int n;

// The system ID starts with the semi-unique name of this node
//
   if (!iName || !*iName) iName = "anon";
   if (!iHost || !*iHost) iHost = "localhost";
   strcpy(sidbuff, iName); strcat(sidbuff, "-");
   sp = sidbuff + strlen(sidbuff);
   *sp++ = iType; *sp++ = ' '; cP = sp;

// Develop a unique cluster name for this cluster
//
   if (!tp) sp += sprintf(sp, "%s@%s", iName, iHost);
      else {tpF = tp;
            fMan = tp->text + strlen(tp->text) - 1;
            while((tp = tp->next))
                 {fp = fMan; xp = tp->text + strlen(tp->text) - 1;
                  do {if (*fp != *xp) break;
                      xp--;
                     } while(fp-- != tpF->text);
                  if ((n = xp - tp->text + 1) > 0)
                     {sp += sprintf(sp, "%d", tp->val);
                      if (sp+n >= sidend) return (char *)0;
                      strncpy(sp, tp->text, n); sp += n;
                     }
                 }
            sp += sprintf(sp, "%d", tpF->val);
            n = strlen(tpF->text);
            if (sp+n >= sidend) return (char *)0;
            strcpy(sp, tpF->text); sp += n;
           }

// Set envar to hold the cluster name
//
   *sp = '\0';
   XrdOucEnv::Export("XRDCMSCLUSTERID", cP);

// Return the system ID
//
   return  strdup(sidbuff);
}
