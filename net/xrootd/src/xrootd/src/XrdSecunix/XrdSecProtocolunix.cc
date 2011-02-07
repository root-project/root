/******************************************************************************/
/*                                                                            */
/*                 X r d S e c P r o t o c o l u n i x . c c                  */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <strings.h>
#include <grp.h>
#include <pwd.h>
#include <sys/types.h>

#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSec/XrdSecInterface.hh"

/******************************************************************************/
/*              X r d S e c P r o t o c o l u n i x   C l a s s               */
/******************************************************************************/

class XrdSecProtocolunix : public XrdSecProtocol
{
public:
friend class XrdSecProtocolDummy; // Avoid stupid gcc warnings about destructor


        int                Authenticate  (XrdSecCredentials *cred,
                                          XrdSecParameters **parms,
                                          XrdOucErrInfo     *einfo=0);

        XrdSecCredentials *getCredentials(XrdSecParameters  *parm=0,
                                          XrdOucErrInfo     *einfo=0);

        XrdSecProtocolunix(const char                *hname,
                           const struct sockaddr     *ipadd)
                          : XrdSecProtocol("unix")
                          {Entity.host = strdup(hname);
                           Entity.name = (char *)"?";
                           credBuff    = 0;
                          }

        void              Delete() {delete this;}

private:

       ~XrdSecProtocolunix() {if (credBuff)    free(credBuff);
                              if (Entity.host) free(Entity.host);
                             } // via Delete()

struct sockaddr           hostaddr;      // Client-side only
char                     *credBuff;      // Credentials buffer (server)
};

/******************************************************************************/
/*             C l i e n t   O r i e n t e d   F u n c t i o n s              */
/******************************************************************************/
/******************************************************************************/
/*                        g e t C r e d e n t i a l s                         */
/******************************************************************************/


XrdSecCredentials *XrdSecProtocolunix::getCredentials(XrdSecParameters *noparm,
                                                      XrdOucErrInfo    *error)
{
   struct passwd *pEnt;
   struct group  *pGrp;
   char Buff[1024], *Bp;
   int Blen;

// Set protocol ID in the buffer
//
   strcpy(Buff, "unix"); Bp = Buff + 5;

// Get the username
//
   if (!(pEnt = getpwuid(geteuid()))) strcpy(Bp, "*");
      else strcpy(Bp, pEnt->pw_name);
   Bp += strlen(Bp);

// Get the group name
//
   if ((pGrp = getgrgid(getegid())))
      {*Bp++ = ' '; strcpy(Bp, pGrp->gr_name); Bp += strlen(Bp);}

// Return the credentials
//
   Blen = Bp-Buff+1;
   Bp = (char *)malloc(Blen);
   memcpy(Bp, Buff, Blen);
   return new XrdSecCredentials(Bp, Blen);
}

/******************************************************************************/
/*               S e r v e r   O r i e n t e d   M e t h o d s                */
/******************************************************************************/
/******************************************************************************/
/*                          A u t h e n t i c a t e                           */
/******************************************************************************/

int XrdSecProtocolunix::Authenticate(XrdSecCredentials *cred,
                                     XrdSecParameters **parms,
                                     XrdOucErrInfo     *erp)
{
   char *bp, *ep;

// Check if we have any credentials or if no credentials really needed.
// In either case, use host name as client name
//
   if (cred->size <= int(4) || !cred->buffer)
      {strncpy(Entity.prot, "host", sizeof(Entity.prot));
       Entity.name = (char *)"?";
       return 0;
      }

// Check if this is our protocol
//
   if (strcmp(cred->buffer, "unix"))
      {char msg[256];
       snprintf(msg, sizeof(msg),
                "Secunix: Authentication protocol id mismatch (unix != %.4s).",
                cred->buffer);
       if (erp) erp->setErrInfo(EINVAL, msg);
          else cerr <<msg <<endl;
       return -1;
      }

// Skip over the protocol ID and copy the buffer
//
   bp = credBuff = strdup((cred->buffer)+5);
   ep = bp + strlen(bp);

// Extract out username
//
   while(*bp && *bp == ' ') bp++;
   Entity.name = bp;
   while(*bp && *bp != ' ') bp++;
   *bp++ = '\0';

// Extract out the group name
//
   if (bp >= ep) return 0;
   while(*bp && *bp == ' ') bp++;
   Entity.grps = bp;

// All done
//
   return 0;
}
  
/******************************************************************************/
/*                X r d S e c p r o t o c o l u n i x I n i t                 */
/******************************************************************************/
  
extern "C"
{
char  *XrdSecProtocolunixInit(const char     mode,
                              const char    *parms,
                              XrdOucErrInfo *erp)
{
   return (char *)"";
}
}

/******************************************************************************/
/*              X r d S e c P r o t o c o l u n i x O b j e c t               */
/******************************************************************************/
  
extern "C"
{
XrdSecProtocol *XrdSecProtocolunixObject(const char              mode,
                                         const char             *hostname,
                                         const struct sockaddr  &netaddr,
                                         const char             *parms,
                                               XrdOucErrInfo    *erp)
{
   XrdSecProtocolunix *prot;

// Return a new protocol object
//
   if (!(prot = new XrdSecProtocolunix(hostname, &netaddr)))
      {const char *msg = "Seckunix: Insufficient memory for protocol.";
       if (erp) erp->setErrInfo(ENOMEM, msg);
          else cerr <<msg <<endl;
       return (XrdSecProtocol *)0;
      }

// All done
//
   return prot;
}
}
