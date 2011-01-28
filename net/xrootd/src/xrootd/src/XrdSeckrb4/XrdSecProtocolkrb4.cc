/******************************************************************************/
/*                                                                            */
/*                 X r d S e c P r o t o c o l k r b 4 . c c                  */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

const char *XrdSecProtocolkrb4CVSID = "$Id$";

#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <sys/param.h>

#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOuc/XrdOucTokenizer.hh"
#include "XrdSec/XrdSecInterface.hh"

#include "kerberosIV/krb.h"
typedef  int krb_rc;

/******************************************************************************/
/*                               D e f i n e s                                */
/******************************************************************************/
  
#define XrdSecPROTOIDENT    "krb4"
#define XrdSecPROTOIDLEN    sizeof(XrdSecPROTOIDENT)
#define XrdSecNOIPCHK       0x0001
#define XrdSecDEBUG         0x1000

#define CLDBG(x) if (options & XrdSecDEBUG) cerr <<"sec_krb4: " <<x <<endl;

/******************************************************************************/
/*              X r d S e c P r o t o c o l k r b 4   C l a s s               */
/******************************************************************************/

class XrdSecProtocolkrb4 : public XrdSecProtocol
{
public:
friend class XrdSecProtocolDummy; // Avoid stupid gcc warnings about destructor


        int                Authenticate  (XrdSecCredentials *cred,
                                          XrdSecParameters **parms,
                                          XrdOucErrInfo     *einfo=0);

        XrdSecCredentials *getCredentials(XrdSecParameters  *parm=0,
                                          XrdOucErrInfo     *einfo=0);

static  char              *getPrincipal() {return Principal;}

static  int                Init_Server(XrdOucErrInfo *einfo, 
                                       char *KP=0, char *kfn=0);

static  void               setOpts(int opts) {options = opts;}

        XrdSecProtocolkrb4(const char                *KP,
                           const char                *hname,
                           const struct sockaddr     *ipadd)
                          {Service = (KP ? strdup(KP) : 0);
                           Entity.host = strdup(hname);
                           memcpy(&hostaddr, ipadd, sizeof(hostaddr));
                           CName[0] = '?'; CName[1] = '\0';
                           Entity.name = CName;
                          }

        void              Delete();

private:

       ~XrdSecProtocolkrb4() {} // Delete() does it all


static char *Append(char *dst, const char *src);
static int Fatal(XrdOucErrInfo *erp,int rc,const char *msg1,char *KP=0,int krc=0);
static int   get_SIR(XrdOucErrInfo *erp, const char *sh, char *sbuff, char *ibuff,
              char *rbuff);

static XrdSysMutex        krbContext;           // Client or server
static int                options;              // Client or server
static char               mySname[SNAME_SZ+1];  // Server
static char               myIname[INST_SZ+1];   // Server
static char               myRname[REALM_SZ+1];  // Server

static char              *keyfile;       // Server-side only
static char              *Principal;     // Server's principal name

struct sockaddr           hostaddr;      // Client-side only
char                      CName[256];    // Kerberos limit
char                     *Service;       // Target principal for client
};

/******************************************************************************/
/*                           S t a t i c   D a t a                            */
/******************************************************************************/
  
XrdSysMutex         XrdSecProtocolkrb4::krbContext;          // Client or server
int                 XrdSecProtocolkrb4::options = 0;         // Client or Server

char                XrdSecProtocolkrb4::mySname[SNAME_SZ+1]; // Server
char                XrdSecProtocolkrb4::myIname[INST_SZ+1];  // Server
char                XrdSecProtocolkrb4::myRname[REALM_SZ+1]; // Server
char               *XrdSecProtocolkrb4::keyfile   = 0;       // Server
char               *XrdSecProtocolkrb4::Principal = 0;       // Server

/******************************************************************************/
/*                                D e l e t e                                 */
/******************************************************************************/
  
void XrdSecProtocolkrb4::Delete()
{
     if (Entity.host) free(Entity.host);
     if (Service)     free(Service);
     delete this;
}

/******************************************************************************/
/*             C l i e n t   O r i e n t e d   F u n c t i o n s              */
/******************************************************************************/
/******************************************************************************/
/*                        g e t C r e d e n t i a l s                         */
/******************************************************************************/


XrdSecCredentials *XrdSecProtocolkrb4::getCredentials(XrdSecParameters *noparm,
                                                      XrdOucErrInfo    *error)
{
   const long cksum = 0L;
   struct ktext katix;       /* Kerberos data */
   krb_rc rc;
   char *buff;
   char  sname[SNAME_SZ+1];
   char  iname[INST_SZ+1];
   char  rname[REALM_SZ+1];

// Extract the "Principal.instance@realm" from the Service name
//
   if (!Service)
      {Fatal(error, EINVAL, "krb4 service Principal name not specified.");
       return (XrdSecCredentials *)0;
      }
   if (get_SIR(error, Service, sname, iname, rname) < 0) 
      return (XrdSecCredentials *)0;
   CLDBG("sname='" <<sname <<"' iname='" <<iname <<"' rname='" <<rname <<"'");

// Supply null credentials if so needed for this protocol
//
   if (!sname[0])
      {CLDBG("Null credentials supplied.");
       return new XrdSecCredentials(0,0);
      }

// Supply kerberos-style credentials
//
   krbContext.Lock();
   rc = krb_mk_req(&katix, sname, iname, rname, cksum);
   krbContext.UnLock();

// Check if all succeeded. If so, copy the ticket into the buffer. We wish
// we could place the ticket directly into the buffer but architectural
// differences won't allow us that optimization.
// Because of some typedef stupidity, we are now reserving the 1st 8 bytes
// of the credentials buffer for identifying information.
//
   if (rc == KSUCCESS)
      {int bsz = XrdSecPROTOIDLEN+katix.length;
       if (!(buff = (char *)malloc(bsz)))
          {Fatal(error,ENOMEM,"Insufficient memory for credentials.",Service);
           return (XrdSecCredentials *)0;
          }
       strcpy(buff, XrdSecPROTOIDENT);
       memcpy((void *)(buff+XrdSecPROTOIDLEN),
              (const void *)katix.dat, (size_t)katix.length);
       CLDBG("Returned " <<bsz <<" bytes of credentials; p=" <<sname);
       return new XrdSecCredentials(buff, bsz);
      }

// Diagnose the failure
//
   {char ebuff[1024];
    snprintf(ebuff, sizeof(ebuff)-1, "Unable to get credentials from %s;",
             Service);
    ebuff[sizeof(ebuff)-1] = '\0';
    Fatal(error, EACCES, ebuff, Service, rc);
    return (XrdSecCredentials *)0;
   }
}

/******************************************************************************/
/*               S e r v e r   O r i e n t e d   M e t h o d s                */
/******************************************************************************/
/******************************************************************************/
/*                          A u t h e n t i c a t e                           */
/******************************************************************************/

int XrdSecProtocolkrb4::Authenticate(XrdSecCredentials *cred,
                                     XrdSecParameters **parms,
                                     XrdOucErrInfo     *error)
{
   struct ktext katix;       /* Kerberos data */
   struct auth_dat pid;
   krb_rc rc;
   char *idp;
   unsigned int ipaddr;  // Should be 32 bits in all supported data models

// Check if we have any credentials or if no credentials really needed.
// In either case, use host name as client name
//
   if (cred->size <= (int)XrdSecPROTOIDLEN || !cred->buffer)
      {strncpy(Entity.prot, "host", sizeof(Entity.prot));
       Entity.name[0] = '?'; Entity.name[1] = '\0';
       return 0;
      }

// Check if this is a recognized protocol
//
   if (strcmp(cred->buffer, XrdSecPROTOIDENT))
      {char emsg[256];
       snprintf(emsg, sizeof(emsg),
                "Authentication protocol id mismatch (%.4s != %.4s).",
                XrdSecPROTOIDENT,  cred->buffer);
       Fatal(error, EINVAL, emsg);
       return -1;
      }

// Indicate who we are
//
   strncpy(Entity.prot, XrdSecPROTOIDENT, sizeof(Entity.prot));

// Create a kerberos style ticket (need to do that, unfortunately)
//
   katix.length = cred->size-XrdSecPROTOIDLEN;
   memcpy((void *)katix.dat, (const void *)&cred->buffer[XrdSecPROTOIDLEN],
                                   (size_t) katix.length);

// Prepare to check the ip address. This is rather poor since K4 "knows"
// that IP addresses are 4 bytes. Well, by the time IPV6 comes along, K4
// will be history (it's almost there now :-).
//
   if (options & XrdSecNOIPCHK) ipaddr = 0;
      else {sockaddr_in *ip = (sockaddr_in *)&hostaddr;
            memcpy((void *)&ipaddr,(void *)&ip->sin_addr.s_addr,sizeof(ipaddr));
           }

// Decode the credentials
//
   krbContext.Lock();
   rc = krb_rd_req(&katix, mySname, myIname, ipaddr, &pid, keyfile);
   krbContext.UnLock();

// Diagnose any errors
//
   if (rc != KSUCCESS)
      {Fatal(error,rc,"Unable to authenticate credentials;",Principal,rc);
       return -1;
      }

// Construct the user's name (use of the fact that names are < 256 chars)
//
   idp = Append(CName, pid.pname);
   if (pid.pinst[0])
      {*idp = '.'; idp++; idp = Append(idp, pid.pinst);}
   if (pid.prealm[0] && strcasecmp(pid.prealm, myRname))
      {*idp = '@'; idp++; idp = Append(idp, pid.prealm);}

// All done
//
   return 0;
}

/******************************************************************************/
/*       P r o t o c o l   I n i t i a l i z a t i o n   M e t h o d s        */
/******************************************************************************/
/******************************************************************************/
/*                           I n i t _ S e r v e r                            */
/******************************************************************************/
  
int XrdSecProtocolkrb4::Init_Server(XrdOucErrInfo *erp, char *KP, char *kfn)
{
   int plen;

// Now, extract the "Principal.instance@realm" from the stream
//
   if (!KP)
      return Fatal(erp, EINVAL, "krb4 service Principal name not specified.");
   if (get_SIR(erp, KP, mySname, myIname, myRname) < 0) return -1;
   CLDBG("sname='" <<mySname <<"' iname='" <<myIname <<"' rname='" <<myRname <<"'");

// Construct appropriate Principal name
//
   plen = strlen(mySname) + strlen(myIname) + strlen(myRname) + 3;
   if (!(Principal = (char *)malloc(plen)))
      {Principal = (char *)"?";
       return Fatal(erp, ENOMEM, "insufficient storage", KP);
      }
   if (*myIname) sprintf((char *)Principal, "%s.%s@%s",mySname,myIname,myRname);
      else       sprintf((char *)Principal, "%s@%s",   mySname,myRname);

// If we have been passed a keyfile name, use it.
//
   if (kfn && *kfn) keyfile = strdup(kfn);
      else keyfile = (char *)"";

// All done
//
   return 0;
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                A p p e n d                                 */
/******************************************************************************/
  
 char *XrdSecProtocolkrb4::Append(char *dst, const char *src)
{
      while(*src) {*dst = *src; dst++; src++;}
      *dst = '\0';
      return dst;
}

/******************************************************************************/
/*                                 F a t a l                                  */
/******************************************************************************/

int XrdSecProtocolkrb4::Fatal(XrdOucErrInfo *erp, int rc, const char *msg,
                             char *KP, int krc)
{
   const char *msgv[8];
   int k, i = 0;

              msgv[i++] = "Seckrb4: ";     //0
              msgv[i++] = msg;             //1
   if (krc)  {msgv[i++] = "; ";            //2
              msgv[i++] = krb_err_txt[rc]; //3
             }
   if (KP)   {msgv[i++] = " (p=";          //4
              msgv[i++] = KP;              //5
              msgv[i++] = ").";            //6
             }
   if (erp) erp->setErrInfo(rc, msgv, i);
      else {for (k = 0; k < i; k++) cerr <<msgv[k];
            cerr <<endl;
           }

   return -1;
}

/******************************************************************************/
/*                               g e t _ S I R                                */
/******************************************************************************/
  
int XrdSecProtocolkrb4::get_SIR(XrdOucErrInfo *erp, const char *sh,
                                       char *sbuff, char *ibuff, char *rbuff)
{
    int h, i, j, k;

    k = strlen(sh);
    if (k > MAX_K_NAME_SZ) 
       return Fatal(erp, EINVAL, "service name is to long", (char *)sh);

    for (j = 0; j < k && sh[j] != '@'; j++) {};
    if (j > k) j = k;
       else {if (j == k-1) 
                return Fatal(erp,EINVAL,"realm name missing after '@'",(char *)sh);
             if (k-j > REALM_SZ) 
                return Fatal(erp, EINVAL, "realm name is to long",(char *)sh);
            }

    for (i = 0; i < j && sh[i] != '.'; i++) {};
    if (i < j) {if (j-i >= INST_SZ) 
                   return Fatal(erp, EINVAL, "instance is too long",(char *)sh);
                if (i+1 == j) 
                   return Fatal(erp,EINVAL,"instance name missing after '.'",(char *)sh);
               }

    if (i == SNAME_SZ) 
       return Fatal(erp, EINVAL, "service name is too long", (char *)sh);
    if (!i) return Fatal(erp, EINVAL, "service name not specified.");

    strncpy(sbuff, sh, i); sbuff[i] = '\0';
    if ( (h = j - i - 1) <= 0) ibuff[0] = '\0';
       else {strncpy(ibuff, &sh[i+1], h); ibuff[h] = '\0';}
    if ( (h = k - j - 1) <= 0) krb_get_lrealm(rbuff, 1);
       else {strncpy(rbuff, &sh[j+1], h); rbuff[h] = '\0';}

    return 1;
}
 
/******************************************************************************/
/*                X r d S e c p r o t o c o l k r b 4 I n i t                 */
/******************************************************************************/
  
extern "C"
{
char  *XrdSecProtocolkrb4Init(const char     mode,
                              const char    *parms,
                              XrdOucErrInfo *erp)
{
   char *op, *KPrincipal=0, *Keytab=0;
   char parmbuff[1024];
   XrdOucTokenizer inParms(parmbuff);
   int options = XrdSecNOIPCHK;

// For client-side one-time initialization, we only need to set debug flag and
// initialize the kerberos context and cache location.
//
   if (mode == 'c')
      {if (getenv("XrdSecDEBUG")) XrdSecProtocolkrb4::setOpts(XrdSecDEBUG);
       return (char *)"";
      }

// Duplicate the parms
//
   if (parms) strlcpy(parmbuff, parms, sizeof(parmbuff));
      else {char *msg = (char *)"Seckrb4: Kerberos parameters not specified.";
            if (erp) erp->setErrInfo(EINVAL, msg);
               else cerr <<msg <<endl;
            return (char *)0;
           }

// Expected parameters: [<keytab>] [-ipchk] <principal>
//
   if (inParms.GetLine())
      {if ((op = inParms.GetToken()) && *op == '/')
          {Keytab = op; op = inParms.GetToken();}
           if (op && !strcmp(op, "-ipchk"))
              {options &= ~XrdSecNOIPCHK;
               op = inParms.GetToken();
              }
           KPrincipal = op;
      }

// Now make sure that we have all the right info
//
   if (!KPrincipal)
      {char *msg = (char *)"Seckrb4: Kerberos principal not specified.";
       if (erp) erp->setErrInfo(EINVAL, msg);
          else cerr <<msg <<endl;
       return (char *)0;
      }

// Now initialize the server
//
   XrdSecProtocolkrb4::setOpts(options);
   return (XrdSecProtocolkrb4::Init_Server(erp, KPrincipal, Keytab)
           ? (char *)0 : XrdSecProtocolkrb4::getPrincipal());
}
}

/******************************************************************************/
/*              X r d S e c P r o t o c o l k r b 4 O b j e c t               */
/******************************************************************************/
  
extern "C"
{
XrdSecProtocol *XrdSecProtocolkrb4Object(const char              mode,
                                         const char             *hostname,
                                         const struct sockaddr  &netaddr,
                                         const char             *parms,
                                               XrdOucErrInfo    *erp)
{
   XrdSecProtocolkrb4 *prot;
   char *KPrincipal=0;

// If this is a client call, then we need to get the target principal from the
// parms (which must be the first and only token). For servers, we use the
// context we established at initialization time.
//
   if (mode == 'c')
      {if ((KPrincipal = (char *)parms)) while(*KPrincipal == ' ') KPrincipal++;
       if (!KPrincipal || !*KPrincipal)
          {char *msg = (char *)"Seckrb4: Kerberos principal not specified.";
           if (erp) erp->setErrInfo(EINVAL, msg);
              else cerr <<msg <<endl;
           return (XrdSecProtocol *)0;
          }
      }

// Get a new protocol object
//
   if (!(prot = new XrdSecProtocolkrb4(KPrincipal, hostname, &netaddr)))
      {char *msg = (char *)"Seckrb4: Insufficient memory for protocol.";
       if (erp) erp->setErrInfo(ENOMEM, msg);
          else cerr <<msg <<endl;
       return (XrdSecProtocol *)0;
      }

// All done
//
   return prot;
}
}
