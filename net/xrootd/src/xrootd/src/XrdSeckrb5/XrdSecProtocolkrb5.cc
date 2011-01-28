/******************************************************************************/
/*                                                                            */
/*                 X r d S e c P r o t o c o l k r b 5 . c c                  */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/*   Modifications:                                                           */
/*    - January 2007: add support for forwarded tickets                       */
/*                   (author: G. Ganis, CERN)                                 */
/******************************************************************************/

//       $Id$

const char *XrdSecProtocolkrb5CVSID = "$Id$";

#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <sys/param.h>
#include <pwd.h>
#include <sys/types.h>
#include <sys/stat.h>


extern "C" {
#include "krb5.h"
#ifdef HAVE_ET_COM_ERR_H
#include "et/com_err.h"
#else
#include "com_err.h"
#endif
}

#include "XrdNet/XrdNetDNS.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOuc/XrdOucTokenizer.hh"
#include "XrdSec/XrdSecInterface.hh"
#include "XrdSys/XrdSysPriv.hh"
  
/******************************************************************************/
/*                               D e f i n e s                                */
/******************************************************************************/

#define krb_etxt(x) (char *)error_message(x)
  
#define XrdSecPROTOIDENT    "krb5"
#define XrdSecPROTOIDLEN    sizeof(XrdSecPROTOIDENT)
#define XrdSecNOIPCHK       0x0001
#define XrdSecEXPTKN        0x0002
#define XrdSecINITTKN       0x0004
#define XrdSecDEBUG         0x1000

#define XrdSecMAXPATHLEN      4096

#define CLDBG(x) if (client_options & XrdSecDEBUG) cerr <<"Seckrb5: " <<x <<endl;
#define CLPRT(x) cerr <<"Seckrb5: " <<x <<endl;

typedef  krb5_error_code krb_rc;

/******************************************************************************/
/*              X r d S e c P r o t o c o l k r b 5   C l a s s               */
/******************************************************************************/

class XrdSecProtocolkrb5 : public XrdSecProtocol
{
public:
friend class XrdSecProtocolDummy; // Avoid stupid gcc warnings about destructor

        int                Authenticate  (XrdSecCredentials *cred,
                                          XrdSecParameters **parms,
                                          XrdOucErrInfo     *einfo=0);

        XrdSecCredentials *getCredentials(XrdSecParameters  *parm=0,
                                          XrdOucErrInfo     *einfo=0);

static  char              *getPrincipal() {return Principal;}

static  int                Init(XrdOucErrInfo *einfo, char *KP=0, char *kfn=0);

static  void               setOpts(int opts) {options = opts;}
static  void               setClientOpts(int opts) {client_options = opts;}
static  void               setParms(char *param) {Parms = param;}
static  void               setExpFile(char *expfile)
                                     {if (expfile)
                                         {int lt = strlen(expfile);
                                          lt = (lt >= XrdSecMAXPATHLEN) ?
                                                      XrdSecMAXPATHLEN -1 : lt;
                                          memcpy(ExpFile, expfile, lt);
                                          ExpFile[lt] = 0;
                                         }
                                     }

        XrdSecProtocolkrb5(const char                *KP,
                           const char                *hname,
                           const struct sockaddr     *ipadd)
                          {Service = (KP ? strdup(KP) : 0);
                           Entity.host = strdup(hname);
                           memcpy(&hostaddr, ipadd, sizeof(hostaddr));
                           CName[0] = '?'; CName[1] = '\0';
                           Entity.name = CName;
                           Step = 0;
                           AuthContext = 0;
                           AuthClientContext = 0;
                           Ticket = 0;
                           Creds = 0;
                          }

        void              Delete();

private:

       ~XrdSecProtocolkrb5() {} // Delete() does it all

static int Fatal(XrdOucErrInfo *erp,int rc,const char *msg1,char *KP=0,int krc=0);
static int get_krbCreds(char *KP, krb5_creds **krb_creds);

static XrdSysMutex        krbContext;    // Server
static XrdSysMutex        krbClientContext;// Client
static int                options;       // Server
static int                client_options;// Client
static krb5_context       krb_context;   // Server
static krb5_context       krb_client_context;   // Client 
static krb5_ccache        krb_client_ccache;    // Client 
static krb5_ccache        krb_ccache;    // Server
static krb5_keytab        krb_keytab;    // Server
static uid_t              krb_kt_uid;// Server
static gid_t              krb_kt_gid;// Server
static krb5_principal     krb_principal; // Server

static char              *Principal;     // Server's principal name
static char              *Parms;         // Server parameters

static char               ExpFile[XrdSecMAXPATHLEN]; // Server: (template for)
                                                     // file to export token
int exp_krbTkn(XrdSecCredentials *cred, XrdOucErrInfo *erp);
int get_krbFwdCreds(char *KP, krb5_data *outdata);

struct sockaddr           hostaddr;      // Client-side only
char                      CName[256];    // Kerberos limit
char                     *Service;       // Target principal for client
char                      Step;          // Indicates at which step we are
krb5_auth_context         AuthContext;   // Authetication context
krb5_auth_context         AuthClientContext;   // Authetication context
krb5_ticket              *Ticket;        // Ticket associated to client authentication
krb5_creds               *Creds;         // Client: credentials
};

/******************************************************************************/
/*                           S t a t i c   D a t a                            */
/******************************************************************************/
  
XrdSysMutex         XrdSecProtocolkrb5::krbContext;        // Server
XrdSysMutex         XrdSecProtocolkrb5::krbClientContext;  // Client

int                 XrdSecProtocolkrb5::client_options = 0;// Client
int                 XrdSecProtocolkrb5::options = 0;       // Server
krb5_context        XrdSecProtocolkrb5::krb_context;       // Server
krb5_context        XrdSecProtocolkrb5::krb_client_context;       // Client 
krb5_ccache         XrdSecProtocolkrb5::krb_client_ccache; // Client
krb5_ccache         XrdSecProtocolkrb5::krb_ccache;        // Server
krb5_keytab         XrdSecProtocolkrb5::krb_keytab = NULL; // Server
uid_t               XrdSecProtocolkrb5::krb_kt_uid = 0;    // Server
gid_t               XrdSecProtocolkrb5::krb_kt_gid = 0;    // Server
krb5_principal      XrdSecProtocolkrb5::krb_principal;     // Server

char               *XrdSecProtocolkrb5::Principal = 0;     // Server
char               *XrdSecProtocolkrb5::Parms = 0;         // Server

char                XrdSecProtocolkrb5::ExpFile[XrdSecMAXPATHLEN] = "/tmp/krb5cc_<uid>";

/******************************************************************************/
/*                                D e l e t e                                 */
/******************************************************************************/
  
void XrdSecProtocolkrb5::Delete()
{
     if (Parms)       free(Parms); Parms = 0;
     if (Creds)       krb5_free_creds(krb_context, Creds);
     if (Ticket)      krb5_free_ticket(krb_context, Ticket);
     if (AuthContext) krb5_auth_con_free(krb_context, AuthContext);
     if (AuthClientContext) krb5_auth_con_free(krb_client_context, AuthClientContext);
     if (Entity.host) free(Entity.host);
     if (Service)     free(Service);
     delete this;
}

/******************************************************************************/
/*                        g e t C r e d e n t i a l s                         */
/******************************************************************************/

XrdSecCredentials *XrdSecProtocolkrb5::getCredentials(XrdSecParameters *noparm,
                                                      XrdOucErrInfo    *error)
{
   char *buff;
   int bsz;
   krb_rc rc;
   krb5_data         outbuf;
   CLDBG("getCredentials");
// Supply null credentials if so needed for this protocol
//
   if (!Service)
      {CLDBG("Null credentials supplied.");
       return new XrdSecCredentials(0,0);
      }

// Set KRB5CCNAME to its dfault value, if not done
//
   if (!getenv("KRB5CCNAME")) {
      char ccname[128];
      sprintf(ccname, "KRB5CCNAME=FILE:/tmp/krb5cc_%d", geteuid());
      putenv(strdup(ccname));
   }
   CLDBG(getenv("KRB5CCNAME"));
	 
// Initialize the context and get the cache default.
//
   if ((rc = krb5_init_context(&krb_client_context)))
      {Fatal(error, ENOPROTOOPT, "Kerberos initialization failed", Service, rc);
       return (XrdSecCredentials *)0;
      }

   CLDBG("init context");
// Obtain the default cache location
//
   if ((rc = krb5_cc_default(krb_client_context, &krb_client_ccache)))
      {Fatal(error, ENOPROTOOPT, "Unable to locate cred cache", Service, rc);
       return (XrdSecCredentials *)0;
      }

   CLDBG("cc cache default");
// Check if the server asked for a forwardable ticket
//
   char *pfwd = 0;
   if ((pfwd = (char *) strstr(Service,",fwd")))
      {
         client_options |= XrdSecEXPTKN;
         *pfwd = 0;
      }

// Clear outgoing ticket and lock the kerberos context
//
   outbuf.length = 0; outbuf.data = 0;

   CLDBG("context lock");
   krbClientContext.Lock();
   CLDBG("context locked");

// If this is not the first call, we are asked to send over a delegated ticket:
// we must create it first
// we save it into a file and return signalling the end of the hand-shake
//

   if (Step > 0)
      {if ((rc = get_krbFwdCreds(Service, &outbuf)))
          {krbClientContext.UnLock();
           Fatal(error, ESRCH, "Unable to get forwarded credentials", Service, rc);
           return (XrdSecCredentials *)0;
          } else
            {bsz = XrdSecPROTOIDLEN+outbuf.length;
             if (!(buff = (char *)malloc(bsz)))
                {krbClientContext.UnLock();
                 Fatal(error, ENOMEM, "Insufficient memory for credentials.", Service);
                 return (XrdSecCredentials *)0;
                }
             strcpy(buff, XrdSecPROTOIDENT);
             memcpy((void *)(buff+XrdSecPROTOIDLEN),
                            (const void *)outbuf.data, (size_t)outbuf.length);
             CLDBG("Returned " <<bsz <<" bytes of creds; p=" <<Service);
             if (outbuf.data)  free(outbuf.data);
             krbClientContext.UnLock();
             return new XrdSecCredentials(buff, bsz);
            }
      }

// Increment the step
//
   Step += 1;

// Get a service ticket for this principal
//
   const char *reinitcmd = (client_options & XrdSecEXPTKN) ? "kinit -f" : "kinit";
   bool notdone = 1;
   bool reinitdone = 0;
   while (notdone)
      {if ((rc = (krb_rc)get_krbCreds(Service, &Creds)))
          { if (!(client_options & XrdSecINITTKN) || reinitdone)
               {krbClientContext.UnLock();
                const char *m = (!(client_options & XrdSecINITTKN)) ?
                                "No or invalid credentials" : "Unable to get credentials";
                Fatal(error, ESRCH, m, Service, rc);
                return (XrdSecCredentials *)0;
               } else {// Need to re-init
                       CLPRT("Ticket missing or invalid: re-init ");
                       rc = system(reinitcmd);
                       CLDBG("getCredentials: return code from '"<<reinitcmd<<
                             "': "<< rc);
                       reinitdone = 1;
                       continue;
                      }
          }
       if (client_options & XrdSecEXPTKN)
          {// Make sure the ticket is forwardable
           if (!(Creds->ticket_flags & TKT_FLG_FORWARDABLE))
              { if (client_options & XrdSecINITTKN)
                   { // Need to re-init
                    CLPRT("Existing ticket is not forwardable: re-init ");
                    rc = system(reinitcmd);
                    CLDBG("getCredentials: return code from '"<<reinitcmd<<
                          "': "<< rc);
                    reinitdone = 1;
                    continue;
                   } else {
                    krbClientContext.UnLock();
                    Fatal(error, ESRCH, "Existing ticket is not forwardable: cannot continue",
                                        Service, rc);
                    return (XrdSecCredentials *)0;
                   }
              }
          }
       // We are done
       notdone = 0;
      }

// Set the RET_TIME flag in the authentication context 
//
   if ((rc = krb5_auth_con_init(krb_client_context, &AuthClientContext)))
      {krbClientContext.UnLock();
       Fatal(error, ESRCH, "Unable to init a new auth context", Service, rc);
       return (XrdSecCredentials *)0;
      }

// Generate a kerberos-style authentication message
//
   rc = krb5_mk_req_extended(krb_client_context, &AuthClientContext,
             AP_OPTS_USE_SESSION_KEY,(krb5_data *)0, Creds,&outbuf);

// Check if all succeeded. If so, copy the ticket into the buffer. We wish
// we could place the ticket directly into the buffer but architectural
// differences won't allow us that optimization.
//
   if (!rc)
      {bsz = XrdSecPROTOIDLEN+outbuf.length;
       if (!(buff = (char *)malloc(bsz)))
          {krbClientContext.UnLock();
           Fatal(error, ENOMEM, "Insufficient memory for credentials.", Service);
           return (XrdSecCredentials *)0;
          }
       strcpy(buff, XrdSecPROTOIDENT);
       memcpy((void *)(buff+XrdSecPROTOIDLEN),
              (const void *)outbuf.data, (size_t)outbuf.length);
       CLDBG("Returned " <<bsz <<" bytes of creds; p=" <<Service);
       if (outbuf.data)  free(outbuf.data);
       krbClientContext.UnLock();
       return new XrdSecCredentials(buff, bsz);
      }

// Diagnose the failure
//
   if (outbuf.data)  free(outbuf.data);
   krbClientContext.UnLock();
   Fatal(error, EACCES, "Unable to get credentials", Service, rc);
   return (XrdSecCredentials *)0;
}

/******************************************************************************/
/*               S e r v e r   O r i e n t e d   M e t h o d s                */
/******************************************************************************/
/******************************************************************************/
/*                          A u t h e n t i c a t e                           */
/******************************************************************************/

int XrdSecProtocolkrb5::Authenticate(XrdSecCredentials *cred,
                                     XrdSecParameters **parms,
                                     XrdOucErrInfo     *error)
{
   krb5_data         inbuf;                 /* Kerberos data */
   krb5_address      ipadd;
   krb_rc rc = 0;
   char *iferror = 0;

// Check if we have any credentials or if no credentials really needed.
// In either case, use host name as client name
//
   if (cred->size <= (int)XrdSecPROTOIDLEN || !cred->buffer)
      {strncpy(Entity.prot, "host", sizeof(Entity.prot));
       return 0;
      }

// Check if this is a recognized protocol
//
   if (strcmp(cred->buffer, XrdSecPROTOIDENT))
      {char emsg[256];
       snprintf(emsg, sizeof(emsg),
                "Authentication protocol id mismatch (%.4s != %.4s).",
                XrdSecPROTOIDENT,  cred->buffer);
       Fatal(error, EINVAL, emsg, Principal);
       return -1;
      }

   CLDBG("protocol check");

   char printit[4096];
   sprintf(printit,"Step is %d",Step);
   CLDBG(printit);
// If this is not the first call the buffer contains a forwarded token:
// we save it into a file and return signalling the end of the hand-shake
//
   if (Step > 0)
      {if ((rc = exp_krbTkn(cred, error)))
          iferror = (char *)"Unable to export the token to file";
       if (rc && iferror) {
          krbContext.UnLock();
          return Fatal(error, EINVAL, iferror, Principal, rc);
       }
       krbContext.UnLock();

       return 0;
      }
   
   CLDBG("protocol check");

// Increment the step
//
   Step += 1;

// Indicate who we are
//
   strncpy(Entity.prot, XrdSecPROTOIDENT, sizeof(Entity.prot));

// Create a kerberos style ticket and obtain the kerberos mutex
//

   CLDBG("Context Lock");

   inbuf.length = cred->size -XrdSecPROTOIDLEN;
   inbuf.data   = &cred->buffer[XrdSecPROTOIDLEN];

   krbContext.Lock();

// Check if whether the IP address in the credentials must match that of
// the incomming host.
//
   CLDBG("Context Locked");
   if (!(XrdSecProtocolkrb5::options & XrdSecNOIPCHK))
      {struct sockaddr_in *ip = (struct sockaddr_in *)&hostaddr;
      // The above is a hack but K5 does it this way
       ipadd.addrtype = ADDRTYPE_INET;
       ipadd.length = sizeof(ip->sin_addr);
       ipadd.contents = (krb5_octet *)&ip->sin_addr;
       iferror = (char *)"Unable to validate ip address;";
       if (!(rc=krb5_auth_con_init(krb_context, &AuthContext)))
             rc=krb5_auth_con_setaddrs(krb_context, AuthContext, NULL, &ipadd);
      }

// Decode the credentials and extract client's name
//
   {  XrdSysPrivGuard pGuard(krb_kt_uid, krb_kt_gid);
      if (pGuard.Valid())
         {if (!rc)
             {if ((rc = krb5_rd_req(krb_context, &AuthContext, &inbuf,
                                  (krb5_const_principal)krb_principal,
                                   krb_keytab, NULL, &Ticket)))
                 iferror = (char *)"Unable to authenticate credentials;";
              else if ((rc = krb5_aname_to_localname(krb_context,
                                          Ticket->enc_part2->client,
                                          sizeof(CName)-1, CName)))
                     iferror = (char *)"Unable to extract client name;";
             }
      } else
         iferror = (char *)"Unable to acquire privileges to read the keytab;";
   }
// Make sure the name is null-terminated
//
   CName[sizeof(CName)-1] = '\0';

// If requested, ask the client for a forwardable token
   int hsrc = 0;
   if (!rc && XrdSecProtocolkrb5::options & XrdSecEXPTKN) {
      // We just ask for more; the client knows what to send over
      hsrc = 1;
      // We need to fill-in a fake buffer
      int len =  strlen("fwdtgt") + 1;
      char *buf = (char *) malloc(len);
      memcpy(buf, "fwdtgt", len-1);
      buf[len-1] = 0;
      *parms = new XrdSecParameters(buf, len);
   }

// Release any allocated storage at this point and unlock mutex
//
   krbContext.UnLock();

// Diagnose any errors
//
   if (rc && iferror)
      return Fatal(error, EACCES, iferror, Principal, rc);

// All done
//
   return hsrc;
}

/******************************************************************************/
/*                I n i t i a l i z a t i o n   M e t h o d s                 */
/******************************************************************************/
/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/

int XrdSecProtocolkrb5::Init(XrdOucErrInfo *erp, char *KP, char *kfn)
{
   krb_rc rc;
   char buff[1024];

// Create a kerberos context. There is one such context per protocol object.
//

// If we have no principal then this is a client-side call
//
   if (!KP) {
     if ((rc = krb5_init_context(&krb_client_context)))
       return Fatal(erp, ENOPROTOOPT, "Kerberos initialization failed", KP, rc);

     // Obtain the default cache location
     //
     if ((rc = krb5_cc_default(krb_client_context, &krb_client_ccache)))
       return Fatal(erp, ENOPROTOOPT, "Unable to locate cred cache", KP, rc);
     
     return 0;
   }

   if ((rc = krb5_init_context(&krb_context)))
      return Fatal(erp, ENOPROTOOPT, "Kerberos initialization failed", KP, rc);

// Obtain the default cache location
//
   if ((rc = krb5_cc_default(krb_context, &krb_ccache)))
      return Fatal(erp, ENOPROTOOPT, "Unable to locate cred cache", KP, rc);

// Try to resolve the keyfile name
//
   if (kfn && *kfn)
      {if ((rc = krb5_kt_resolve(krb_context, kfn, &krb_keytab)))
          {snprintf(buff, sizeof(buff), "Unable to find keytab '%s';", kfn);
           return Fatal(erp, ESRCH, buff, Principal, rc);
          }
      } else {
       krb5_kt_default(krb_context, &krb_keytab);
      }

// Keytab name
//
   char krb_kt_name[1024];
   if ((rc = krb5_kt_get_name(krb_context, krb_keytab, &krb_kt_name[0], 1024)))
      {snprintf(buff, sizeof(buff), "Unable to get keytab name;");
       return Fatal(erp, ESRCH, buff, Principal, rc);
      }

// Find out if need to acquire privileges to open and read the keytab file and
// set the required uid,gid accordingly
//
   krb_kt_uid = 0;
   krb_kt_gid = 0;
   char *pf = 0;
   if ((pf = (char *) strstr(krb_kt_name, "FILE:")))
      {pf += strlen("FILE:");
       if (strlen(pf) > 0)
          {struct stat st;
           if (!stat(pf, &st))
              {if (st.st_uid != geteuid() || st.st_gid != getegid())
                  {krb_kt_uid = st.st_uid;
                   krb_kt_gid = st.st_gid;
                  }
              }
          }
      }

// Now, extract the "principal/instance@realm" from the stream
//
   if ((rc = krb5_parse_name(krb_context,KP,&krb_principal)))
     return Fatal(erp, EINVAL, "Cannot parse service name", KP, rc);

// Establish the correct principal to use
//
   if ((rc = krb5_unparse_name(krb_context,(krb5_const_principal)krb_principal,
                              (char **)&Principal)))
      return Fatal(erp, EINVAL, "Unable to unparse principal;", KP, rc);

// All done
//
   return 0;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                 F a t a l                                  */
/******************************************************************************/

int XrdSecProtocolkrb5::Fatal(XrdOucErrInfo *erp, int rc, const char *msg,
                             char *KP, int krc)
{
   const char *msgv[8];
   int k, i = 0;

              msgv[i++] = "Seckrb5: ";    //0
              msgv[i++] = msg;            //1
   if (krc)  {msgv[i++] = "; ";           //2
              msgv[i++] =  krb_etxt(krc); //3
             }
   if (KP)   {msgv[i++] = " (p=";         //4
              msgv[i++] = KP;             //5
              msgv[i++] = ").";           //6
             }
   if (erp) erp->setErrInfo(rc, msgv, i);
      else {for (k = 0; k < i; k++) cerr <<msgv[k];
            cerr <<endl;
           }

   return -1;
}

/******************************************************************************/
/*                          g e t _ k r b C r e d s                           */
/******************************************************************************/

// Warning! The krbClientContext lock must be held prior to calling this routine

int XrdSecProtocolkrb5::get_krbCreds(char *KP, krb5_creds **krb_creds)
{
    krb_rc rc;
    krb5_principal the_principal;
    krb5_creds mycreds;

// Clear my credentials
//
   memset((char *)&mycreds, 0, sizeof(mycreds));

// Setup the "principal/instance@realm"
//
   if ((rc = krb5_parse_name(krb_client_context,KP,&the_principal)))
      {CLDBG("get_krbCreds: Cannot parse service name;" <<krb_etxt(rc));
       return rc;
      }

// Copy the current target principal into the credentials
//
   if ((rc = krb5_copy_principal(krb_client_context, the_principal, &mycreds.server)))
      {CLDBG("get_krbCreds: err copying principal to creds; " <<krb_etxt(rc));
       return rc;
      }

// Get our principal name
//
   if ((rc = krb5_cc_get_principal(krb_client_context, krb_client_ccache, &mycreds.client)))
      {krb5_free_cred_contents(krb_client_context, &mycreds);
       CLDBG("get_krbCreds: err copying client name to creds; " <<krb_etxt(rc));
       return rc;
      }

// Now get the credentials (free our local info)
//
   rc = krb5_get_credentials(krb_client_context, 0, krb_client_ccache, &mycreds,  krb_creds);
   krb5_free_cred_contents(krb_client_context, &mycreds);

// Check if all went well
//
   if (rc) {CLDBG("get_krbCreds: unable to get creds; " <<krb_etxt(rc));}
   return rc;
}

/******************************************************************************/
/*                        g e t _ k r b F w d C r e d s                       */
/******************************************************************************/

int XrdSecProtocolkrb5::get_krbFwdCreds(char *KP, krb5_data *outdata)
{
    int rc;
    krb5_principal client, server;

// Fill-in our principal
//
   if ((rc = krb5_cc_get_principal(krb_client_context, krb_client_ccache, &client)))
      {CLDBG("get_krbFwdCreds: err filling client principal; " <<krb_etxt(rc));
       return rc;
      }

// Fill-in target (service) principal
//
   if ((rc = krb5_parse_name(krb_client_context, KP, &server)))
      {CLDBG("get_krbFwdCreds: Cannot parse service principal;" <<krb_etxt(rc));
       return rc;
      }

// Set the timestamp in the authentication context
//
   if ((rc = krb5_auth_con_setflags(krb_client_context, AuthClientContext,
                                   KRB5_AUTH_CONTEXT_RET_TIME)))
      {CLDBG("Unable to set KRB5_AUTH_CONTEXT_RET_TIME"
                           " in the authentication context" << krb_etxt(rc));
       return rc;
      }

// Acquire a TGT for use at a remote host system
//
   if ((rc = krb5_fwd_tgt_creds(krb_client_context, AuthClientContext, 0 /*host*/,
                                     client, server, krb_client_ccache, true,
                                     outdata)))
      {CLDBG("get_krbFwdCreds: err getting forwarded ticket;" <<krb_etxt(rc));
       return rc;
      }

// Done
//
   return rc;
}

/******************************************************************************/
/*                          e x p _ k r b T k n                               */
/******************************************************************************/

int XrdSecProtocolkrb5::exp_krbTkn(XrdSecCredentials *cred, XrdOucErrInfo *erp)
{
    int rc = 0;

// Create the cache filename, expanding the keywords, if needed
//
    char ccfile[XrdSecMAXPATHLEN];
    strcpy(ccfile, XrdSecProtocolkrb5::ExpFile);
    int nlen = strlen(ccfile);
    char *pusr = (char *) strstr(&ccfile[0], "<user>");
    if (pusr)
       {int ln = strlen(CName);
        if (ln != 6) {
           // Adjust the space
           int lm = strlen(ccfile) - (int)(pusr + 6 - &ccfile[0]); 
           memmove(pusr+ln, pusr+6, lm);
        }
        // Copy the name
        memcpy(pusr, CName, ln);
        // Adjust the length
        nlen += (ln - 6);
        }
    char *puid = (char *) strstr(&ccfile[0], "<uid>");
    struct passwd *pw = getpwnam(CName);
    if (puid)
       {char cuid[20] = {0};
        if (pw)
           sprintf(cuid, "%d", pw->pw_uid);
        int ln = strlen(cuid);
        if (ln != 5) {
           // Adjust the space
           int lm = strlen(ccfile) - (int)(puid + 5 - &ccfile[0]); 
           memmove(puid+ln, pusr+5, lm);
        }
        // Copy the name
        memcpy(puid, cuid, ln);
        // Adjust the length
        nlen += (ln - 5);
        }

// Terminate to the new length
//
    ccfile[nlen] = 0;

// Point the received creds
//
    krbContext.Lock();
    krb5_data forwardCreds;
    forwardCreds.data = &cred->buffer[XrdSecPROTOIDLEN];
    forwardCreds.length = cred->size -XrdSecPROTOIDLEN;

// Get the replay cache
//
    krb5_rcache rcache;
    if ((rc = krb5_get_server_rcache(krb_context,
                                     krb5_princ_component(krb_context, krb_principal, 0),
                                     &rcache)))
       return rc;
    if ((rc = krb5_auth_con_setrcache(krb_context, AuthContext, rcache)))
       return rc;

// Fill-in remote address
//
    if ((rc = krb5_auth_con_setaddrs(krb_context, AuthContext, 0, (krb5_address *)&hostaddr)))
       return rc;

// Readout the credentials
//
    krb5_creds **creds = 0;
    if ((rc = krb5_rd_cred(krb_context, AuthContext,
                           &forwardCreds, &creds, 0)))
       return rc;

// Resolve cache name
    krb5_ccache cache = 0;
    if ((rc = krb5_cc_resolve(krb_context, ccfile, &cache)))
       return rc;

// Need privileges from now on
//
    XrdSysPrivGuard pGuard((uid_t)0, (gid_t)0);
    if (!pGuard.Valid())
       return Fatal(erp, EINVAL, "Unable to acquire privileges;", ccfile, 0);

// Init cache
//
    if ((rc = krb5_cc_initialize(krb_context, cache,
                                 Ticket->enc_part2->client)))
       return rc;

// Store credentials in cache
//
    if ((rc = krb5_cc_store_cred(krb_context, cache, *creds)))
       return rc;

// Close cache
    if ((rc = krb5_cc_close(krb_context, cache)))
       return rc;

// Change permission and ownership of the file
//
    if (chown(ccfile, pw->pw_uid, pw->pw_gid) == -1)
       return Fatal(erp, errno, "Unable to change file ownership;", ccfile, 0);
    if (chmod(ccfile, 0600) == -1)
       return Fatal(erp, errno, "Unable to change file permissions;", ccfile, 0);

// Done
//
   return 0;
}
 
/******************************************************************************/
/*                X r d S e c p r o t o c o l k r b 5 I n i t                 */
/******************************************************************************/
  
extern "C"
{
char  *XrdSecProtocolkrb5Init(const char     mode,
                              const char    *parms,
                              XrdOucErrInfo *erp)
{
   char *op, *KPrincipal=0, *Keytab=0, *ExpFile=0;
   char parmbuff[1024];
   XrdOucTokenizer inParms(parmbuff);
   int options = XrdSecNOIPCHK;
   static bool serverinitialized = false;

// For client-side one-time initialization, we only need to set debug flag and
// initialize the kerberos context and cache location.
//
   if ((mode == 'c') || (serverinitialized))
      {
       int opts = 0;
       if (getenv("XrdSecDEBUG")) opts |= XrdSecDEBUG;
       if (getenv("XrdSecKRB5INITTKN")) opts |= XrdSecINITTKN;
       XrdSecProtocolkrb5::setClientOpts(opts);
       return (XrdSecProtocolkrb5::Init(erp) ? (char *)0 : (char *)"");
      }

   if (!serverinitialized) {
     serverinitialized = true;
   }

// Duplicate the parms
//
   if (parms) strlcpy(parmbuff, parms, sizeof(parmbuff));
      else {char *msg = (char *)"Seckrb5: Kerberos parameters not specified.";
            if (erp) erp->setErrInfo(EINVAL, msg);
               else cerr <<msg <<endl;
            return (char *)0;
           }

// Expected parameters: [<keytab>] [-ipchk] [-exptkn[:filetemplate]] <principal>
//
   if (inParms.GetLine())
      {if ((op = inParms.GetToken()) && *op == '/')
          {Keytab = op; op = inParms.GetToken();}
           if (op && !strcmp(op, "-ipchk"))
              {options &= ~XrdSecNOIPCHK;
               op = inParms.GetToken();
              }
           if (op && !strncmp(op, "-exptkn", 7))
              {options |= XrdSecEXPTKN;
               if (op[7] == ':') ExpFile = op+8;
               op = inParms.GetToken();
              }
           KPrincipal = strdup(op);
      }

    if (ExpFile)
       fprintf(stderr,"Template for exports: %s\n", ExpFile);
    else
       fprintf(stderr,"Template for exports not set\n");

// Now make sure that we have all the right info
//
   if (!KPrincipal)
      {char *msg = (char *)"Seckrb5: Kerberos principal not specified.";
       if (erp) erp->setErrInfo(EINVAL, msg);
          else cerr <<msg <<endl;
       return (char *)0;
      }

// Expand possible keywords in the principal
//
    int plen = strlen(KPrincipal);
    int lkey = strlen("<host>");
    char *phost = (char *) strstr(&KPrincipal[0], "<host>");
    if (phost)
       {char *hn = XrdNetDNS::getHostName();
        if (hn)
           {int lhn = strlen(hn);
            if (lhn != lkey) {
              // Allocate, if needed
              int lnew = plen - lkey + lhn;
              if (lnew > plen) {
                 KPrincipal = (char *) realloc(KPrincipal, lnew+1);
                 KPrincipal[lnew] = 0;
                 phost = (char *) strstr(&KPrincipal[0], "<host>");
              }
              // Adjust the space
              int lm = plen - (int)(phost + lkey - &KPrincipal[0]); 
              memmove(phost + lhn, phost + lkey, lm);
            }
            // Copy the name
            memcpy(phost, hn, lhn);
            // Cleanup
            free(hn);
           }
       }

// Now initialize the server
//
   options |= XrdSecDEBUG;
   XrdSecProtocolkrb5::setExpFile(ExpFile);
   XrdSecProtocolkrb5::setOpts(options);
   if (!XrdSecProtocolkrb5::Init(erp, KPrincipal, Keytab))
      {free(KPrincipal);
       int lpars = strlen(XrdSecProtocolkrb5::getPrincipal());
       if (options & XrdSecEXPTKN)
          lpars += strlen(",fwd");
       char *params = (char *)malloc(lpars+1);
       if (params)
          {memset(params,0,lpars+1);
           strcpy(params,XrdSecProtocolkrb5::getPrincipal());
           if (options & XrdSecEXPTKN)
              strcat(params,",fwd");
           XrdSecProtocolkrb5::setParms(params);
           return params;
          }
      }

// Failure
//
   free(KPrincipal);
   return (char *)0;
}
}

/******************************************************************************/
/*              X r d S e c P r o t o c o l k r b 5 O b j e c t               */
/******************************************************************************/
  
extern "C"
{
XrdSecProtocol *XrdSecProtocolkrb5Object(const char              mode,
                                         const char             *hostname,
                                         const struct sockaddr  &netaddr,
                                         const char             *parms,
                                               XrdOucErrInfo    *erp)
{
   XrdSecProtocolkrb5 *prot;
   char *KPrincipal=0;

// If this is a client call, then we need to get the target principal from the
// parms (which must be the first and only token). For servers, we use the
// context we established at initialization time.
//
   if (mode == 'c')
      {if ((KPrincipal = (char *)parms)) while(*KPrincipal == ' ') KPrincipal++;
       if (!KPrincipal || !*KPrincipal)
          {char *msg = (char *)"Seckrb5: Kerberos principal not specified.";
           if (erp) erp->setErrInfo(EINVAL, msg);
              else cerr <<msg <<endl;
           return (XrdSecProtocol *)0;
          }
      }

// Get a new protocol object
//
   if (!(prot = new XrdSecProtocolkrb5(KPrincipal, hostname, &netaddr)))
      {char *msg = (char *)"Seckrb5: Insufficient memory for protocol.";
       if (erp) erp->setErrInfo(ENOMEM, msg);
          else cerr <<msg <<endl;
       return (XrdSecProtocol *)0;
      }

// All done
//
   return prot;
}

void
      __eprintf (const char *string, const char *expression,
                 unsigned int line, const char *filename)
      {
        fprintf (stderr, string, expression, line, filename);
        fflush (stderr);
        abort ();
      }
}
