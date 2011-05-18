#ifndef _SECPROTOCOLSSS_
#define _SECPROTOCOLSSS_
/******************************************************************************/
/*                                                                            */
/*                  X r d S e c P r o t o c o l s s s . h h                   */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdCrypto/XrdCryptoLite.hh"
#include "XrdSec/XrdSecInterface.hh"
#include "XrdSecsss/XrdSecsssID.hh"
#include "XrdSecsss/XrdSecsssKT.hh"
#include "XrdSecsss/XrdSecsssRR.hh"

class XrdOucErrInfo;

class XrdSecProtocolsss : public XrdSecProtocol
{
public:
friend class XrdSecProtocolDummy; // Avoid stupid gcc warnings about destructor


        int                Authenticate  (XrdSecCredentials *cred,
                                          XrdSecParameters **parms,
                                          XrdOucErrInfo     *einfo=0);

        void               Delete();

static  int                eMsg(const char *epn, int rc, const char *txt1,
                                const char *txt2=0,      const char *txt3=0, 
                                const char *txt4=0);

static  int                Fatal(XrdOucErrInfo *erP, const char *epn, int rc,
                                                     const char *etxt);

        XrdSecCredentials *getCredentials(XrdSecParameters  *parms=0,
                                          XrdOucErrInfo     *einfo=0);

        int   Init_Client(XrdOucErrInfo *erp, const char *Parms);

        int   Init_Server(XrdOucErrInfo *erp, const char *Parms);

static  char *Load_Client(XrdOucErrInfo *erp, const char *Parms);

static  char *Load_Server(XrdOucErrInfo *erp, const char *Parms);

static  void  setOpts(int opts) {options = opts;}

        XrdSecProtocolsss(const char                *hname,
                          const struct sockaddr     *ipadd)
                         : XrdSecProtocol("sss"),
                           keyTab(0), Crypto(0), idBuff(0), Sequence(0)
                         {urName = strdup(hname);}

struct Crypto {const char *cName; char cType;};

private:
       ~XrdSecProtocolsss() {} // Delete() does it all

int                Decode(XrdOucErrInfo *error, XrdSecsssKT::ktEnt &decKey,
                          char *iBuff, XrdSecsssRR_Data *rrData, int iSize);
XrdSecCredentials *Encode(XrdOucErrInfo *error, XrdSecsssKT::ktEnt &encKey,
                          XrdSecsssRR_Hdr *rrHdr, XrdSecsssRR_Data *rrData,
                          int dLen);
int            getCred(XrdOucErrInfo *, XrdSecsssRR_Data &);
int            getCred(XrdOucErrInfo *, XrdSecsssRR_Data &, XrdSecParameters *);
char          *getLID(char *buff, int blen);
static
XrdCryptoLite *Load_Crypto(XrdOucErrInfo *erp, const char *eN);
static
XrdCryptoLite *Load_Crypto(XrdOucErrInfo *erp, const char  eT);
int            myClock();
char          *setID(char *id, char **idP);

static struct Crypto  CryptoTab[];

static const char    *myName;
static int            myNLen;
       char          *urName;
static int            options;
static int            isMutual;
static int            deltaTime;
static int            ktFixed;

static XrdSecsssKT   *ktObject;  // Both:   Default Key Table object
       XrdSecsssKT   *keyTab;    // Both:   Active  Key Table

static XrdCryptoLite *CryptObj;  // Both:   Default Cryptogrophy object
       XrdCryptoLite *Crypto;    // Both:   Active  Cryptogrophy object

static XrdSecsssID   *idMap;     // Client: Registry
       char          *idBuff;    // Server: Underlying buffer for XrdSecEntity
static char          *staticID;  // Client: Static identity
static int            staticIDsz;// Client: Static identity length
       int            Sequence;  // Client: Check for sequencing
};
#endif
