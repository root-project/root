/******************************************************************************/
/*                                                                            */
/*                  X r d S e c P r o t o c o l s s s . c c                   */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <sys/param.h>

#include "XrdNet/XrdNetDNS.hh"
#include "XrdOuc/XrdOucCRC.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdOuc/XrdOucPup.hh"
#include "XrdOuc/XrdOucTokenizer.hh"
#include "XrdSecsss/XrdSecProtocolsss.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"

/******************************************************************************/
/*                               D e f i n e s                                */
/******************************************************************************/
  
#define XrdSecPROTOIDENT    "sss"
#define XrdSecPROTOIDLEN    sizeof(XrdSecPROTOIDENT)
#define XrdSecDEBUG         0x1000

#define CLDBG(x) if (options & XrdSecDEBUG) cerr <<"sec_sss: " <<x <<endl;

/******************************************************************************/
/*                           S t a t i c   D a t a                            */
/******************************************************************************/

const char    *XrdSecProtocolsss::myName;
int            XrdSecProtocolsss::myNLen;

XrdCryptoLite *XrdSecProtocolsss::CryptObj   = 0;
XrdSecsssKT   *XrdSecProtocolsss::ktObject   = 0;
XrdSecsssID   *XrdSecProtocolsss::idMap      = 0;
char          *XrdSecProtocolsss::staticID   = 0;
int            XrdSecProtocolsss::staticIDsz = 0;
int            XrdSecProtocolsss::options    = 0;
int            XrdSecProtocolsss::isMutual   = 0;
int            XrdSecProtocolsss::deltaTime  =13;
int            XrdSecProtocolsss::ktFixed    = 0;

struct XrdSecProtocolsss::Crypto XrdSecProtocolsss::CryptoTab[] = {
       {"bf32", XrdSecsssRR_Hdr::etBFish32},
       {0, '0'}
       };

/******************************************************************************/
/*                          A u t h e n t i c a t e                           */
/******************************************************************************/

int XrdSecProtocolsss::Authenticate(XrdSecCredentials *cred,
                                    XrdSecParameters **parms,
                                    XrdOucErrInfo     *einfo)
{
   XrdSecsssRR_Hdr    *rrHdr = (XrdSecsssRR_Hdr *)(cred->buffer);
   XrdSecsssRR_Data    rrData;
   XrdSecsssKT::ktEnt  decKey;
   XrdSecEntity        myID("sss");
   char lidBuff[16],  eType, *idP, *dP, *eodP, *theHost = 0;
   int idTLen = 0, idSz, dLen;

// Decode the credentials
//
   if ((dLen = Decode(einfo, decKey, cred->buffer, &rrData, cred->size)) <= 0)
      return -1;

// Check if we should echo back the LID
//
   if (rrData.Options == XrdSecsssRR_Data::SndLID)
      {rrData.Options = 0;
       getLID(lidBuff, sizeof(lidBuff));
       dP = rrData.Data;
       *dP++ = XrdSecsssRR_Data::theLgid;
       XrdOucPup::Pack(&dP, lidBuff);
       *parms = Encode(einfo, decKey, rrHdr, &rrData, dP-(char *)&rrData);
       return (*parms ? 1 : -1);
      }

// Extract out the entity ID
//
   dP = rrData.Data; eodP = dLen + (char *)&rrData;
   while(dP < eodP)
        {eType = *dP++;
         if (!XrdOucPup::Unpack(&dP, eodP, &idP, idSz) || *idP == '\0')
            {Fatal(einfo, "Authenticate", EINVAL, "Invalid id string.");
             return -1;
            }
         idTLen += idSz;
         switch(eType)
               {case XrdSecsssRR_Data::theName: myID.name         = idP; break;
                case XrdSecsssRR_Data::theVorg: myID.vorg         = idP; break;
                case XrdSecsssRR_Data::theRole: myID.role         = idP; break;
                case XrdSecsssRR_Data::theGrps: myID.grps         = idP; break;
                case XrdSecsssRR_Data::theEndo: myID.endorsements = idP; break;
                case XrdSecsssRR_Data::theHost: theHost           = idP; break;
                case XrdSecsssRR_Data::theRand: idTLen -= idSz;          break;
                default: Fatal(einfo,"Authenticate",EINVAL,"Invalid id type.");
                         return -1;
               }
        }

// Verify that we have some kind of identification
//
   if (!idTLen)
      {Fatal(einfo, "Authenticate", ENOENT, "No id specified.");
       return -1;
      }

// Verify the source of the information to largely prevent packet stealing
//
   if (!theHost)
      {Fatal(einfo, "Authenticate", ENOENT, "No hostname specified.");
       return -1;
      }
   if (strcmp(theHost, urName))
      {Fatal(einfo, "Authenticate", EINVAL, "Hostname mismatch.");
       return -1;
      }

// Set correct username
//
        if (decKey.Data.Opts & XrdSecsssKT::ktEnt::anyUSR)
           {if (!myID.name) myID.name = (char *)"nobody";}
   else myID.name = decKey.Data.User;

// Set correct group
//
         if (decKey.Data.Opts & XrdSecsssKT::ktEnt::usrGRP) myID.grps = 0;
   else {if (decKey.Data.Opts & XrdSecsssKT::ktEnt::anyGRP)
            {if (!myID.grps) myID.grps = (char *)"nogroup";}
            else myID.grps = decKey.Data.Grup;
        }

// Complete constructing our identification
//
   if (idBuff) free(idBuff);
   idBuff = idP = (char *)malloc(idTLen);
   Entity.name         = setID(myID.name,         &idP);
   Entity.vorg         = setID(myID.vorg,         &idP);
   Entity.role         = setID(myID.role,         &idP);
   Entity.grps         = setID(myID.grps,         &idP);
   Entity.endorsements = setID(myID.endorsements, &idP);

// All done
//
   return 0;
}

/******************************************************************************/
/*                                D e l e t e                                 */
/******************************************************************************/
  
void XrdSecProtocolsss::Delete()
{
// Delete things that get re-allocated every time. The staticID is allocated
// only once so it must stick around for every instance of this object.
//
     if (Entity.host)         free(Entity.host);
     if (urName)              free(urName);
     if (idBuff)              free(idBuff);
     if (keyTab && keyTab != ktObject) delete keyTab;

     delete this;
}

/******************************************************************************/
/*                                  e M s g                                   */
/******************************************************************************/

int XrdSecProtocolsss::eMsg(const char *epname, int rc,
                            const char *txt1, const char *txt2,
                            const char *txt3, const char *txt4)
{
              cerr <<"Secsss (" << epname <<"): ";
              cerr <<txt1;
   if (rc>0)  cerr <<"; " <<strerror(rc);
   if (txt2)  cerr <<txt2;
   if (txt3)  cerr <<txt3;
   if (txt4)  cerr <<txt4;
              cerr <<endl;

   return (rc ? (rc < 0 ? rc : -rc) : -1);
}

/******************************************************************************/
/*                                 F a t a l                                  */
/******************************************************************************/

int XrdSecProtocolsss::Fatal(XrdOucErrInfo *erP, const char *epn, int rc,
                                                 const char *etxt)
{
   if (erP) {erP->setErrInfo(rc, etxt);
             CLDBG(epn <<": " <<etxt);
            }
      else  eMsg(epn, rc, etxt);
   return 0;
}

/******************************************************************************/
/*                        g e t C r e d e n t i a l s                         */
/******************************************************************************/

XrdSecCredentials *XrdSecProtocolsss::getCredentials(XrdSecParameters *parms,
                                                      XrdOucErrInfo   *einfo)
{
   XrdSecsssRR_Hdr    rrHdr;
   XrdSecsssRR_Data   rrData;
   XrdSecsssKT::ktEnt encKey;
   int dLen;

// Get the actual data portion
//
   if ((dLen=(Sequence ? getCred(einfo,rrData,parms) 
                       : getCred(einfo,rrData      )))<=0)
      return (XrdSecCredentials *)0;

// Get an encryption key
//
   if (keyTab->getKey(encKey))
      {Fatal(einfo, "getCredentials", ENOENT, "Encryption key not found.");
       return (XrdSecCredentials *)0;
      }

// Fill out the header
//
   strcpy(rrHdr.ProtID, XrdSecPROTOIDENT);
   memset(rrHdr.Pad, 0, sizeof(rrHdr.Pad));
   rrHdr.KeyID = htonll(encKey.Data.ID);
   rrHdr.EncType = Crypto->Type();

// Now simply encode the data and return the result
//
   return Encode(einfo, encKey, &rrHdr, &rrData, dLen);
}

/******************************************************************************/
/*                           I n i t _ C l i e n t                            */
/******************************************************************************/

int XrdSecProtocolsss::Init_Client(XrdOucErrInfo *erp, const char *pP)
{
   XrdSecsssKT *ktP;
   struct stat buf;
   char *Colon;
   int lifeTime;

// We must have <enctype>:[<ktpath>]
//
   if (!pP || !*pP) return Fatal(erp, "Init_Client", EINVAL,
                                 "Client parameters missing.");

// Get encryption object
//
   if (!*pP || *(pP+1) != '.') return Fatal(erp, "Init_Client", EINVAL,
                                 "Encryption type missing.");
   if (!(Crypto = Load_Crypto(erp, *pP))) return 0;
   pP += 2;

// The next item is the cred lifetime
//
   lifeTime = strtol(pP, &Colon, 10);
   if (!lifeTime || *Colon != ':') return Fatal(erp, "Init_Client", EINVAL,
                                          "Credential lifetime missing.");
   deltaTime = lifeTime; pP = Colon+1;

// Get the correct keytab
//
        if (ktFixed || (ktObject && ktObject->Same(pP))) keyTab = ktObject;
   else if (*pP == '/' && !stat(pP, &buf))
           {if (!(ktP=new XrdSecsssKT(erp,pP,XrdSecsssKT::isClient,3600)))
               return Fatal(erp, "Init_Client", ENOMEM,
                                 "Unable to create keytab object.");
            if (erp->getErrInfo()) {delete ktP; return 0;}
            if (!ktObject) ktObject = ktP;
            keyTab = ktP;
            CLDBG("Client keytab='" <<pP <<"'");
           } else keyTab = ktObject;

   if (!keyTab)
      return Fatal(erp, "Init_Client", ENOENT, 
                        "Unable to determine keytab location.");

// All done
//
   return 1;
}


/******************************************************************************/
/*                           I n i t _ S e r v e r                            */
/******************************************************************************/

int XrdSecProtocolsss::Init_Server(XrdOucErrInfo *erp, const char *pP)
{

// This is a trivial init
//
   keyTab = ktObject;
   Crypto = CryptObj;
   return 1;
}

/******************************************************************************/
/*                           L o a d _ C l i e n t                            */
/******************************************************************************/
  
char *XrdSecProtocolsss::Load_Client(XrdOucErrInfo *erp, const char *parms)
{
   static const char *KTPath = XrdSecsssKT::genFN();
   static const int   rfrHR = 60*60;
   struct stat buf;
   XrdSecsssID::authType aType = XrdSecsssID::idStatic;
   const char *kP = 0;

// Get our full host name
//
   if (!(myName = XrdNetDNS::getHostName()))
      {Fatal(erp, "Load_Client", ENOENT, "Unable to obtain local hostname.");
       return (char *)0;
      }
   myNLen = strlen(myName)+1;

// Check for the presence of a registry object
//
   idMap = XrdSecsssID::getObj(aType, &staticID, staticIDsz);
   switch(aType)
         {case XrdSecsssID::idDynamic:  isMutual = 1; break;
          case XrdSecsssID::idStaticM:  isMutual = 1;
          case XrdSecsssID::idStatic:
               default:                 idMap    = 0; break;
          }

// We want to establish the default location of the keytable
//
   if ((kP = getenv("XrdSecsssKT")) && *kP && !stat(kP, &buf)) ktFixed = 1;
      else kP = 0;

   if (!kP && !stat(KTPath, &buf)) kP = KTPath;

// Build the keytable if we actual have a path (if none, then the server
// will have to supply the path)
//
   if (kP)
      {if (!(ktObject=new XrdSecsssKT(erp,kP,XrdSecsssKT::isClient,rfrHR)))
          {Fatal(erp, "Load_Client", ENOMEM, "Unable to create keytab object.");
           return (char *)0;
          }
       if (erp->getErrInfo())
          {delete ktObject, ktObject = 0; return (char *)0;}
       CLDBG("Client keytab='" <<kP <<"'");
      }

// All done
//
   return strdup("");
}
  
/******************************************************************************/
/*                           L o a d _ C r y p t o                            */
/******************************************************************************/
  
XrdCryptoLite *XrdSecProtocolsss::Load_Crypto(XrdOucErrInfo *erp,
                                              const char    *eN)
{
   XrdCryptoLite *cP;
   char buff[128];
   int rc, i = 0;

// Find correct crypto object
//
   while(CryptoTab[i].cName && strcmp(CryptoTab[i].cName, eN)) i++;

// If we didn't find it, complain
//
   if (!CryptoTab[i].cName)
      {sprintf(buff, "Secsss: %s cryptography not supported.", eN);
       Fatal(erp, "Load_Crypto", EINVAL, buff);
       return (XrdCryptoLite *)0;
      }

// Return load result
//
   if ((cP = XrdCryptoLite::Create(rc, eN, CryptoTab[i].cType))) return cP;
   sprintf(buff,"Secsss: %s cryptography load failed; %s",eN,strerror(rc));
   Fatal(erp, "Load_Crypto", EINVAL, buff);
   return (XrdCryptoLite *)0;
}

/******************************************************************************/
  
XrdCryptoLite *XrdSecProtocolsss::Load_Crypto(XrdOucErrInfo *erp,
                                              const char     eT)
{
   XrdCryptoLite *cP;
   char buff[128];
   int rc, i = 0;

// Check if we can use the satic object
//
   if (CryptObj && eT == CryptObj->Type()) return CryptObj;

// Find correct crypto object
//
   while(CryptoTab[i].cName && CryptoTab[i].cType != eT) i++;

// If we didn't find it, complain
//
   if (!CryptoTab[i].cName)
      {sprintf(buff, "Secsss: 0x%hhx cryptography not supported.", eT);
       Fatal(erp, "Load_Crypto", EINVAL, buff);
       return (XrdCryptoLite *)0;
      }

// Return load result
//
   if ((cP = XrdCryptoLite::Create(rc, CryptoTab[i].cName, eT))) return cP;
   sprintf(buff,"Secsss: 0x%hhx cryptography load failed; %s",eT,strerror(rc));
   Fatal(erp, "Load_Crypto", EINVAL, buff);
   return (XrdCryptoLite *)0;
}

/******************************************************************************/
/*                           L o a d _ S e r v e r                            */
/******************************************************************************/
  
char *XrdSecProtocolsss::Load_Server(XrdOucErrInfo *erp, const char *parms)
{
   const char *msg = 0;
   const char *encName = "bf32", *ktClient = "", *ktServer = 0;
   char buff[2048], parmbuff[2048], *op, *od, *eP;
   int lifeTime = 13, rfrTime = 60*60;
   XrdOucTokenizer inParms(parmbuff);

// Duplicate the parms
//
   if (parms) strlcpy(parmbuff, parms, sizeof(parmbuff));

// Expected parameters: [-c <ckt_path>] [-e <enctype>]
//                      [-r <minutes>] [-l <seconds>]  [-s <skt_path>]
//
   if (parms && inParms.GetLine())
      while((op = inParms.GetToken()))
           {if (!(od = inParms.GetToken()))
               {sprintf(buff,"Secsss: Missing %s parameter argument",op);
                msg = buff; break;
               }
                 if (!strcmp("-c", op)) ktClient = od;
            else if (!strcmp("-e", op)) encName  = od;
            else if (!strcmp("-l", op))
                    {lifeTime = strtol(od, &eP, 10) * 60;
                     if (errno || *eP || lifeTime < 1)
                        {msg = "Secsss: Invalid life time"; break;}
                    }
            else if (!strcmp("-r", op))
                    {rfrTime = strtol(od, &eP, 10) * 60;
                     if (errno || *eP || rfrTime < 600)
                        {msg = "Secsss: Invalid refresh time"; break;}
                    }
            else if (!strcmp("-s", op)) ktServer = od;
            else {sprintf(buff,"Secsss: Invalid parameter - %s",op);
                  msg = buff; break;
                 }
           }

// Check for errors
//
   if (msg) {Fatal(erp, "Load_Server", EINVAL, msg); return (char *)0;}

// Load the right crypto object
//
   if (!(CryptObj = Load_Crypto(erp, encName))) return (char *)0;

// Supply default keytab location if not specified
//
   if (!ktServer) ktServer = XrdSecsssKT::genFN();

// Set the delta time used to expire credentials
//
   deltaTime = lifeTime;

// Create a keytab object (only one for the server)
//
   if (!(ktObject = new XrdSecsssKT(erp, ktServer, XrdSecsssKT::isServer,
                                         rfrTime)))
      {Fatal(erp, "Load_Server", ENOMEM, "Unable to create keytab object.");
       return (char *)0;
      }
   if (erp->getErrInfo()) return (char *)0;
   ktFixed = 1;
   CLDBG("Server keytab='" <<ktServer <<"'");

// Construct client parameter <enccode>:<keytab>
//
   sprintf(buff, "%c.%d:%s", CryptObj->Type(), lifeTime, ktClient);
   CLDBG("client parms='" <<buff <<"'");
   return strdup(buff);
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                D e c o d e                                 */
/******************************************************************************/

int                XrdSecProtocolsss::Decode(XrdOucErrInfo      *error,
                                             XrdSecsssKT::ktEnt &decKey,
                                             char               *iBuff,
                                             XrdSecsssRR_Data   *rrData,
                                             int                 iSize)
{
   static const int maxLen = sizeof(XrdSecsssRR_Hdr) + sizeof(XrdSecsssRR_Data);
   static const int minLen = maxLen - XrdSecsssRR_Data::DataSz;
   XrdSecsssRR_Hdr  *rrHdr  = (XrdSecsssRR_Hdr  *)iBuff;
   int rc, genTime, dLen = iSize - sizeof(XrdSecsssRR_Hdr);

// Verify that some credentials exist
//
   if (iSize <= minLen || !iBuff || iSize >= maxLen)
      return Fatal(error,"Decode",EINVAL,"Credentials missing or of invalid size.");

// Check if this is a recognized protocol
//
   if (strcmp(rrHdr->ProtID, XrdSecPROTOIDENT))
      {char emsg[256];
       snprintf(emsg, sizeof(emsg),
                "Authentication protocol id mismatch (%.4s != %.4s).",
                XrdSecPROTOIDENT,  rrHdr->ProtID);
       return Fatal(error, "Decode", EINVAL, emsg);
      }

// Verify decryption method
//
   if (rrHdr->EncType != Crypto->Type())
      return Fatal(error, "Decode", ENOTSUP, "Crypto type not supported.");

// Get the key
//
   decKey.Data.ID = ntohll(rrHdr->KeyID);
   decKey.Data.Name[0] = '\0';
   if (keyTab->getKey(decKey))
      return Fatal(error, "Decode", ENOENT, "Decryption key not found.");

// Decrypt
//
   if ((rc = Crypto->Decrypt(decKey.Data.Val, decKey.Data.Len,
                             iBuff+sizeof(XrdSecsssRR_Hdr), dLen,
                             (char *)rrData, sizeof(XrdSecsssRR_Data))) <= 0)
      return Fatal(error, "Decode", -rc, "Unable to decrypt credentials.");

// Verify that the packet has not expired (OK to do before CRC check)
//
   genTime = ntohl(rrData->GenTime);
   if (genTime + deltaTime <= myClock())
      return Fatal(error, "Decode", ESTALE, "Credentials expired.");

// Return success (size of decrypted info)
//
   return rc;
}
  
/******************************************************************************/
/*                                E n c o d e                                 */
/******************************************************************************/

XrdSecCredentials *XrdSecProtocolsss::Encode(XrdOucErrInfo      *einfo,
                                             XrdSecsssKT::ktEnt &encKey,
                                             XrdSecsssRR_Hdr    *rrHdr,
                                             XrdSecsssRR_Data   *rrData,
                                             int                 dLen)
{
   static const int hdrSZ = sizeof(XrdSecsssRR_Hdr);
   char *credP, *eodP = ((char *)rrData) + dLen;
   int knum, cLen;

// Make sure we have enought space left in the buffer
//
   if (dLen > (int)sizeof(rrData->Data) - (16+myNLen))
      {Fatal(einfo,"Encode",ENOBUFS,"Insufficient buffer space for credentials.");
       return (XrdSecCredentials *)0;
      }

// Add in our host name for source verification
//
   if (myName)
      {*eodP++ = XrdSecsssRR_Data::theHost;
       XrdOucPup::Pack(&eodP, myName, myNLen);
       dLen = eodP - (char *)rrData;
      }

// Make sure we have at least 128 bytes of encrypted data
//
   if (dLen < 128)
      {char  rBuff[128];
       int   rLen = 128 - dLen;
       *eodP++ = XrdSecsssRR_Data::theRand;
       XrdSecsssKT::genKey(rBuff, rLen);
       if (!(*rBuff)) *rBuff = ~(*rBuff);
       XrdOucPup::Pack(&eodP, rBuff, rLen);
       dLen = eodP - (char *)rrData;
      }

// Complete the packet
//
   XrdSecsssKT::genKey(rrData->Rand, sizeof(rrData->Rand));
   rrData->GenTime = htonl(myClock());
   memset(rrData->Pad, 0, sizeof(rrData->Pad));

// Allocate an output buffer
//
   cLen = hdrSZ + dLen + Crypto->Overhead();
   if (!(credP = (char *)malloc(cLen)))
      {Fatal(einfo, "Encode", ENOMEM, "Insufficient memory for credentials.");
       return (XrdSecCredentials *)0;
      }

// Copy the header and encrypt the data
//
   memcpy(credP, (const void *)rrHdr, hdrSZ);
   if ((dLen = Crypto->Encrypt(encKey.Data.Val, encKey.Data.Len, (char *)rrData,
                               dLen, credP+hdrSZ, cLen-hdrSZ)) <= 0)
      {Fatal(einfo, "Encode", -dLen, "Unable to encrypt credentials.");
       return (XrdSecCredentials *)0;
      }

// Return new credentials
//
   dLen += hdrSZ; knum = encKey.Data.ID&0x7fffffff;
   CLDBG("Ret " <<dLen <<" bytes of credentials; k=" <<knum);
   return new XrdSecCredentials(credP, dLen);
}

/******************************************************************************/
/*                               g e t C r e d                                */
/******************************************************************************/

int XrdSecProtocolsss::getCred(XrdOucErrInfo    *einfo,
                               XrdSecsssRR_Data &rrData)
{
// Indicate we have been here
//
   Sequence = 1;

// If we need mutual authentication
//
   if (isMutual)
      {rrData.Options = XrdSecsssRR_Data::SndLID;
       return XrdSecsssRR_Data_HdrLen;
      }

// Send the static ID
//
   memcpy(rrData.Data, staticID, staticIDsz);
   rrData.Options = 0;
   return XrdSecsssRR_Data_HdrLen + staticIDsz;
}

/******************************************************************************/

int XrdSecProtocolsss::getCred(XrdOucErrInfo    *einfo,
                               XrdSecsssRR_Data &rrData,
                               XrdSecParameters *parm)
{
   XrdSecsssKT::ktEnt  decKey;
   XrdSecsssRR_Data    prData;
   char *lidP = 0, *idP, *dP, *eodP, idType;
   int lidSz, idSz, dLen;

// Decode the credentials
//
   if ((dLen = Decode(einfo, decKey, parm->buffer, &prData, parm->size)) <= 0)
      return -1;

// The only thing allowed here is an echoed loginid
//
   if (prData.Options 
   ||  dLen >= (int)sizeof(XrdSecsssRR_Data)
   ||  prData.Data[0] != XrdSecsssRR_Data::theLgid)
      return Fatal(einfo, "getCred", EINVAL, "Invalid server response.");

// Extract out the loginid
//
   dP = prData.Data; eodP = dLen + (char *)&prData;
   while(dP < eodP)
        {idType = *dP++;
         if (!XrdOucPup::Unpack(&dP, eodP, &idP, idSz) 
         ||  !idP || *idP == '\0')
            return Fatal(einfo, "getCred", EINVAL, "Invalid id string.");
         switch(idType)
               {case XrdSecsssRR_Data::theLgid: lidP = idP; lidSz = idSz; break;
                case XrdSecsssRR_Data::theRand:                           break;
                default: return Fatal(einfo,"getCred",EINVAL,"Invalid id type.");
               }
        }

// Verify that we have the loginid
//
   if (!lidP) return Fatal(einfo, "getCred", ENOENT, "No loginid specified.");

// Try to map the id appropriately
//
   if (!idMap)
      {if (staticID && staticIDsz < (int)sizeof(rrData.Data))
          {memcpy(rrData.Data, staticID, staticIDsz); 
           idSz = staticIDsz;
           return XrdSecsssRR_Data_HdrLen + idSz;
          }
       return Fatal(einfo, "getCred", ENAMETOOLONG, "Authinfo too big.");
      }

// Map the loginid
//
   if ((dLen = idMap->Find(lidP, rrData.Data, sizeof(rrData.Data))) <= 0)
      return Fatal(einfo, "getCred", ESRCH, "No loginid mapping.");

// All done
//
   rrData.Options = XrdSecsssRR_Data::UseData;
   return XrdSecsssRR_Data_HdrLen + dLen;
}

/******************************************************************************/
/*                                g e t L I D                                 */
/******************************************************************************/
  
char *XrdSecProtocolsss::getLID(char *buff, int blen)
{
   char *dot;

// Extract out the loginid from the trace id
//
   if (!Entity.tident 
   ||  !(dot = index(Entity.tident,'.'))
   ||  dot == Entity.tident
   ||  dot >= (Entity.tident+blen)) strcpy(buff,"nobody");
      else {int idsz = dot - Entity.tident;
            strncpy(buff, Entity.tident, idsz);
            *(buff+idsz) = '\0';
           }

// All done
//
   return buff;
}

/******************************************************************************/
/*                               m y C l o c k                                */
/******************************************************************************/
  
int XrdSecProtocolsss::myClock()
{
   static const time_t baseTime = 1222183880;

   return static_cast<int>(time(0)-baseTime);
}

/******************************************************************************/
/*                                 s e t I D                                  */
/******************************************************************************/
  
char *XrdSecProtocolsss::setID(char *id, char **idP)
{
   if (id)
      {int n = strlen(id);
       strcpy(*idP, id); id = *idP; *idP = *idP + n + 1;
      }
   return id;
}

/******************************************************************************/
/*                 X r d S e c P r o t o c o l s s s I n i t                  */
/******************************************************************************/
  
extern "C"
{
char  *XrdSecProtocolsssInit(const char     mode,
                             const char    *parms,
                             XrdOucErrInfo *erp)
{

// Set debug option
//
   if (getenv("XrdSecDEBUG")) XrdSecProtocolsss::setOpts(XrdSecDEBUG);

// Perform load-time initialization
//
   return (mode == 'c' ? XrdSecProtocolsss::Load_Client(erp, parms)
                       : XrdSecProtocolsss::Load_Server(erp, parms));
}
}

/******************************************************************************/
/*               X r d S e c P r o t o c o l s s s O b j e c t                */
/******************************************************************************/
  
extern "C"
{
XrdSecProtocol *XrdSecProtocolsssObject(const char              mode,
                                        const char             *hostname,
                                        const struct sockaddr  &netaddr,
                                        const char             *parms,
                                              XrdOucErrInfo    *erp)
{
   XrdSecProtocolsss *prot;
   int Ok;

// Get a new protocol object
//
   if (!(prot = new XrdSecProtocolsss(hostname, &netaddr)))
      XrdSecProtocolsss::Fatal(erp, "sss_Object", ENOMEM,
                         "Secsss: Insufficient memory for protocol.");
      else {Ok = (mode == 'c' ? prot->Init_Client(erp, parms)
                              : prot->Init_Server(erp, parms));

            if (!Ok) {prot->Delete(); prot = 0;}
           }

// All done
//
   return (XrdSecProtocol *)prot;
}
}
