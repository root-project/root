/******************************************************************************/
/*                                                                            */
/*                        X r d S e c s s s I D . c c                         */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//       $Id$

const char *XrdSecsssIDCVSID = "$Id$";

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <grp.h>
#include <pwd.h>
#include <sys/types.h>

#include "XrdSecsss/XrdSecsssID.hh"
#include "XrdSecsss/XrdSecsssRR.hh"

#include "XrdOuc/XrdOucPup.hh"
#include "XrdSys/XrdSysHeaders.hh"

/******************************************************************************/
/*                               D e f i n e s                                */
/******************************************************************************/
  
#define XRDSECSSSID "XrdSecsssID"

XrdSysMutex         XrdSecsssID::InitMutex;

XrdSecsssID::sssID *XrdSecsssID::defaultID = 0;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdSecsssID::XrdSecsssID(authType aType, XrdSecEntity *idP)
{
   static char buff[64];
   union {unsigned long val; XrdSecsssID *myP;} p2i;
   int nID = 1;

// Check if we have initialized already. If so, indicate warning
//
   InitMutex.Lock();
   if (getenv(XRDSECSSSID))
      {InitMutex.UnLock();
       cerr <<"SecsssID: Already instantiated; new instance ineffective!" <<endl;
       return;
      }

// Verify the authType
//
   switch(aType)
         {case idMutual:  idP = 0; break;
          case idDynamic:          break;
          case idStatic:  nID = 0; break;
          case idStaticM: nID = 0; break;
          default:        idP = 0; aType = idLogin; break;
         }
   myAuth = aType;

// Check if we need to generate default of fixed identity
//

   if (!idP || !(defaultID = genID(idP)))
      {if (nID) defaultID = 0;
          else {XrdSecEntity myID;
                struct passwd *pEnt;
                struct group  *pGrp;
                if (!(pEnt = getpwuid(geteuid()))) myID.name = (char *)"nobody";
                   else myID.name = pEnt->pw_name;
                if ((pGrp = getgrgid(getegid())))  myID.grps = (char *)"nogroup";
                   else myID.grps = pGrp->gr_name;
                defaultID = genID(&myID);
               }
      }


// Establish a pointer to this object so that the shared library can use it
// We only do this once!
//
   p2i.myP = this;
   sprintf(buff, XRDSECSSSID"=%lx", p2i.val);
   putenv(buff);

// All done with initialization
//
   InitMutex.UnLock();
}

/******************************************************************************/
/*                                  F i n d                                   */
/******************************************************************************/
  
int XrdSecsssID::Find(const char *lid, char *Buff, int Blen)
{
   sssID *fP;
   int rc;

// Lock the hash table and find the entry
//
   myMutex.Lock();
   if (!(fP = Registry.Find(lid)) && !(fP = defaultID)) fP = genID(lid);
   if (!fP || fP->iLen > Blen) {myMutex.UnLock(); return 0;}

// Return the data
//
   memcpy(Buff, fP->iData, fP->iLen);
   rc = fP->iLen;
   myMutex.UnLock();
   return rc;
}

/******************************************************************************/
/*                                 g e t I D                                  */
/******************************************************************************/

const char *XrdSecsssID::getID(int &idLen)
{
   if (defaultID) {idLen = defaultID->iLen; return defaultID->iData;}
   idLen = 0;
   return (const char *)0;
}
  
/******************************************************************************/
/*                                g e t O b j                                 */
/******************************************************************************/
  
XrdSecsssID::authType XrdSecsssID::getObj(XrdSecsssID **objP)
{
   char *eP, *xP;
   authType aType = idLogin;
   union {long long llval; long lval; XrdSecsssID *idP;} i2p;

// Prevent changes
//
   InitMutex.Lock();

// Convert to pointer
//
   if ((eP = getenv(XRDSECSSSID)) && *eP)
      {if (sizeof(XrdSecsssID *) > 4) i2p.llval = strtoll(eP, &xP, 16);
          else                        i2p.lval  = strtol (eP, &xP, 16);
       if (*xP)                       i2p.idP   = 0;
          else aType = i2p.idP->myAuth;
      }

// Return result
//
   *objP = i2p.idP;
   InitMutex.UnLock();
   return aType;
}

/******************************************************************************/
/*                              R e g i s t e r                               */
/******************************************************************************/

int XrdSecsssID::Register(const char *lid, XrdSecEntity *eP, int doRep)
{
   sssID *idP;
   int    rc;

// Check if we are simply deleting an entry
//
   if (!eP)
      {myMutex.Lock(); Registry.Del(lid); myMutex.UnLock(); return 1;}

// Generate an ID and add it to registry
//
   if (!(idP = genID(eP))) return 0;
   myMutex.Lock(); 
   rc = (Registry.Add(lid, idP, (doRep ? Hash_replace : Hash_default)) ? 0:1);
   myMutex.UnLock();
   return rc;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                 g e n I D                                  */
/******************************************************************************/
  
XrdSecsssID::sssID *XrdSecsssID::genID(const char *lid)
{
   XrdSecEntity   myID;
   struct passwd *pEnt;
   struct group  *pGrp;

// Construct entity corresponding to the loginid
//
   if (!(pEnt = getpwnam(lid))
   ||  !(pGrp = getgrgid(pEnt->pw_gid))) myID.grps = (char *)"nogroup";
      else myID.grps = pGrp->gr_name;
   myID.name = (char *)lid;

// Just return the sssID
//
   return genID(&myID);
}

/******************************************************************************/

XrdSecsssID::sssID *XrdSecsssID::genID(XrdSecEntity *eP)
{
   sssID *idP;
   char *bP;
   int tLen;

// Calculate the length needed for the entity (4 bytes overhead for each item)
//
   tLen = (eP->name         ? strlen(eP->name)         + 4 : 0)
        + (eP->vorg         ? strlen(eP->vorg)         + 4 : 0)
        + (eP->role         ? strlen(eP->role)         + 4 : 0)
        + (eP->grps         ? strlen(eP->grps)         + 4 : 0)
        + (eP->endorsements ? strlen(eP->endorsements) + 4 : 0);

// If no identity information, return failure otherwise allocate a struct
//
   if (!tLen || !(idP = (sssID *)malloc(tLen + sizeof(sssID)))) return 0;

// Now stick each entry into the iData field
//
   bP = idP->iData;
   if (eP->name)
      {*bP++ = XrdSecsssRR_Data::theName; XrdOucPup::Pack(&bP,eP->name);}
   if (eP->vorg)
      {*bP++ = XrdSecsssRR_Data::theVorg; XrdOucPup::Pack(&bP,eP->vorg);}
   if (eP->role)
      {*bP++ = XrdSecsssRR_Data::theRole; XrdOucPup::Pack(&bP,eP->role);}
   if (eP->grps)
      {*bP++ = XrdSecsssRR_Data::theGrps; XrdOucPup::Pack(&bP,eP->grps);}
   if (eP->endorsements)
      {*bP++ = XrdSecsssRR_Data::theEndo; XrdOucPup::Pack(&bP,eP->endorsements);}
   idP->iLen = bP - (idP->iData);

// All done
//
   return idP;
}
