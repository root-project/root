/******************************************************************************/
/*                                                                            */
/*                      X r d C m s M a n L i s t . c c                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

// Original Version: 1.4 2007/01/22 03:40:24 abh

const char *XrdCmsManListCVSID = "$Id$";

#include <ctype.h>
#include <unistd.h>

#include "XrdNet/XrdNetDNS.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdCms/XrdCmsManList.hh"

using namespace XrdCms;

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

       XrdCmsManList  XrdCms::myMans;

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
  
class XrdCmsManRef
{
public:

XrdCmsManRef *Next;
char         *Manager;
unsigned int  ManRef;
int           ManPort;
int           ManLvl;

              XrdCmsManRef(unsigned int ref, char *name, int port, int lvl)
                          : Next(0), Manager(name), ManRef(ref),
                            ManPort(port), ManLvl(lvl) {};

             ~XrdCmsManRef() {if (Manager) free(Manager);}
};

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdCmsManList::~XrdCmsManList()
{
   XrdCmsManRef *prp, *mrp = allMans;

   while(mrp) {prp = mrp; mrp = mrp->Next; delete prp;}
}


/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
void XrdCmsManList::Add(unsigned int ref, char *manp, int manport, int lvl)
{
   XrdCmsManRef *prp = 0, *mrp;
   struct sockaddr InetAddr;
   char *cp, *ipname;
   int port;

// Find the colon in the host name
//
   if (!(cp = index(manp, int(':')))) port = manport;
      else {if (!(port=atoi(cp+1)) || port > 0xffff) port=manport;
            *cp = '\0';
           }

// Check if we need to translate ip address to name
//
   if (!isdigit((int)*manp) || !XrdNetDNS::getHostAddr(manp, InetAddr)
   || !(ipname =  XrdNetDNS::getHostName(InetAddr))) ipname = strdup(manp);
   if (cp) *cp = ':';

// Start up
//
   mlMutex.Lock();
   mrp = allMans;

// Chck if this is a duplicate
//
   while(mrp)
        {if (!strcmp(mrp->Manager, ipname)) 
            {mlMutex.UnLock(); free(ipname); return;}
         if (mrp->Next)
            {if (mrp->Next->ManLvl > lvl) prp = mrp;}
            else if (!prp) prp = mrp;
         mrp = mrp->Next;
        }

// Create new entry
//
   mrp = new XrdCmsManRef(ref, ipname, port, lvl);
   if (!prp) nextMan = allMans = mrp;
      else {mrp->Next = prp->Next; prp->Next = mrp;
            if (nextMan->ManLvl > lvl) nextMan = mrp;
           }
   mlMutex.UnLock();
}

/******************************************************************************/
/*                                   D e l                                    */
/******************************************************************************/
  
void XrdCmsManList::Del(unsigned int ref)
{
   XrdCmsManRef *nrp, *prp = 0, *mrp;

// Start up
//
   mlMutex.Lock();
   mrp = allMans;

// Delete all ref entries
//
   while(mrp)
        {if (mrp->ManRef == ref)
            {nrp = mrp->Next;
             if (!prp) allMans  = nrp;
                else {prp->Next = nrp;
                      if (mrp == allMans) allMans = nrp;
                     }
             if (mrp == nextMan) nextMan = nrp;
             delete mrp;
             mrp = nrp;
            } else {prp = mrp; mrp = mrp->Next;}
        }

// All done
//
   mlMutex.UnLock();
}

/******************************************************************************/
/*                                  N e x t                                   */
/******************************************************************************/
  
int XrdCmsManList::Next(int &port, char *buff, int bsz)
{
   XrdCmsManRef *np;
   int lvl;

   mlMutex.Lock();
   if (!(np = nextMan)) nextMan = allMans;
      else {strlcpy(buff, np->Manager, bsz);
            port = np->ManPort;
            nextMan = np->Next;
           }
   lvl = (np ? np->ManLvl : 0);
   mlMutex.UnLock();
   return lvl;
}
