/******************************************************************************/
/*                                                                            */
/*                   X r d A c c C a p a b i l i t y . c c                    */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdAccCapabilityCVSID = "$Id$";

#include "XrdAcc/XrdAccCapability.hh"

/******************************************************************************/
/*                   E x t e r n a l   R e f e r e n c e s                    */
/******************************************************************************/
  
extern unsigned long XrdOucHashVal2(const char *KeyVal, int KeyLen);

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdAccCapability::XrdAccCapability(char *pathval, XrdAccPrivCaps &privval)
{
   int i;

// Do common initialization
//
   next = 0; ctmp = 0;
   priv.pprivs = privval.pprivs; priv.nprivs = privval.nprivs;
   plen = strlen(pathval); pins = 0; prem = 0;
   pkey = XrdOucHashVal2((const char *)pathval, plen);
   path = strdup(pathval);

// Now set up for @= insertions. We do this eventhough it might never be used
//
   for (i = 0; i < plen; i++)
       if (path[i] == '@' && path[i+1] == '=')
          {pins = i; prem = plen - i - 2; break;}
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
// This is a tricky destructor because deleting any item in the list must
// delete all subsequent items in the list (but only once).
//
XrdAccCapability::~XrdAccCapability()
{
     XrdAccCapability *cp, *np = next;

     if (path) {free(path); path = 0;}

     while(np) {cp = np; np = np->next; cp->next = 0; delete cp;}
     next = 0;
}
/******************************************************************************/
/*                                 P r i v s                                  */
/******************************************************************************/
  
int XrdAccCapability::Privs(      XrdAccPrivCaps &pathpriv,
                            const char           *pathname,
                            const int             pathlen,
                            const unsigned long   pathhash,
                            const char           *pathsub)
{XrdAccCapability *cp=this;
 const int psl = (pathsub ? strlen(pathsub) : 0);

 do {if (cp->ctmp)
       {if (cp->ctmp->Privs(pathpriv,pathname,pathlen,pathhash,pathsub))
           return 1;
       }
        else if (pathlen >= cp->plen)
                if ((!pathsub && !strncmp(pathname, cp->path, cp->plen))
                ||  (pathsub &&  cp->Subcomp(pathname,pathlen,pathsub,psl)))
                   {pathpriv.pprivs = (XrdAccPrivs)(pathpriv.pprivs |
                                                    cp->priv.pprivs);
                    pathpriv.nprivs = (XrdAccPrivs)(pathpriv.nprivs |
                                                    cp->priv.nprivs);
                    return 1;
                   }
    } while ((cp = cp->next));
 return 0;
}

/******************************************************************************/
/*                               S u b c o m p                                */
/******************************************************************************/
  
int XrdAccCapability::Subcomp(const char *pathname, const int pathlen,
                              const char *pathsub,  const int sublen)
{  int ncmp;

// First check if the prefix matches
//
   if (strncmp(pathname, path, pins)) return 0;

// Now, check if the substitution appears in the source path
//
   if (strncmp(&pathname[pins], pathsub, sublen)) return 0;

// Now check if we can match the tail
//
   ncmp = pins + sublen;
   if ((pathlen - ncmp) < prem) return 0;

// Return the results of matching the tail (prem should never be 0, but hey)
//
   if (prem) return !strncmp(&path[pins+2], &pathname[ncmp], prem);
   return 1;
}

/******************************************************************************/
/*                         X r d A c c C a p N a m e                          */
/******************************************************************************/
/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdAccCapName::~XrdAccCapName()
{
   XrdAccCapName *cp, *np = next;

// Free regular storage
//
   next = 0;
   if (CapName) free(CapName);
   if (C_List)  delete C_List;

// Delete list in a non-recursive way
//
   while(np) {cp = np; np = np->next; cp->next = 0; delete cp;}
}
  
/******************************************************************************/
/*                                  F i n d                                   */
/******************************************************************************/
  
XrdAccCapability *XrdAccCapName::Find(const char *name)
{
   int nlen = strlen(name);
   XrdAccCapName *ncp = this;

   do {if (ncp->CNlen <= nlen && !strcmp(ncp->CapName,name+(nlen - ncp->CNlen)))
          return ncp->C_List;
       ncp = ncp->next;
      } while(ncp);
   return (XrdAccCapability *)0;
}
