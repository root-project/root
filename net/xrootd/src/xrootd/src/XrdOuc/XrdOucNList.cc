/******************************************************************************/
/*                                                                            */
/*                        X r d O u c N L i s t . c c                         */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//        $Id$

const char *XrdOucNListCVSID = "$Id$";

#include <string.h>
#include "XrdOuc/XrdOucNList.hh"
  
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdOucNList::XrdOucNList(const char *name, int nval)
{
   char *ast;

// Do the default assignments
//
   nameL = strdup(name);
   next  = 0;
   flags = nval;

// First find the asterisk, if any in the name
//
   if ((ast = index(nameL, '*')))
      {namelenL = ast - nameL;
       *ast  = 0;
       nameR = ast+1;
       namelenR = strlen(nameR);
      } else {
       namelenL = strlen(nameL);
       namelenR = -1;
      }
}
 
/******************************************************************************/
/*                                N a m e O K                                 */
/******************************************************************************/
  
int XrdOucNList::NameOK(const char *pd, const int pl)
{

// Check if exact match wanted
//
   if (namelenR < 0) return !strcmp(pd, nameL);

// Make sure the prefix matches
//
   if (namelenL && namelenL <= pl && strncmp(pd,nameL,namelenL))
      return 0;

// Make sure suffix matches
//
   if (!namelenR)     return 1;
   if (namelenR > pl) return 0;
   return !strcmp((pd + pl - namelenR), nameR);
}

/******************************************************************************/
/*                               R e p l a c e                                */
/******************************************************************************/
  
void XrdOucNList_Anchor::Replace(const char *name, int nval)
{
   XrdOucNList *xp = new XrdOucNList(name, nval);

   Replace(xp);
}


void XrdOucNList_Anchor::Replace(XrdOucNList *xp)
{
   XrdOucNList *np, *pp = 0;

// Lock ourselves
//
   Lock();
   np = next;

// Find the matching item or the place to insert the item
//
   while(np && np->namelenL >= xp->namelenL)
        {if (np->namelenL == xp->namelenL
         &&  np->namelenR == xp->namelenR
         && (np->nameL && xp->nameL && !strcmp(np->nameL, xp->nameL))
         && (np->nameR && xp->nameR && !strcmp(np->nameR, xp->nameR)))
            {np->Set(xp->flags);
             UnLock();
             delete xp;
             return;
            }
          pp = np; np = np->next;
         }

// Must insert a new item
//
   if (pp) {xp->next = np; pp->next = xp;}
      else {xp->next = next;   next = xp;}

// All done
//
   UnLock();
}
