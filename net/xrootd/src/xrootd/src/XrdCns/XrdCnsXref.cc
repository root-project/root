/******************************************************************************/
/*                                                                            */
/*                         X r d C n s X r e f . c c                          */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

const char *XrdCnsXrefCVSID = "$Id$";

#include "XrdCns/XrdCnsXref.hh"

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/

char *XrdCnsXref::xIndex = (char *)"0123456789:;<=>?@"
                                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
                                   "abcdefghijklmnopqrstuvwxyz{|}~";

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdCnsXref::XrdCnsXref(const char *dflt, int MTProt)
{
   memset(yTable, 0, sizeof(yTable));
   isMT = MTProt;
   if (dflt) Default(dflt);
   availIdx = 1;
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdCnsXref::~XrdCnsXref()
{
   int i;

   for (i = 0; i < yTSize; i++) if (yTable[i]) free(yTable[i]);
}

/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
char XrdCnsXref::Add(const char *kval, char idx)
{
   XrdSysMutexHelper xHelp(isMT ? &xMutex : 0);
   char *oldx, *xKey = strdup(kval);
   int i, j;

// If a character was specified, try to use it.
//
        if (idx) i = c2i(idx);
   else if ((oldx = xTable.Find(xKey))) {free(xKey); return *oldx;}
   else if ((i = availI()) < 0)         {free(xKey); return 0;}

// Try to add the new entry
//
   if (!(oldx = xTable.Add(xKey, xIndex+i, 0, Hash_keep))) yTable[i] = xKey;
      else if (*oldx != idx)
              {if ((j = c2i(*oldx)) >= 0)
                  {if (yTable[j])
                      {xTable.Del(yTable[j]); free(yTable[j]); yTable[j] = 0;}
                   xTable.Rep(xKey, xIndex+i, 0, Hash_keep);
                   yTable[j] = xKey;
                  }
              } else free(xKey);

// Return the assigned character index
//
   return xIndex[i];
}

/******************************************************************************/
/*                               D e f a u l t                                */
/******************************************************************************/
  
char XrdCnsXref::Default(const char *kval)
{
   return (kval ? Add(kval, xIndex[0]) : xIndex[0]);
}

/******************************************************************************/
/*                                   K e y                                    */
/******************************************************************************/
  
char *XrdCnsXref::Key(char idc)
{
   XrdSysMutexHelper xHelp(isMT ? &xMutex : 0);
   int i = c2i(idc);

   if (i >= 0 && yTable[i]) return yTable[i];
   return yTable[0];
}

/******************************************************************************/
/*                                  F i n d                                   */
/******************************************************************************/
  
char XrdCnsXref::Find(const char *kval)
{
   XrdSysMutexHelper xHelp(isMT ? &xMutex : 0);
   char *xdat;

   return ((xdat = xTable.Find(kval)) ? *xdat : char(0));
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/* Private:                       A v a i l I                                 */
/******************************************************************************/
  
int XrdCnsXref::availI()
{
   int idn;

   for (idn = availIdx; idn < yTSize; idn++)
       if (!yTable[idn]) {availIdx = idn+1; return idn;}

   return -1;
}

/******************************************************************************/
/* Private:                          c 2 i                                    */
/******************************************************************************/
  
int XrdCnsXref::c2i(char xCode)
{
   int n = static_cast<int>(xCode);

   return (n <= 127 ? xCode - '0' : -1);
}
