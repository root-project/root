/******************************************************************************/
/*                                                                            */
/*                         X r d C m s N a s h . h h                          */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdCmsNashCVSID = "$Id$";

#include <stdlib.h>

#include "XrdCms/XrdCmsNash.hh"

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdCmsNash::XrdCmsNash(int psize, int csize)
{
     prevtablesize = psize;
     nashtablesize = csize;
     Threshold     = (csize * LoadMax) / 100;
     nashnum       = 0;
     nashtable     = (XrdCmsKeyItem **)
                     malloc( (size_t)(csize*sizeof(XrdCmsKeyItem *)) );
     memset((void *)nashtable, 0, (size_t)(csize*sizeof(XrdCmsKeyItem *)));
}

/******************************************************************************/
/* public                            A d d                                    */
/******************************************************************************/
  
XrdCmsKeyItem *XrdCmsNash::Add(XrdCmsKey &Key)
{
   XrdCmsKeyItem *hip;
   unsigned int kent;

// Allocate the entry
//
   if (!(hip = XrdCmsKeyItem::Alloc(Key.TOD))) return (XrdCmsKeyItem *)0;

// Check if we should expand the table
//
   if (++nashnum > Threshold) Expand();

// Fill out the key data
//
   if (!Key.Hash) Key.setHash();
   hip->Key = Key;

// Add the entry to the table
//
   kent = Key.Hash % nashtablesize;
   hip->Next = nashtable[kent];
   nashtable[kent] = hip;
   return hip;
}
  
/******************************************************************************/
/* private                        E x p a n d                                 */
/******************************************************************************/
  
void XrdCmsNash::Expand()
{
   int newsize, newent, i;
   size_t memlen;
   XrdCmsKeyItem **newtab, *nip, *nextnip;

// Compute new size for table using a fibonacci series
//
   newsize = prevtablesize + nashtablesize;

// Allocate the new table
//
   memlen = (size_t)(newsize*sizeof(XrdCmsKeyItem *));
   if (!(newtab = (XrdCmsKeyItem **) malloc(memlen))) return;
   memset((void *)newtab, 0, memlen);

// Redistribute all of the current items
//
   for (i = 0; i < nashtablesize; i++)
       {nip = nashtable[i];
        while(nip)
             {nextnip = nip->Next;
              newent  = nip->Key.Hash % newsize;
              nip->Next = newtab[newent];
              newtab[newent] = nip;
              nip = nextnip;
             }
       }

// Free the old table and plug in the new table
//
   free((void *)nashtable);
   nashtable     = newtab;
   prevtablesize = nashtablesize;
   nashtablesize = newsize;

// Compute new expansion threshold
//
   Threshold = static_cast<int>((static_cast<long long>(newsize)*LoadMax)/100);
}

/******************************************************************************/
/* public                           F i n d                                   */
/******************************************************************************/
  
XrdCmsKeyItem *XrdCmsNash::Find(XrdCmsKey &Key)
{
  XrdCmsKeyItem *nip;
  unsigned int kent;

// Check if we already have a hash value and get one if not
//
   if (!Key.Hash) Key.setHash();

// Compute position of the hash table entry
//
   kent = Key.Hash%nashtablesize;

// Find the entry
//
   nip = nashtable[kent];
   while(nip && nip->Key != Key) nip = nip->Next;
   return nip;
}

/******************************************************************************/
/* public                        R e c y c l e                                */
/******************************************************************************/
  
// The item must have been previously unload which will place the original
// hash value in Loc.HashSave. Yes, not very OO but very fast.
//
int XrdCmsNash::Recycle(XrdCmsKeyItem *rip)
{
   XrdCmsKeyItem *nip, *pip = 0;
   unsigned int kent;

// Compute position of the hash table entry
//
   kent = rip->Loc.HashSave%nashtablesize;

// Find the entry
//
   nip = nashtable[kent];
   while(nip && nip != rip) {pip = nip; nip = nip->Next;}

// Remove and recycle if found
//
   if (nip)
      {if (pip) pip->Next = nip->Next;
          else nashtable[kent] = nip->Next;
          rip->Recycle();
          nashnum--;
      }
   return nip != 0;
}
