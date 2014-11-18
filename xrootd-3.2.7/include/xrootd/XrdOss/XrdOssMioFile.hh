#ifndef __OSSMIOFILE_H__
#define __OSSMIOFILE_H__
/******************************************************************************/
/*                                                                            */
/*                      X r d O s s M i o F i l e . h h                       */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

#include <time.h>
#include <sys/types.h>
  
class XrdOssMioFile
{
public:
friend class XrdOssMio;

off_t Export(void **Addr) {*Addr = Base; return Size;}

       XrdOssMioFile(char *hname)
                    {strcpy(HashName, hname); 
                     inUse = 1; Next = 0; Size = 0;
                    }
      ~XrdOssMioFile();

private:

XrdOssMioFile *Next;
dev_t          Dev;
ino_t          Ino;
int            Status;
int            inUse;
void          *Base;
off_t          Size;
char           HashName[64];
};
#endif
