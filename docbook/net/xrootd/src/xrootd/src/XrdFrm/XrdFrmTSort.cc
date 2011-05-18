/******************************************************************************/
/*                                                                            */
/*                        X r d F r m T S o r t . c c                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdFrmTSortCVSID = "$Id$";

#include "XrdFrm/XrdFrmFiles.hh"
#include "XrdFrm/XrdFrmTSort.hh"
//#include "iostream.h"

/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
int XrdFrmTSort::Add(XrdFrmFileset *fsp)
{
   XrdOucNSWalk::NSEnt *nsp = fsp->baseFile();
   int n;

// Make sure we can actuall add this entry
//
   if (baseT < nsp->Stat.st_atime) return 0;

// Get the relative time
//
   fsp->Age = static_cast<int>(baseT - nsp->Stat.st_atime);

// Insert into the table
//
   n = fsp->Age/dVal;
   if (n > 63) n = 63;
   fsp->Next = FSTab[0][n];
   FSTab[0][n] = fsp;
   if (n > DYent) DYent = n;
//cerr <<"Add " <<std::hex <<fsp->Age <<' ' <<std::dec <<0 <<',' <<n <<' ' <<fsp->basePath() <<endl;
   numEnt++;
   return 1;
}

/******************************************************************************/
/* Private:                          B i n                                    */
/******************************************************************************/

int XrdFrmTSort::Bin(XrdFrmFileset *fsp, int j, int Shift)
{
   XrdFrmFileset *fsq;
   int k, n = 0;

   while((fsq = fsp))
        {fsp = fsp->Next;
         k = (fsq->Age >> Shift) & tMask;
         if (k > n) n = k;
         if (Shift || !sortSZ) fsq->Next = FSTab[j][k];
            else fsq = Insert(fsq, FSTab[j][k]);
         FSTab[j][k] = fsq;
//cerr <<"Bin " <<std::hex <<fsq->Age <<' ' <<std::dec <<j <<',' <<k <<' ' <<fsq->basePath() <<endl;
        }
   return n;
}

/******************************************************************************/
/* Private:                       I n s e r t                                 */
/******************************************************************************/
  
XrdFrmFileset *XrdFrmTSort::Insert(XrdFrmFileset *newP, XrdFrmFileset *oldP)
{
   XrdFrmFileset *prvP = 0, *nowP = oldP;
   off_t newSize = newP->baseFile()->Stat.st_size;

// Find insertion point of new element (decreasing size order)
//
   while(nowP && newSize < nowP->baseFile()->Stat.st_size)
        {prvP = nowP; nowP = nowP->Next;}

// Perform insertion
//
   if (prvP) {prvP->Next = newP; newP->Next = nowP;}
      else    newP->Next = nowP;

// Return correct head of list
//
   return (prvP ? oldP : newP);
}

/******************************************************************************/
/*                                O l d e s t                                 */
/******************************************************************************/

XrdFrmFileset *XrdFrmTSort::Oldest()
{
   XrdFrmFileset *fsp = 0;

// Work backwards on the list, resorting as needed
//
   do {while(SCent >= 0)
            {if ((fsp = FSTab[3][SCent]))
                {if (!( FSTab[3][SCent] = fsp->Next)) SCent--;
                 numEnt--;
//cerr <<"Oldest " <<fsp->Age <<' ' <<fsp->basePath() <<endl;
                 return fsp;
                } else SCent--;
            }
//cerr <<"SC=" <<SCent <<" MN=" <<MNent <<" HR=" <<HRent <<" DY=" <<DYent <<endl;
       fsp = 0;
       while(MNent >= 0 && !fsp) fsp = FSTab[2][MNent--];
       if (fsp) {FSTab[2][MNent+1]=0; SCent = Bin(fsp, 3, SCshift); continue;}
       while(HRent >= 0 && !fsp) fsp = FSTab[1][HRent--];
       if (fsp) {FSTab[1][HRent+1]=0; MNent = Bin(fsp, 2, MNshift); continue;}
       while(DYent >= 0 && !fsp) fsp = FSTab[0][DYent--];
       if (fsp) {FSTab[0][DYent+1]=0; HRent = Bin(fsp, 1, HRshift); continue;}
      } while(numEnt);
   return 0;
}

/******************************************************************************/
/*                                 P u r g e                                  */
/******************************************************************************/
  
void XrdFrmTSort::Purge()
{
   XrdFrmFileset *fsp, *csp;
   int i, j, aBeg[4] = {DYent, HRent, MNent, SCent};

   for (i = 0; i < 4; i++)
       for (j = aBeg[i]; j >= 0; j--)
           {if ((fsp = FSTab[i][j]))
               while((csp = fsp)) {fsp = fsp->Next; delete csp;}
           }
   Reset();
}

/******************************************************************************/
/* Private:                        R e s e t                                  */
/*                                                                            */
/******************************************************************************/

void XrdFrmTSort::Reset()
{

// Clear the base table and set base time
//
   memset(FSTab, 0, sizeof(FSTab));
   DYent = HRent = MNent = SCent = -1;
   baseT = time(0);
   numEnt = 0;
}
