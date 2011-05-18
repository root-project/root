#ifndef __CMS_MANLIST__H
#define __CMS_MANLIST__H
/******************************************************************************/
/*                                                                            */
/*                      X r d C m s M a n L i s t . h h                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdSys/XrdSysPthread.hh"

//         $Id$

class XrdCmsManRef;

class XrdCmsManList
{
public:

// Add() adds an alternate manager to the list of managers (duplicates not added)
//
void     Add(unsigned int refp, char *manp, int manport, int lvl);

// Del() removes all entries added under refp
//
void     Del(unsigned int refp);

// haveAlts() returns true if alternates exist, false otherwise
//
int      haveAlts() {return allMans != 0;}

// Next() returns the next manager in the list and its level or 0 if none are left.
//        The next call to Next() will return the first manager in the list.
//
int      Next(int &port, char *buff, int bsz);

         XrdCmsManList() {allMans = nextMan = 0;}
        ~XrdCmsManList();

private:

XrdSysMutex   mlMutex;
XrdCmsManRef *nextMan;
XrdCmsManRef *allMans;
};

namespace XrdCms
{
extern    XrdCmsManList myMans;
}
#endif
