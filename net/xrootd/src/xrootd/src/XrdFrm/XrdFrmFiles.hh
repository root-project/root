#ifndef __FRMFILES__HH
#define __FRMFILES__HH
/******************************************************************************/
/*                                                                            */
/*                        X r d F r m F i l e s . h h                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

#include <string.h>
#include <sys/types.h>

#include "XrdOss/XrdOssPath.hh"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucNSWalk.hh"

/******************************************************************************/
/*                   C l a s s   X r d F r m F i l e s e t                    */
/******************************************************************************/
  
class  XrdFrmFileset
{
public:
friend class XrdFrmFiles;

// These are the basic set of files related to the base file. Two other file
// suffixes are ignore for fileset purposes (".anew" and ".stage"). To take
// ownership of the entry, simply set the pointer to zero after saving it.
//
XrdOucNSWalk::NSEnt *File[XrdOssPath::sfxNum];

                     XrdFrmFileset(XrdFrmFileset *sP=0) : Next(sP)
                                   {memset(File, 0, sizeof(File));}
                    ~XrdFrmFileset() {int i;
                                      for (i = 0; i < XrdOssPath::sfxNum; i++)
                                          if(File[i]) delete File[i];
                                     }
private:
XrdFrmFileset *Next;
};

/******************************************************************************/
/*                     C l a s s   X r d F r m F i l e s                      */
/******************************************************************************/
  
class  XrdFrmFiles
{
public:

XrdFrmFileset *Get(int &rc, int noBase=0);

static const int Recursive = 0x0001;

            XrdFrmFiles(const char *dname, int opts=Recursive);

           ~XrdFrmFiles() {}

private:
int  Process(XrdOucNSWalk::NSEnt *nP);

XrdOucHash<XrdFrmFileset>fsTab;

XrdOucNSWalk             nsObj;
XrdFrmFileset           *fsList;
};
#endif
