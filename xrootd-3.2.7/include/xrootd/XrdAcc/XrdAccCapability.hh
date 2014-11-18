#ifndef __ACC_CAPABILITY__
#define __ACC_CAPABILITY__
/******************************************************************************/
/*                                                                            */
/*                   X r d A c c C a p a b i l i t y . h h                    */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "XrdAcc/XrdAccPrivs.hh"

/******************************************************************************/
/*                      X r d A c c C a p a b i l i t y                       */
/******************************************************************************/
  
class XrdAccCapability
{
public:
void                Add(XrdAccCapability *newcap) {next = newcap;}

XrdAccCapability   *Next() {return next;}

// Privs() searches the associated capability for a prefix matching path. If one
// is found, the privileges are or'd into the passed XrdAccPrivCaps struct and
// a 1 is returned. Otherwise, 0 is returned and XrdAccPrivCaps is unchanged.
//
int                 Privs(      XrdAccPrivCaps &pathpriv,
                          const char           *pathname,
                          const int             pathlen,
                          const unsigned long   pathhash,
                          const char           *pathsub=0);

int                 Privs(      XrdAccPrivCaps &pathpriv,
                          const char           *pathname,
                          const int             pathlen,
                          const char           *pathsub=0)
                          {extern unsigned long XrdOucHashVal2(const char *,int);
                           return Privs(pathpriv, pathname, pathlen,
                                  XrdOucHashVal2(pathname,(int)pathlen),pathsub);}

int                 Privs(      XrdAccPrivCaps &pathpriv,
                          const char           *pathname,
                          const char           *pathsub=0)
                          {extern unsigned long XrdOucHashVal2(const char *,int);
                           int pathlen = strlen(pathname);
                           return Privs(pathpriv, pathname, pathlen,
                                  XrdOucHashVal2(pathname, pathlen), pathsub);}

int                 Subcomp(const char *pathname, const int pathlen,
                            const char *pathsub,  const int sublen);

                  XrdAccCapability(char *pathval, XrdAccPrivCaps &privval);

                  XrdAccCapability(XrdAccCapability *taddr)
                        {next = 0; ctmp = taddr;
                         pkey = 0; path = 0; plen = 0; pins = 0; prem = 0;
                        }

                 ~XrdAccCapability();
private:
XrdAccCapability *next;      // -> Next capability
XrdAccCapability *ctmp;      // -> Capability template

/*----------- The below fields are valid when template is zero -----------*/

XrdAccPrivCaps   priv;
unsigned long    pkey;
char            *path;
int              plen;
int              pins;    // index of @=
int              prem;    // remaining length after @=
};

/******************************************************************************/
/*                         X r d A c c C a p N a m e                          */
/******************************************************************************/

class XrdAccCapName
{
public:
void              Add(XrdAccCapName *cnp) {next = cnp;}

XrdAccCapability *Find(const char *name);

       XrdAccCapName(char *name, XrdAccCapability *cap)
                    {next = 0; CapName = strdup(name); CNlen = strlen(name);
                     C_List = cap;
                    }
      ~XrdAccCapName();
private:
XrdAccCapName    *next;
char             *CapName;
int               CNlen;
XrdAccCapability *C_List;
};
#endif
