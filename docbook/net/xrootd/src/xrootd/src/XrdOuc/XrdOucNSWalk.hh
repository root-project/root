#ifndef __XRDOUCNSWALK_HH
#define __XRDOUCNSWALK_HH
/******************************************************************************/
/*                                                                            */
/*                       X r d O u c N S W a l k . h h                        */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
  
//         $Id$

class XrdOucTList;
class XrdSysError;

class XrdOucNSWalk
{
public:

struct NSEnt
{
struct NSEnt *Next;   // -> Next entry in the indexed directory
char         *Path;   // Path name to file  if opts & noPath  (Path == File)
char         *File;   // File name component
int           Plen;   // strlen(Path)
struct stat   Stat;   // stat() of Path     if opts & retStat
char         *Link;   // -> Link data       if opts & retLink
int           Lksz;   // Length of link data

enum   Etype {isBad = 0, isDir, isFile, isLink, isMisc};

Etype         Type;   // One of the above. If isLink then Link is invalid!

              NSEnt() : Next(0), Path(0), Plen(0), Link(0), Lksz(0) {}
             ~NSEnt() {if (Path) free(Path);
                       if (Link) free(Link);
                      }
};

// Calling Index() provides the requested directory entries with a return code:
// NSEnts != 0 && rc == 0: Normal ending.
// NSEnts != 0 && rc != 0: Potentially short list as indexing aborted w/ err.
// NSEnts == 0 && rc == 0: End of indexing no more entries can be returned.
// NSEnts == 0 && rc != 0: Abort occured before any entries could be returned.
//
// When opts & skpErrs is true, then rc may be zero even when an error occured.
//
// If opts & Recurse, indexing will traverse the directory tree, one directory
// at a time. For a complete traversal you sould keep calling Index() until
// it returns 0 with rc == 0. When dPath is supplied, a pointer to the base
// directory is returned as well (see noPath).
//
NSEnt        *Index(int &rc, const char **dPath=0);

// The CallBack class is used to intercept empty directories. When set by a
// call to setCallBack(); should an empty directory (i.e., one with no entries
// or only with a lock file) in encountered a call is made to to the isEmpty()
// method. If lkFn is zero, the directory is empty; otherwise, lkFn is the name
// of the singleton lock file. To unset the callback use setCallBack(0);
//
class CallBack
{public:
virtual
void     isEmpty(struct stat *dStat, const char *dPath, const char *lkFn)=0;

         CallBack() {}
virtual ~CallBack() {}
};

void         setCallBack(CallBack *cbP=0) {edCB = cbP;}

// The following are processing options passed to the constructor
//
static const int retDir =  0x0001; // Return directories (implies retStat)
static const int retFile=  0x0002; // Return files       (implies retStat)
static const int retLink=  0x0004; // Return link data   (implies retStat)
static const int retMisc=  0x0008; // Return other types (implies retStat)
static const int retAll =  0x000f; // Return everything

static const int retStat=  0x0010; // return stat() information
static const int retIDLO=  0x0020; // Names returned in decreasing length order
static const int retIILO=  0x0040; // Names returned in increasing length order
static const int Recurse=  0x0080; // Recursive traversal, 1 Level per Index()
static const int noPath =  0x0100; // Do not include the full directory path
static const int skpErrs=  0x8000; // Skip any entry causing an error

             XrdOucNSWalk(XrdSysError *erp,  // Error msg object. If 0->silent
                         const char *dname,  // Initial directory path
                         const char *LKfn=0, // Lock file name (see note below)
                         int opts=retAll,    // Options        (see above)
                         XrdOucTList *xP=0); // 1st Level dir exclude list
            ~XrdOucNSWalk();

// Note: When Lkfn is supplied and it exists in a directory about to be indexed
//       then the file is opened in r/w mode and an exclusive lock is obtained.
//       If either fails, the the directory is not indexed and Index() will
//       return null pointer with rc != 0. Note that the lkfn is not returned
//       as a directory entry if an empty directory call back has been set.

private:
void          addEnt(XrdOucNSWalk::NSEnt *eP);
int           Build();
int           getLink(XrdOucNSWalk::NSEnt *eP);
int           getStat(XrdOucNSWalk::NSEnt *eP, int doLstat=0);
int           getStat();
int           inXList(const char *dName);
int           isSymlink();
int           LockFile();
void          setPath(char *newpath);

XrdSysError  *eDest;
XrdOucTList  *DList;
XrdOucTList  *XList;
struct NSEnt *DEnts;
struct stat   dStat;
CallBack     *edCB;
char          DPath[1032];
char         *File;
char         *LKFn;
int           LKfd;
int           DPfd;
int           Opts;
int           errOK;
int           isEmpty;
};
#endif
