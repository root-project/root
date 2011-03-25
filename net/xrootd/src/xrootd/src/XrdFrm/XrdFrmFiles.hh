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

#include <string.h>
#include <sys/types.h>

#include "XrdFrm/XrdFrmXAttr.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucNSWalk.hh"
#include "XrdOuc/XrdOucXAttr.hh"

class  XrdOucTList;

/******************************************************************************/
/*                   C l a s s   X r d F r m F i l e s e t                    */
/******************************************************************************/
  
class  XrdFrmFileset
{
public:
friend class XrdFrmFiles;

// The following are the extended attributes describing file characteristics
//
XrdOucXAttr<XrdFrmXAttrCpy> cpyInfo;   // Last copy time
XrdOucXAttr<XrdFrmXAttrPin> pinInfo;   // Pin information

// These are inline function to return most common file information
//
inline XrdOucNSWalk::NSEnt *baseFile() {return File[XrdOssPath::isBase];}
const  char                *basePath() {return Mkfn(baseFile());}
inline XrdOucNSWalk::NSEnt *failFile() {return File[XrdOssPath::isFail];}
const  char                *failPath() {return Mkfn(failFile());}
inline XrdOucNSWalk::NSEnt *lockFile() {return File[XrdOssPath::isLock];}
const  char                *lockPath() {return Mkfn(lockFile());}
inline XrdOucNSWalk::NSEnt * pfnFile() {return File[XrdOssPath::isPfn ];}
const  char                * pfnPath() {return Mkfn(pfnFile());}
inline XrdOucNSWalk::NSEnt * pinFile() {return File[XrdOssPath::isPin ];}
const  char                * pinPath() {return Mkfn(pinFile());}

inline XrdOucNSWalk::NSEnt * xyzFile(XrdOssPath::theSfx sfx) {return File[sfx];}
const  char                * xyzPath(XrdOssPath::theSfx sfx)
                                    {return Mkfn(File[sfx]);}

int                         dirPath(char *dBuff, int dBlen);

static void                 Purge() {BadFiles.Purge();}

int                         Refresh(int isMig=0, int doLock=1);

int                         Screen(int needLF=1);

int                         setCpyTime(int Refresh=0);

                     XrdFrmFileset(XrdFrmFileset *sP=0, XrdOucTList *diP=0);
                    ~XrdFrmFileset();

// The following are public to ease management of this object
//
XrdFrmFileset *Next;
int            Age;

private:
int         chkLock(const char *Path);
const char *Mkfn(XrdOucNSWalk::NSEnt *fP);
void        Remfix(const char *fType, const char *fPath);

// These are the basic set of files related to the base file. Two other file
// suffixes are ignore for fileset purposes (".anew" and ".stage").
//
XrdOucNSWalk::NSEnt *File[XrdOssPath::sfxNum];

XrdOucTList         *dInfo;     // Shared directory information

static XrdOucHash<char> BadFiles;

static const int     dLen = 0;  // Index to directory path length in dInfo
static const int     dRef = 1;  // Index to the reference counter in dInfo
};

/******************************************************************************/
/*                     C l a s s   X r d F r m F i l e s                      */
/******************************************************************************/
  
class  XrdFrmFiles
{
public:

XrdFrmFileset *Get(int &rc, int noBase=0);

static const int Recursive = 0x0001;   // List filesets recursively
static const int CompressD = 0x0002;   // Use shared directory object (not MT)
static const int NoAutoDel = 0x0004;   // Do not automatically delete objects
static const int GetCpyTim = 0x0008;   // Initialize cpyInfo attribute on Get()

            XrdFrmFiles(const char *dname, int opts=Recursive,
                        XrdOucTList *XList=0, XrdOucNSWalk::CallBack *cbP=0);

           ~XrdFrmFiles();

private:
void Complain(const char *dPath);
int  oldFile(XrdOucNSWalk::NSEnt *fP, XrdOucTList *dP, int fType);
int  Process(XrdOucNSWalk::NSEnt *nP, const char *dPath);

XrdOucHash<XrdFrmFileset>fsTab;

XrdOucNSWalk             nsObj;
XrdFrmFileset           *fsList;
XrdOucHash_Options       manMem;
int                      shareD;
int                      getCPT;
};
#endif
