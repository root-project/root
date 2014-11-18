#ifndef __XRDCKSMANAGER_HH__
#define __XRDCKSMANAGER_HH__
/******************************************************************************/
/*                                                                            */
/*                      X r d C k s M a n a g e r . h h                       */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdCks/XrdCks.hh"
#include "XrdCks/XrdCksData.hh"

/* This class defines the checksum management interface. It may also be used
   as the base class for a plugin. This allows you to replace selected methods
   which may be needed for handling certain filesystems (see protected ones).
*/

class XrdCksCalc;
class XrdSysError;
  
class XrdCksManager : public XrdCks
{
public:
virtual int         Calc( const char *Pfn, XrdCksData &Cks, int doSet=1);

virtual int         Config(const char *Token, char *Line);

virtual int         Del(  const char *Pfn, XrdCksData &Cks);

virtual int         Get(  const char *Pfn, XrdCksData &Cks);

virtual int         Init(const char *ConfigFN, const char *AddCalc=0);

virtual char       *List(const char *Pfn, char *Buff, int Blen, char Sep=' ');

virtual const char *Name(int seqNum=0);

virtual int         Size( const char  *Name=0);

virtual int         Set(  const char *Pfn, XrdCksData &Cks, int myTime=0);

virtual int         Ver(  const char *Pfn, XrdCksData &Cks);

                    XrdCksManager(XrdSysError *erP, int iosz=0);
virtual            ~XrdCksManager();

protected:

/* Calc()     returns 0 if the checksum was successfully calculated using the
              supplied CksObj and places the file's modification time in MTime.
              Otherwise, it returns -errno. The default implementation uses
              open(), fstat(), mmap(), and unmap() to calculate the results.
*/
virtual int         Calc(const char *Pfn, time_t &MTime, XrdCksCalc *CksObj);

/* ModTime()  returns 0 and places file's modification time in MTime. Otherwise,
              it return -errno. The default implementation uses stat().
*/
virtual int         ModTime(const char *Pfn, time_t &MTime);

private:

struct csInfo
      {char          Name[XrdCksData::NameSize];
       XrdCksCalc   *Obj;
       char         *Path;
       char         *Parms;
       XrdSysPlugin *Plugin;
       int           Len;
                     csInfo() : Obj(0), Path(0), Parms(0), Plugin(0), Len(0)
                                {memset(Name, 0, sizeof(Name));}
      };

int     Config(const char *cFN, csInfo &Info);
csInfo *Find(const char *Name);

static const int csMax = 4;
csInfo           csTab[csMax];
int              csLast;
int              segSize;
};
#endif
