#ifndef __XRDCKSXATTR_HH__
#define __XRDCKSXATTR_HH__
/******************************************************************************/
/*                                                                            */
/*                        X r d C k s X A t t r . h h                         */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <sys/types.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <sys/types.h>

#include "XrdCks/XrdCksData.hh"
#include "XrdSys/XrdSysPlatform.hh"

/* XrdCksXAttr encapsulates the extended attributes needed to save a checksum.
*/
  
class XrdCksXAttr
{
public:

XrdCksData Cks;         // Check sum information

/* postGet() will put fmTime and csTime in host byte order (see preSet()).
*/
       int             postGet(int Result)
                              {if (Result > 0)
                                  {Cks.fmTime = ntohll(Cks.fmTime);
                                   Cks.csTime = ntohl (Cks.csTime);
                                  }
                               return Result;
                              }

/* preSet() will put fmTime and csTime in network byte order to allow the
            attribute to be copied to different architectures and still work.
*/
       XrdCksXAttr    *preSet(XrdCksXAttr &tmp)
                             {memcpy(&tmp.Cks, &Cks, sizeof(Cks));
                              tmp.Cks.fmTime = htonll(Cks.fmTime);
                              tmp.Cks.csTime = htonl (Cks.csTime);
                              return &tmp;
                             }

/* Name() returns the extended attribute name for this object.
*/
       const char     *Name() {if (!(*VarName))    //01234567
                                  {strcpy(VarName,  "XrdCks.");
                                   strcpy(VarName+7, Cks.Name);
                                  }
                               return VarName;
                              }

/* sizeGet() and sizeSet() return the actual size of the object is used.
*/
       int             sizeGet() {return sizeof(Cks);}
       int             sizeSet() {return sizeof(Cks);}

       XrdCksXAttr() {*VarName = 0;}
      ~XrdCksXAttr() {}

private:

char VarName[XrdCksData::NameSize+8];
};
#endif
