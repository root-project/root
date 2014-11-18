#ifndef __XRDCMSRRDATA_H__
#define __XRDCMSRRDATA_H__
/******************************************************************************/
/*                                                                            */
/*                       X r d C m s R R D a t a . h h                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include <stdlib.h>

#include "XProtocol/YProtocol.hh"

class XrdCmsRLData
{
public:

       char          *theAuth;
       char          *theSID;
       char          *thePaths;
       int            totLen;

//     XrdCmsRLData() {}  Lack of constructor makes this a POD type
//    ~XrdCmsRLData() {}  Lack of destructor  makes this a POD type
};


class XrdCmsRRData
{
public:
XrdCms::CmsRRHdr       Request;     // all
        char          *Path;        // all -prepcan
        char          *Opaque;      // all -prepcan
        char          *Path2;       // mv
        char          *Opaque2;     // mv
        char          *Avoid;       // locate, select
        char          *Reqid;       // prepadd, prepcan
        char          *Notify;      // prepadd
        char          *Prty;        // prepadd
        char          *Mode;        // chmod, mkdir, mkpath, prepadd
        char          *Ident;       // all
        unsigned int   Opts;        // locate, select
                 int   PathLen;     // locate, prepadd, select (inc null byte)
        unsigned int   dskFree;     // avail, load
union  {unsigned int   dskUtil;     // avail
                 int   waitVal;
       };
        char          *Buff;        // Buffer underlying the pointers
        int            Blen;        // Length of buffer
        int            Dlen;        // Length of data in the buffer
        int            Routing;     // Routing options

enum ArgName
{    Arg_Null=0,   Arg_AToken,    Arg_Avoid,     Arg_Datlen,
     Arg_Ident,    Arg_Info,      Arg_Mode,      Arg_Notify,
     Arg_Opaque2,  Arg_Opaque,    Arg_Opts,      Arg_Path,
     Arg_Path2,    Arg_Port,      Arg_Prty,      Arg_Reqid,
     Arg_dskFree,  Arg_dskUtil,   Arg_theLoad,   Arg_SID,
     Arg_dskTot,   Arg_dskMinf,

     Arg_Count     // Always the last item which equals the number of elements
};

static XrdCmsRRData *Objectify(XrdCmsRRData *op=0);

       int           getBuff(size_t bsz);

//      XrdCmsRRData() {}  Lack of constructor makes this a POD type
//     ~XrdCmsRRData() {}  Lack of destructor  makes this a POD type

XrdCmsRRData *Next;   // POD types canot have private members so virtual private
};
#endif
