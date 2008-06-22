/*****************************************************************************/
/*                                                                           */
/*                          XrdMonSndTraceEntry.hh                           */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONSNDTRACEENTRY_HH
#define XRDMONSNDTRACEENTRY_HH

#include "XrdMon/XrdMonTypes.hh"
#include "XrdSys/XrdSysHeaders.hh"

using std::ostream;

class XrdMonSndTraceEntry {
public:
    XrdMonSndTraceEntry(kXR_int64 offset,
                        kXR_int32  length,
                        kXR_int32 id);

    kXR_int64 offset() const  { return _offset; }
    kXR_int32 length() const  { return _length; }
    kXR_int32 id()     const  { return _id;     }
    
private:
    kXR_int64 _offset;
    kXR_int32 _length;
    kXR_int32 _id;

    friend ostream& operator<<(ostream& o, 
                               const XrdMonSndTraceEntry& m);
};

#endif /* XRDMONSNDTRACEENTRY_HH */
