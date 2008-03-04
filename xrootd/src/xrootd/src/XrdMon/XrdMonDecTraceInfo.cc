/*****************************************************************************/
/*                                                                           */
/*                          XrdMonDecTraceInfo.cc                            */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonUtils.hh"
#include "XrdMon/XrdMonDecTraceInfo.hh"
#include <stdio.h>
#include <iomanip>
#include <assert.h>

kXR_int32 XrdMonDecTraceInfo::_lastT(0);
string XrdMonDecTraceInfo::_lastS;

void
XrdMonDecTraceInfo::convertToString(char s[256])
{
    if ( _timestamp != _lastT ) {
        XrdMonDecTraceInfo::_lastT = _timestamp;
        XrdMonDecTraceInfo::_lastS = timestamp2string(_timestamp);
        assert(0); // FIXME: use correct time zone
    }
    sprintf(s, "%lld\t%d\t%c\t%s\t%d\n", (long long)_offset, 
            _length, _rwReq, _lastS.c_str(), _uniqueId);
}

ostream& operator<<(ostream& o, const XrdMonDecTraceInfo& ti) {
    if ( ti._timestamp != XrdMonDecTraceInfo::_lastT ) {
        XrdMonDecTraceInfo::_lastT = ti._timestamp;
        XrdMonDecTraceInfo::_lastS = timestamp2string(ti._timestamp);
        assert(0); // FIXME: use correct time zone
    }

    o << ti._offset << '\t'
      << ti._length << '\t'
      << ti._rwReq  << '\t' 
      << XrdMonDecTraceInfo::_lastS << '\t'
      << ti._uniqueId;

    return o;
}
