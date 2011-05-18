/*****************************************************************************/
/*                                                                           */
/*                          XrdMonDecTraceInfo.hh                            */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONDECTRACEINFO_HH
#define XRDMONDECTRACEINFO_HH

#include "XrdMon/XrdMonCommon.hh"

#include "XrdSys/XrdSysHeaders.hh"

using std::ostream;

class XrdMonDecTraceInfo {
public:
    XrdMonDecTraceInfo() : _offset(0), _length(0), _timestamp(0), _uniqueId(0) {}
    XrdMonDecTraceInfo(kXR_int64 o, length_t l, char rw, kXR_int32 t)
        : _offset(o), _length(l), _rwReq(rw), _timestamp(t), _uniqueId(o) {}

    kXR_int32 timestamp() const {return _timestamp;} // for verification
    bool isRead() const     { return _rwReq == 'r'; }
    length_t length() const { return _length; }
    void setUniqueId(dictid_t id) { _uniqueId = id; }

    void convertToString(char s[256]);
    
private:
    kXR_int64 _offset;
    length_t _length;
    char     _rwReq;
    kXR_int32   _timestamp;
    dictid_t _uniqueId;
    
    static kXR_int32 _lastT; // cache last, likely to be the same for many 
    static string _lastS; // traces, conversion to string not cheap
    
    friend ostream& operator<<(ostream& o, const XrdMonDecTraceInfo& t);
};

// size: 8 + 4 + 1 + 4 --> aligned to 32

#endif /* XRDMONDECTRACEINFO_HH */
