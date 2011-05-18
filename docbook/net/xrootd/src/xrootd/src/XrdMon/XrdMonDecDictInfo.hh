/*****************************************************************************/
/*                                                                           */
/*                           XrdMonDecDictInfo.hh                            */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONDECDICTINFO_HH
#define XRDMONDECDICTINFO_HH

#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonTypes.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include <string>
#include <strings.h>
#include <string.h>

using std::ostream;
using std::string;

class XrdMonDecTraceInfo;

class XrdMonDecDictInfo {
public:

    enum TYPE { OPEN, CLOSE };
    
    XrdMonDecDictInfo();
    XrdMonDecDictInfo(dictid_t id,
                      dictid_t uniqueId,
                      const char* theString,
                      int len,
                      senderid_t senderId);
    XrdMonDecDictInfo(const char* buf, int& pos);
    
    dictid_t xrdId() const { return _myXrdId; }
    dictid_t uniqueId() const { return _myUniqueId; }
    senderid_t senderId() const { return _senderId; }
    
    bool isClosed() const   { return 0 != _close; }
    int stringSize() const;
    const char* convert2string() const;
    const char* writeRT2BufferOpenFile(kXR_int64 fSize) const;
    const char* writeRT2BufferCloseFile() const;
    void writeSelf2buf(char* buf, int& pos) const;
    
    void openFile(kXR_int32 t, kXR_int64 fSize);
    void closeFile(kXR_int64 bytesR, kXR_int64 bytesW, kXR_int32 t);
    bool addTrace(const XrdMonDecTraceInfo& trace);

    int mySize() const;
    
private:
    int doOne(const char* s, char* buf, int len, char delim) {
        int x = 0;
        while ( x < len && *(s+x) != delim ) {
            ++x;
        }
        if ( x >= len ) {
            return -1;
        }
        
        memcpy(buf, s, x);
        *(buf+x) = '\0';
        return x;
    }

    dictid_t _myXrdId;     // the one that come inside packet, not unique
    dictid_t _myUniqueId; // unique (across all dictIds for given xrd server)

    string  _user;
    kXR_int16 _pid;
    string  _cHost;  // client host
    string  _path;
    senderid_t _senderId;
    kXR_int32  _open;
    kXR_int32  _close;

    kXR_int64 _fSize;
    kXR_int64 _noRBytes;  // no bytes read
    kXR_int64 _noWBytes;  // no bytes writen
    
    friend ostream& operator<<(ostream& o, 
                               const XrdMonDecDictInfo& m);
};

#endif /* XRDMONDECDICTINFO_HH */
