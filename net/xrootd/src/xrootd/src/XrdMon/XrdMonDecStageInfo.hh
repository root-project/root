/*****************************************************************************/
/*                                                                           */
/*                           XrdMonDecStageInfo.hh                           */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONDECSTAGEINFO_HH
#define XRDMONDECSTAGEINFO_HH

#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonTypes.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include <string>
#include <strings.h>
#include <string.h>

using std::ostream;
using std::string;

class XrdMonDecStageInfo {
public:

    XrdMonDecStageInfo();
    XrdMonDecStageInfo(dictid_t id,
                       dictid_t uniqueId,
                       const char* theString,
                       int len,
                       senderid_t senderId);
    XrdMonDecStageInfo(const char* buf, int& pos);
    
    dictid_t xrdId() const { return _myXrdId; }
    dictid_t uniqueId() const { return _myUniqueId; }
    senderid_t senderId() const { return _senderId; }
    
    const char* convert2string() const;
    const char* writeRT2Buffer() const;

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
    dictid_t _myUniqueId;  // unique (across all dictIds for given xrd server)

    string     _user;
    kXR_int16  _pid;
    string     _cHost;  // client host
    string     _path;
    kXR_int32  _tod;
    senderid_t _senderId;
    kXR_int32  _bytes;
    kXR_int32  _seconds;


    friend ostream& operator<<(ostream& o, 
                               const XrdMonDecStageInfo& m);
};

#endif /* XRDMONDECSTAGEINFO_HH */
