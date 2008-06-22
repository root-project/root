/*****************************************************************************/
/*                                                                           */
/*                           XrdMonDecUserInfo.hh                            */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONDECUSERINFO_HH
#define XRDMONDECUSERINFO_HH

#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonTypes.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include <string>
#include <strings.h>
#include <string.h>

using std::ostream;
using std::string;

class XrdMonDecTraceInfo;

class XrdMonDecUserInfo {
public:
    enum TYPE { CONNECT, DISCONNECT };

    XrdMonDecUserInfo();
    XrdMonDecUserInfo(dictid_t id,
                      dictid_t uniqueId,
                      const char* theString,
                      int len,
                      senderid_t senderId);

    inline bool readyToBeStored() const {return _dTime > 0;}
    
    void setDisconnectInfo(kXR_int32 sec, kXR_int32 timestamp);
    
    dictid_t xrdId() const { return _myXrdId; }
    dictid_t uniqueId() const { return _myUniqueId; }
    senderid_t senderId() const { return _senderId; }
    
    const char* convert2string() const;
    const char* writeRT2Buffer(TYPE t) const;
    string convert2stringRTDisconnect() const;
    int mySize();
    
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

    dictid_t _myXrdId;    // the one that come inside packet, not unique
    dictid_t _myUniqueId; // unique (across all dictIds for given xrd server)

    string    _user;
    kXR_int16 _pid;   // client process id
    string    _cHost; // client host

    senderid_t _senderId;
    
    kXR_int32 _sec;   // number of seconds that client was connected
    kXR_int32 _dTime; // disconnect time
    
    friend ostream& operator<<(ostream& o, const XrdMonDecUserInfo& m);
};

#endif /* XRDMONDECUSERINFO_HH */
