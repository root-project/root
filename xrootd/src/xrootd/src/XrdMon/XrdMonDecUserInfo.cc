
/*****************************************************************************/
/*                                                                           */
/*                           XrdMonDecUserInfo.cc                            */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonException.hh"
#include "XrdMon/XrdMonDecUserInfo.hh"
#include "XrdMon/XrdMonErrors.hh"
#include "XrdMon/XrdMonSenderInfo.hh"
#include "XrdMon/XrdMonUtils.hh"
#include "XrdMon/XrdMonDecTraceInfo.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include <netinet/in.h>
#include <stdio.h>
#include <sys/time.h>

using std::cout;
using std::cerr;
using std::endl;

XrdMonDecUserInfo::XrdMonDecUserInfo()
    : _myXrdId(0),
      _myUniqueId(0),
      _user("InvalidUser"),
      _pid(-1),
      _cHost("InvalidHost"),
      _senderId(INVALID_SENDER_ID),
      _sec(0),
      _dTime(0)
{}

XrdMonDecUserInfo::XrdMonDecUserInfo(dictid_t id,
                                     dictid_t uniqueId,
                                     const char* s, 
                                     int len,
                                     senderid_t senderId)
    : _myXrdId(id),
      _myUniqueId(uniqueId),
      _senderId(senderId),
      _sec(0),
      _dTime(0)
{
    // uncomment all 3 below if you want to print the string
    //char*b = new char [len+1];strncpy(b, s, len);b[len] = '\0';
    //cout << "Decoding string in UserInfo " << b << endl;
    //delete [] b;
    
    // decode theString, format: <user>.<pid>:<fd>@<host>
    int x1 = 0, x2 = 0;
    char* buf = new char [len+1];

    x1 = doOne(s, buf, len, '.');
    if (x1 == -1 ) {
        delete [] buf;
        string es("Cannot find "); es+='.'; es+=" in "; es+=s;
        throw XrdMonException(ERR_INVDICTSTRING, es);
    }
    _user = (x1 != 0 ? buf : "unknown");

    x2 += x1+1;
    x1 = doOne(s+x2, buf, len-x2, ':');
    if ( x1 == -1 ) {
        delete [] buf;
        string es("Cannot find "); es+=':'; es+=" in "; es+=s;
        throw XrdMonException(ERR_INVDICTSTRING, es);
    }
    _pid = atoi(buf);

    x2 += x1+1;
    x1 = doOne(s+x2, buf, len-x2, '@');
    if ( x1 == -1 ) {
        delete [] buf;
        string es("Cannot find "); es+='@'; es+=" in "; es+=s;
        throw XrdMonException(ERR_INVDICTSTRING, es);
    }
    //kXR_int16 fd = atoi(buf);

    x2 += x1+1;
    memcpy(buf, s+x2, len-x2);
    *(buf+len-x2) = '\0';
    _cHost = buf;

    delete [] buf;
}

void
XrdMonDecUserInfo::setDisconnectInfo(kXR_int32 sec,
                                     kXR_int32 timestamp)
{
    _sec   = sec;
    _dTime = timestamp;
}

// this goes to history data ascii file
const char*
XrdMonDecUserInfo::convert2string() const
{
    static char buf[512];
    char tBuf[24];
    timestamp2string(_dTime, tBuf, GMT);
    
    sprintf(buf, "%s\t%i\t%s\t%i\t%s\t%s\n", 
            _user.c_str(), _pid, _cHost.c_str(), 
            _sec, tBuf, XrdMonSenderInfo::id2Host(_senderId));

    return buf;
}

int
XrdMonDecUserInfo::mySize()
{
    return sizeof(*this) + _user.size() + _cHost.size();
}

// this goes to real time log file
const char*
XrdMonDecUserInfo::writeRT2Buffer(TYPE t) const
{
    static char buf[512];
    
    if ( t == CONNECT ) {
        struct timeval tv;
        gettimeofday(&tv, 0);
        static char timeNow[24];
        timestamp2string(tv.tv_sec, timeNow, GMT);
        sprintf(buf, "u\t%i\t%s\t%i\t%s\t%s\t%s\n", 
                _myUniqueId, _user.c_str(), _pid, _cHost.c_str(), 
                XrdMonSenderInfo::id2Host(_senderId), timeNow);
    } else {
        static char b[24];
        timestamp2string(_dTime, b, GMT);
        sprintf(buf, "d\t%i\t%i\t%s\n", _myUniqueId, _sec, b);
    }
    return buf;
}

// this is for debugging
ostream& 
operator<<(ostream& o, const XrdMonDecUserInfo& m)
{
   o << ' ' << m._myXrdId
     << ' ' << m._myUniqueId
     << ' ' << m._user
     << ' ' << m._pid
     << ' ' << m._cHost
     << ' ' << m._senderId 
     << " (" << XrdMonSenderInfo::id2Host(m._senderId) << ")";
      
    return o;
}
