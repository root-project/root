/*****************************************************************************/
/*                                                                           */
/*                           XrdMonDecStageInfo.cc                           */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonException.hh"
#include "XrdMon/XrdMonDecStageInfo.hh"
#include "XrdMon/XrdMonErrors.hh"
#include "XrdMon/XrdMonSenderInfo.hh"
#include "XrdMon/XrdMonUtils.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include <netinet/in.h>
#include <stdio.h>
using std::cout;
using std::cerr;
using std::endl;

XrdMonDecStageInfo::XrdMonDecStageInfo()
    : _myXrdId(0),
      _myUniqueId(0),
      _user("InvalidUser"),
      _pid(-1),
      _cHost("InvalidHost"),
      _path("InvalidPath"),
      _senderId(INVALID_SENDER_ID),
      _bytes(0),
      _seconds(0)
{}

XrdMonDecStageInfo::XrdMonDecStageInfo(dictid_t id,
                                       dictid_t uniqueId,
                                       const char* s, 
                                       int len, 
                                       senderid_t senderId)
    : _myXrdId(id),
      _myUniqueId(uniqueId),
      _senderId(senderId),
      _bytes(0),
      _seconds(0)
{
    // uncomment all 3 below if you want to print the string
    char*b = new char [len+1];strncpy(b, s, len);b[len] = '\0';
    cout << "Decoding string in StageInfo " << b << endl;
    delete [] b;
    
    // decode theString, format: <user>.<pid>:<fd>@<host>\n<path>\n&sz=bytes&tm=secs&tod=secs
    int x1 = 0, x2 = 0;
    char* buf = new char [len+1];

    x1 = doOne(s, buf, len, '.');
    if (x1 == -1 ) {
        delete [] buf;
        string es("Cannot find "); es+='.'; es+=" in "; es+=s;
        throw XrdMonException(ERR_INVDICTSTRING, es);
    }
    _user = buf;
    
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
    x1 = doOne(s+x2, buf, len-x2, '\n');
    if ( x1 == -1 ) {
        delete [] buf;
        string es("Cannot find "); es+='\n'; es+=" in "; es+=s;
        throw XrdMonException(ERR_INVDICTSTRING, es);
    }
    _cHost = buf;

    x2 += x1+1;
    x1 = doOne(s+x2, buf, len-x2, '\n');
    if ( x1 == -1 ) {
        delete [] buf;
        string es("Cannot find "); es+='\n'; es+=" in "; es+=s;
        throw XrdMonException(ERR_INVDICTSTRING, es);
    }
    _path = buf;

    x2 += 1; // skip \n
    
    do {
        x2 += x1+1; // skip &
        x1 = doOne(s+x2, buf, len-x2, '=');
        if ( x1 == -1 ) break;
        if ( 0 == strcmp(buf, "sz") ) {
            x2 += x1+1;
            x1 = doOne(s+x2, buf, len-x2, '&');
            _bytes = ( x1 == -1 ? atoi(s+x2) : atoi(buf) );
        } else if ( 0 == strcmp(buf, "tm") ) {
            x2 += x1+1;
            x1 = doOne(s+x2, buf, len-x2, '&');
            _seconds = ( x1 == -1 ? atoi(s+x2) : atoi(buf) );
        } else if ( 0 == strcmp(buf, "tod") ) {
            x2 += x1+1;
            x1 = doOne(s+x2, buf, len-x2, '&');
            _tod = ( x1 == -1 ? atoi(s+x2) : atoi(buf) );
        } else {
            delete [] buf;
            string es("Cannot find "); es+='\n'; es+=" in "; es+=s;
            throw XrdMonException(ERR_INVDICTSTRING, es);
        }
    } while (1);

    delete [] buf;
}

// this goes to history data ascii file
const char*
XrdMonDecStageInfo::convert2string() const
{
    static char buf[1024];
    char tBuf[24];
    timestamp2string(_tod, tBuf, GMT);

    sprintf(buf, "%s\t%i\t%s\t%s\t%s\t%d\t%d\t%s\n", 
            _user.c_str(), _pid, _cHost.c_str(), _path.c_str(),
            tBuf, _bytes, _seconds, 
            XrdMonSenderInfo::id2Host(_senderId));

    return buf;
}

// this goes to real time log file
const char*
XrdMonDecStageInfo::writeRT2Buffer() const
{
    static char buf[1024];
    static char tBuf[24];

    timestamp2string(_tod, tBuf, GMT);
    sprintf(buf, "s\t%i\t%s\t%i\t%s\t%s\t%s\t%d\t%d\t%s\n", 
            _myUniqueId, _user.c_str(), _pid, _cHost.c_str(), _path.c_str(),
            tBuf, _bytes, _seconds, XrdMonSenderInfo::id2Host(_senderId));

    return buf;
}


// this is for debugging
ostream& 
operator<<(ostream& o, const XrdMonDecStageInfo& m)
{
   o << ' ' << m._myXrdId
     << ' ' << m._myUniqueId
     << ' ' << m._user
     << ' ' << m._pid
     << ' ' << m._cHost
     << ' ' << m._path
     << ' ' << m._senderId 
            << " (" << XrdMonSenderInfo::id2Host(m._senderId)<<")"
     << ' ' << timestamp2string(m._tod)
     << ' ' << m._bytes
     << ' ' << m._seconds;

    return o;
}
