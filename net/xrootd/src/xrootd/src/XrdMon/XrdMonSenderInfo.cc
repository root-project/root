/*****************************************************************************/
/*                                                                           */
/*                            XrdMonSenderInfo.cc                            */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonException.hh"
#include "XrdMon/XrdMonErrors.hh"
#include "XrdMon/XrdMonSenderInfo.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include <sys/socket.h>
#include <string.h>
#include <netdb.h>
using std::cout;
using std::endl;

map<kXR_int64, senderid_t> XrdMonSenderInfo::_ids;
vector<hp_t>               XrdMonSenderInfo::_hps;

senderid_t
XrdMonSenderInfo::convert2Id(struct sockaddr_in sAddr)
{
    // convert sAddr to myid. If not regiserted yet, 
    // register and also build <hostname>:<port> and register it
    kXR_int64 myhash = (sAddr.sin_addr.s_addr << 16) + sAddr.sin_port;

    map<kXR_int64, senderid_t>::const_iterator itr = _ids.find(myhash);
    if ( itr != _ids.end() ) {
        return itr->second;
    }
    senderid_t id;
    id = _ids[myhash] = _hps.size();
    registerSender(sAddr);
    return id;
}

void
XrdMonSenderInfo::printSelf()
{
    cout << "SenderId mapping: \n";
    int i, s = _hps.size();
    for (i=0 ; i<s ; ++i) {
        cout << "    " << i << " --> " 
             << _hps[i].first << ":" << _hps[i].second << endl;
    }
}

void
XrdMonSenderInfo::shutdown()
{
    _ids.clear();
     
    int i, s = _hps.size();
    for (i=0 ; i<s ; ++i) {
        delete [] _hps[i].first;
    }
    _hps.clear();
}

void
XrdMonSenderInfo::registerSender(struct sockaddr_in sAddr)
{
    char hostName[256];
    char servInfo[256];
    memset((char*)hostName, 0, sizeof(hostName));
    memset((char*)servInfo, 0, sizeof(servInfo));
            
    if ( 0 != getnameinfo((sockaddr*) &sAddr,
                          sizeof(sockaddr),
                          hostName,
                          256,
                          servInfo,
                          256,
                          0) ) {
        throw XrdMonException(ERR_INVALIDADDR, "Cannot resolve ip");
    }

    char* h = new char [strlen(hostName) + 1];
    strcpy(h, hostName);
    senderid_t p = ntohs(sAddr.sin_port);
    
    _hps.push_back(hp_t(h, p));
}

