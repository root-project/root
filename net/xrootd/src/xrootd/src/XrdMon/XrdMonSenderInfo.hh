/*****************************************************************************/
/*                                                                           */
/*                            XrdMonSenderInfo.hh                            */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONSENDERINFO_HH
#define XRDMONSENDERINFO_HH

#include "XrdMon/XrdMonTypes.hh"
#include <stdio.h>
#include <netinet/in.h>
#include <map>
#include <vector>
using std::map;
using std::vector;

class XrdMonSenderInfo {
public:
    static senderid_t convert2Id(struct sockaddr_in sAddr);
    static hp_t addr2HostPort(struct sockaddr_in sAddr) {
        return id2HostPort(convert2Id(sAddr));
    }
    static hp_t id2HostPort(senderid_t id) {
        if ( id >= _hps.size() ) {
            return hp_t((char*) "Error, invalid offset", 0);
        }
        return _hps[id];
    }
    static const char* id2HostPortStr(senderid_t id) {
        hp_t hp = id2HostPort(id);
        static char x[256];
        sprintf(x, "%s:%d", hp.first, hp.second);
        return x;
    }
    
    static const char* id2Host(senderid_t id) {
        if ( id >= _hps.size() ) {
            return "Error, invalid offset";
        }
        return _hps[id].first;
    }
    static void printSelf();
    
    static void shutdown();
    
private:
    static void registerSender(struct sockaddr_in sAddr);

private:
    // Maps hash of sockaddr_in --> id.
    // Used as offset in various vectors
    static map<kXR_int64, senderid_t> _ids;

    static vector<hp_t> _hps; // {host, port}
};

#endif /* XRDMONSENDERINFO_HH */
