/*****************************************************************************/
/*                                                                           */
/*                          XrdMonSndTransmitter.hh                          */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONSNDTRANSMITTER_HH
#define XRDMONSNDTRANSMITTER_HH

#include "XrdMon/XrdMonSndPacket.hh"
#include <netinet/in.h>
class XrdMonSndTraceCache;

#include <arpa/inet.h>

class XrdMonSndTransmitter {
public:
    XrdMonSndTransmitter();
    
    int initialize(const char* receiverHost, 
                   kXR_int16 receiverPort);
    int operator()(const XrdMonSndPacket& packet);
    void shutdown();

private:
    bool messThingsUp(const XrdMonSndPacket& packet, int packetNo);
    
private:
    int _socket;
    struct sockaddr_in _sAddress;
};

#endif /* XRDMONSNDTRANSMITTER_HH */
