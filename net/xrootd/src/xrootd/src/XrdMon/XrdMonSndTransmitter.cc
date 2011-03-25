/*****************************************************************************/
/*                                                                           */
/*                         XrdMonSndTransmitter.cc                           */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonUtils.hh"
#include "XrdMon/XrdMonSndDebug.hh"
#include "XrdMon/XrdMonSndTransmitter.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include <string>
using std::cerr;
using std::cout;
using std::endl;

XrdMonSndTransmitter::XrdMonSndTransmitter()
    : _socket(-1)
{}

int
XrdMonSndTransmitter::initialize(const char* receiverHost, kXR_int16 receiverPort)
{
    _socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    
    if ( -1 == _socket ) {
        cerr << "Failed to initialize socket" << endl;
        return 1;
    }

    memset((char *) &_sAddress, 0, sizeof(_sAddress));
    _sAddress.sin_family = AF_INET;
    _sAddress.sin_port = htons(receiverPort);

    if ( *receiverHost >= '0' && *receiverHost <= '9' ) {
        if (inet_aton(receiverHost, &_sAddress.sin_addr)==0) {
            cerr << "inet_aton() failed, host: " << receiverHost << endl;
            return 2;
        }
    } else {
        struct hostent* h = gethostbyname(receiverHost);
        if ( h == NULL ) {
            cout << "Error, invalid host \"" << receiverHost << "\"" << endl;
            return 3;
        }
        memcpy(&(_sAddress.sin_addr.s_addr), h->h_addr, h->h_length);
    }
    
    return 0; // success
}

int
XrdMonSndTransmitter::operator()(const XrdMonSndPacket& packet)
{
    static int packetNo = 0;
    ++ packetNo;

    //if ( messThingsUp(packet,  packetNo) ) {
    //    return 0;
    //}
    
    if ( XrdMonSndDebug::verbose(XrdMonSndDebug::Sending) ) {
        cout << "Sending packet no " << packetNo << endl;
    }
    int ret = sendto(_socket, packet.data(), packet.size(), 0, 
                     (sockaddr *)& _sAddress, 
                     sizeof(_sAddress) );

    if ( ret == -1 ) {
        cerr << "Failed to send data" << endl;
        return 2;
    }

    //usleep(500);
    
    return 0; // success
}

void
XrdMonSndTransmitter::shutdown()
{
    close(_socket);
}

bool
XrdMonSndTransmitter::messThingsUp(const XrdMonSndPacket& packet, int packetNo)
{
    static XrdMonSndPacket *outOfOrder25 = 0, *outOfOrder270 = 0;
    if ( packetNo == 25 ) {
        outOfOrder25 = new XrdMonSndPacket(packet);
        cout << "JJJ XrdMonSndPacket no " << packetNo << " will be sent later." << endl;
        return true;
    }

    if ( packetNo == 15 || 
         packetNo == 23 ||
         packetNo == 24 ||
         packetNo == 26 ||
         packetNo == 27    ) {
        cout << "JJJ Loosing packet no " << packetNo << endl;
        return true;
    }
    if ( packetNo == 270 ) {
        outOfOrder270 = new XrdMonSndPacket(packet);
        cout << "JJJ XrdMonSndPacket no " << packetNo << " will be sent later." << endl;
        return true;
    }
    if ( packetNo == 280 ) {
        cout << "JJJ Sending packet no 270 now." << endl;
        sendto(_socket, outOfOrder270->data(), outOfOrder270->size(), 0, 
               (sockaddr *)& _sAddress, 
               sizeof(_sAddress) );
        delete outOfOrder270;
        outOfOrder270 = 0;
    }

    if ( packetNo == 55 ) {
        cout << "JJJ Sending packet no 25 now." << endl;
        sendto(_socket, outOfOrder25->data(), outOfOrder25->size(), 0, 
               (sockaddr *)& _sAddress, 
               sizeof(_sAddress) );
        delete outOfOrder25;
        outOfOrder25 = 0;
    }
    return false;
}

