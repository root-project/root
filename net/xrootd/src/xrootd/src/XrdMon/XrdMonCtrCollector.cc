/*****************************************************************************/
/*                                                                           */
/*                          XrdMonCtrCollector.cc                            */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonTimer.hh"
#include "XrdMon/XrdMonCtrBuffer.hh"
#include "XrdMon/XrdMonCtrPacket.hh"
#include <sys/socket.h>
#include <assert.h>

//#define DEBUG
#define PRINT_SPEED

// for DEBUG/PRINT_SPEED only
#include "XrdMon/XrdMonCtrDebug.hh"
#include "XrdMon/XrdMonSenderInfo.hh"

#include <iomanip>
#include "XrdSys/XrdSysHeaders.hh"
using std::cout;

namespace XrdMonCtrCollector {
    int port = DEFAULT_PORT;
}

void
printSpeed()
{
    static kXR_int64 noP = 0;
    static XrdMonTimer t;
    if ( 0 == noP ) {
        t.start();
    }
    ++noP;
    const kXR_int64 EVERY = 1001;
    if ( noP % EVERY == EVERY-1) {
        double elapsed = t.stop();
        cout << noP << " packets received in " << elapsed 
             << " sec (" << EVERY/elapsed << " Hz)" << endl;
        t.reset(); t.start();
    }
}

extern "C" void* receivePackets(void*)
{
    struct sockaddr_in sAddress;

    int socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);    
    assert( -1 != socket_ );

    memset((char *) &sAddress, 0, sizeof(sAddress));
    sAddress.sin_family = AF_INET;
    sAddress.sin_port = htons(XrdMonCtrCollector::port);
    sAddress.sin_addr.s_addr = htonl(INADDR_ANY);

    if ( -1 == bind(socket_, 
                    (struct sockaddr*)&sAddress, 
                    sizeof(sAddress)) ) {
        cerr << "Failed to bind, likely port " 
             << XrdMonCtrCollector::port << " in use" << endl;
        ::abort();
    }

    XrdMonCtrBuffer* pb = XrdMonCtrBuffer::instance();
    cout << "Ready to receive data..." << endl;
    while ( 1 ) {
        XrdMonCtrPacket* packet = new XrdMonCtrPacket(MAXPACKETSIZE);
        socklen_t slen = sizeof(packet->sender);
        if ( -1 == recvfrom(socket_, 
                            packet->buf, 
                            MAXPACKETSIZE, 
                            0, 
                            (sockaddr* )(&(packet->sender)), 
                            &slen) ) {
            cerr << "Failed to receive data" << endl;
            ::abort();
        }
#ifdef DEBUG
        static kXR_int32 packetNo = 0;
        ++packetNo;
        {
            XrdMonCtrXrdSysMutexHelper mh; mh.Lock(&XrdMonCtrDebug::_mutex);
            cout << "Received packet no " 
                 << setw(5) << packetNo << " from " 
                 << XrdMonSenderInfo::hostPort(packet->sender) << endl;
        }
#endif

        pb->push_back(packet);

#ifdef PRINT_SPEED
        printSpeed();
#endif
    }
    return 0;
}
