/*****************************************************************************/
/*                                                                           */
/*                            XrdMonCtrPacket.hh                             */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONCTRPACKET_HH
#define XRDMONCTRPACKET_HH

#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <netinet/in.h>

// struct carries contents of one packet and its sender
struct XrdMonCtrPacket {
    XrdMonCtrPacket(int size) : buf( (char*)malloc(size) ) {
        memset((char*)buf, size, 0);
    }
    ~XrdMonCtrPacket() {
        free(buf);
    }
    char* buf;
    struct sockaddr_in sender;
};

#endif /* XRDMONCTRPACKET_HH */
