/*****************************************************************************/
/*                                                                           */
/*                            XrdMonCtrAdmin.hh                              */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONCTRADMIN_HH
#define XRDMONCTRADMIN_HH

#include "XrdMon/XrdMonCommon.hh"
#include "XrdMon/XrdMonHeader.hh"

// class responsible for interpreting admin packets
// and taking appropriete action

class XrdMonCtrAdmin {

public:
    static bool isAdminPacket(const XrdMonHeader& header) {
        return header.packetType() == PACKET_TYPE_ADMIN;
    }
    
    static void doIt(kXR_int16 command, kXR_int16 arg);

    static void decodeAdminPacket(const char* packet,
                                  kXR_int16& command, 
                                  kXR_int16& arg);
};

#endif /* XRDMONCTRADMIN_HH */
