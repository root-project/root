/*****************************************************************************/
/*                                                                           */
/*                             XrdMonCtrAdmin.cc                             */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonCtrAdmin.hh"
#include "XrdMon/XrdMonErrors.hh"
#include "XrdMon/XrdMonException.hh"

#include "XrdSys/XrdSysHeaders.hh"

#include <netinet/in.h> /* ntohs */
#include <string.h>

using std::cout;
using std::endl;

void
XrdMonCtrAdmin::doIt(kXR_int16 command, kXR_int16 arg)
{
    switch (command) {
        case c_shutdown: {
            throw XrdMonException(SIG_SHUTDOWNNOW);
        }
        default: {
            cout << "Invalid admin command: " << command << " ignored" << endl;
            throw XrdMonException(ERR_UNKNOWN);
        }
    }
}

void
XrdMonCtrAdmin::decodeAdminPacket(const char* packet,
                                  kXR_int16& command,
                                  kXR_int16& arg)
{
    kXR_int16 x16;
    int8_t offset = HDRLEN;
    memcpy(&x16, packet+offset, sizeof(kXR_int16));
    offset += sizeof(kXR_int16);
    command = ntohs(x16);

    memcpy(&x16, packet+offset, sizeof(kXR_int16));
    offset += sizeof(kXR_int16);
    arg = ntohs(x16);
}
