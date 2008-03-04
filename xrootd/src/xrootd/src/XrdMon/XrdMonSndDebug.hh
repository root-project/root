/*****************************************************************************/
/*                                                                           */
/*                            XrdMonSndDebug.hh                              */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef DEBUG_HH
#define DEBUG_HH

// class responsible for handling log/debug/error messages

class XrdMonSndDebug {
public:
    enum Verbosity {
        Quiet     = 0x0000, // No printing

        Generator = 0x0001, // related to generating dummy data
        SCache    = 0x0002, // related to keeping dummy generated data in cache
        Sending   = 0x0008, // related to sending data from xrootd
        SPacket   = 0x0010, // building sender's packet
        All       = 0xFFFF  // Everything
    };

    static void initialize();
    
    inline static bool verbose(Verbosity val) {
        return _verbose & val;
    }

private:
    static Verbosity _verbose;
};

#endif /* DEBUG_HH */
