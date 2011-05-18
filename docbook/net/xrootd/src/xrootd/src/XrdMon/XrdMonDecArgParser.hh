/*****************************************************************************/
/*                                                                           */
/*                           XrdMonDecArgParser.cc                           */
/*                                                                           */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONDECARGPARSER_HH
#define XRDMONDECARGPARSER_HH

#include "XProtocol/XPtypes.hh"
#include <string>
using std::string;
#include <sys/time.h>

class XrdMonDecArgParser {
public:
    static void parseArguments(int argc, char* argv[]);
    static bool   _forceCloseOnly;
    static kXR_int32 _upToTime;
    static kXR_int32 _ignoreIfBefore;
    static string _fPath;
    static string _hostPort; // of the sender - xrd host
    static string _baseDir;
    static bool   _saveTraces;
    static int    _maxTraceLogSize;
    
    // these below used for dumpPackets app only 
   static kXR_int64 _offset2Dump;
    
private:
    static void convertTime(int nr, char* argv[]);
    static string parsePath();
};

#endif /* XRDMONDECARGPARSER_HH */
