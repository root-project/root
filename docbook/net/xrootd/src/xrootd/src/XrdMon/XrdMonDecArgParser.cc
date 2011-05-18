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

#include "XrdMonDecArgParser.hh"
#include "XrdMon/XrdMonException.hh"
#include "XrdMon/XrdMonErrors.hh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> /* access */

bool    XrdMonDecArgParser::_forceCloseOnly(false);
kXR_int32  XrdMonDecArgParser::_upToTime(0);
kXR_int32  XrdMonDecArgParser::_ignoreIfBefore(0);
string  XrdMonDecArgParser::_fPath;
kXR_int64 XrdMonDecArgParser::_offset2Dump(0);
string  XrdMonDecArgParser::_hostPort;
string  XrdMonDecArgParser::_baseDir("logs/decoder"); // FIXME configurable
bool    XrdMonDecArgParser::_saveTraces(false);       // FIXME configurable
int     XrdMonDecArgParser::_maxTraceLogSize(2048);   // [MB] FIXME configurable

void
XrdMonDecArgParser::parseArguments(int argc, char* argv[])
{
    if ( argc < 2 ) {
        throw XrdMonException(ERR_INVALIDARG, "Expected input file name");
    }
    bool isActiveFlag = false;
    
    int nr = 1;
    while ( nr < argc ) {
        if ( ! (strcmp(argv[nr],  "-isActive")) ) {
            if ( argc < nr+8 ) {
                throw XrdMonException(ERR_INVALIDARG, 
                        "Expected 7 arguments after -isActive specified");
            }
            if ( 0 == strchr(argv[nr+1], ':') ) {
                string ss("Expected <host>:<port>, found ");
                ss += argv[nr+1];
                throw XrdMonException(ERR_INVALIDARG, ss);
            }
            _hostPort = argv[nr+1];
            convertTime(nr, argv);            
            nr += 7;
            isActiveFlag = true;
        } else if ( ! strcmp(argv[nr], "-forceCloseOnly") ) {
            _forceCloseOnly = true;
        } else if ( ! strcmp(argv[nr], "-ignoreIfBefore") ) {
            if ( argc < nr+2 ) {
                throw XrdMonException(ERR_INVALIDARG, 
                                  "Missing argument after -ignoreIfBefore");
            }
            sscanf(argv[nr+1], "%d", &_ignoreIfBefore);
            ++nr;
        } else if ( ! strcmp(argv[nr], "-offset2Dump") ) {
            if ( argc < nr+2 ) {
                throw XrdMonException(ERR_INVALIDARG, 
                                  "Missing argument after -offset2Dump");
            }
            sscanf(argv[nr+1], "%lld", &_offset2Dump);
            ++nr;
        } else if ( _fPath.empty() ) {
            if ( 0 != access(argv[nr], F_OK) ) {
                string ss("Invalid argument logFilePath ");
                ss += argv[nr];
                ss += " (file does not exist)";
                throw XrdMonException(ERR_INVALIDARG, ss);
            }
            _fPath = argv[nr];
        } else {
            throw XrdMonException(ERR_INVALIDARG, "Invalid argument");
        }
        ++nr;
    }

    if ( isActiveFlag ) {
        const char* fName = _fPath.c_str();
        int len = strlen(fName);
        if ( len < 11 ) {
            throw XrdMonException(ERR_INVALIDARG, 
                  "Expected active.rcv after -isActive");
        }
        if ( 0 != strcmp(fName+len-10, "active.rcv") ) {
            throw XrdMonException(ERR_INVALIDARG, "Expected active.rcv after -isActive");
        }
    } else if ( ! _fPath.empty() ) {
        _hostPort = parsePath();
    }
    if ( _hostPort.empty() ) {
        throw XrdMonException(ERR_INVALIDARG, "HostPort is invalid");
    }
}

void
XrdMonDecArgParser::convertTime(int nr, char* argv[])
{
    struct tm tt;
    int x = atoi(argv[nr+2])-1900;
    if ( x < 100 ) {
        string ss("Invalid arg: year "); ss += argv[nr+2];
        throw XrdMonException(ERR_INVALIDARG, ss);
    }
    tt.tm_year = x;
    
    x = atoi(argv[nr+3])-1;
    if ( x < 0 || x > 11 ) {
        string ss("Invalid arg: month "); ss += argv[nr+3];
        throw XrdMonException(ERR_INVALIDARG, ss);
    }
    tt.tm_mon  = x;
 
    x = atoi(argv[nr+4]);
    if ( x < 1 || x > 31 ) {
        string ss("Invalid arg: day "); ss += argv[nr+4];
        throw XrdMonException(ERR_INVALIDARG, ss);
    }
    tt.tm_mday = x;
 
    x = atoi(argv[nr+5]);
    if ( x < 0 || x > 23 ) {
        string ss("Invalid arg: hour "); ss += argv[nr+5];
        throw XrdMonException(ERR_INVALIDARG, ss);
    }
    tt.tm_hour = x;

    x = atoi(argv[nr+6]);
    if ( x < 0 || x > 59 ) {
        string ss("Invalid arg: min "); ss += argv[nr+6];
        throw XrdMonException(ERR_INVALIDARG, ss);
    }
    tt.tm_min  = x;

    x = atoi(argv[nr+7]);
    if ( x < 0 || x > 59 || (argv[nr+7][0] < '0' || argv[nr+7][0] > '9') ) {
        string ss("Invalid arg: sec "); ss += argv[nr+7];
        throw XrdMonException(ERR_INVALIDARG, ss);
    }
    tt.tm_sec  = x;
    
    _upToTime = mktime(&tt);
}

// returns <host>:<port>
string
XrdMonDecArgParser::parsePath()
{   
    if ( 0 != access(_fPath.c_str(), R_OK) ) {
        string se("Cannot open file "); se+=_fPath; se+=" for reading";
        throw XrdMonException(ERR_INVALIDARG, se);
    }

    // parse input file name, format:
    // <path>/YYYYMMDD_HH:MM:SS.MSC_<sender name>:<sender port>.rcv
        // if filePath contains '/', remove it
    int beg = _fPath.rfind('/', _fPath.size());
    if ( beg == -1 ) {
        beg = 0; // start from the beginning
    } else {
        ++beg;   // skip / and all before
    }
    if ( _fPath[ 8+beg] != '_' || _fPath[11+beg] != ':' ||
         _fPath[14+beg] != ':' || _fPath[17+beg] != '.' ||
         _fPath[21+beg] != '_' || _fPath.size()-beg < 27   ) {
        string se("Incorrect format of log file name, expected: ");
        se += "<path>/YYYYMMDD_HH:MM:SS.MSC_<sender name>:<sender port>.rcv";
        throw XrdMonException(ERR_INVALIDARG, se);
    }
    beg += 22; // skip the timestamp
    int mid = _fPath.find(':', beg);
    if ( mid-beg < 1 ) {
        throw XrdMonException(ERR_INVALIDARG, 
                "Log file does not contain ':' after timestamp");
    }
    int end = _fPath.find(".rcv", beg);
    if ( end < 1 ) {
        throw XrdMonException(ERR_INVALIDARG, "Log file does not end with \".rcv\"");
    }
    string strHost(_fPath, beg, mid-beg);
    string strPort(_fPath, mid+1, end-(mid+1));

    int portNr = 0;
    sscanf(strPort.c_str(), "%d", &portNr);
    if ( portNr < 1 ) {
        char buf[256];
        sprintf(buf, "Decoded port number is invalid: %i", portNr);
        throw XrdMonException(ERR_INVALIDARG, buf);
    }
    strHost += ':';
    strHost += portNr;

    return strHost;
}


