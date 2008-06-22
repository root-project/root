/*****************************************************************************/
/*                                                                           */
/*                              XrdMonUtils.cc                               */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonException.hh"
#include "XrdMon/XrdMonErrors.hh"
#include "XrdMon/XrdMonUtils.hh"
#include "XrdSys/XrdSysHeaders.hh"

#include <errno.h>
#include <string.h>     /* strerror */
#include <stdio.h>
#include <sys/stat.h>   /* mkdir  */
#include <sys/time.h>
#include <sys/types.h>  /* mkdir  */
#include <unistd.h>     /* access */

using std::cout;
using std::endl;


string
generateTimestamp()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    const time_t sec = tv.tv_sec;
    struct tm *t = localtime(&sec);

    char buf[24];
    sprintf(buf, "%02d%02d%02d_%02d:%02d:%02d.%03d",
            t->tm_year+1900,
            t->tm_mon+1,
            t->tm_mday,
            t->tm_hour,
            t->tm_min,
            t->tm_sec,
            (int)tv.tv_usec/1000);
    return string(buf);
}

string
timestamp2string(kXR_int32 timestamp, bool gmt)
{
    char s[24];
    timestamp2string(timestamp, s, gmt);
    return string(s);
}

void
timestamp2string(kXR_int32 timestamp, char s[24], bool gmt)
{
    if ( 0 == timestamp ) {
        strcpy(s, "0000-00-00 00:00:00");
        return;
    }

    time_t tt = timestamp;
    struct tm *t = (gmt ? gmtime(&tt) : localtime(&tt));
    
    // Format: YYYY-MM-DD HH:MM:SS
    sprintf(s, "%4d-%02d-%02d %02d:%02d:%02d",
            t->tm_year+1900,
            t->tm_mon+1,
            t->tm_mday,
            t->tm_hour,
            t->tm_min,
            t->tm_sec);
}

// converts string host:port to a pair<host, port>
pair<string, string>
breakHostPort(const string& hp)
{
    int colonPos = hp.rfind(':', hp.size());
    if ( colonPos == -1 ) {
        string se("No : in "); se += hp;
        throw XrdMonException(ERR_INVALIDADDR, se);
    }
    string host(hp, 0, colonPos);
    string port(hp, colonPos+1, hp.size()-colonPos-1);
    return pair<string, string>(host, port);
}

void
mkdirIfNecessary(const char* dir)
{
    if ( 0 == access(dir, F_OK) ) {
        return;
    }

    // find non-existing directory in the path, 
    // then create all missing directories
    string org(dir);
    int size = org.size();
    int pos = 0;
    vector<string> dirs2create;
    if ( org[size-1] == '/' ) {
        org = string(org, 0, size-1); // remove '/' at the end
    }
    dirs2create.push_back(org);
    do {
        pos = org.rfind('/', size-1);
        if ( pos == -1 ) {
            break;
        }
        org = string(dir, pos);
        if ( 0 == access(org.c_str(), F_OK) ) {
            break;
        }
        dirs2create.push_back(org);
    } while ( pos > 0 );

    size = dirs2create.size();
    for ( int i=size-1 ; i>=0 ; --i ) {
        const char*d = dirs2create[i].c_str();
        if ( 0 != mkdir(d, 0x3FD) ) {
            char buf[256];
            sprintf(buf, "Failed to mkdir %s. Error: '%s'", 
                    dir, strerror (errno));
            throw XrdMonException(ERR_NODIR, buf);
        }
        cout << "mkdir " << d << " OK" << endl;
    }
}

