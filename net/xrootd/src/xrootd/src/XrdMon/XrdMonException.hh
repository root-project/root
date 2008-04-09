/*****************************************************************************/
/*                                                                           */
/*                            XrdMonException.hh                             */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONEXCEPTION_HH
#define XRDMONEXCEPTION_HH

#include "XrdMon/XrdMonTypes.hh"
#include <map>
#include <string>
#include <vector>

using std::map;
using std::string;
using std::vector;


typedef int err_t;

class XrdMonException {
public:
    XrdMonException(err_t err);
    XrdMonException(err_t err,
                      const string& s);
    XrdMonException(err_t err,
                      const char* s);

    err_t err() const { return _err; }
    const string msg() const { return _msg; }
    void printIt() const;
    void printItOnce() const;

private:
    struct ErrInfo {
        vector<string> msgs;
        int count;
    };
    
    static map<err_t, ErrInfo> _oneTime;
    
    err_t  _err;
    string _msg;

};

#endif /* XRDMONEXCEPTION_HH */
