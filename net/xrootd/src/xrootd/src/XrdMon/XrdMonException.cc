/*****************************************************************************/
/*                                                                           */
/*                             XrdMonException.cc                            */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonException.hh"
#include "XrdSys/XrdSysHeaders.hh"
using std::cerr;
using std::cout;
using std::endl;

map<err_t, XrdMonException::ErrInfo> XrdMonException::_oneTime;

XrdMonException::XrdMonException(err_t err)
    : _err(err)
{}

XrdMonException::XrdMonException(err_t err,
                                     const string& s)
    : _err(err),
      _msg(s)
{}

XrdMonException::XrdMonException(err_t err,
                                     const char* s)
    : _err(err),
      _msg(s)
{}

void
XrdMonException::printIt() const
{
    cerr << "Caught exception " << err() 
         << " \"" << msg() << "\"" << endl;
}

void
XrdMonException::printItOnce() const
{
    map<err_t, ErrInfo >::iterator itr = _oneTime.find(err());
    if ( itr != _oneTime.end() ) {
        vector<string> v = itr->second.msgs;
        int i, size = v.size();
        for (i=0 ; i<size ; ++i ) {
            if ( v[i] == msg() ) {
                ++itr->second.count;
                cout << "this exception (" << err() << ") already "
                     << "printed " << itr->second.count << " times" << endl;
                return; // this exception was already thrown
            }
        }
        v.push_back(msg());
        itr->second.count = 1;
    } else {
        ErrInfo ei;
        ei.msgs.push_back(msg());
        ei.count = 1;
        _oneTime[err()] = ei;
    }
    
    printIt();
}

