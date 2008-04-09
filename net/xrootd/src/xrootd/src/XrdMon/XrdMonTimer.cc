/*****************************************************************************/
/*                                                                           */
/*                              XrdMonTimer.cc                               */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#include "XrdMon/XrdMonTimer.hh"
#include <stdio.h>
using std::cout;
using std::endl;

void
XrdMonTimer::printAll() const
{
    cout << "Counter:\n";
    printOne(_tbeg,    "     Beg:     ");
    cout << "     Elapsed: " << _elapsed << endl;
    cout << "---------" << endl;
}

void
XrdMonTimer::printOne(const timeval& t,
                const char* prefix) const
{
    if ( 0 != prefix ) {
        cout << prefix;
    }

    char buf[128];
    // sprintf(buf, "%lu.%03lu", t.tv_sec, t.tv_usec/100);
    sprintf(buf, "%.3f", convert2Double(t));

    cout << buf << endl;
}

void
XrdMonTimer::printElapsed(const char* str)
{
    cout << str << ": " << calcElapsed() << endl;
}
 
