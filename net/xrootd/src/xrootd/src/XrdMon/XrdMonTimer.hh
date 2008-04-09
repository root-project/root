/*****************************************************************************/
/*                                                                           */
/*                              XrdMonTimer.hh                               */
/*                                                                           */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University */
/*                            All Rights Reserved                            */
/*       Produced by Jacek Becla for Stanford University under contract      */
/*              DE-AC02-76SF00515 with the Department of Energy              */
/*****************************************************************************/

// $Id$

#ifndef XRDMONTIMER_HH
#define XRDMONTIMER_HH

#include "sys/time.h"

class XrdMonTimer {
public:
    XrdMonTimer() { reset(); }    

    inline void reset();                  // resets the counter

    // working with elapsed time
    inline int  start();                  // starts the timer
    inline double stop();                 // stops the timer, returns elapsed time
    inline double getElapsed() const;     // returns elapsed time

    void printElapsed(const char* str);
    // for debugging only
    void printAll() const;

private:
    // modifiers
    inline void resetTBeg();
    inline void resetTElapsed();

    inline double calcElapsed();       // calculates, sets, and returns total elapsed time

    // selectors
    inline int timerOn() const;
    inline int isOn(const struct timeval& t) const;

    inline double calcDif(const struct timeval& start, 
                          const struct timeval& stop) const;

    void printOne(const timeval& t, const char* prefix=0) const;

    double convert2Double(const timeval& t) const;

private:
    struct timeval _tbeg;       // most recent "start"
    double _elapsed;   // elapsed time between all "starts" and "stops",
       // excluding most recent "start" which has no corresponding "stop"
};

#include "XrdMonTimer.icc"


#endif /* XRDMONTIMER_HH */

