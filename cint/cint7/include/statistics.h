/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*************************************************************************
* statistics.h
*
*  Standard deviation calculation library
*
*************************************************************************/
#ifndef G__STATISTICS_H
#define G__STATISTICS_H

#pragma security level0

#ifndef G__STATISTICSSL

#ifdef G__SHAREDLIB
#pragma include_noerr <statistics.dll>
# ifndef G__STATISTICSSL
#include <statistics.c>
# endif
#else
#include <statistics.c>
#endif
#include <iostream.h>

#endif

class statistics : private _statistics {
#ifndef G__STATISTICSSL
	double min,max;
	double sigma,sigma2;
	int nsample;
	int error;
#endif
public:
	statistics(void);
	init(void);
	int add(double data,double lowlimit=-1e99, double uplimit=1e99);
	statistics& operator<<(double d);
	void disp(void);
	double min(void);
	double max(void);
	double stddev(void);
	double mean(void);
	int nsample(void);
	int error(void);
};

void statistics::disp(void)
{
  cout << *this;
}

ostream& operator<<(ostream& ios,statistics& stat)
{
	ios << "min=" << stat.min() << "  max=" << stat.max() ;
	ios << "  stddev=" << stat.stddev() << "  mean=" << stat.mean() ;
	ios << "  nsample=" << stat.nsample() << "  error=" << stat.error() ;
	ios << "\n";
	return(ios);
}

statistics& statistics::operator<<(double d)
{
	add(d);
	return(*this);
}

statistics::statistics(void)
{
	G__init_statistics((__statistics*)this);
}

statistics::init(void)
{
	G__init_statistics((__statistics*)this);
}

int statistics::add(double data,double lowlimit,double uplimit)
{
	G__add_statistics((__statistics*)this,data,lowlimit,uplimit);
}


double statistics::min(void)
{
	return(G__min_statistics((__statistics*)this));
}

double statistics::max(void)
{
	return(G__max_statistics((__statistics*)this));
}

double statistics::stddev(void)
{
	return(G__stddev_statistics((__statistics*)this));
}

double statistics::mean(void)
{
	return(G__mean_statistics((__statistics*)this));
}

int statistics::nsample(void)
{
	return(G__nsample_statistics((__statistics*)this));
}

int statistics::error(void)
{
	return(G__error_statistics((__statistics*)this));
}

#endif
