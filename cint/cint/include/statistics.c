/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***************************************************************************
* statistics.c
*
* shared library source for standard deviation.
*
*  makecint -dl statistics.sl -c statistics.c
*
***************************************************************************/
#include <stdio.h>
#include <math.h>

#define G__STATISTICSSL

typedef struct _statistics{
	double min,max;
	double sigma,sigma2;
	int nsample;
	int error;
} __statistics ;

int G__init_statistics(__statistics *stat)
{
	stat->min = 1e99;
	stat->max = -1e99;
	stat->sigma = 0.0;
	stat->sigma2 = 0.0;
	stat->error = 0;
	stat->nsample = 0;
	return(0);
}

int G__add_statistics(__statistics *stat,double data
		      ,double lowlimit,double uplimit)
{
	stat->sigma += data;
	stat->sigma2 += (data*data);
	++stat->nsample ;

	if(data>stat->max) stat->max=data;
	if(data<stat->min) stat->min=data;

	if(data < lowlimit || uplimit < data) {
	        /* fprintf(stderr,"%g\n",data); */
		++stat->error;
		return(stat->error);
	}
	return(0);
}

double G__min_statistics(__statistics *stat)
{
	return(stat->min);
}

double G__max_statistics(__statistics *stat)
{
	return(stat->max);
}

int G__nsample_statistics(__statistics *stat)
{
	return(stat->nsample);
}

int G__error_statistics(__statistics *stat)
{
	return(stat->error);
}

double G__stddev_statistics(__statistics *stat)
{
	double result;
	if(0>=stat->nsample) 
		result=0.0;
	else
		result=sqrt((stat->sigma2 -
			     stat->sigma*stat->sigma/stat->nsample)
			    /(stat->nsample-1)
			    );
	return(result);
}

double G__mean_statistics(__statistics *stat)
{
	if(0<=stat->nsample) 
		return(stat->sigma/stat->nsample);
	else 
		return(0);
}

