/* @(#)root/main:$Id$ */
/* Author: Valery Fine(fine@vxcern.cern.ch)   08/12/96 */

/*
 * This fortran subroutine is needed to set the right size
 * for the /PAWC/ common block for h2root C++ utility.
 * This common is defined as "external" in h2root and
 * its size is ignored by linker.
 */

int PAWC[4000000];
void setpawc(){}
