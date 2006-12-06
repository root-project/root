/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include "IPair.h"

/*!
Prints the components of a IPair to an output stream.  Example usage: cout << myIPair;
\param os The output stream
\param v The IPair to print
\return The output stream
*/
std::ostream &operator << ( std::ostream &os, const IPair &v )
{
	return os << v[0] << " " << v[1];
}


