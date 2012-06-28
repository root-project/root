/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <ostream>

#include "IPair.cpp"

using namespace std;

// -----------------------------

// Test wrapper

static IPair pass( const IPair &pr );

static IPair pass( const IPair &pr )
{
	return pr;
}

// -----------------------------

int main()
{
	IPair p( 0, 100 );
	IPair q( 100, 100 );

	cout << pass( p ) << endl;
	cout << pass( q ) << endl;
	cout << ( pass( p ) - pass( q ) ) << endl;
	IPair r = pass( p ) - pass( q );

	cout << "The following three lines are correct: " << endl;
	cout << ( p - q ).Mag() << endl;
	cout << r.Mag() << endl;
	cout << ( r ).Mag() << endl;

	cout << "The following line used to not work when interpreted." << endl;
	cout << ( pass( p ) - pass( q ) ).Mag() << endl;

	return 0;
}

// EOF
/*

The provided examples demonstrates a problem when running in the interpreter :

	$CINTSYSDIR/cint IPair.cpp main.cpp
	0 100
	100 100
	-100 0
	The following three lines are correct: 
	100
	100
	100
	The following line does not work when interpreted.
	Error: non class,struct,union object pass(p)-pass(q) used with . or -> FILE:main.cpp LINE:36
	!!! return from main() function
	
	cint> quit

But when compiled it is fine:

	g++ IPair.cpp main.cpp -o test
	./test
	0 100
	100 100
	-100 0
	The following three lines are correct: 
	100
	100
	100
	The following line does not work when interpreted.
	100

The problem occurs in main.cpp on line 36 where the pass() functions are returning a copy of "p" & "q".

	cout << ( pass( p ) - pass( q ) ).Mag() << endl;

Note that the simplified case of "(p - q).Mag()" does work.

*/

