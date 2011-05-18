/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// debug.c
//
// cint debug interface demo program. "$ cint debug.c" to start
//
#include <iostream.h>

main()
{
	int a=1,b=2,c;
	cout << "\n";
	cout << "**********************************************************\n";
	cout << "* Welcome to cint debug capability demonstration.\n" ;
	cout << "* Please follow instruction written as comment\n" ;
	cout << "**********************************************************\n";
	cout << "\n";
	cout << "To debug source code, you need to start cint with -S or -b\n";
	cout << "\n";
	cout << "    $ cint -S [source].c     # step execution\n";
	cout << "    $ cint -b 10 [source].c  # break at line 10\n";
	cout << "\n";
	cout << "Then, stopped source line and following prompt is shown.\n";

	G__stepmode(1);
	// Follow the instruction,
	//
	// Input "S" and RETURN to step to next statement
	c=a+b;
	// If you want to step into function, input "s" and RETURN
	func(a,b,&c);
	// You step out from the function
	// Suppose you want to continue upto line 40, input "c 40" and RETURN

	c=a-b;

	a++;
	b++;

	func(b,c,&a);

	// You came to line 40.

	// Suppose you want to step over the function but want to know
	// what is going on. 

	// Input "T" and RETURN to turn on trace mode
	// Then "S" and RETURN" to step over the function.
	func(c,a,&b);
	// You saw the source code is traced.

	// Now this time, lets not execute following function.
	// Input "i" and RETURN to ignore the statement.
	func(c,a,&b);
	// Statement (function call) was ignored.

	// Input "T" and RETURN to toggle off trace mode 
	// Then "S" and RETURN to step.
	a=c*b;

	// Suppose you want to continue to the end
	// Input "c" and RETURN

	int i;
	int dmy[20001];
	for(i=0;i<10000;i++) dmy[i]=i;

	cout << "\n";
	cout << "       // Suppose you changed your mind and want to break\n";
	cout << "       // Press Break key or Cntl-C once.\n";
	cout << "       // Then \"v\" and return to look at source code\n";
	cout << "       // and read the comment carefully\n";
        cout << endl;
	i=1;
	while(i<20000) { //  This is where you are.
		dmy[i]=i;
		//
		// Now, you know why program does not finish.
		// This is an infinite loop. Very bad.
		// 
		// Lets change i value and exit from the loop
		// Input "p i" and RETURN to monitor i value
		// Input "p i=20000" and RETURN to modify i value
		//
		// Then "c" to continue
	} 
	cout << "\n   // exit from infinit loop\n";

	cout << "\n";
	cout << "   // Program is stopped by explicit G__pause() call.\n";
	cout << "   // To terminate process, input \"q\" and RETURN\n";
	cout << "   // To get more information, input \"h\" and RETURN\n";
        cout << endl;
	while(G__pause()==0) ;

}


func(int a,int b,int *pc)
{
	// You came into the function. 

	// Input "v" and return to list source code
	*pc=a+b;

	int i;
	double ary[1000];
	for(i=0;i<1000;i++) ary[i] = i;
}

//   '->' represents where you are.  
//
//   Input "e" and Return to step out from the function
//
