/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/



#if defined(interp) && defined(makecint)
#include "test.dll"
#else
#include "t1277.h"
#endif

#include <string>
#include <vector>
#include <iostream>

using namespace std;

int main() 
{
   vector<vector<float>* > IndexVec;		// OK

   IndexVec.push_back(new vector<float>);		// OK
   IndexVec.push_back(new vector<float>);		// OK

   IndexVec[0]->push_back(0.1f);		// OK
   IndexVec[0]->push_back(0.2f);		// OK

   IndexVec[1]->push_back(1.1f);		// OK
   IndexVec[1]->push_back(1.2f);		// OK

   vector<vector<float>* >::iterator fiter;	// OK
   vector<float> *ptr;			// OK

   fiter = IndexVec.begin();		// OK
   //ptr = *IndexVec.begin();		// OK
   //fiter;
   ptr = *fiter;				// problem is here (or typeing "p *fiter" in debugger)

   cout << (*ptr)[0] << endl;		// 

   cout << "first vector, first element = " << (*IndexVec[0])[0] << endl;// ERROR
   cout << "first vector, second element = " << (*IndexVec[0])[1] << endl;
   cout << "second vector, first element = " << (*IndexVec[1])[0] << endl;
   cout << "second vector, second element = " << (*IndexVec[1])[1] << endl;

   return 0;
}

