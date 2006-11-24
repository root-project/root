/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <vector>
#include <list>
#include <string>
#include <map>
#include <deque>
#include <algorithm>
//#include <bitset>
//#include <set>
using namespace std;

class A {
 public:
  vector<float> x;
  list<float> y;
  string z;
  deque<float> b;
  map<string,float> a;
  //map<char*,float> a;
  //bitset<8> c;
  //set<float> d;
};

// Example of container container
class fvector : public vector<float> { }; // Trick to make cint happy
vector<fvector> xxx;
// vector<vector<float> > xxx;  // Limitation, this does not work


#ifdef __MAKECINT__
#pragma link C++ nestedtypedefs;
#pragma link C++ nestedclasses;
#pragma link C++ function reverse(deque<float>::iterator,deque<float>::iterator);
#endif

