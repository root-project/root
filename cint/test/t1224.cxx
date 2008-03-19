/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <iostream>
#include <map>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::map;
using std::string;

int main()
{
  string s;
  map<string, int> counters; // store each word and an associated counter
  map<string, int>::iterator iter;
  
  // read the input, keeping track of each word and how often we see it
  //while (cin >> s) {++counters[s];}
  ++counters["This"];
  ++counters["is"];
  ++counters["a"];
  ++counters["test"];
  ++counters["."];
  ++counters["test"];
  ++counters["of"];
  ++counters["iterator"];
  ++counters["operation"];
  ++counters["."];
  
  // write the words and associated counts
  for (iter = counters.begin(); iter != counters.end(); ++iter) {
    cout << iter->first << "\t" << iter->second << endl;
  }
  
  for (iter = counters.begin(); iter != counters.end(); ++iter) {
    cout << (*iter).first << "\t" << (*iter).second << endl;
  }
  
  return 0;
}


