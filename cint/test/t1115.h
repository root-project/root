/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// 030819tmplt.txt
     // 030829map.txt, 030829map/...

#include <map>
#include <iostream>

class helper {
 public:
};
bool operator==(const helper& a,const helper& b) { return false; }
bool operator!=(const helper& a,const helper& b) { return true; }
bool operator>(const helper& a,const helper& b) { return true; }
bool operator<(const helper& a,const helper& b) { return true; }

class holder1 {
public:
  void f(std::map<const helper*,int> &) {
    std::cout << "f compiled okey" << std::endl;
  }
  void g(std::map<const helper,int> &) {
    std::cout << "g compiled okey" << std::endl;
  }
};
