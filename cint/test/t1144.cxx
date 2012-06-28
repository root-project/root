/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <stdio.h>
#include <string>

namespace crap {
  static std::string s_unknown=std::string("unknown");
  //the previous line core dumps, however the next line works:
  static const std::string s_unknown2("unknown2");
}

std::string names[] = {"test1","test2"};

int main() {
  static std::string s_unknown3=std::string("unknown3");
  //the previous line core dumps, however the next line works:
  static const std::string s_unknown4("unknown4");
  
  printf("%s\n",crap::s_unknown.c_str());
  printf("%s\n",crap::s_unknown2.c_str());
  printf("%s\n",s_unknown3.c_str());
  printf("%s\n",s_unknown4.c_str());
  printf("%s\n",names[0].c_str());
  printf("%s\n",names[1].c_str());

  return 0;
}
