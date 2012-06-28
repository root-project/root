/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "rflx_tools.h"
#include <iostream>

std::string rflx_tools::escape_class_name(const std::string & name)
{

   std::string rpl_chars = "<>,*: ./~&";
   std::string lname = name;
   for (size_t i = 0; i < name.length(); ++i) {
      if (rpl_chars.find(lname[i]) != std::string::npos)
         lname[i] = '_';
   }

   return lname;

}


std::string rflx_tools::rm_end_ref(const std::string & name)
{
   if (name[name.length() - 1] == '&')
      return name.substr(0, name.length() - 1);
   return name;
}


std::string rflx_tools::decorate_stl_type(const std::string & name)
{
   // Note: this is extremely ugly, but CINT doesn't seem to know about
   // the std namespace. Until a better solution for distinguishing stl
   // from non-stl types is found we have to rely on the name ....

   /*
      4 map set
      5 list
      6 queue stack deque
      7 vector bitset string
      9 multimap multiset
      10 allocator
    */

   bool isSTLType = false;

   std::string nam = "";

   nam = name.substr(0, 10);
   if ((nam == "allocator<"))
      isSTLType = true;

   nam = name.substr(0, 9);
   if ((nam == "multimap<") || (nam == "multiset<"))
      isSTLType = true;

   nam = name.substr(0, 7);
   if ((nam == "vector<") || (nam == "bitset<"))
      isSTLType = true;

   nam = name.substr(0, 6);
   if ((nam == "queue<") || (nam == "deque<") || (nam == "stack<"))
      isSTLType = true;

   nam = name.substr(0, 5);
   if ((nam == "list<"))
      isSTLType = true;

   nam = name.substr(0, 4);
   if ((nam == "map<") || (nam == "set<"))
      isSTLType = true;

   if (isSTLType) {
      return "std::" + name;
      //std::string lname = "std::" + name;
      //std::cout << lname << std::endl;
      //return lname;
   } else
      return name;
}

std::string rflx_tools::un_const(const std::string & name)
{
   if (name.substr(0, 6) == "const ")
      return name.substr(6);
   return name;
}

std::string rflx_tools::stub_type_name(const std::string & name)
{

   std::string lname = name;
   lname = un_const(lname);
   //lname = decorate_stl_type(lname);
   lname = rm_end_ref(lname);

   return lname;
}
