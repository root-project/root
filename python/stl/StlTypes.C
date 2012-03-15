/*
  File: roottest/python/stl/StlTypes.C
  Author: Wim Lavrijsen@lbl.gov
  Created: 10/25/05
  Last: 10/25/05
*/

#include <list>
#include <map>
#include <string>
#include <vector>


class JustAClass {
public:
   int m_i;
};

class NoDictAvailable;

template< class T >
class STLLikeClass {
public:
   NoDictAvailable* begin() { return 0; }
   NoDictAvailable* end() { return 0; }
   int size() { return 4; }
   int operator[]( int i ) { return i; }
   std::string operator[]( double ) { return "double"; }
   std::string operator[]( const std::string& ) { return "string"; }
};

class StringyClass {
public:
   std::string GetString1() { return m_string; }
   void GetString2( std::string& s ) { s = m_string; }

   void SetString1( const std::string& s ) { m_string = s; }
   void SetString2( std::string s ) { m_string = s; }

   std::string m_string;
};

namespace {
   std::vector< std::string > jv1;
   std::vector< JustAClass >  jv2;
   std::list< JustAClass >    jl1;

   STLLikeClass< int >       js1;
   STLLikeClass< double>     js2;
}

#ifdef __MAKECINT__
using namespace std;  // instead of std:: to make cint7 happy
#pragma link C++ class vector< JustAClass >-;
#pragma link C++ class vector< JustAClass >::iterator-;
#pragma link C++ class vector< JustAClass >::const_iterator-;
#pragma link C++ class list< JustAClass* >-;
#pragma link C++ class list< JustAClass* >::iterator-;
#pragma link C++ class list< JustAClass* >::const_iterator-;
#pragma link C++ class map< std::string, unsigned int >-;
#pragma link C++ class map< std::string, unsigned int >::iterator-;
#pragma link C++ class map< std::string, unsigned int >::const_iterator-;
#pragma link C++ class pair< std::string, unsigned int >-;
#pragma link C++ class map< std::string, unsigned long >-;
#pragma link C++ class map< std::string, unsigned long >::iterator-;
#pragma link C++ class map< std::string, unsigned long >::const_iterator-;
#pragma link C++ class pair< std::string, unsigned long >-;
#ifdef G__WIN32
#pragma link C++ class iterator<random_access_iterator_tag,JustAClass,long,JustAClass*,JustAClass&>-;
#endif
#endif
