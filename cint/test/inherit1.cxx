/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//
// class inheritance test 1
//
// constructor, destructor
//
#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class String {
      private:
	char *p;
      public:
	String(const char *set) { 
		cout << "construct String(" << set << ")\n";
		p=(char*)malloc(strlen(set)+1);
		strcpy(p,set);
	}
	int len() { return(strlen(p)); }
	char *s() { return(p); }

	~String() { 
		cout << "destruct ~String(" << p << ")\n"; 
		free(p); 
	}
};

ostream& operator <<(ostream& ios,String& x) 
{
	ios << x.s() ;
	return(ios);
}

class gender {
      private:
	char *male;
	String Gender;
      public:
	gender(const char *set) : Gender(set) { 
		cout << "construct gender(" << set << ")\n";
		male=(char*)malloc(strlen(set)+1);
		strcpy(male,set);
	}
	char *S() { return(male); }
	~gender() { 
		cout << "destruct ~gender(" << male<< ")\n";
		free(male); 
	}
};

class person : public gender {
public:
  String address;
  String name;
  int age;
  person(void);
  person(const char *adr,const char *nm,int ag,const char *setgender);
};

person::person(void) : gender("MALE"),address("SAITAMA"),name("MASAHARU")
{
  age = 33;
  cout << "construct person(" << age << ")\n";
}

person::person(const char *adr,const char *nm,int ag,const char *setgender) 
  : gender(setgender),address(adr) , name(nm) 
{
  cout << "construct person(" << ag << ")\n";
  age = ag;
}


int main()
{
	person me=person("Saitama","Masa",32,"male");

	cout << " address=" << me.address 
	     << " name=" << me.name 
	     << " age=" << me.age 
	     << " Gender=" << me.S()
	     << "\n";

	person *pme;
	pme = new person;

	cout << " address=" << pme->address 
	     << " name=" << pme->name 
	     << " age=" << pme->age 
	     << " Gender=" << pme->S()
	     << "\n";

	delete pme;

	return 0;
}
