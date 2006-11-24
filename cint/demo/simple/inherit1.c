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
#include <iostream.h>

class string {
      private:
	char *p;
      public:
	string(char *set) { 
		cout << "construct string(" << set << ")\n";
		p=(char*)malloc(strlen(set)+1);
		strcpy(p,set);
	}
	int len() { return(strlen(p)); }
	char *s() { return(p); }

	~string() { 
		cout << "destruct ~string(" << p << ")\n"; 
		free(p); 
	}
};

ostream& operator <<(ostream& ios,string& x) 
{
	ios << x.s() ;
	return(ios);
}

class gender {
      private:
	char *male;
	string Gender;
      public:
	gender(char *set) : Gender(set) { 
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
	string address;
	string name;
	int age;
	person(void);
	person(char *adr,char *nm,int ag,char *setgender);
};

person::person(void) 
:address("SAITAMA"),name("MASAHARU"),gender("MALE")
{
	age = 33;
	cout << "construct person(" << age << ")\n";
}

person::person(char *adr,char *nm,int ag,char *setgender) 
: address(adr) , name(nm) , gender(setgender)
{
	cout << "construct person(" << ag << ")\n";
	age = ag;
}


main()
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
}
