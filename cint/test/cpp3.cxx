/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***************************************************************
* cpp3.cxx
*
*  function overloading
*  constructor is overloaded
*
* BUGS:
*  If constructor is defined, default constructor shouldn't 
* exist. But now, even if constructor is defined, default constructor
* still exist. Thus, 'X g4;' won't be an error , which should
* be.
***************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

double global=0;

//############################################################
// Class X 
//############################################################
class X {
 public:
  char *string;
  X(const char *set);
  X(int set);
  ~X() { free(string); } // delete operator
  void print(void);
};

X::X(const char *set)
{
  string = (char *)malloc(strlen(set)+1);  // new operator
  strcpy(string,set);
}

X::X(int set)
{
  char temp[20];
  sprintf(temp,"%d",set);
  string = (char *)malloc(strlen(temp)+1);
  strcpy(string,temp);
}

void X::print(void)
{
  printf("class object X = %s\n",string);
}

//############################################################
// main
//############################################################
X g1 = X((int)3.5), g2=X("global2"), g3(3229);

// X g4; This should be an error, because default constructor shouldn't exist


void funcX(const char *in)
{
  X a = X(in) ;  // calling constructor 
  printf("%s\n",a.string);
}

void funcX(const int in)
{
  X a = X(in) ;  // calling constructor 
  printf("%s\n",a.string);
}

void funcX2(const char *in1="default string", const int in2=-1)
{
  X a = X(in1),b=X(in2); // calling constructor twice
  printf("%s\n",a.string);
  printf("%s\n",b.string);
}

int main()
{
  g1.print();
  g2.print();
  g3.print();

  funcX("abcdefg");
  funcX(25);
  funcX2();
  funcX2("actual",1011);

  return 0;
}
