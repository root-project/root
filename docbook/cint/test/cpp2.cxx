/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/***************************************************************
* cpp2.cxx 
*
*  constructor,destructor
*
* not include copy constructor
*
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
  ~X() { free(string); } // delete operator
  void print(void);
};

X::X(const char *set)
{
  string = (char *)malloc(strlen(set)+1);  // new operator
  strcpy(string,set);
}

void X::print(void)
{
  printf("class object X = %s\n",string);
}

//############################################################
// Class Y 
//############################################################
class Y {
public:
  double *ary;
  int  n;
  Y(int number);
  ~Y();
 void print(void);
};

// int dmy;

Y::Y(int number){
  ary=(double *)malloc(number*8);
  for(n=0;n<number;n++)
    ary[n]=0.;
  n=number;
} 

Y::~Y()
{
  int i;
  for(i=0;i<n;i++) {
    printf("ary[%d]=%g ",i,ary[i]);
  }
  //putchar('\n');
  printf("\n");
  free(ary);  // delete operator
}

void Y::print(void)
{
  int i;
  for(i=0;i<n;i++) {
    printf("%d:%g ",i,ary[i]);
  }
  printf("\n");
}

//############################################################
// main
//############################################################
X g = X("global"), g2=X("global2");
Y gy = Y(10);
Y gy2(15);


void funcX(const char *in)
{
  X a = X(in) ;  // calling constructor 
  printf("%s\n",a.string);
}

void funcX2(const char *in1, const char *in2)
{
  X a = X(in1),b=X(in2); // calling constructor twice
  printf("%s\n",a.string);
  printf("%s\n",b.string);
}

void funcY()
{
  Y d(5);  // calling constructor
  int i;
  for(i=0;i<5;i++) {
    d.ary[i] = global;
    global = global+1;
  }
}

void funcY2()
{
  int i;
  Y a(6),b(7) ; // calling constructor twice
  for(i=0;i<6;i++) {
    a.ary[i] = global;
    global = global+1;
  }
  for(i=0;i<7;i++) {
    b.ary[i] = global;
    global = global+1;
  }
}

int main()
{
  g.print();
  g2.print();
  gy.print();
  gy2.print();

  funcX("abcdefg");
  funcY();
  funcX("defhijklmn");
  funcY();
  funcY2();
  funcX2("hijklmn","opqrstu");

  return 0;
}
