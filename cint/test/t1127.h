/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>


struct vadat {
  char buf[256];
};

struct A {
  //A() {}
  //A(int ain,double bin,short cin,char din,double ein) : a(ain) , b(bin) , c(cin) , d(din) , e(ein) { }
  int a;
  double b;
  short c;
  char d;
  double e;
  //void disp() {printf("'a=%d b=%g c=%d d=%c e=%g'",a,b,c,d,e);}
};

void disp(A& x) {printf("'a=%d b=%g c=%d d=%c e=%g'",x.a,x.b,x.c,x.d,x.e);}

void xdump(char* x,int size) {
  int i;
  for(i=0;i<size;i++) {
    if(i%4==0) printf("\n%p   :",&x[i]);
    printf("%3x ",x[i]&0xff);
    //if(isprint(x[i])) printf("%c ");
    //else              printf("  ");
  } 
  printf("\n");
}

void f(const char* fmt,int argn, ...) {
  va_list argptr;
  A a;
  printf("%s %d :",fmt,argn);
  //printf("%p %p %s\n",&fmt,fmt,fmt);
  //printf("%p %d\n",&argn,argn);
  char *tmp;
  va_start(argptr,argn);
#ifdef TEST
  xdump((char*)argptr,100);
#endif
  while(*fmt) {
    switch(*fmt) {
    case 's':
      tmp = va_arg(argptr,char*);
      printf("(char*)%s ",tmp);
      break;
    case 'd':
      printf("(double)%g ",va_arg(argptr,double));
      break;
    case 'u':
      a = va_arg(argptr,A);
      disp(a);
      break;
#if defined(G__MSC_VER) || defined(_MSC_VER)
    case 'c':
      printf("(char)%c ",va_arg(argptr,char));
      break;
    case 'r':
      printf("(short)%d ",va_arg(argptr,short));
      break;
#else
    case 'c':
      printf("(char)%c ",va_arg(argptr,int));
      break;
    case 'r':
      printf("(short)%d ",va_arg(argptr,int));
      break;
#endif
    case 'i':
    default:
      printf("(int)%d ",va_arg(argptr,int));
      break;
    }
    ++fmt;
  }
#ifdef TEST
  xdump((char*)argptr,100);
#endif
  va_end(argptr);
  printf("\n");
}


void g(const char *fmt,int argn,struct vadat x) {
  printf("%s %d",fmt,argn);
  printf("%p\n",&x);
}


int testc() {
  struct vadat x;
  A a = { 345, 6.28, 3229, 'x', 1.4142 };
  x.buf[0] = 0;
  f("sdisrcu",2,"abcdefghijklmn",3.14,1234,"A",(short)12,'a',a);
  //g("sdis",2,x);
  return 0;
}




