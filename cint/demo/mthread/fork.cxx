/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// fork.cxx
//   This is not a multi-thread program. Simply using fork(), you can run
//  background job on UNIX.

#include <unistd.h>
#include <iostream.h>

// Data for running thread function /////////////////////////////////
class FuncData {
 public:
  FuncData(int nin,char *msgin) { n=nin; msg=msgin; }
  int n;
  char *msg;
};


// Interpreted foreground function //////////////////////////////////////
int InterpretedFunc(void *p) {
  struct FuncData *px=(struct FuncData*)p;
  cout << "ProcessID=" << getpid() << endl;
  for(int i=0;i<px->n;i++) {
    cout << px->msg << " i=" << i << endl;    
  }
  return(i);
}

// RunBackground //////////////////////////////////////////////////////
typedef int (*JobFunc)(void*);
void RunBackground(JobFunc p2f,void *p) {
  pid_t pid = fork();
  if(0==pid) {
    (*p2f)(p);
    exit(0);
  }
  return;
}

// Running 1 foreground and 1 background job ///////////////////////////
void test1() {

  // create data for jobs 
  FuncData background(100,"backgrond");
  FuncData foreground(100,"foreground");

  // start background job
  RunBackground(InterpretedFunc,&background);

  // start foreground jobs
  InterpretedFunc(&foreground);

}

// Running 1 foreground and 2 background jobs //////////////////////////
int test2() {

  // create data for jobs 
  FuncData background(100,"backgrond");
  FuncData background2(100,"backgrond2");
  FuncData foreground(100,"foreground");

  // start background jobs
  RunBackground(InterpretedFunc,&background);
  RunBackground(InterpretedFunc,&background2);

  // start foreground jobs
  InterpretedFunc(&foreground);
}

// Test program main /////////////////////////////////////////////
int main() {
  test1();
  test2();
  return(0);
}
