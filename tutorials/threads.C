//example of a simple script creating 3 threads
//this script can only be executed via ACLIC .x threads.C+
//before executing the script, load the Thread library with
//  gSystem->Load("libThread");
   
#include "TThread.h"
#include <Riostream.h>

void* handle(void* ptr) {
  int nr = (int) ptr;

  for (Int_t i=0;i<10;i++) {
    TThread::Lock();
    printf("Here I am loop index: %3d , thread: %d\n",i,nr);
    TThread::UnLock();
    sleep(1);
  }
  return 0;
}

void threads() {
  gDebug = 1;

  printf("Starting Thread 1\n");
  TThread *h1 = new TThread("h1", handle, (void*) 1);
  h1->Run();
  printf("Starting Thread 2\n");
  TThread *h2 = new TThread("h2", handle, (void*) 2);
  h2->Run();
  printf("Starting Thread 3\n");
  TThread *h3 = new TThread("h3", handle, (void*) 3);
  h3->Run();
}
