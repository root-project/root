/// \file
/// \ingroup tutorial_thread
/// Example of a simple script creating 3 threads.
/// This script can only be executed via ACliC: .x threads.C++.
/// Before executing the script, load the Thread library with:
///
/// ~~~{.cpp}
///   gSystem->Load("libThread");
/// ~~~
///
/// This is not needed anymore due to the rootmap facility which
/// automatically loads the needed libraries.
///
/// \macro_code
///
/// \author Victor Perevovchikov

#include "TThread.h"
#include <Riostream.h>

void *handle(void *ptr)
{
   long nr = (long) ptr;

   for (int i = 0; i < 10; i++) {
      TThread::Printf("Here I am loop index: %d , thread: %ld", i, nr);
      gSystem->Sleep(10);
   }
   return 0;
}

void threads()
{

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

   TThread::Ps();

   h1->Join();
   TThread::Ps();
   h2->Join();
   h3->Join();
   TThread::Ps();
}
