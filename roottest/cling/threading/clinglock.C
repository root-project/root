#include "TROOT.h"
#include "TInterpreter.h"
#include <thread>
void clinglock(){

ROOT::EnableThreadSafety();

auto w = [&](){gInterpreter->ProcessLine("gInterpreterMutex->Lock();gInterpreterMutex->UnLock();");};
std::thread t(w);
t.join();

}
