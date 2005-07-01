#ifndef SINGLETON_H
#define SINGLETON_H

#include <iostream>

using namespace std;

template <class T>
class Singleton
{
  static Singleton<T>* instance;
  Singleton() {}
 public:
  void DoIt(bool output) { if (output) cout<<__PRETTY_FUNCTION__<<endl;}
  static Singleton& Instance()
  {  
    if(!instance)
      instance = new Singleton<T>();
    return *instance;
  }
};

// Here's the bug
// If this isn't hidden from rootcint, the build fails
template <class T> Singleton<T>* Singleton<T>::instance = 0;

#ifdef __MAKECINT__
#pragma link C++ class Singleton<int>;
#pragma link C++ class Singleton<double>;
#endif

#endif
