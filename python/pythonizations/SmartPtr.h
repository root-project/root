#ifndef SMARTPTR_H
#define SMARTPTR_H

#include <memory>
#include <iostream>

using namespace std;

class MyShareable {
public:
   MyShareable() { ++sInstances; }
   MyShareable(const MyShareable&) { ++sInstances; }
   MyShareable& operator=(const MyShareable&) { return *this; }
   ~MyShareable() { --sInstances; }

public:
   virtual const char* say_hi() { return "Hi!"; }

public:
   static int sInstances;
};

int MyShareable::sInstances = 0;

auto_ptr<MyShareable> mine = auto_ptr<MyShareable>(new MyShareable);

void renew_mine() { mine = auto_ptr<MyShareable>(new MyShareable); }

auto_ptr<MyShareable> gime_mine();
auto_ptr<MyShareable>* gime_mine_ptr();
auto_ptr<MyShareable>& gime_mine_ref();

void pass_mine_sp(auto_ptr<MyShareable> p);
void pass_mine_sp_ref(auto_ptr<MyShareable>& p);
void pass_mine_sp_ptr(auto_ptr<MyShareable>* p);

void pass_mine_rp(MyShareable);
void pass_mine_rp_ref(const MyShareable&);
void pass_mine_rp_ptr(const MyShareable*);

#ifdef __GCCXML__
// GCCXML explicit template instantiation block
template class auto_ptr<MyShareable>;
#endif

#endif

