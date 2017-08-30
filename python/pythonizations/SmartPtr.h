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

unique_ptr<MyShareable> mine = unique_ptr<MyShareable>(new MyShareable);

void renew_mine() { mine = unique_ptr<MyShareable>(new MyShareable); }

unique_ptr<MyShareable> gime_mine();
unique_ptr<MyShareable>* gime_mine_ptr();
unique_ptr<MyShareable>& gime_mine_ref();

void pass_mine_sp(unique_ptr<MyShareable> p);
void pass_mine_sp_ref(unique_ptr<MyShareable>& p);
void pass_mine_sp_ptr(unique_ptr<MyShareable>* p);

void pass_mine_rp(MyShareable);
void pass_mine_rp_ref(const MyShareable&);
void pass_mine_rp_ptr(const MyShareable*);

#endif

