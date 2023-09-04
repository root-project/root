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

shared_ptr<MyShareable> mine = std::make_shared<MyShareable>();

void renew_mine() { mine = std::make_shared<MyShareable>(); }

shared_ptr<MyShareable> gime_mine();
shared_ptr<MyShareable>* gime_mine_ptr();
shared_ptr<MyShareable>& gime_mine_ref();

void pass_mine_sp(shared_ptr<MyShareable> p);
void pass_mine_sp_ref(shared_ptr<MyShareable>& p);
void pass_mine_sp_ptr(shared_ptr<MyShareable>* p);

void pass_mine_rp(MyShareable);
void pass_mine_rp_ref(const MyShareable&);
void pass_mine_rp_ptr(const MyShareable*);

#endif
