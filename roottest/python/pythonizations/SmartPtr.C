#include "SmartPtr.h"

shared_ptr<MyShareable> gime_mine() { return mine; }
shared_ptr<MyShareable>* gime_mine_ptr() { return &mine; }
shared_ptr<MyShareable>& gime_mine_ref() { return mine; }

void pass_mine_sp(shared_ptr<MyShareable> )
//{ cout << "Use count " << p->use_count() << "\n"; }
{ } // cout << "pass_mine_sp: underlying ptr " << p.get() << "\n"; }

void pass_mine_sp_ref(shared_ptr<MyShareable>& )
//{ cout << "Use count " << p->use_count() << "\n"; }
{ } // cout << "pass_mine_sp_ref: underlying ptr " << p.get() << "\n"; }

void pass_mine_sp_ptr(shared_ptr<MyShareable>* )
//{ cout << "Use count " << p->use_count() << "\n"; }
{ } // cout << "pass_mine_sp_ptr: underlying ptr " << p->get() << "\n"; }

void pass_mine_rp(MyShareable) {}
void pass_mine_rp_ref(const MyShareable&) {}
void pass_mine_rp_ptr(const MyShareable*) {}
