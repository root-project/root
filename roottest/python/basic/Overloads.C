#include "Overloads.h"


//===========================================================================
OverloadA::OverloadA() { i1 = 42; i2 = -1; }

NamespaceA::OverloadA::OverloadA() { i1 = 88; i2 = -34; }
int NamespaceA::OverloadB::f(const std::vector<int>* v) { return (*v)[0]; }

NamespaceB::OverloadA::OverloadA() { i1 = -33; i2 = 89; }

OverloadB::OverloadB() { i1 = -2; i2 = 13; }


//===========================================================================
OverloadC::OverloadC() {}
int OverloadC::get_int(OverloadA* a)                 { return a->i1; }
int OverloadC::get_int(NamespaceA::OverloadA* a)     { return a->i1; }
int OverloadC::get_int(NamespaceB::OverloadA* a)     { return a->i1; }
int OverloadC::get_int(short* p)                     { return *p; }
int OverloadC::get_int(OverloadB* b)                 { return b->i2; }
int OverloadC::get_int(int* p)                       { return *p; }


//===========================================================================
OverloadD::OverloadD() {}
int OverloadD::get_int(int* p)                       { return *p; }
int OverloadD::get_int(OverloadB* b)                 { return b->i2; }
int OverloadD::get_int(short* p)                     { return *p; }
int OverloadD::get_int(NamespaceB::OverloadA* a)     { return a->i1; }
int OverloadD::get_int(NamespaceA::OverloadA* a)     { return a->i1; }
int OverloadD::get_int(OverloadA* a)                 { return a->i1; }


//===========================================================================
OlBB* get_OlBB() { static long buf; return (OlBB*)&buf; }
OlDD* get_OlDD() { static long buf; return (OlDD*)&buf; }


//===========================================================================
MoreOverloads::MoreOverloads() {}
std::string MoreOverloads::call(const OlAA&) { return "OlAA"; }
std::string MoreOverloads::call(const OlBB&, void* n) { n = 0; return "OlBB"; }
std::string MoreOverloads::call(const OlCC&) { return "OlCC"; }
std::string MoreOverloads::call(const OlDD&) { return "OlDD"; }

std::string MoreOverloads::call_unknown(const OlDD&) { return "OlDD"; }

std::string MoreOverloads::call(double)  { return "double"; }
std::string MoreOverloads::call(int)     { return "int"; }
std::string MoreOverloads::call1(int)    { return "int"; }
std::string MoreOverloads::call1(double) { return "double"; }


//===========================================================================
MoreOverloads2::MoreOverloads2() {}
std::string MoreOverloads2::call(const OlBB&) { return "OlBB&"; }
std::string MoreOverloads2::call(const OlBB*) { return "OlBB*"; }

std::string MoreOverloads2::call(const OlDD*, int) { return "OlDD*"; }
std::string MoreOverloads2::call(const OlDD&, int) { return "OlDD&"; }


//===========================================================================
std::string MoreBuiltinOverloads::method(int)    { return "int"; }
std::string MoreBuiltinOverloads::method(double) { return "double"; }
std::string MoreBuiltinOverloads::method(bool)   { return "bool"; }
std::string MoreBuiltinOverloads::method2(int)   { return "int"; }
std::string MoreBuiltinOverloads::method3(bool)  { return "bool"; }
std::string MoreBuiltinOverloads::method4(int, double)  { return "double"; }
std::string MoreBuiltinOverloads::method4(int, char)    { return "char"; }


//===========================================================================
std::string global_builtin_overload(int, double) { return "double"; }
std::string global_builtin_overload(int, char)   { return "char"; }


//===========================================================================
double calc_mean(long n, const float* a)  { return calc_mean_templ<float>(n, a); }
double calc_mean(long n, const double* a) { return calc_mean_templ<double>(n, a); }
double calc_mean(long n, const int* a)    { return calc_mean_templ<int>(n, a); }
double calc_mean(long n, const short* a)  { return calc_mean_templ<short>(n, a); }
double calc_mean(long n, const long* a)   { return calc_mean_templ<long>(n, a); }
