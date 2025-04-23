#include <iostream>

#ifndef T_H
#define T_H

class t {
public:
   t(): fT(42.31) {}
   ~t() {}

   template <typename T>
   T get() const {
      std::cout << "called T get()" << std::endl;
      return fT; }
   float getfloat() const {
      return fT; }

   template <typename T>
   void set(T targ) {
      std::cout << "called set(T)" << std::endl;
      fT = targ; }
private:
   float fT;
};

template <> int t::get<int>() const {
   std::cout << "called get<int>()" << std::endl;
   return 1; }
template <> float t::get<float>() const {
   std::cout << "called get<float>()" << std::endl;
   return 2.;}
template <> double t::get<double>() const {
   std::cout << "called get<double>()" << std::endl;
   return 3.;}
template <> void t::set<int>(int) {
   std::cout << "called set<int>()" << std::endl;
   fT = 7; }
template <> void t::set<float>(float) {
   std::cout << "called set<float>()" << std::endl;
   fT = 8; }
template <> void t::set<double>(double) {
   std::cout << "called set<double>()" << std::endl;
   fT = 9; }

#endif
