template <class T> class SomeTemplate{};
template <template <class T> class C> class TheTemplTempl{};
TheTemplTempl<SomeTemplate> b;
