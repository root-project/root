// Two wrappers are needed to reproduce the assertion with broken redecl chain.
template <typename T>
class Wrapper1 {};
template <typename T>
class Wrapper2 {};

class RandomClass {};
Wrapper1<Wrapper2<RandomClass>> var;
