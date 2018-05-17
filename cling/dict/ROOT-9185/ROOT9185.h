// See https://sft.its.cern.ch/jira/browse/ROOT-9185
// rootcling was trying to reason about a dependent type.

template <typename T>
struct Traits {
    static const int isStatic = 1;
};

template <int I>
struct InnerBase {};

template <typename T>
class Outer {
public:
    struct Inner : InnerBase<Traits<T>::isStatic> {};
};

class ThisShouldBeSelected{};
