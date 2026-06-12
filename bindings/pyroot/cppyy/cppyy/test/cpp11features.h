#include <functional>
#include <memory>
#include <vector>


//===========================================================================
class TestSmartPtr {         // for std::shared/unique_ptr<> testing
public:
    static int s_counter;

public:
    TestSmartPtr() { ++s_counter; }
    TestSmartPtr(const TestSmartPtr&) { ++s_counter; }
    virtual ~TestSmartPtr() { --s_counter; }

public:
    virtual int get_value();
};

std::shared_ptr<TestSmartPtr> create_shared_ptr_instance();
std::unique_ptr<TestSmartPtr> create_unique_ptr_instance();

class DerivedTestSmartPtr : TestSmartPtr {
public:
    DerivedTestSmartPtr(int i) : m_int(i) {}
    virtual int get_value();

public:
    int m_int;
};

int pass_shared_ptr(std::shared_ptr<TestSmartPtr> p);
int move_shared_ptr(std::shared_ptr<TestSmartPtr>&& p);
int move_unique_ptr(std::unique_ptr<TestSmartPtr>&& p);
int move_unique_ptr_derived(std::unique_ptr<DerivedTestSmartPtr>&& p);

TestSmartPtr create_TestSmartPtr_by_value();

// for auto-downcast of objects returned through a smart pointer
class PubDerivedTestSmartPtr : public TestSmartPtr {
public:
    int only_in_derived() { return 27; }
};

// second base so that the cross-cast to the most derived type needs a
// non-zero pointer adjustment, which the smart pointer's dereferencer can
// not apply consistently (so no down-cast should happen in that case)
class TestSmartPtrIface {
public:
    virtual ~TestSmartPtrIface() {}
    long m_pad = 0;
    int only_in_iface() { return 37; }
};

class MultiDerivedTestSmartPtr : public PubDerivedTestSmartPtr, public TestSmartPtrIface {
};

std::shared_ptr<TestSmartPtr> create_shared_ptr_to_derived();
std::unique_ptr<TestSmartPtr> create_unique_ptr_to_derived();
std::unique_ptr<TestSmartPtrIface> create_unique_ptr_to_offset_derived();

// sinks expecting a smart pointer to the *derived* type; a base-class smart
// pointer (even when its object was auto-down-cast) must not be accepted here
int pass_unique_ptr_to_derived(std::unique_ptr<PubDerivedTestSmartPtr> p) {
    return p->only_in_derived();
}

int pass_shared_ptr_to_derived(std::shared_ptr<PubDerivedTestSmartPtr> p) {
    return p->only_in_derived();
}

// Overloaded function to check if automatic downcasting is consistently
// applied for regular proxy objects and smart pointer proxies.
std::string pass_ptr_overloaded(TestSmartPtr *) { return "TestSmartPtr"; }
std::string pass_ptr_overloaded(PubDerivedTestSmartPtr *) { return "PubDerivedTestSmartPtr"; }
std::string pass_ref_overloaded(TestSmartPtr &) { return "TestSmartPtr"; }
std::string pass_ref_overloaded(PubDerivedTestSmartPtr &) { return "PubDerivedTestSmartPtr"; }
std::string pass_val_overloaded(TestSmartPtr) { return "TestSmartPtr"; }
std::string pass_val_overloaded(PubDerivedTestSmartPtr) { return "PubDerivedTestSmartPtr"; }


//===========================================================================
class TestMoving1 {          // for move ctors etc.
public:
    static int s_move_counter;
    static int s_instance_counter;

public:
    TestMoving1() { ++s_instance_counter; }
    TestMoving1(TestMoving1&&) { ++s_move_counter; ++s_instance_counter; }
    TestMoving1(const TestMoving1&) { ++s_instance_counter; }
    TestMoving1& operator=(TestMoving1&&) { ++s_move_counter; return *this; }
    TestMoving1& operator=(TestMoving1&) { return *this; }
    ~TestMoving1() { --s_instance_counter; }
};

class TestMoving2 {          // note opposite method order from TestMoving1
public:
    static int s_move_counter;
    static int s_instance_counter;

public:
    TestMoving2() { ++s_instance_counter; }
    TestMoving2(const TestMoving1&) { ++s_instance_counter; }
    TestMoving2(const TestMoving2&) { ++s_instance_counter; }
    TestMoving2(TestMoving2&& other) { ++s_move_counter; ++s_instance_counter; }
    TestMoving2& operator=(TestMoving2&) { return *this; }
    TestMoving2& operator=(TestMoving2&&) { ++s_move_counter; return *this; }
    ~TestMoving2() { --s_instance_counter; }
};

void implicit_converion_move(TestMoving2&&);


//===========================================================================
struct TestData {            // for initializer list construction
    TestData(int i=0) : m_int(i) {}
    int m_int;
};

struct TestData2 {
    TestData2(int i=0) : m_int(i) {}
    virtual ~TestData2() {}
    int m_int;
};

template<class T>
class WithInitList {
public:
    WithInitList(std::initializer_list<T> ll) : m_data(ll) {}
    const T& operator[](int i) { return m_data[i]; }

    int size() { return m_data.size(); }

private:
    std::vector<T> m_data;
};


//===========================================================================
struct FNTestStruct {        // for std::function<> testing
     FNTestStruct(int i) : t(i) {}
     int t;
};
std::function<int(const FNTestStruct& t)> FNCreateTestStructFunc();

namespace FunctionNS {
    struct FNTestStruct { FNTestStruct(int i) : t(i) {} int t; };
    std::function<int(const FNTestStruct& t)> FNCreateTestStructFunc();
}
