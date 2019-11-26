#if __cplusplus >= 201103L

#include <functional>
#include <memory>
#include <vector>


//===========================================================================
class TestSharedPtr {        // for std::shared_ptr<> testing
public:
    static int s_counter;

public:
    TestSharedPtr() { ++s_counter; }
    TestSharedPtr(const TestSharedPtr&) { ++s_counter; }
    virtual ~TestSharedPtr() { --s_counter; }

public:
    virtual int get_value();
};

std::shared_ptr<TestSharedPtr> create_shared_ptr_instance();

class DerivedTestSharedPtr : TestSharedPtr {
public:
    DerivedTestSharedPtr(int i) : m_int(i) {}
    virtual int get_value();

public:
    int m_int;
};

int pass_shared_ptr(std::shared_ptr<TestSharedPtr> p);


//===========================================================================
class TestMoving1 {          // for move ctors etc.
public:
    static int s_move_counter;

public:
    TestMoving1() {}
    TestMoving1(TestMoving1&&) { ++s_move_counter; }
    TestMoving1(const TestMoving1&) {}
    TestMoving1& operator=(TestMoving1&&) { ++s_move_counter; return *this; }
    TestMoving1& operator=(TestMoving1&) { return *this; }
};

class TestMoving2 {          // note opposite method order from TestMoving1
public:
    static int s_move_counter;

public:
    TestMoving2() {}
    TestMoving2(const TestMoving2&) {}
    TestMoving2(TestMoving2&& other) { ++s_move_counter; }
    TestMoving2& operator=(TestMoving2&) { return *this; }
    TestMoving2& operator=(TestMoving2&&) { ++s_move_counter; return *this; }
};


//===========================================================================
struct TestData {            // for initializer list construction
    TestData(int i) : m_int(i) {}
    int m_int;
};

struct TestData2 {
    TestData2(int i) : m_int(i) {}
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

#endif // c++11 and later
