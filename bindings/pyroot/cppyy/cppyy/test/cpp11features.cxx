#if __cplusplus >= 201103L

#include "cpp11features.h"


// for std::shared/unique_ptr<> testing
int TestSmartPtr::s_counter = 0;

std::shared_ptr<TestSmartPtr> create_shared_ptr_instance() {
    return std::shared_ptr<TestSmartPtr>(new TestSmartPtr);
}

std::unique_ptr<TestSmartPtr> create_unique_ptr_instance() {
    return std::unique_ptr<TestSmartPtr>(new TestSmartPtr);
}

int TestSmartPtr::get_value() {
    return 17;
}

int DerivedTestSmartPtr::get_value() {
    return m_int + 76;
}

int pass_shared_ptr(std::shared_ptr<TestSmartPtr> p) {
    return p->get_value();
}

int move_shared_ptr(std::shared_ptr<TestSmartPtr>&& p) {
    return p->get_value();
}

int move_unique_ptr(std::unique_ptr<TestSmartPtr>&& p) {
    return p->get_value();
}

int move_unique_ptr_derived(std::unique_ptr<DerivedTestSmartPtr>&& p) {
    return p->get_value();
}

TestSmartPtr create_TestSmartPtr_by_value() {
    return TestSmartPtr{};
}


// for move ctors etc.
int TestMoving1::s_move_counter = 0;
int TestMoving1::s_instance_counter = 0;
int TestMoving2::s_move_counter = 0;
int TestMoving2::s_instance_counter = 0;

void implicit_converion_move(TestMoving2&&) {
    /* empty */
}


// for std::function testing
std::function<int(const FNTestStruct& t)> FNCreateTestStructFunc() { return [](const FNTestStruct& t) { return t.t; }; }
std::function<int(const FunctionNS::FNTestStruct& t)> FunctionNS::FNCreateTestStructFunc() { return [](const FNTestStruct& t) { return t.t; }; }

#endif // c++11 and later
