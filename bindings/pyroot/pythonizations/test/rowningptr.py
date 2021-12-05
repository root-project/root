import unittest

import ROOT


# Probe class to track the number of destructor calls. Includes static
# functions that return class as raw pointer, owner pointer, or unique_ptr.
# Finally there is another raw pointer version that we will flag with
# `__creates__` on the Python side.
ROOT.gInterpreter.Declare(
    """
class MyClass {
public:
    static constexpr int x_val = 666;

    MyClass() : _x{x_val}, _vec{2500-1} {}
    ~MyClass() { ++_n_destructor_calls; }
    int x() const { return _x; }
    static int n_destructor_calls() { return _n_destructor_calls; }

    static MyClass* make_raw() { return new MyClass; }
    static ROOT::ROwningPtr<MyClass> make_owner() { return new MyClass; }
    static std::unique_ptr<MyClass> make_unique() { return std::make_unique<MyClass>(); }
    static MyClass* make_creates() { return new MyClass; }
private:
    int _x = 0; // One value that we check to see if the memory was corrupted.
    std::vector<int> _vec;
    int _mem[2500-1]; // Make this class fat (10 kB) so we can test for memory leaks easily.
    static int _n_destructor_calls;
};

int MyClass::_n_destructor_calls = 0;
"""
)

MyClass = ROOT.MyClass
MyClass.make_creates.__creates__ = True


class TestROOTROwningPtr(unittest.TestCase):
    """
    Test for ROOT::ROwningPtr (raw pointer wrapper to indicate ownership of return
    values to Python).
    """

    def test_successul_creation(self):
        self.assertEqual(MyClass.make_raw().x(), MyClass.x_val)
        self.assertEqual(MyClass.make_unique().x(), MyClass.x_val)
        self.assertEqual(MyClass.make_creates().x(), MyClass.x_val)
        self.assertEqual(MyClass.make_owner().x(), MyClass.x_val)

    def test_successul_deletion(self):
        def test_creator(func, should_work):
            n_prev_calls = MyClass.n_destructor_calls()
            x = func()
            del x
            self.assertEqual(MyClass.n_destructor_calls(), n_prev_calls + int(should_work))

        test_creator(MyClass.make_raw, False)
        test_creator(MyClass.make_unique, True)
        test_creator(MyClass.make_creates, True)
        test_creator(MyClass.make_owner, True)

    def test_if_memory_leaks(self):
        # The class is 10 kB, so if we create 5000 of them we should get a
        # leak of 50 MB that will be clearly visible despite Pythons memory
        # management.
        n_iter = 5000

        pinfo = ROOT.ProcInfo_t()

        def mem_increase(func):
            ROOT.gSystem.GetProcInfo(pinfo)
            initial = pinfo.fMemResident

            for i in range(n_iter):
                x = func()
                del x

            ROOT.gSystem.GetProcInfo(pinfo)
            final = pinfo.fMemResident

            return final - initial

        incr_raw = mem_increase(MyClass.make_raw)  # will leak!
        incr_unique = mem_increase(MyClass.make_unique)
        incr_creates = mem_increase(MyClass.make_creates)
        incr_owner = mem_increase(MyClass.make_owner)

        # We can't be sure that there will be no memory increase for the good
        # solutions, but the increase should be significantly less.
        max_allowed_increase = 0.2 * incr_raw
        self.assertLess(incr_unique, max_allowed_increase)
        self.assertLess(incr_creates, max_allowed_increase)
        self.assertLess(incr_owner, max_allowed_increase)


if __name__ == "__main__":
    unittest.main()
