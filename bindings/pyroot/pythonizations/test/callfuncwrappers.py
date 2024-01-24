import unittest
import ROOT


class CallFuncWrappers(unittest.TestCase):
    """
    Tests for pass-by-value semantics of classes with deleted copy constructor
    """

    def test_pass_by_value_unique_ptr(self):
        """
        It is possible to pass a unique_ptr by value in a function call,
        whether it is a default argument or not
        """
        ROOT.gInterpreter.Declare(r'''

        struct uptrtype{
            int mVal{42};
            uptrtype() {}
            uptrtype(int val): mVal(val) {}
        };

        int foo_uniqueptr(std::unique_ptr<uptrtype> ptr = std::make_unique<uptrtype>()) { return ptr->mVal; }

        ''')

        # The correct value is returned by the default argument
        self.assertEqual(ROOT.foo_uniqueptr(), 42)

        # Creating a unique_ptr explicitly is also valid
        ptr = ROOT.std.make_unique[ROOT.uptrtype](33)
        self.assertEqual(ROOT.foo_uniqueptr(ptr), 33)

    def test_pass_by_value_templated_move_constructor(self):
        """
        It is possible to pass a class with a deleted copy constructor
        and a templated move constructor in a function by value
        """
        ROOT.gInterpreter.Declare(r'''
        struct templmovetype{
            int mVal{42};
            templmovetype() {}
            templmovetype(int val): mVal(val) {}
            templmovetype(const templmovetype&) = delete;
            template<typename T = int>
            templmovetype(templmovetype &&other): mVal(other.mVal) {}
        };
        int foo_templmovetype(templmovetype val = templmovetype{}) { return val.mVal; }
        ''')

        # The correct value is returned by the default argument
        self.assertEqual(ROOT.foo_templmovetype(), 42)

        val = ROOT.templmovetype(33)
        self.assertEqual(ROOT.foo_templmovetype(val), 33)

        self.assertEqual(ROOT.foo_templmovetype(ROOT.templmovetype(55)), 55)


if __name__ == '__main__':
    unittest.main()
