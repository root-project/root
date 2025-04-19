import unittest


class Regression18441(unittest.TestCase):
    # https://github.com/root-project/root/issues/18441
    def test(self):
        # Imports are done here instead than at module level to reproduce
        # the issue with import order at the link
        import cppyy

        cppyy.cppdef(r"""

        template <typename T, typename U>
        class Class18441{
            T value;
        public:
            Class18441(T val): value(val) {}
        };

        struct Value18441{
            int my_int{0};
        };
        """)

        from ROOT import pythonization

        # 2 template arguments to make the instantiation less trivial
        myclass = cppyy.gbl.Class18441[cppyy.gbl.Value18441, float]

        @pythonization("Class18441", is_prefix=True)
        def _(klass):
            if not hasattr(klass, "pythonization_counter"):
                klass.pythonization_counter = 0

            klass.pythonization_counter += 1

        self.assertEqual(myclass.pythonization_counter, 1)


if __name__ == "__main__":
    unittest.main()
