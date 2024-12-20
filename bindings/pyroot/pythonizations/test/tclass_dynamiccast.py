import unittest

from ROOT import TClass, TObject, TObjString


class TClassDynamicCast(unittest.TestCase):
    """
    Test for the pythonization of TClass::DynamicCast, which adds an
    an extra cast before returning the Python proxy to the user so that
    it has the right type.
    """

    # Tests
    def test_dynamiccast(self):
        tobj_class = TClass.GetClass("TObject")
        tobjstr_class = TClass.GetClass("TObjString")

        o = TObjString("a")

        # Upcast: TObject <- TObjString
        o_upcast = tobjstr_class.DynamicCast(tobj_class, o)
        self.assertEqual(type(o_upcast), TObject)

        # Downcast: TObject -> TObjString
        o_downcast = tobjstr_class.DynamicCast(tobj_class, o_upcast, False)
        self.assertEqual(type(o_downcast), TObjString)


if __name__ == '__main__':
    unittest.main()
