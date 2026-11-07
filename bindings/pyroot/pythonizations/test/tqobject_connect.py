import gc
import unittest

import ROOT

ROOT.gInterpreter.Declare("""
class PyTestEmitter : public TQObject {
public:
   void Go(Int_t i) { Emit("Go(Int_t)", i); }
   void Go3(Int_t i, Int_t j, Int_t k) { EmitVA("Go3(Int_t,Int_t,Int_t)", 3, i, j, k); }
   void Ping() { Emit("Ping()"); }
   // Connect() determines the sender class via IsA(); compiled classes get
   // this override from their ClassDef macro
   TClass *IsA() const override { return TClass::GetClass("PyTestEmitter"); }
};
""")


class TQObjectConnect(unittest.TestCase):
    """
    Test the pythonization of TQObject::Connect and Disconnect that directly
    accepts Python callables as slots.
    """

    def test_connect_no_args(self):
        emitter = ROOT.PyTestEmitter()
        calls = []
        self.assertTrue(emitter.Connect("Ping()", lambda: calls.append(1)))
        emitter.Ping()
        self.assertEqual(calls, [1])

    def test_connect_forwards_signal_args(self):
        emitter = ROOT.PyTestEmitter()
        calls = []
        self.assertTrue(emitter.Connect("Go(Int_t)", calls.append))
        emitter.Go(42)
        emitter.Go(7)
        self.assertEqual(calls, [42, 7])

    def test_connect_callable_with_fewer_args(self):
        # A callable accepting fewer arguments than the signal emits is called
        # with the arguments it can accept
        emitter = ROOT.PyTestEmitter()
        calls = []

        def no_args_slot():
            calls.append("called")

        self.assertTrue(emitter.Connect("Go(Int_t)", no_args_slot))
        emitter.Go(42)
        self.assertEqual(calls, ["called"])

    def test_disconnect(self):
        emitter = ROOT.PyTestEmitter()
        calls = []

        def slot(i):
            calls.append(i)

        self.assertTrue(emitter.Connect("Go(Int_t)", slot))
        emitter.Go(1)
        self.assertTrue(emitter.Disconnect("Go(Int_t)", slot))
        emitter.Go(2)
        self.assertEqual(calls, [1])
        # Disconnecting a callable that is not connected returns False
        self.assertFalse(emitter.Disconnect("Go(Int_t)", slot))

    def test_bound_method_slot(self):
        emitter = ROOT.PyTestEmitter()

        class Receiver:
            def __init__(self):
                self.calls = []

            def on_go(self, i):
                self.calls.append(i)

        receiver = Receiver()
        self.assertTrue(emitter.Connect("Go(Int_t)", receiver.on_go))
        emitter.Go(3)
        self.assertEqual(receiver.calls, [3])
        # Bound methods are recreated on each attribute access, so Disconnect
        # must still find the connection
        self.assertTrue(emitter.Disconnect("Go(Int_t)", receiver.on_go))
        emitter.Go(4)
        self.assertEqual(receiver.calls, [3])

    def test_connection_keeps_callable_alive(self):
        emitter = ROOT.PyTestEmitter()
        calls = []

        def make_slot():
            return lambda i: calls.append(i)

        emitter.Connect("Go(Int_t)", make_slot())
        gc.collect()
        emitter.Go(5)
        self.assertEqual(calls, [5])

    def test_multi_arg_signal(self):
        emitter = ROOT.PyTestEmitter()
        calls = []

        def slot(i, j, k):
            calls.append((i, j, k))

        self.assertTrue(emitter.Connect("Go3(Int_t,Int_t,Int_t)", slot))
        emitter.Go3(1, 2, 3)
        self.assertEqual(calls, [(1, 2, 3)])

    def test_exception_in_slot(self):
        # An exception raised in the slot is printed and does not propagate
        # through the C++ signal emission
        emitter = ROOT.PyTestEmitter()
        calls = []

        def raising_slot(i):
            calls.append(i)
            raise RuntimeError("problem in slot")

        self.assertTrue(emitter.Connect("Go(Int_t)", raising_slot))
        emitter.Go(1)
        emitter.Go(2)
        self.assertEqual(calls, [1, 2])


if __name__ == "__main__":
    unittest.main()
