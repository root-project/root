import unittest

import ROOT
from ROOT import pythonization


class PythonizationDecorator(unittest.TestCase):
    """
    Test the @pythonization decorator for user-defined classes.
    """

    # Some already instantiated ROOT classes may match targets of @pythonization
    # in some tests, and because of immediate pythonization they will be
    # processed by the pythonizors. Just ignore them
    exclude = [ 'TClass', 'TSystem', 'TUnixSystem', 'TMacOSXSystem',
                'TWinNTSystem', 'TDictionary', 'TEnv', 'TInterpreter',
                'TObject', 'TNamed', 'TROOT', 'TIter', 'TDirectory', 'TString' ]

    # Helpers
    def _define_class(self, class_name, namespace=None):
        if namespace is None:
            ROOT.gInterpreter.ProcessLine('class {cn} {{ }};'.format(cn=class_name))
        else:
            ROOT.gInterpreter.ProcessLine('''
            namespace {ns} {{
            class {cn} {{}};
            }}'''.format(ns=namespace, cn=class_name))

    # Test @pythonization('MyClass')
    def test_single_class_global(self):
        # Define test class
        class_name = 'MyClass'
        self._define_class(class_name)

        executed = []

        # Define pythonizor function for class
        @pythonization(class_name)
        def my_func(klass, name):
            self.assertEqual(klass.__cpp_name__, class_name)
            self.assertEqual(name, class_name)
            executed.append(name)

        # Trigger execution of pythonizor
        getattr(ROOT, class_name)

        self.assertEqual(executed.pop(), class_name)
        self.assertTrue(not executed)

    # Test @pythonization('NS::MyClass')
    def test_single_class_in_ns(self):
        # Define test class
        ns = 'NS'
        class_name = 'MyClass1'
        fqn = ns + '::' + class_name
        self._define_class(class_name, ns)
 
        executed = []

        # Define pythonizor function for class
        @pythonization(class_name, ns=ns)
        def my_func(klass, name):
            self.assertEqual(klass.__cpp_name__, fqn)
            self.assertEqual(name, fqn)
            executed.append(name)

        # Trigger execution of pythonizor
        getattr(getattr(ROOT, ns), class_name)

        self.assertEqual(executed.pop(), fqn)
        self.assertTrue(not executed)

    # Test @pythonization('Prefix', is_prefix=True)
    def test_single_prefix_global(self):
        # Define test classes
        prefix = 'OnePrefix'
        class1 = prefix + 'Class1'
        class2 = prefix + 'Class2'
        for class_name in class1, class2:
            self._define_class(class_name)

        executed = []

        # Define pythonizor function for prefix
        @pythonization(prefix, is_prefix=True)
        def my_func(klass, name):
            self.assertTrue(klass.__cpp_name__.startswith(prefix))
            self.assertTrue(name.startswith(prefix))
            executed.append(name)

        # Trigger execution of pythonizor
        for class_name in class1, class2:
            getattr(ROOT, class_name)

        for class_name in class1, class2:
            self.assertEqual(executed.pop(0), class_name)
        self.assertTrue(not executed)

    # Test @pythonization('NS::Prefix', is_prefix=True)
    def test_single_prefix_in_ns(self):
        # Define test classes
        ns = 'NS'
        prefix = 'TwoPrefix'
        class1 = prefix + 'Class1'
        fqn1 = ns + '::' + class1
        class2 = prefix + 'Class2'
        fqn2 = ns + '::' + class2
        ns_prefix = ns + '::' + prefix
        for class_name in class1, class2:
            self._define_class(class_name, ns)

        executed = []

        # Define pythonizor function for prefix
        @pythonization(prefix, ns=ns, is_prefix=True)
        def my_func(klass, name):
            self.assertTrue(klass.__cpp_name__.startswith(ns_prefix))
            self.assertTrue(name.startswith(ns_prefix))
            executed.append(name)

        # Trigger execution of pythonizor
        getattr(getattr(ROOT, ns), class1)
        getattr(getattr(ROOT, ns), class2)

        self.assertEqual(executed.pop(0), fqn1)
        self.assertEqual(executed.pop(0), fqn2)
        self.assertTrue(not executed)

    # Test @pythonization(['OneClass', 'AnotherClass'])
    def test_multiple_classes_global(self):
        # Define test classes
        class1 = 'MyClass2'
        class2 = 'MyClass3'
        for class_name in class1, class2:
            self._define_class(class_name)

        executed = []
        target_classes = [ class1, class2 ]

        # Define pythonizor function for classes
        @pythonization(target_classes)
        def my_func(klass, name):
            self.assertTrue(klass.__cpp_name__ in target_classes)
            self.assertTrue(name in target_classes)
            executed.append(name)

        # Trigger execution of pythonizor
        for class_name in class1, class2:
            getattr(ROOT, class_name)

        for class_name in class1, class2:
            self.assertEqual(executed.pop(0), class_name)
        self.assertTrue(not executed)

    # Test @pythonization(['OneClass', 'AnotherClass'], ns='NS')
    def test_multiple_classes_in_ns(self):
        # Define test classes
        class1 = 'MyClass4'
        class2 = 'MyClass5'
        ns = 'NS'
        fqns = []
        for class_name in class1, class2:
            self._define_class(class_name, ns)
            fqns.append(ns + '::' + class_name)

        executed = []
        target_classes = [ class1, class2 ]

        # Define pythonizor function for classes
        @pythonization(target_classes, ns=ns)
        def my_func(klass, name):
            self.assertTrue(klass.__cpp_name__ in fqns)
            self.assertTrue(name in fqns)
            executed.append(name)

        # Trigger execution of pythonizor
        for class_name in class1, class2:
            getattr(getattr(ROOT, ns), class_name)

        for class_name in fqns:
            self.assertEqual(executed.pop(0), class_name)
        self.assertTrue(not executed)

    # Test @pythonization(['OnePrefix', 'AnotherPrefix'], is_prefix=True)
    def test_multiple_prefixes_global(self):
        # Define test classes
        prefix1 = 'ThreePrefix'
        prefix2 = 'FourPrefix'
        class1 = prefix1 + 'Class1'
        class2 = prefix2 + 'Class2'
        for class_name in class1, class2:
            self._define_class(class_name)

        executed = []
        target_prefixes = [ prefix1, prefix2 ]

        # Define pythonizor function for prefixes
        @pythonization(target_prefixes, is_prefix=True)
        def my_func(klass, name):
            self.assertTrue(any(klass.__cpp_name__.startswith(p) for p in target_prefixes))
            self.assertTrue(any(name.startswith(p) for p in target_prefixes))
            executed.append(name)

        # Trigger execution of pythonizor
        for class_name in class1, class2:
            getattr(ROOT, class_name)

        for class_name in class1, class2:
           self.assertEqual(executed.pop(0), class_name)
        self.assertTrue(not executed)

    # Test @pythonization(['OnePrefix', 'AnotherPrefix'], ns='NS', is_prefix=True)
    def test_multiple_prefixes_in_ns(self):
        # Define test classes
        ns = 'NS'
        prefix1 = 'FivePrefix'
        prefix2 = 'SixPrefix'
        class1 = prefix1 + 'Class1'
        class2 = prefix2 + 'Class2'
        ns_prefixes = []
        for class_name in class1, class2:
            self._define_class(class_name, ns)
        for prefix in prefix1, prefix2:
            ns_prefixes.append(ns + '::' + prefix)

        executed = []
        target_prefixes = [ prefix1, prefix2 ]

        # Define pythonizor function for prefixes
        @pythonization(target_prefixes, ns=ns, is_prefix=True)
        def my_func(klass, name):
            self.assertTrue(any(klass.__cpp_name__.startswith(p) for p in ns_prefixes))
            self.assertTrue(any(name.startswith(p) for p in ns_prefixes))
            executed.append(name)

        # Trigger execution of pythonizor
        for class_name in class1, class2:
            getattr(getattr(ROOT, ns), class_name)

        for class_name in class1, class2:
           self.assertEqual(executed.pop(0), ns + '::' + class_name)
        self.assertTrue(not executed)

    # Test @pythonization('', is_prefix=True)
    def test_all_global_classes_prefix(self):
        # Define test classes
        class1 = 'Foo'
        class2 = 'Bar'
        for class_name in class1, class2:
            self._define_class(class_name)

        executed = []
        target_prefix = ''

        # Define pythonizor function for prefix.
        # Match all classes in global namespace
        @pythonization(target_prefix, is_prefix=True)
        def my_func(klass, name):
            if name not in self.exclude:
                self.assertTrue(klass.__cpp_name__.startswith(target_prefix))
                self.assertTrue(name.startswith(target_prefix))
                executed.append(name)

        # Trigger execution of pythonizor
        for class_name in class1, class2:
            getattr(ROOT, class_name)

        for class_name in class1, class2:
           self.assertEqual(executed.pop(0), class_name)
        self.assertTrue(not executed)

    # Test @pythonization('', ns='NS', is_prefix=True)
    def test_all_classes_in_ns_prefix(self):
        # Define test classes
        ns = 'NS'
        class1 = 'Foo'
        class2 = 'Bar'
        for class_name in class1, class2:
            self._define_class(class_name, ns)

        executed = []
        target_prefix = ''
        ns_prefix = ns + '::' + target_prefix

        # Define pythonizor function for prefix.
        # Match all classes in NS namespace
        @pythonization(target_prefix, ns=ns, is_prefix=True)
        def my_func(klass, name):
            self.assertTrue(klass.__cpp_name__.startswith(ns_prefix))
            self.assertTrue(name.startswith(ns_prefix))
            executed.append(name)

        # Trigger execution of pythonizor
        for class_name in class1, class2:
            getattr(getattr(ROOT, ns), class_name)

        for class_name in class1, class2:
            self.assertEqual(executed.pop(0), ns + '::' + class_name)
        self.assertTrue(not executed)

    # Test @pythonization(['MyClass', 'MyClass'])
    def test_repeated_targets(self):
        # Define test class
        class_name = 'MyClass6'
        self._define_class(class_name)

        executed = []
        target_classes = [ class_name, class_name ]

        # Define pythonizor function for classes
        @pythonization(target_classes)
        def my_func(klass, name):
            self.assertTrue(klass.__cpp_name__ in target_classes)
            self.assertTrue(name in target_classes)
            executed.append(name)

        # Trigger execution of pythonizor
        getattr(ROOT, class_name)

        self.assertEqual(executed.pop(0), class_name)
        # The following should be true since @pythonization removes repetitions
        self.assertTrue(not executed)

    # Test that @pythonization filters out non-matching classes
    def test_non_pythonized_classes(self):
        ns = 'NS'
        prefix = 'Mat'
        ns_prefix = ns + '::' + prefix
        class1 = 'Matches'
        class2 = 'DoesNotMatch'

        for class_name in class1, class2:
            self._define_class(class_name)
            self._define_class(class_name, ns)

        executed = []

        @pythonization(class1)
        def my_func1(klass, name):
            self.assertEqual(klass.__cpp_name__, class1)
            self.assertTrue(name, class1)
            executed.append(name)

        @pythonization(prefix, ns=ns, is_prefix=True)
        def my_func2(klass, name):
            self.assertTrue(klass.__cpp_name__.startswith(ns_prefix))
            self.assertTrue(name.startswith(ns_prefix))
            executed.append(name)

        # Trigger execution of pythonizors.
        # Non-matching classes should be filtered out
        for class_name in class1, class2:
            getattr(ROOT, class_name)
            getattr(getattr(ROOT, ns), class_name)

        self.assertEqual(executed.pop(0), class1)
        self.assertEqual(executed.pop(0), ns + '::' + class1)
        # The list should be empty if non-matching classes have been discarded
        self.assertTrue(not executed)

    # Test passing as target an iterable that is not a list
    def test_iterable(self):
        # Define test classes
        class1 = 'MyClass7'
        class2 = 'MyClass8'
        for class_name in class1, class2:
            self._define_class(class_name)

        executed = []
        target_classes_tuple = ( class1, class2 )

        # Define pythonizor function for classes
        @pythonization(target_classes_tuple)
        def my_func(klass, name):
            self.assertTrue(klass.__cpp_name__ in target_classes_tuple)
            self.assertTrue(name in target_classes_tuple)
            executed.append(name)

        # Trigger execution of pythonizor
        for class_name in class1, class2:
            getattr(ROOT, class_name)

        for class_name in class1, class2:
            self.assertEqual(executed.pop(0), class_name)
        self.assertTrue(not executed)

    # Test pythonizor with a single parameter (the class proxy)
    def test_single_parameter_pythonizor(self):
        # Define test class
        class_name = 'MyClass9'
        self._define_class(class_name)

        executed = []

        # Define pythonizor function for class
        @pythonization(class_name)
        def my_func(klass):
            self.assertEqual(klass.__cpp_name__, class_name)
            executed.append(class_name)

        # Trigger execution of pythonizor
        getattr(ROOT, class_name)

        self.assertEqual(executed.pop(0), class_name)
        self.assertTrue(not executed)

    # Test pythonizor with wrong number of parameters
    def test_wrong_pars_pythonizor(self):
        # Define test class
        class_name = 'MyClass10'

        # Define pythonizor function for class.
        # Registration of pythonizor should fail
        with self.assertRaises(TypeError):
            @pythonization(class_name)
            def my_func(klass, name, wrong_par):
                pass

    # Test @pythonization where class_name (wrongly) includes namespace
    def test_wrong_class_name(self):
        # Define test class
        class_name = 'NS::MyClass11'

        # Define pythonizor function for class.
        # Registration of pythonizor should fail
        with self.assertRaises(ValueError):
            @pythonization(class_name)
            def my_func(klass, name):
                pass

    # Test stacking of @pythonization decorators
    def test_stacking_decorator(self):
        # Define test classes
        ns = 'NS'
        class_name = 'MyClass12'
        ns_class_name = ns + '::' + class_name
        self._define_class(class_name)
        self._define_class(class_name, ns)

        executed = []
        targets = [ class_name, ns_class_name ]

        # Stack two @pythonization
        @pythonization(class_name)
        @pythonization(class_name, ns=ns)
        def my_func(klass, name):
            self.assertTrue(klass.__cpp_name__ in targets)
            self.assertTrue(name in targets)
            executed.append(name)

        # Trigger execution of pythonizor
        getattr(ROOT, class_name)
        getattr(getattr(ROOT, ns), class_name)

        # The pythonizor should have been triggered twice
        self.assertEqual(executed.pop(0), class_name)
        self.assertEqual(executed.pop(0), ns_class_name)
        self.assertTrue(not executed)

    # Test pythonization of already instantiated classes
    def test_instantiated_classes(self):
        # Define test classes
        ns = 'NS'
        class_name = 'MyClass13'
        ns_class_name = ns + '::' + class_name
        self._define_class(class_name)
        self._define_class(class_name, ns)
        class_template = 'ClassTemplate'
        ROOT.gInterpreter.ProcessLine('template <class T> class {ct} {{ }};'
                                      .format(ct=class_template))
        templ_type = 'int'

        # Instantiate classes
        getattr(ROOT, class_name)
        getattr(getattr(ROOT, ns), class_name)
        getattr(ROOT, class_template)[templ_type]

        executed = []
        targets = [ class_name, ns_class_name ]

        # Immediate pythonization should happen.
        # Accesses classes are cached by cppyy using their class name as key in
        # their namespace
        @pythonization(class_name)
        @pythonization(class_name, ns=ns)
        def my_func1(klass, name):
            self.assertTrue(klass.__cpp_name__ in targets)
            self.assertTrue(name in targets)
            executed.append(name)

        self.assertTrue(class_name in executed)
        self.assertTrue(ns_class_name in executed)
        self.assertEqual(len(executed), 2)
        executed.clear()

        # Immediate pythonization should happen.
        # Instantiated templates are also tested because they are cached by
        # cppyy using their fully-qualified name as key in their namespace
        @pythonization(class_template, is_prefix=True)
        def my_func2(klass, name):
            self.assertTrue(klass.__cpp_name__.startswith(class_template))
            self.assertTrue(name.startswith(class_template))
            executed.append(name)

        self.assertEqual(executed.pop(0), class_template + '<' + templ_type + '>')
        self.assertTrue(not executed)


if __name__ == '__main__':
    unittest.main()
