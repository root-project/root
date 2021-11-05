import unittest

import ROOT
from ROOT import pythonization


class PythonizationDecorator(unittest.TestCase):
    """
    Test the @pythonization decorator for user-defined classes.
    """

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
        @pythonization(fqn)
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
        prefix = 'Prefix'
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

        # Trigger execution of pythonizors
        getattr(ROOT, class1)
        getattr(ROOT, class2)

        self.assertEqual(executed.pop(0), class1)
        self.assertEqual(executed.pop(0), class2)
        self.assertTrue(not executed)

    # Test @pythonization('NS::Prefix', is_prefix=True)
    def test_single_prefix_in_ns(self):
        # Define test classes
        ns = 'NS'
        prefix = 'Prefix'
        class1 = prefix + 'Class1'
        fqn1 = ns + '::' + class1
        class2 = prefix + 'Class2'
        fqn2 = ns + '::' + class2
        ns_prefix = ns + '::' + prefix
        for class_name in class1, class2:
            self._define_class(class_name, ns)

        executed = []

        # Define pythonizor function for prefix
        @pythonization(ns_prefix, is_prefix=True)
        def my_func(klass, name):
            self.assertTrue(klass.__cpp_name__.startswith(ns_prefix))
            self.assertTrue(name.startswith(ns_prefix))
            executed.append(name)

        # Trigger execution of pythonizors
        getattr(getattr(ROOT, ns), class1)
        getattr(getattr(ROOT, ns), class2)

        self.assertEqual(executed.pop(0), fqn1)
        self.assertEqual(executed.pop(0), fqn2)
        self.assertTrue(not executed)

    # Test @pythonization(['MyClass', 'NS::MyClass'])
    def test_multiple_classes(self):
        # Define test classes
        class_name = 'MyClass2'
        ns = 'NS'
        fqn = ns + '::' + class_name
        self._define_class(class_name)
        self._define_class(class_name, ns)

        executed = []
        target_classes = [ class_name, fqn ]

        # Define pythonizor function for classes
        @pythonization(target_classes)
        def my_func(klass, name):
            self.assertTrue(klass.__cpp_name__ in target_classes)
            self.assertTrue(name in target_classes)
            executed.append(name)

        # Trigger execution of pythonizor
        getattr(ROOT, class_name)
        getattr(getattr(ROOT, ns), class_name)

        self.assertEqual(executed.pop(0), class_name)
        self.assertEqual(executed.pop(0), fqn)
        self.assertTrue(not executed)

    # Test @pythonization(['Prefix', 'NS::Prefix'], is_prefix=True)
    def test_multiple_prefixes(self):
        # Define test classes
        ns = 'NS'
        prefix = 'Prefix'
        ns_prefix = ns + '::' + prefix
        class1 = prefix + 'Class11'
        class2 = prefix + 'Class22'
        for class_name in class1, class2:
            self._define_class(class_name)
            self._define_class(class_name, ns)

        executed = []
        target_prefixes = [ prefix, ns_prefix ]

        # Define pythonizor function for prefixes
        @pythonization(target_prefixes, is_prefix=True)
        def my_func(klass, name):
            self.assertTrue(any(klass.__cpp_name__.startswith(p) for p in target_prefixes))
            self.assertTrue(any(name.startswith(p) for p in target_prefixes))
            executed.append(name)

        # Trigger execution of pythonizors
        for class_name in class1, class2:
            getattr(ROOT, class_name)
            getattr(getattr(ROOT, ns), class_name)

        for class_name in class1, class2:
           self.assertEqual(executed.pop(0), class_name)
           self.assertEqual(executed.pop(0), ns + '::' + class_name)
        self.assertTrue(not executed)

    # Test @pythonization(['', 'NS::'], is_prefix=True)
    def test_all_classes_in_ns_prefixes(self):
        # Define test classes
        ns = 'NS'
        class1 = 'Foo'
        class2 = 'Bar'
        for class_name in class1, class2:
            self._define_class(class_name)
            self._define_class(class_name, ns)

        executed = []
        target_prefixes = [ '',       # matches all classes in global namespace
                            ns + '::' # matches all classes in `ns` namespace
                          ]

        # Define pythonizor function for prefixes
        @pythonization(target_prefixes, is_prefix=True)
        def my_func(klass, name):
            self.assertTrue(any(klass.__cpp_name__.startswith(p) for p in target_prefixes))
            self.assertTrue(any(name.startswith(p) for p in target_prefixes))
            executed.append(name)

        # Trigger execution of pythonizors
        for class_name in class1, class2:
            getattr(ROOT, class_name)
            getattr(getattr(ROOT, ns), class_name)

        for class_name in class1, class2:
           self.assertEqual(executed.pop(0), class_name)
           self.assertEqual(executed.pop(0), ns + '::' + class_name)
        self.assertTrue(not executed)

    # Test @pythonization(['MyClass', 'MyClass', 'NS::MyClass', 'NS::MyClass'])
    def test_repeated_targets(self):
        # Define test classes
        class_name = 'MyClass3'
        ns = 'NS'
        fqn = ns + '::' + class_name
        self._define_class(class_name)
        self._define_class(class_name, ns)

        executed = []
        target_classes = [ class_name, class_name, fqn, fqn ]

        # Define pythonizor function for classes
        @pythonization(target_classes)
        def my_func(klass, name):
            self.assertTrue(klass.__cpp_name__ in target_classes)
            self.assertTrue(name in target_classes)
            executed.append(name)

        # Trigger execution of pythonizor
        getattr(ROOT, class_name)
        getattr(getattr(ROOT, ns), class_name)

        self.assertEqual(executed.pop(0), class_name)
        self.assertEqual(executed.pop(0), fqn)
        # The following should be true since @pythonization removes repetitions
        self.assertTrue(not executed)

    # Test that @pythonization filters out non-matching classes
    def test_non_pythonized_classes(self):
        ns1 = 'NS1'
        ns2 = 'NS2'
        prefix = 'Mat'
        class1 = 'Matches'
        class2 = 'DoesNotMatch'

        for class_name,ns in (class1,ns1), (class2,ns2):
            self._define_class(class_name)
            self._define_class(class_name, ns)

        executed = []
        target_classes = [ class1, ns1 + '::' + class1 ]
        target_prefixes = [ prefix, ns1 + '::' + prefix ]

        @pythonization(target_classes)
        def my_func1(klass, name):
            self.assertTrue(klass.__cpp_name__ in target_classes)
            self.assertTrue(name in target_classes)
            executed.append(name)

        @pythonization(target_prefixes, is_prefix=True)
        def my_func2(klass, name):
            self.assertTrue(any(klass.__cpp_name__.startswith(p) for p in target_prefixes))
            self.assertTrue(any(name.startswith(p) for p in target_prefixes))
            executed.append(name)

        # Trigger execution of pythonizors.
        # Non-matching classes should be filtered out
        for class_name, ns in (class1,ns1), (class2,ns2):
            getattr(ROOT, class_name)
            getattr(getattr(ROOT, ns), class_name)

        # Each first class access triggers the execution of the two pythonizors
        for _ in range(2):
            self.assertEqual(executed.pop(0), class1)
        for _ in range(2):
            self.assertEqual(executed.pop(0), ns1 + '::' + class1)
        self.assertTrue(not executed)

    # Test passing as target an iterable that is not a list
    def test_iterable(self):
        # Define test classes
        class1 = 'MyClass4'
        class2 = 'MyClass5'
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


if __name__ == '__main__':
    unittest.main()
