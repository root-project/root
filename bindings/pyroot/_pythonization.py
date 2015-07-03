""" Pythonization API.
"""

# TODO: externalize this (have PythonizationScope and UserPythonization as
# globals here and picked up from this module

# TODO: set explicit export list

# TODO: move cast to cppyy.lowlevel or some sort

# TODO: remove all need for accessing _backend
def _set_backend( backend ):
   global _backend
   _backend = backend

#--- Pythonization factories --------------------------------------------

def set_pythonization_scope(scope):
   _backend.PythonizationScope = scope
   if scope not in _backend.UserPythonization:
      _backend.UserPythonization[scope] = []


def install_pythonization(pythonization):
   scope = _backend.PythonizationScope
   _backend.UserPythonizations[scope].append(pythonization)


def add_class_method(match_class, match_method, addition):
   class method_pythonizor:
      def __init__(self, match_class, match_method, backend):
         import re
         self.backend = backend
         self.match_class = re.compile(match_class)
         self.match_method = re.compile(match_method)

      def __call__(self, obj, name):
         if not self.match_class.match(name):
            return
         for k in obj.__dict__:
            tmp = getattr(obj, k)
            if self.match_method.match(k):
               if isinstance(tmp, self.backend.MethodProxy):
                  tmp.__add_overload__(addition)
   return method_pythonizor(match_class, match_method, _backend)


def compose_method(match_class, match_method, g):
   class composition_pythonizor:
      def __init__(self, match_class, match_method, g):
         import re
         self.match_class = re.compile(match_class)
         self.match_method = re.compile(match_method)
         self.g = g

      def __call__(self, obj, name):
         if not self.match_class.match(name):
            return
         g = self.g
         for k in obj.__dict__:
            if not self.match_method.match(k):
               continue
            f = getattr(obj, k)
            def h(self, *args, **kwargs):
               return g(self, f(self, *args, **kwargs))
            setattr(obj, k, h)
   return composition_pythonizor(match_class, match_method, g)


def decorate_method(match_class, match_method, decorator):
   class decoration_pythonizor:
      def __init__(self, match_class, match_method, decorator):
         import re
         self.match_class = re.compile(match_class)
         self.match_method = re.compile(match_method)
         self.decorator = decorator

      def __call__(self, obj, name):
         if not self.match_class.match(name):
            return
         for k in obj.__dict__:
            if not self.match_method.match(k):
               continue
            setattr(obj, k, self.decorator(getattr(obj, k)))
   return decoration_pythonizor(match_class, match_method, decorator)


def set_methodproxy_property(match_class, match_method, prop, value):
   class method_pythonizor:
      def __init__(self, match_class, match_method, prop, value, backend):
         import re
         self.backend = backend
         self.match_class = re.compile(match_class)
         self.match_method = re.compile(match_method)
         self.prop = prop
         self.value = value

      def __call__(self, obj, name):
         if not self.match_class.match(name):
            return
         for k in obj.__dict__:
            tmp = getattr(obj, k)
            if self.match_method.match(k):
               if isinstance(tmp, self.backend.MethodProxy):
                  setattr(tmp, self.prop, self.value)
   return method_pythonizor(match_class, match_method, prop, value, _backend)


def set_gil_policy(match_class, match_method, release_gil=True):
   return set_methodproxy_property(match_class, match_method, '_threaded', int(release_gil))


def make_property(match_class, match_get, match_set=None, match_del=None, prop_name=None):
   class property_pythonizor:
      def __init__(self, match_class, match_get, match_set, match_del, prop_name):
         import re
         self.match_class = re.compile(match_class)

         self.match_get = re.compile(match_get)
         match_many_getters = self.match_get.groups == 1

         if match_set:
            self.match_set = re.compile(match_set)
            match_many_setters = self.match_set.groups == 1
            if match_many_getters ^ match_many_setters:
               raise ValueError('Must match getters and setters equally')
         else:
            self.match_set = None

         if match_del:
            self.match_del = re.compile(match_del)
            match_many_deleters = self.match_del.groups == 1
            if match_many_getters ^ match_many_deleters:
               raise ValueError('Must match getters and deleters equally')
         else:
            self.match_del = None

         self.match_many = match_many_getters
         if not (self.match_many or prop_name):
            raise ValueError("If not matching properties by regex, need a property name with exactly one substitution field")
         if self.match_many and prop_name:
            if prop_name.format(').!:(') == prop_name:
               raise ValueError("If matching properties by regex and providing a property name, the name needs exactly one substitution field")

         self.prop_name = prop_name

      def make_get_del_proxy(self, getter):
         class proxy:
            def __init__(self, getter):
               self.getter = getter

            def __call__(self, obj):
               return getattr(obj, self.getter)()
         return proxy(getter)

      def make_set_proxy(self, setter):
         class proxy:
            def __init__(self, setter):
               self.setter = setter

            def __call__(self, obj, arg):
               return getattr(obj, self.setter)(arg)
         return proxy(setter)

      def __call__(self, obj, name):
         if not self.match_class.match(name):
            return

         names = []
         named_getters = {}
         named_setters = {}
         named_deleters = {}

         if not self.match_many:
            fget, fset, fdel = None, None, None

         for k in obj.__dict__:
            match = self.match_get.match(k)
            if match and hasattr(getattr(obj, k), '__call__'):
               if self.match_many:
                  name = match.group(1)
                  named_getters[name] = k
               else:
                  fget = self.make_get_del_proxy(k)
                  break

         if self.match_set:
            for k in obj.__dict__:
               match = self.match_set.match(k)
               if match and hasattr(getattr(obj, k), '__call__'):
                  if self.match_many:
                     name = match.group(1)
                     named_setters[name] = k
                  else:
                     fset = self.make_set_proxy(k)
                     break

         if self.match_del:
            for k in obj.__dict__:
               match = self.match_del.match(k)
               if match and hasattr(getattr(obj, k), '__call__'):
                  if self.match_many:
                     name = match.group(1)
                     named_deleters[name] = k
                  else:
                     fdel = self.make_get_del_proxy(k)
                     break

         if not self.match_many:
            new_prop = property(fget, fset, fdel)
            setattr(obj, self.prop_name, new_prop)
            return

         names += list(named_getters.keys())
         names += list(named_setters.keys())
         names += list(named_deleters.keys())
         names = set(names)

         properties = []
         for name in names:
            if name in named_getters:
               fget = self.make_get_del_proxy(named_getters[name])
            else:
               fget = None

            if name in named_setters:
               fset = self.make_set_proxy(named_setters[name])
            else:
               fset = None

            if name in named_deleters:
               fdel = self.make_get_del_proxy(named_deleters[name])
            else:
               fdel = None

            new_prop = property(fget, fset, fdel)
            if self.prop_name:
               prop_name = self.prop_name.format(name)
            else:
               prop_name = name

            setattr(obj, prop_name, new_prop)

   return property_pythonizor(match_class, match_get, match_set, match_del, prop_name)


def rename_attribute(match_class, orig_attribute, new_attribute, keep_orig=False):
   class attribute_pythonizor:
      def __init__(self, match_class, orig_attribute, new_attribute, keep_orig):
         import re
         self.match_class = re.compile(match_class)
         self.match_attr = re.compile(orig_attribute)
         self.new_attr = new_attribute
         self.keep_orig = keep_orig

      def __call__(self, obj, name):
         if not self.match_class.match(name):
            return
         for k in obj.__dict__:
            tmp = getattr(obj, k)
            if self.match_attr.match(k):
               setattr(obj, self.new_attr, tmp)
               if not self.keep_orig: delattr(obj, k)
   return method_pythonizor(match_class, orig_attribute, new_attribute, keep_orig)


def pin_type(derived_type, base_type):
   _backend.SetTypePinning(match_class, cast_to)


def make_interface(base_type):
   pin_type(base_type, base_type)


def ignore_type_pinning(some_type):
   _backend.IgnoreTypePinning(some_type)


def cast(some_object, new_type):
   return _backend.Cast(some_object, new_type)


def add_exception_mapping(cpp_exception, py_exception):
   _backend.UserExceptions[cpp_exception] = py_exception
