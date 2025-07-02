\defgroup Pythonizations Python interface
\brief Python-specific functionalities offered by ROOT

This page lists the so-called "pythonizations", that is those functionalities offered by ROOT for classes and functions which are specific to Python usage of the package and provide a more pythonic experience.

### Pythonization example 

This example shows how to use the `@pythonization` decorator to add extra
behaviour to C++ user classes that are used from Python via PyROOT.
Let's first define a new C++ class. In this tutorial, we will see how we can
"pythonize" this class, i.e. how we can add some extra behaviour to it to
make it more pythonic or easier to use from Python.
Note: In this example, the class is defined dynamically for demonstration
purposes, but it could also be a C++ class defined in some library or header.
For more information about loading C++ user code to be used from Python with
PyROOT, please see:
https://root.cern.ch/manual/python#loading-user-libraries-and-just-in-time-compilation-jitting

~~~{.py}
ROOT.gInterpreter.Declare("""
class MyClass {};
""")
~~~

Next, we define a pythonizor function: the function that will be responsible
for injecting new behaviour in our C++ class `MyClass`. 
To convert a given Python function into a pythonizor, we need to decorate it
with the @pythonization decorator. Such decorator allows us to define which
which class we want to pythonize by providing its class name and its
namespace (if the latter is not specified, it defaults to the global
namespace, i.e. '::').  
The decorated function - the pythonizor - must accept either one or two
parameters:
1. The class to be pythonized (proxy object where new behaviour can be
injected)
2. The fully-qualified name of that class (optional).   
Let's see all this with a simple example. Suppose I would like to define how
`MyClass` objects are represented as a string in Python (i.e. what would be
shown when I print that object). For that purpose, I can define the following
pythonizor function. There are two important things to be noted here:
- The @pythonization decorator has one argument that specifies our target
class is `MyClass`.
- The pythonizor function `pythonizor_of_myclass` provides and injects a new
implementation for `__str__`, the mechanism that Python provides to define
how to represent objects as strings. This new implementation
always returns the string "This is a MyClass object".

~~~{.py}
@pythonization('MyClass')
def pythonizor_of_myclass(klass):
    klass.__str__ = lambda o : 'This is a MyClass object'
~~~
Once we have defined our pythonizor function, let's see it in action.
We will now use the `MyClass` class for the first time from Python: we will
create a new instance of that class. At this moment, the pythonizor will
execute and modify the class - pythonizors are always lazily run when a given
class is used for the first time from a Python script.

~~~{.py}
my_object = ROOT.MyClass()
~~~

Since the pythonizor already executed, we should now see the new behaviour.
For that purpose, let's print `my_object` (should show "This is a MyClass
object").

~~~{.py}
print(my_object)
~~~

The previous example is just a simple one, but there are many ways in which a
class can be pythonized. Typical examples are the redefinition of dunder
methods (e.g. `__iter__` and `__next__` to make your objects iterable from
Python). If you need some inspiration, many ROOT classes are pythonized in
the way we just saw; their pythonizations can be seen at:
<https://github.com/root-project/root/tree/master/bindings/pyroot/pythonizations/python/ROOT/_pythonization>
The @pythonization decorator offers a few more options when it comes to
matching classes that you want to pythonize. We saw that we can match a
single class, but we can also specify a list of classes to pythonize.
The following code defines a couple of new classes:

~~~{.py}
ROOT.gInterpreter.Declare("""
namespace NS {
    class Class1 {};
    class Class2 {};
}
""")
~~~

Note that these classes belong to the `NS` namespace. As mentioned above, the
@pythonization decorator accepts a parameter with the namespace of the class
or classes to be pythonized. Therefore, a pythonizor that matches both classes
would look like this:

~~~{.py}
@pythonization(['Class1', 'Class2'], ns='NS')
def pythonize_two_classes(klass):
    klass.new_attribute = 1
~~~

Both classes will have the new attribute:

~~~{.py}
o1 = ROOT.NS.Class1()
o2 = ROOT.NS.Class2()
print("Printing new attribute")
for o in o1, o2:
    print(o.new_attribute)
~~~

In addition, @pythonization also accepts prefixes of classes in a certain
namespace in order to match multiple classes in that namespace. To signal that
what we provide to @pythonization is a prefix, we need to set the `is_prefix`
argument to `True` (default is `False`).    
A common case where matching prefixes is useful is when we have a templated
class and we want to pythonize all possible instantiations of that template.
For example, we can pythonize the `std::vector` (templated) class like so:

~~~{.py}
@pythonization('vector<', ns='std', is_prefix=True)
def vector_pythonizor(klass):
    # first_elem returns the first element of the vector if it exists
    klass.first_elem = lambda v : v[0] if v else None
~~~

Since we defined a prefix to do the match, the pythonization will be applied
both if we instantiate e.g. a vector of integers and a vector of doubles.

~~~{.py}
v_int = ROOT.std.vector['int']([1,2,3])
v_double = ROOT.std.vector['double']([4.,5.,6.])
print("First element of integer vector: {}".format(v_int.first_elem()))
print("First element of double vector: {}".format(v_double.first_elem()))
~~~

These are some examples of combinations of prefixes and namespaces and the
corresponding classes that they match:
- '' : all classes in the global namespace.
- '', ns='NS1::NS2' : all classes in the `NS1::NS2` namespace.
- 'Prefix' : classes whose name starts with `Prefix` in the global namespace.
- 'Prefix', ns='NS' : classes whose name starts with `Prefix` in the `NS`
namespace
Moreover, a pythonizor function can have a second optional parameter that
contains the fully-qualified name of the class being pythonized. This can be
useful e.g. if we would like to do some more complex filtering of classes in
our pythonizor, for instance using regular expressions.

~~~{.py}
@pythonization('pair<', ns='std', is_prefix=True)
def pair_pythonizor(klass, name):
    print('Pythonizing class ' + name)
~~~

The pythonizor above will be applied to any instantiation of `std::pair` - we
can see this with the print we did inside the pythonizor.
Note that we could use the `name` parameter to e.g. further filter which
particular instantiations we would like to pythonize.

~~~{.py}
p1 = ROOT.std.pair['int','int'](1,2) # prints 'Pythonizing class std::pair<int,int>'
p2 = ROOT.std.pair['int','double'](1,2.) # prints 'Pythonizing class std::pair<intdouble>'
~~~

Note that, to pythonize multiple classes in different namespaces, we can
stack multiple @pythonization decorators. For example, if we define these
classes:

~~~{.py}
ROOT.gInterpreter.Declare("""
class FirstClass {};
namespace NS {
    class SecondClass {};
}
""")
~~~

We can pythonize both of them with a single pythonizor function like so:

~~~{.py}
@pythonization('FirstClass')
@pythonization('SecondClass', ns='NS')
def pythonizor_for_first_and_second(klass, name):
    print('Executed for class ' + name)
~~~

If we now access both classes, we should see that the pythonizor runs twice.

~~~{.py}
f = ROOT.FirstClass()
s = ROOT.NS.SecondClass()
~~~

So far we have seen how pythonizations can be registered for classes that
have not been used yet. We have discussed how, in that case, the pythonizor
functions are executed lazily when their target class/es are used for the
first time in the application.
However, it can also happen that our target class/es have already been
accessed by the time we register a pythonization. In such a scenario, the
pythonizor is applied immediately (at registration time) to the target
class/es    
Let's see an example of what was just explained. We will define a new class
and immediately create an object of that class. We can check how the object
still does not have a new attribute `pythonized` that we are going to inject
in the next step.

~~~{.py}
ROOT.gInterpreter.Declare("""
class MyClass2 {};
""")
o = ROOT.MyClass2()
try:
    print(o.pythonized)
except AttributeError:
    print("Object has not been pythonized yet!")
~~~

After that, we will register a pythonization for `MyClass2`. Since the class
has already been used, the pythonization will happen right away.

~~~{.py}
@pythonization('MyClass2')
def pythonizor_for_myclass2(klass):
    klass.pythonized = True
~~~

Now our object does have the `pythonized` attribute:

~~~{.py}
print(o.pythonized) # prints True
~~~

### Pythonization printing example 
This example illustrates the pretty printing feature of PyROOT, which reveals
the content of the object if a string representation is requested, e.g., by
Python's print statement. The printing behaves similar to the ROOT prompt
powered by the C++ interpreter cling.   
Create an object with PyROOT

~~~{.py}
obj = ROOT.std.vector("int")(3)
for i in range(obj.size()):
    obj[i] = i
~~~

Print the object, which reveals the content. Note that `print` calls the special
method `__str__` of the object internally.

~~~{.py}
print(obj)
~~~

The output can be retrieved as string by any function that triggers the `__str__`
special method of the object, e.g., `str` or `format`.

~~~{.py}
print(str(obj))
print("{}".format(obj))
~~~

Note that the interactive Python prompt does not call `__str__`, it calls
`__repr__`, which implements a formal and unique string representation of
the object.

~~~{.py}
print(repr(obj))
obj
~~~

The print output behaves similar to the ROOT prompt, e.g., here for a ROOT histogram.

~~~{.py}
hist = ROOT.TH1F("name", "title", 10, 0, 1)
print(hist)
~~~

If cling cannot produce any nice representation for the class, we fall back to a
"<ClassName at address>" format, which is what `__repr__` returns

~~~{.py}
ROOT.gInterpreter.Declare('class MyClass {};')
m = ROOT.MyClass()
print(m)
print(str(m) == repr(m))
~~~
