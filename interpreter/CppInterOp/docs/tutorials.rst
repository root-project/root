Tutorials
----------
This tutorial emphasises the abilities and usage of CppInterOp. Let's get 
started! The tutorial demonstrates two examples, one in C and one in Python,
for interoperability.

**Note:This example library shown below is to illustrate the concept on which 
CppInterOp is based.** 

Python:
=======

.. code-block:: python

   libInterop = ctypes.CDLL(libpath, mode = ctypes.RTLD_GLOBAL)
   _cpp_compile = libInterop.Clang_Parse
   _cpp_compile.argtypes = [ctypes.c_char_p]

  # We are using ctypes for inducting our library, and *Clang_Parse*, which is 
  # part of the library, for parsing the C++ code.

.. code-block:: cpp

    Giving a glance at how the header file looks for our library :
    The header keeps our function declarations for the functions used in our
    library.

    # This basically parses our C++ code.
    void Clang_Parse(const char* Code);

    # Looks up an entity with the given name, possibly in the given Context.
    Decl_t Clang_LookupName(const char* Name, Decl_t Context);

    # Returns the address of a JIT'd function of the corresponding declaration.
    FnAddr_t Clang_GetFunctionAddress(Decl_t D);

    # Returns the name of any named decl (class, namespace) & template arguments
    std::string GetCompleteName(Decl_t A);

    # Allocates memory of underlying size of the passed declaration.
    void * Clang_CreateObject(Decl_t RecordDecl);

    # Instantiates a given templated declaration.
    Decl_t Clang_InstantiateTemplate(Decl_t D, const char* Name, const char* Args);

The C++ code that is to be used in Python comes under this below section. This 
code is parsed by the CppInterOp library in the previous snippet and further 
compilation goes on.

.. code-block:: cpp

   def cpp_compile(arg):
    return _cpp_compile(arg.encode("ascii"))

   cpp_compile(r"""\
   void* operator new(__SIZE_TYPE__, void* __p) noexcept;
   extern "C" int printf(const char*,...);
   class A {};
   class C {};
   struct B : public A {
   template<typename T, typename S, typename U>
   void callme(T, S, U*) { printf(" call me may B! \n"); }
   };
   """)

.. code-block:: python
   
   class CppInterOpLayerWrapper:
  _get_scope = libInterop.Clang_LookupName
  _get_scope.restype = ctypes.c_size_t
  _get_scope.argtypes = [ctypes.c_char_p]

  _construct = libInterop.Clang_CreateObject
  _construct.restype = ctypes.c_void_p
  _construct.argtypes = [ctypes.c_size_t]

  _get_template_ct = libInterop.Clang_InstantiateTemplate
  _get_template_ct.restype = ctypes.c_size_t
  _get_template_ct.argtypes = [ctypes.c_size_t, ctypes.c_char_p, ctypes.c_char_p]

  def _get_template(self, scope, name, args):
    return self._get_template_ct(scope, name.encode("ascii"), args.encode("ascii"))

  def get_scope(self, name):
    return self._get_scope(name.encode("ascii"))

  def get_template(self, scope, name, tmpl_args = [], tpargs = []):
    if tmpl_args:
      # Instantiation is explicit from full name
      full_name = name + '<' + ', '.join([a for a in tmpl_args]) + '>'
      meth = self._get_template(scope, full_name, '')
    elif tpargs:
      # Instantiation is implicit from argument types
      meth = self._get_template(scope, name, ', '.join([a.__name__ for a in tpargs]))
    return CallCPPFunc(meth)

  def construct(self, cpptype):
    return self._construct(cpptype)

The class CppInterOpLayerWrapper is supposed to provide a Python wrapper over
the cppinterop layer. Here, we have the functions *Clang_LookupName,
Clang_CreateObject and Clang_InstantiateTemplate* are being used, so these
are being wrapped to be used in the Python language.

We get to know the scope of the attribute of class by using get_scope. In a 
similar manner, we can use get_namespace.

.. code-block::python

   class TemplateWrapper:
   def __init__(self, scope, name):
      self._scope = scope
      self._name  = name

   def __getitem__(self, *args, **kwds):
      # Look up the template and return the overload.
      return gIL.get_template(
         self._scope, self._name, tmpl_args = args)

   def __call__(self, *args, **kwds):
      # Keyword arguments are not supported for this demo.
      assert not kwds

      # Construct the template arguments from the types and find the overload.
      ol = gIL.get_template(
         self._scope, self._name, tpargs = [type(a) for a in args])

      # Call actual method.
      ol(*args, **kwds)

In this example for instantiating templates, we need the wrapper for the
function being used, which is responsible for finding a template that matches
the arguments.

.. code-block::python

   gIL = CppInterOpLayerWrapper()

   def cpp_allocate(proxy):
   pyobj = object.__new__(proxy)
   proxy.__init__(pyobj)
   pyobj.cppobj = gIL.construct(proxy.handle)
   return pyobj


.. code-block::python

   if __name__ == '__main__':
   # create a couple of types to play with
   CppA = type('A', (), {
         'handle'  : gIL.get_scope('A'),
         '__new__' : cpp_allocate
   })
   h = gIL.get_scope('B')
   CppB = type('B', (CppA,), {
         'handle'  : h,
         '__new__' : cpp_allocate,
         'callme'  : TemplateWrapper(h, 'callme')
   })
   CppC = type('C', (), {
         'handle'  : gIL.get_scope('C'),
         '__new__' : cpp_allocate
   })

   # call templates
   a = CppA()
   b = CppB()
   c = CppC()

   b.callme(a, 42, c) 

* In the main, we create types to access the class attributes and the wrapper 
can be supplied as the parameter in the map of the type given.

* We are using a python wrapper around functions to be supplied to the map for 
the identification and usage of the function.

* Finally, the objects are created for the respective class and the desired
function is called, which is `callme` function in this case.

The complete example can found below:
`Example <https://github.com/compiler-research/pldi-tutorials-2023
/blob/main/examples/p3-ex4/instantiate_cpp_template.py>`_.

C:
===

Include **p3-ex4-lib.h**, which contains the declarations for the functions used
in this code. The detailed summary of header comes in the latter part.


The variable Code is given as a C-style string, it contains the C++ code
to be parsed. It has two classes, class `A` and a templated class `B` with a member
function callme.

   .. code-block:: C

      const char* Code = "void* operator new(__SIZE_TYPE__, void* __p) noexcept;"
                  "extern \"C\" int printf(const char*,...);"
                  "class A {};"
                  "\n #include <typeinfo> \n"
                  "class B {"
                  "public:"
                  "  template<typename T>"
                  "  void callme(T) {"
                  "    printf(\" Instantiated with [%s] \\n \", typeid(T).name());"
                  " }"
                  "};";

The main() begins with the call to **Clang_Parse** from interop library responsible
for parsing the provided C++ code.

Next there a number of initializations, **Instantiation** is initialized to zero,
it will be used to store the instantiated template. The **InstantiationArgs**
is initialized to "A", it will be used as the argument when instantiating the template.
`T` is initialized to zero, used to store the declaration of the type "T" used for
instantiation.

   .. code-block:: C

      Decl_t Instantiation = 0;
      const char * InstantiationArgs = "A";
      Decl_t TemplatedClass = Clang_LookupName("B", /*Context=*/0);
      Decl_t T = 0;

This snippet checks command-line arguments were provided by the argc arguments.
We take the first argument (`argv[1]`), parse it, then take the second argument
(`argv[2]`) using **Clang_LookupName**, and reassigns **InstantiationArgs** to
the third argument (`argv[3]`). In the else case, we decide to go with the "A".

The code proceeds to instantiate the template `B::callme` with the given
type, using the **Clang_InstantiateTemplate** function from the
library. The instantiated template is stored in the **Instantiation**.

   .. code-block:: C

      Instantiation = Clang_InstantiateTemplate(TemplatedClass, "callme", InstantiationArgs);


A function pointer **callme_fn_ptr** is declared with a type `fn_def` that represents
the function taking a `void*` argument and returning void. The result of
**Clang_GetFunctionAddress** is casted by the function pointer.

   .. code-block:: C

      typedef void (*fn_def)(void*);
      fn_def callme_fn_ptr = (fn_def) Clang_GetFunctionAddress(Instantiation);

The code then creates an object of type `A` using **Clang_CreateObject**, and the
pointer to this object is stored in `NewA`.

   .. code-block:: C

      void* NewA = Clang_CreateObject(T);

Then the function pointer **callme_fn_ptr** is called with the `NewA`, which
calls the member function `B::callme` with the instantiated type. Thus, the instantiation
happens with type `A` and we get the below result.

You get the **output** as :
   .. code-block:: bash

      Instantiated with [1A]

In conclusion, this C code uses the CppInterOp library to dynamically instantiate
templates and call member functions based on provided types. This example was to
show how we can instantiate templates, in a similar manner we can use the
CppInterOp library to use many other features and attributes of languages to
interoperate.

**Header File:**

We wrap the library functions within the extern "C" for its usage in C programs.
All the functions of the library which are to be used in the C program have to
be under the extern C for the compiler to know the C++ code within.

.. code-block:: C

   extern "C" {
   #endif // __cplusplus
   /// Process C++ code.
   ///
   void Clang_Parse(const char* Code);

   /// Looks up an entity with the given name, possibly in the given Context.
   ///
   Decl_t Clang_LookupName(const char* Name, Decl_t Context);

   /// Returns the address of a JIT'd function of the corresponding declaration.
   ///
   FnAddr_t Clang_GetFunctionAddress(Decl_t D);

   /// Allocates memory of underlying size of the passed declaration.
   ///
   void * Clang_CreateObject(Decl_t RecordDecl);

   /// Instantiates a given templated declaration.
   Decl_t Clang_InstantiateTemplate(Decl_t D, const char* Name, const char* Args);
   #ifdef __cplusplus
   }


The complete example can found below:
`Example <https://github.com/compiler-research/pldi-tutorials-2023/blob/main/examples/p3-ex4/p3-ex4.c>`_.