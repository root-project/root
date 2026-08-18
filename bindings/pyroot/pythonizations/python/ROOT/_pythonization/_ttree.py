# Author: Enric Tejedor CERN  06/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
\pythondoc TTree

The TTree class has several additions for its use from Python, which are also
available in its subclasses e.g. TChain and TNtuple.

First, TTree instances are iterable in Python. Therefore, assuming `t` is
a TTree instance, we can do:
\code{.py}
for entry in t:
    x = entry.branch_name
    ...
\endcode

At each iteration, a new entry of the tree will be read. In the code above,
`entry` allows to access the branch values for the current entry. This can be
done with the syntax `entry.branch_name` or, if the branch name is incompatible
with Python naming rules, with e.g. "getattr(entry, '1_branch_name')".

<em>Please note</em> that iterating in Python can be slow, so only iterate over
a tree as described above if performance is not an issue or when dealing with
a small dataset. To read and process the entries of a tree in a much faster
way, please use ROOT::RDataFrame.

Two methods of TTree have been pythonized to facilitate their: TTree::Branch and
TTree::SetBranchAddress.

### Pythonization of TTree::Branch

The following example shows how we can create different types of branches of a TTree.
`Branch` links the new branch with a given Python object. It is therefore possible to
fill such object with the desired content before calling TTree::Fill.

\code{.py}
from array import array
import numpy as np
import ROOT

# We create the file and the tree
with ROOT.TFile("outfile.root", "RECREATE") as ofile:
    t = ROOT.TTree("mytree", "mytree")

    # Basic type branch (float) - use array of length 1
    n = array('f', [ 1.5 ])
    t.Branch('floatb', n, 'floatb/F')

    # Array branch - use array of length N
    N = 10
    a = array('d', N*[ 0. ])
    t.Branch('arrayb', a, 'arrayb[' + str(N) + ']/D')

    # Array branch - use NumPy array of length N
    npa = np.array(N*[ 0. ])
    t.Branch('nparrayb', npa, 'nparrayb[' + str(N) + ']/D')

    # std::vector branch
    v = ROOT.std.vector('double')(N*[ 0. ])
    t.Branch('vectorb0', v)

    # Class branch / struct in single branch
    cb = ROOT.TH1D("myHisto", "myHisto", 64, -4, 4)
    # This could have been any class known to ROOT, also custom
    #cb = ROOT.MyCustomClass()
    t.Branch('classb', cb)

    # Struct as leaflist. This is interpreted on the fly,
    # but could be known to ROOT by other means, such as
    # header inclusion or dictionary load.
    ROOT.gInterpreter.Declare('''
    struct MyStruct {
    int myint;
    float myfloat;
    };
    ''')
    ms = ROOT.MyStruct()
    t.Branch('structll', ms, 'myint/I:myfloat/F')

    # Store struct members individually
    ms = ROOT.MyStruct()
    # Use the `addressof` function in the ROOT module
    # to get the address of the struct members
    t.Branch('myintb', ROOT.addressof(ms, 'myint'), 'myint/I')
    t.Branch('myfloatb', ROOT.addressof(ms, 'myfloat'), 'myfloat/F')

    # Let's write one entry in our tree
    t.Fill()
    # Finally flush the content of the tree to the file
    t.Write()
\endcode

### Pythonization of TTree::SetBranchAddress

This section is to be considered for advanced users. Simple event
loops reading tree entries in Python can be performed as shown above.

Below an example is shown of reading different types tree branches.
Note that `SetBranchAddress` will just link a given branch with a
certain Python object; after that, in order to read the content of such
branch for a given TTree entry `x`, TTree::GetEntry(x) must be
invoked.

\code{.py}
from array import array
import numpy as np
import ROOT

with ROOT.TFile('outfile.root') as infile:

    t = infile['mytree']

    # Basic type branch (float) - use array of length 1
    n = array('f', [ 0. ])
    t.SetBranchAddress('floatb', n)

    # Array branch - use array of length N
    N = 10
    a = array('d', N*[ 0. ])
    t.SetBranchAddress('arrayb', a)

    # Array branch - use NumPy array of length N
    npa = np.array(N*[ 0. ])
    t.SetBranchAddress('nparrayb', a)

    # std::vector branch
    v = ROOT.std.vector('double')()
    t.SetBranchAddress('vectorb', v)

    # Class branch
    cb = ROOT.TH1D()
    # Any other class known to ROOT would have worked
    #cb = ROOT.MyClass()
    t.SetBranchAddress('classb', cb)

    # Struct as leaflist. This is interpreted on the fly,
    # but could be known to ROOT by other means, such as
    # header inclusion or dictionary load.
    ROOT.gInterpreter.Declare('''
    struct MyStruct {
    int myint;
    float myfloat;
    };
    ''')
    ms = ROOT.MyStruct()
    t.SetBranchAddress('structll', ms)

    t.GetEntry(0)
\endcode

\endpythondoc
"""

from . import pythonization
from ._memory_utils import (
    _constructor_releasing_ownership,
    _SetDirectory_SetOwnership,
    _should_give_up_ownership,
)
from ._rvec import _get_cpp_type_from_numpy_type


_branch_lookups = None
_branch_ptr_to_ptr = None


class _BranchLookups(object):
    """The entities that reading a branch needs, looked up once.

    Every name resolved through the ROOT module goes through the facade's
    __getattr__, which is far too expensive to repeat per branch access: it
    dominated the cost of `tree.branch` when these were looked up inline.
    None of them can change over the lifetime of a session, so they are
    resolved on first use and kept.
    """

    __slots__ = ("helpers", "branch_element", "branch_object", "leaf_element", "leaf_object", "instance", "ll")


def _lookups():
    """Return the cached branch lookups, doing the one-off setup if needed.

    TBranch::GetAddress() and TBranchElement::GetObject() return char*, which
    cppyy faithfully turns into a Python str, throwing the pointer value away.
    Neither class offers a void* accessor and fAddress is protected, so the
    addresses are only reachable through wrappers with a different return type.
    These are declared on first use rather than at import, so that sessions
    that never touch a branch do not pay for compiling them.
    """
    global _branch_lookups

    if _branch_lookups is None:
        import ROOT
        from cppyy import ll

        ROOT.gInterpreter.Declare("""
        namespace ROOT::Internal::PyROOT {
        inline intptr_t GetBranchAddress(TBranch *branch)
        {
           return reinterpret_cast<intptr_t>(branch->GetAddress());
        }
        inline intptr_t GetBranchAddressDeref(TBranch *branch)
        {
           char *address = branch->GetAddress();
           return address ? reinterpret_cast<intptr_t>(*reinterpret_cast<void **>(address)) : 0;
        }
        inline intptr_t GetBranchElementObject(TBranchElement *branch)
        {
           return reinterpret_cast<intptr_t>(branch->GetObject());
        }
        }
        """)

        lookups = _BranchLookups()
        lookups.helpers = ROOT.Internal.PyROOT
        lookups.branch_element = ROOT.TBranchElement.Class()
        lookups.branch_object = ROOT.TBranchObject.Class()
        lookups.leaf_element = ROOT.TLeafElement.Class()
        lookups.leaf_object = ROOT.TLeafObject.Class()
        lookups.instance = ROOT._cppyy.types.Instance
        lookups.ll = ll
        _branch_lookups = lookups

    return _branch_lookups


def _ptr_to_ptr_brancher():
    """Return the C++ helper that branches on the address of a pointer.

    The T** overloads of TTree::Branch want the address of the pointer to the
    object. Where that address lives depends on the kind of proxy holding it: a
    proxy for an object keeps the object pointer itself, a proxy for a reference
    to a pointer keeps the address of the caller's pointer. Deriving it here
    would mean restating that rule, so instead the helper takes a T** and lets
    cppyy apply the rule, as it does for any other C++ function taking one.

    Declared on first use rather than at import, so that sessions that never
    branch on an object do not pay for compiling it.
    """
    global _branch_ptr_to_ptr

    if _branch_ptr_to_ptr is None:
        import ROOT

        ROOT.gInterpreter.Declare("""
        namespace ROOT::Internal::PyROOT {
        template <class T>
        TBranch *BranchPtrToPtr(TTree &tree, const char *name, const char *className, T **obj,
                                Int_t bufsize = 32000, Int_t splitlevel = 99)
        {
           return tree.Branch(name, className, reinterpret_cast<void **>(obj), bufsize, splitlevel);
        }
        }
        """)
        _branch_ptr_to_ptr = ROOT.Internal.PyROOT.BranchPtrToPtr

    return _branch_ptr_to_ptr


# TTree iterator
def _TTree__iter__(self):
    i = 0
    bytes_read = self.GetEntry(i)
    while 0 < bytes_read:
        yield self
        i += 1
        bytes_read = self.GetEntry(i)

    if bytes_read == -1:
        raise RuntimeError("TTree I/O error")


def _pythonize_branch_addr(branch, addr_orig):
    """Helper for the SetBranchAddress pythonization, extracting the relevant
    address from a Python object if possible.
    """
    import ctypes

    import ROOT

    is_leaf_list = branch.IsA() is ROOT.TBranch.Class()

    if is_leaf_list:
        # If the branch is a leaf list, SetBranchAddress expects the
        # address of the object that has the corresponding data members.
        return ctypes.c_void_p(ROOT._cppyy.addressof(instance=addr_orig, byref=False))

    # Otherwise, SetBranchAddress is expecting a pointer to the address of
    # the object, and the pointer needs to stay alive. Therefore, we create
    # a container for the pointer and cache it in the original cppyy proxy.
    addr_view = ROOT.array["std::intptr_t", 1]([ROOT._cppyy.addressof(instance=addr_orig, byref=False)])

    if not hasattr(addr_orig, "_set_branch_cached_pointers"):
        addr_orig._set_branch_cached_pointers = []
    addr_orig._set_branch_cached_pointers.append(addr_view)

    # Finally, we have to return the address of the container
    return ctypes.c_void_p(ROOT._cppyy.addressof(instance=addr_view, byref=False))


def _get_cpp_type_from_array_typecode(typecode):
    # Complete list from https://docs.python.org/3/library/array.html
    c_type_names = {
        "b": "signed char",
        "B": "unsigned char",
        "u": "wchar_t",
        "h": "signed short",
        "H": "unsigned short",
        "i": "signed int",
        "I": "unsigned int",
        "l": "signed long",
        "L": "unsigned long",
        "q": "signed long long",
        "Q": "unsigned long long",
        "f": "float",
        "d": "double",
    }
    return c_type_names[typecode]


def _determine_data_type(addr):
    """Figure out data_type in case addr is a numpy.ndarray or array.array."""

    # For NumPy arrays
    if hasattr(addr, "__array_interface__"):
        return _get_cpp_type_from_numpy_type(addr.__array_interface__["typestr"][1:])

    # For the builtin array library
    if hasattr(addr, "buffer_info"):
        return _get_cpp_type_from_array_typecode(addr.typecode)

    return None


def _SetBranchAddress(self, bname, addr, *args, **kwargs):
    """
    Pythonization for TTree::SetBranchAddress.

    Modify the behaviour of SetBranchAddress so that proxy references can be passed
    as arguments from the Python side, more precisely in cases where the C++
    implementation of the method expects the address of a pointer.

    For example:
    ```
    v = ROOT.std.vector('int')()
    t.SetBranchAddress("my_vector_branch", v)
    ```
    """
    import cppyy

    import ROOT

    branch = self.GetBranch(bname)

    # Pythonization for cppyy proxies (of type CPPInstance)
    if isinstance(addr, ROOT._cppyy.types.Instance):
        addr = _pythonize_branch_addr(branch, addr)

    # Figure out data_type in case addr is a numpy.ndarray or array.array
    data_type = _determine_data_type(addr)

    if data_type is None:
        return self._OriginalSetBranchAddress(bname, addr, *args, **kwargs)

    # In the case the data_type is available, we would like to call the
    # template overload of SetBranchAddress instantiatied for that type.
    # However, there are two such overloads candidates:
    #
    #   template <class T> int TTree::SetBranchAddress(const char *bname, T **add, ...);
    #   template <class T> int TTree::SetBranchAddress(const char *bname, T *add, ...);
    #
    # The cppyy bindings can't make a meaningful selection here as Python is
    # lacking pointer semantics, so it considers both overloads as valid
    # choices. In the past, we just happened to be lucky that it tried the T *
    # overload first, which is the one we need. But as cppyy becomes more
    # strict about overload resolution ambiguity errors, this won't work
    # anymore. That's why we re-implement what happens in the template overload
    # on the Python side.

    cl = ROOT.TClass.GetClass[data_type]()
    tp = ROOT.kOther_t
    if not cl:
        tp = ROOT.TDataType.GetType(cppyy.typeid(getattr(ROOT, data_type)))

    # Extract the TBranch ptr argument if available
    tbranch_ptr = ROOT.nullptr
    if len(args) > 0:
        tbranch_ptr = args[0]
    elif "ptr" in kwargs:
        tbranch_ptr = kwargs["ptr"]

    return self._OriginalSetBranchAddress(bname, addr, ptr=tbranch_ptr, realClass=cl, datatype=tp, isptr=False)


def _get_address_of(obj):
    """Return the address of the buffer or proxied object `obj` points at.

    Returns None for anything that does not carry an address, and deliberately
    also for text-like objects, which do but are never meant as a branch buffer.
    """
    import ROOT

    if isinstance(obj, ROOT._cppyy.types.Instance):
        return ROOT._cppyy.addressof(instance=obj, byref=False)

    if isinstance(obj, (str, bytes)):
        return None

    try:
        return ROOT._cppyy.ll.addressof(obj)
    except TypeError:
        return None


def _try_branch_leaf_list_overload(self, args):
    """Try to match TTree::Branch(const char*, void*, const char*, Int_t = 32000)."""
    if not (3 <= len(args) <= 4):
        return None
    name, address, leaflist = args[0], args[1], args[2]
    if not isinstance(name, str) or not isinstance(leaflist, str):
        return None
    if len(args) == 4 and not isinstance(args[3], int):
        return None

    import ctypes

    buf = _get_address_of(address)
    if not buf:
        return None

    return self._OriginalBranch(name, ctypes.c_void_p(buf), leaflist, *args[3:])


def _try_branch_ptr_to_ptr_overloads(self, args):
    """Try to match one of the TTree::Branch overloads taking a T**:

    - ( const char*, const char*, T**, Int_t = 32000, Int_t = 99 )
    - ( const char*,              T**, Int_t = 32000, Int_t = 99 )
    """
    import ROOT

    if len(args) < 2 or not isinstance(args[0], str):
        return None

    name = args[0]
    if isinstance(args[1], str):
        # the class name is given explicitly
        class_name, address, rest = args[1], args[2] if len(args) > 2 else None, args[3:]
    else:
        class_name, address, rest = None, args[1], args[2:]

    if address is None or any(not isinstance(arg, int) for arg in rest) or len(rest) > 2:
        return None

    if isinstance(address, ROOT._cppyy.types.Instance):
        # Hand the proxy to a helper taking a T** and let cppyy work out which
        # address that is. T is the proxied type rather than class_name, which
        # the caller is free to give as a base of it, or as an equivalent
        # spelling that is not the one cppyy knows the proxy by.
        proxy_type = type(address).__cpp_name__
        return _ptr_to_ptr_brancher()[proxy_type](self, name, class_name or proxy_type, address, *rest)

    buf = _get_address_of(address)
    if not buf or not class_name:
        return None

    import ctypes

    return self._OriginalBranch(name, class_name, ctypes.c_void_p(buf), *rest)


def _Branch(self, *args):
    """
    Pythonization for TTree::Branch.

    Modify the behaviour of Branch so that proxy references can be passed as
    arguments from the Python side, more precisely in cases where the C++
    implementation of the method expects the address of a pointer.

    For example:
    ```
    v = ROOT.std.vector('int')()
    t.Branch('my_vector_branch', v)
    ```

    The following signatures are treated in this pythonization:
    - ( const char*, void*, const char*, Int_t = 32000 )
    - ( const char*, const char*, T**, Int_t = 32000, Int_t = 99 )
    - ( const char*, T**, Int_t = 32000, Int_t = 99 )
    """
    if len(args) >= 2:
        res = _try_branch_leaf_list_overload(self, args)
        if res is not None:
            return res

        res = _try_branch_ptr_to_ptr_overloads(self, args)
        if res is not None:
            return res

    # Fall back to the original implementation for the rest of overloads
    return self._OriginalBranch(*args)


def _search_for_branch(tree, name):
    branch = tree.GetBranch(name)
    if not branch:
        # for benefit of naming of sub-branches, the actual name may have a
        # trailing '.'
        branch = tree.GetBranch(name + ".")
    return branch


def _has_single_leaf(branch):
    leaves = branch.GetListOfLeaves()
    # i.e. if unambiguously only this one
    return bool(leaves.GetSize()) and leaves.First() == leaves.Last()


def _search_for_leaf(tree, name, branch):
    leaf = tree.GetLeaf(name)
    if branch and not leaf:
        leaf = branch.GetLeaf(name)
        if not leaf and _has_single_leaf(branch):
            leaf = branch.GetListOfLeaves().At(0)
    return leaf


def _resolve_branch(tree, name, branch):
    """Return the address and type name of the object held by a branch.

    Returns (None, "") if the branch does not hold an object, in which case the
    caller should look for a leaf instead. An address of 0 with a non-empty type
    name means failure, and is reported to the user as a typed null object.
    """
    lookups = _lookups()
    helpers = lookups.helpers

    # for partial return of a split object
    if branch.InheritsFrom(lookups.branch_element):
        current_class = branch.GetCurrentClass()
        if current_class and current_class != branch.GetTargetClass() and branch.GetID() >= 0:
            offset = branch.GetInfo().GetElements().At(branch.GetID()).GetOffset()
            return helpers.GetBranchElementObject(branch) + offset, current_class.GetName()

    # for return of a full object
    if branch.IsA() in (lookups.branch_element, lookups.branch_object):
        if helpers.GetBranchAddress(branch):
            return helpers.GetBranchAddressDeref(branch), branch.GetClassName()

        # try leaf, otherwise indicate failure by returning a typed null object
        if not tree.GetLeaf(name) and not _has_single_leaf(branch):
            return 0, branch.GetClassName()

    return None, ""


def _get_multi_dims(title):
    """Extract the static dimensions from the title of a TLeaf.

    The title of a multi-dimensional leaf carries its dimensions as
    `name[dim1][dim2]...`. In the current implementation of TLeaf there is no
    way to get at them other than by parsing that string.
    """
    import re

    return [int(dim) for dim in re.findall(r"\[([^\]]*)\]", title) if dim]


def _wrap_leaf(leaf):
    """Read the value of a leaf for the entry the tree is currently on."""
    lookups = _lookups()
    ll = lookups.ll

    if leaf.GetLenStatic() > 1 or leaf.GetLeafCount():
        # array types
        is_static = leaf.GetLenStatic() > 1
        type_name = leaf.GetTypeName()

        dims = [leaf.GetNdata()]
        title = leaf.GetTitle()
        if title.count("[") >= 2:
            # multidimensional array case
            dims = _get_multi_dims(title)

        address = 0
        branch = leaf.GetBranch()
        if branch:
            address = lookups.helpers.GetBranchAddress(branch)
        if not address:
            address = ll.addressof(leaf.GetValuePointer())

        return ll.value_from_memory(type_name + ("[]" if is_static else "*"), address, dims)

    value_pointer = leaf.GetValuePointer()
    if value_pointer:
        # value types
        address = ll.addressof(value_pointer)
        if leaf.IsA() in (lookups.leaf_element, lookups.leaf_object):
            # the leaf holds a pointer to the value, rather than the value
            address = ll.value_from_memory("intptr_t", address)
        return ll.value_from_memory(leaf.GetTypeName(), address)

    return None


def _TTree__getattr__(self, key):
    """
    Allow branches to be accessed as attributes of a tree.

    Allow access to branches/leaves as if they were Python data attributes of
    the tree (e.g. mytree.branch).

    Parameters:
    self (TTree): The instance of the TTree object from which the attribute is being retrieved.
    key (str): The name of the branch to retrieve from the TTree object.
    """
    ll = _lookups().ll

    # deal with possible aliasing
    name = self.GetAlias(key) or key

    # search for branch first (typical for objects)
    branch = _search_for_branch(self, name)

    if branch:
        # found a branched object, wrap its address for the object it represents
        address, type_name = _resolve_branch(self, name, branch)
        if type_name:
            return ll.cast[type_name + "*"](address)

    # if not, try leaf
    leaf = _search_for_leaf(self, name, branch)
    if leaf:
        # found a leaf, extract value and wrap with a Python object
        # according to its type
        value = _wrap_leaf(leaf)
        if value is not None:
            return value

    # confused
    raise AttributeError("'{}' object has no attribute '{}'".format(self.IsA().GetName(), name))


def _TTree_CloneTree(self, *args, **kwargs):
    """
    Forward the arguments to the C++ function and give up ownership if the
    TTree is attached to a TFile, which is the owner in that case.
    """
    import ROOT

    out_tree = self._CloneTree(*args, **kwargs)
    if _should_give_up_ownership(out_tree):
        ROOT.SetOwnership(out_tree, False)

    return out_tree


@pythonization("TTree")
def pythonize_ttree(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    # Functions that need to drop the ownership if the current directory is a TFile

    klass._cpp_constructor = klass.__init__
    klass.__init__ = _constructor_releasing_ownership

    klass._CloneTree = klass.CloneTree
    klass.CloneTree = _TTree_CloneTree

    # Pythonizations that are common to TTree and its subclasses.
    # To avoid duplicating the same logic in the pythonizors of
    # the subclasses, inject the pythonizations for all the target
    # classes here.

    # Pythonic iterator
    klass.__iter__ = _TTree__iter__

    # tree.branch syntax
    klass.__getattr__ = _TTree__getattr__

    # SetBranchAddress
    klass._OriginalSetBranchAddress = klass.SetBranchAddress
    klass.SetBranchAddress = _SetBranchAddress

    # Branch
    klass._OriginalBranch = klass.Branch
    klass.Branch = _Branch

    klass._Original_SetDirectory = klass.SetDirectory
    klass.SetDirectory = _SetDirectory_SetOwnership


@pythonization("TChain")
def pythonize_tchain(klass):
    # Parameters:
    # klass: class to be pythonized

    # TChain needs to be explicitly pythonized because it redefines
    # SetBranchAddress in C++. As a consequence, TChain does not
    # inherit TTree's pythonization for SetBranchAddress, which
    # needs to be injected to TChain too. This is not the case for
    # other classes like TNtuple, which will inherit all the
    # pythonizations added here for TTree.

    # SetBranchAddress
    klass._OriginalSetBranchAddress = klass.SetBranchAddress
    klass.SetBranchAddress = _SetBranchAddress


@pythonization("TNtuple")
def pythonize_tntuple(klass):

    # The constructor needs to be explicitly pythonized for derived classes.
    klass._cpp_constructor = klass.__init__
    klass.__init__ = _constructor_releasing_ownership
