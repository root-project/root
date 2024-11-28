# Author: Enric Tejedor CERN  06/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
/**
\class TTree
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly
## PyROOT

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

Second, a couple of TTree methods have been modified to facilitate their use
from Python: TTree::Branch and TTree::SetBranchAddress.

Regarding TTree::Branch, the following example shows how we can create
different types of branches of a TTree. Note that `Branch` will just link
the new branch with a given Python object, so it is still necessary to fill
such object with the desired content before calling TTree::Fill.
\code{.py}
from array import array
import numpy as np
import ROOT
from ROOT import addressof

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
cb = ROOT.MyClass()
t.Branch('classb', cb)

# Struct as leaflist
# Assuming:
# struct MyStruct {
#   int myint;
#   float myfloat;
# };
ms = ROOT.MyStruct()
t.Branch('structll', ms, 'myint/I:myfloat/F')

# Store struct members individually
ms = ROOT.MyStruct()
# Use `addressof` to get the address of the struct members
t.Branch('myintb', addressof(ms, 'myint'), 'myint/I')
t.Branch('myfloatb', addressof(ms, 'myfloat'), 'myfloat/F')
\endcode

Concerning TTree::SetBranchAddress, below is an example of prepare
the reading of different types of branches of a TTree. Note that
`SetBranchAddress` will just link a given branch with a certain
Python object; after that, in order to read the content of such
branch for a given TTree entry `x`, TTree::GetEntry(x) must be
invoked.
\code{.py}
from array import array
import numpy as np
import ROOT

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
cb = ROOT.MyClass()
t.SetBranchAddress('classb', cb)

# Struct branch (both single-branch and leaf list)
ms = ROOT.MyStruct()
ds.SetBranchAddress('structb', ms)
\endcode
\htmlonly
</div>
\endhtmlonly
*/
"""

from libROOTPythonizations import GetBranchAttr, BranchPyz
from ._rvec import _array_interface_dtype_map, _get_cpp_type_from_numpy_type
from . import pythonization


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
    import cppyy
    import ctypes

    is_leaf_list = branch.IsA() is cppyy.gbl.TBranch.Class()

    if is_leaf_list:
        # If the branch is a leaf list, SetBranchAddress expects the
        # address of the object that has the corresponding data members.
        return ctypes.c_void_p(cppyy.addressof(instance=addr_orig, byref=False))

    # Otherwise, SetBranchAddress is expecting a pointer to the address of
    # the object, and the pointer needs to stay alive. Therefore, we create
    # a container for the pointer and cache it in the original cppyy proxy.
    addr_view = cppyy.gbl.array["std::intptr_t", 1]([cppyy.addressof(instance=addr_orig, byref=False)])

    if not hasattr(addr_orig, "_set_branch_cached_pointers"):
        addr_orig._set_branch_cached_pointers = []
    addr_orig._set_branch_cached_pointers.append(addr_view)

    # Finally, we have to return the address of the container
    return ctypes.c_void_p(cppyy.addressof(instance=addr_view, byref=False))


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
    """ Figure out data_type in case addr is a numpy.ndarray or array.array.
    """

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

    branch = self.GetBranch(bname)

    # Pythonization for cppyy proxies (of type CPPInstance)
    if isinstance(addr, cppyy._backend.CPPInstance):
        addr = _pythonize_branch_addr(branch, addr)

    # Figure out data_type in case addr is a numpy.ndarray or array.array
    data_type = _determine_data_type(addr)

    # We call the template specialization if we know the data type
    func = self._OriginalSetBranchAddress if data_type is None else self._OriginalSetBranchAddress[data_type]

    return func(bname, addr, *args, **kwargs)


def _Branch(self, *args):
    # Modify the behaviour if args is one of:
    # ( const char*, void*, const char*, Int_t = 32000 )
    # ( const char*, const char*, T**, Int_t = 32000, Int_t = 99 )
    # ( const char*, T**, Int_t = 32000, Int_t = 99 )
    res = BranchPyz(self, *args)

    if res is None:
        # Fall back to the original implementation for the rest of overloads
        res = self._OriginalBranch(*args)

    return res


def _TTree__getattr__(self, key):
    """
    Allow branches to be accessed as attributes of a tree.

    Allow access to branches/leaves as if they were Python data attributes of
    the tree (e.g. mytree.branch).

    To avoid using the CPyCppyy API, any necessary cast is done here on the
    Python side. The GetBranchAttr() function encodes a necessary cast in the
    second element of the output tuple, which is a string with the required
    type name.

    Parameters:
    self (TTree): The instance of the TTree object from which the attribute is being retrieved.
    key (str): The name of the branch to retrieve from the TTree object.
    """

    import cppyy.ll

    out, cast_type = GetBranchAttr(self, key)
    if cast_type:
        out = cppyy.ll.cast[cast_type](out)
    return out

def _TTree_Constructor(self, *args, **kwargs):
    """
    Forward the arguments to the C++ constructor and give up ownership if the
    TTree is attached to a TFile, which is the owner in that case.
    """
    import ROOT

    self._cpp_constructor(*args, **kwargs)
    tdir = self.GetDirectory()
    if tdir and type(tdir).__cpp_name__ == "TFile":
        ROOT.SetOwnership(self, False)

@pythonization("TTree")
def pythonize_ttree(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    klass._cpp_constructor = klass.__init__
    klass.__init__ = _TTree_Constructor

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
def pythonize_tchain(klass):

    # The constructor needs to be explicitly pythonized for derived classes.
    klass._cpp_constructor = klass.__init__
    klass.__init__ = _TTree_Constructor
