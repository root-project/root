# Author: Enric Tejedor CERN  06/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from libROOTPython import AddBranchAttrSyntax, SetBranchAddressPyz, BranchPyz

import cppyy
from ROOT import pythonization

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

def _SetBranchAddress(self, *args):
    # Modify the behaviour if args is (const char*, void*)
    res = SetBranchAddressPyz(self, *args)

    if res is None:
        # Fall back to the original implementation for the rest of overloads
        res = self._OriginalSetBranchAddress(*args)

    return res

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

# TTree.AsMatrix functionality
def _TTreeAsMatrix(self, columns=None, exclude=None, dtype="double", return_labels=False):
    """Read-out the TTree as a numpy array.

    Note that the reading is performed in multiple threads if the implicit
    multi-threading of ROOT is enabled.

    Parameters:
        columns: If None return all branches as columns, otherwise specify names in iterable.
        exclude: Exclude branches from selection.
        dtype: Set return data-type of numpy array.
        return_labels: Return additionally to the numpy array the names of the columns.

    Returns:
        array(, labels): Numpy array(, labels of columns)
    """

    # Import numpy lazily
    try:
        import numpy as np
    except:
        raise ImportError("Failed to import numpy during call of TTree.AsMatrix.")

    # Check that tree has entries
    if self.GetEntries() == 0:
        raise Exception("Tree {} has no entries.".format(self.GetName()))

    # Get all columns of the tree if no columns are specified
    if columns is None:
        columns = [branch.GetName() for branch in self.GetListOfBranches()]

    # Exclude columns
    if exclude == None:
        exclude = []
    columns = [col for col in columns if not col in exclude]

    if not columns:
        raise Exception("Arguments resulted in no selected branches.")

    # Check validity of branches
    supported_branch_dtypes = ["Float_t", "Double_t", "Char_t", "UChar_t", "Short_t", "UShort_t",
            "Int_t", "UInt_t", "Long64_t", "ULong64_t"]
    col_dtypes = []
    invalid_cols_notfound = []
    invalid_cols_dtype = {}
    invalid_cols_multipleleaves = {}
    invalid_cols_leafname = {}
    for col in columns:
        # Check that column exists
        branch = self.GetBranch(col)
        if branch == None:
            invalid_cols_notfound.append(col)
            continue

        # Check that the branch has only one leaf with the name of the branch
        leaves = [leaf.GetName() for leaf in branch.GetListOfLeaves()]
        if len(leaves) != 1:
            invalid_cols_multipleleaves[col] = len(leaves)
            continue
        if leaves[0] != col:
            invalid_cols_leafname[col] = len(leaves[0])
            continue

        # Check that the leaf of the branch has an arithmetic data-type
        col_dtype = self.GetBranch(col).GetLeaf(col).GetTypeName()
        col_dtypes.append(col_dtype)
        if not col_dtype in supported_branch_dtypes:
            invalid_cols_dtype[col] = col_dtype

    exception_template = "Reading of branch {} is not supported ({})."
    if invalid_cols_notfound:
        raise Exception(exception_template.format(invalid_cols_notfound, "branch not existent"))
    if invalid_cols_multipleleaves:
        raise Exception(exception_template.format([k for k in invalid_cols_multipleleaves], "branch has multiple leaves"))
    if invalid_cols_leafname:
        raise Exception(exception_template.format(
            [k for k in invalid_cols_leafname], "name of leaf is different from name of branch {}".format(
                [invalid_cols_leafname[k] for k in invalid_cols_leafname])))
    if invalid_cols_dtype:
        raise Exception(exception_template.format(
            [k for k in invalid_cols_dtype], "branch has unsupported data-type {}".format(
                [invalid_cols_dtype[k] for k in invalid_cols_dtype])))

    # Check that given data-type is supported
    supported_output_dtypes = ["int", "unsigned int", "long", "unsigned long", "float", "double"]
    if not dtype in supported_output_dtypes:
        raise Exception("Data-type {} is not supported, select from {}.".format(
            dtype, supported_output_dtypes))

    # Convert columns iterable to std.vector("string")
    columns_vector = cppyy.gbl.std.vector["string"](len(columns))
    for i, col in enumerate(columns):
        columns_vector[i] = col

    # Allocate memory for the read-out
    flat_matrix = cppyy.gbl.std.vector[dtype](self.GetEntries()*len(columns))

    # Read the tree as flat std.vector(dtype)
    tree_ptr = cppyy.gbl.ROOT.Internal.RDF.GetAddress(self)
    columns_vector_ptr = cppyy.gbl.ROOT.Internal.RDF.GetAddress(columns_vector)
    flat_matrix_ptr = cppyy.gbl.ROOT.Internal.RDF.GetVectorAddress[dtype](flat_matrix)
    jit_code = "ROOT::Internal::RDF::TTreeAsFlatMatrixHelper<{dtype}, {col_dtypes}>(*reinterpret_cast<TTree*>({tree_ptr}), *reinterpret_cast<std::vector<{dtype}>* >({flat_matrix_ptr}), *reinterpret_cast<std::vector<string>* >({columns_vector_ptr}));".format(
            col_dtypes = ", ".join(col_dtypes),
            dtype = dtype,
            tree_ptr = tree_ptr,
            flat_matrix_ptr = flat_matrix_ptr,
            columns_vector_ptr = columns_vector_ptr)
    cppyy.gbl.gInterpreter.Calc(jit_code)

    # Convert the std.vector(dtype) to a numpy array by memory-adoption and
    # reshape the flat array to the correct shape of the matrix
    flat_matrix_np = np.asarray(flat_matrix)
    reshaped_matrix_np = np.reshape(flat_matrix_np,
            (int(len(flat_matrix)/len(columns)), len(columns)))

    if return_labels:
        return (reshaped_matrix_np, columns)
    else:
        return reshaped_matrix_np

@pythonization()
def pythonize_ttree(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    to_pythonize = [ 'TTree', 'TChain' ]
    if name in to_pythonize:
        # Pythonizations that are common to TTree and its subclasses.
        # To avoid duplicating the same logic in the pythonizors of
        # the subclasses, inject the pythonizations for all the target
        # classes here.
        # TChain needs to be explicitly pythonized because it redefines
        # SetBranchAddress in C++. As a consequence, TChain does not
        # inherit TTree's pythonization for SetBranchAddress, which
        # needs to be injected to TChain too. This is not the case for
        # other classes like TNtuple, which will inherit all the
        # pythonizations added here for TTree.

        # Pythonic iterator
        klass.__iter__ = _TTree__iter__

        # tree.branch syntax
        AddBranchAttrSyntax(klass)

        # SetBranchAddress
        klass._OriginalSetBranchAddress = klass.SetBranchAddress
        klass.SetBranchAddress = _SetBranchAddress

        # Branch
        klass._OriginalBranch = klass.Branch
        klass.Branch = _Branch

        # AsMatrix
        klass.AsMatrix = _TTreeAsMatrix

    return True
