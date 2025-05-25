import ROOT
import numpy as np

def CreateData():
    """
    This function generates the root files of various datatypes with random values to test them.
    Datatypes could be generated are Strings, Char_t, UChar_t
    """
    # function to create random numbers.. gRandom did not give me signed integers
    @ROOT.Numba.Declare(['int', 'bool'], 'long')
    def random_long(bits, signed):
        if signed:
            low = -1*2**(bits - 1)
            high = 2**(bits - 1) -1
        else:
            low = 0
            high = 2**bits
        return np.random.randint(low, high)
    
    N = 100 # df with 100 entries
    df = ROOT.RDataFrame(N)

    col_name = "Short_t"
    df = df.Define(col_name, f"({col_name}) Numba::random_long(16, true)")

    col_name = "UShort_t"
    df = df.Define(col_name, f"({col_name}) Numba::random_long(16, false)")

    col_name = "Int_t"
    df = df.Define(col_name, f"({col_name}) Numba::random_long(32, true)") 

    col_name = "UInt_t"
    df = df.Define(col_name, f"({col_name}) Numba::random_long(32, false)")

    col_name = "Float_t"
    df = df.Define(col_name, f"({col_name}) gRandom->Gaus()")

    col_name = "Float16_t"
    df = df.Define(col_name, f"({col_name}) gRandom->Gaus()")

    col_name = "Double_t"
    df = df.Define(col_name, f"({col_name}) gRandom->Gaus()")

    col_name = "Double32_t"
    df = df.Define(col_name, f"({col_name}) gRandom->Gaus()")

    col_name = "Long64_t"
    df = df.Define(col_name, f"({col_name}) rdfentry_")

    col_name = "ULong64_t"
    df = df.Define(col_name, f"({col_name}) rdfentry_")

    col_name = "Long_t"
    df = df.Define(col_name, f"({col_name}) rdfentry_")

    col_name = "ULong_t"
    df = df.Define(col_name, f"({col_name}) rdfentry_")

    col_name = "Bool_t"
    df = df.Define(col_name, f"({col_name}) gRandom->Integer(2)")

    df.Snapshot("TestData", "./RDF_Filter_Pyz_TestData.root") 

def filter_general(col, x):
    return bool(col > x)

def filter_C(String, x):
    pass

def filter_B(Char_t, x):
    return bool(Char_t > x)

def filter_b(UChar_t, x):
    return bool(UChar_t > x)

def filter_S(Short_t, x):
    return bool(Short_t > x)

def filter_s(UShort_t, x):
    return bool(UShort_t > x)

def filter_I(Int_t, x):
    return bool(Int_t > x)

def filter_i(UInt_t, x):
    return bool(UInt_t > x)

def filter_F(Float_t, x):
    return bool(Float_t > x)

def filter_f(Float16_t, x):
    return bool(Float16_t > x)

def filter_D(Double_t, x):
    return bool(Double_t > x)

def filter_d(Double32_t, x):
    return bool(Double32_t > x)

def filter_L(Long64_t, x):
    return bool(Long64_t > x)

def filter_l(ULong64_t, x):
    return bool(ULong64_t > x)

def filter_G(Long_t, x):
    return bool(Long_t > x)

def filter_g(ULong_t, x):
    return bool(ULong_t > x)

def filter_O(Bool_t, x):
    return bool(x == Bool_t)

TREE_TYPES = ["String","Char_t", "UChar_t", "Short_t", "UShort_t", "Int_t", "UInt_t", "Float_t", "Float16_t", "Double_t",  "Double32_t", "Long64_t", "ULong64_t", "Long_t", "ULong_t", "Bool_t"]
TREE_SYMS = ['C', 'B', 'b', 'S', 's', 'I', 'i', 'F', 'f', 'D', 'd', 'L', 'l', 'G', 'g', 'O']  # 16 Data Types
TYPE_TO_SYMBOL = dict(zip(TREE_TYPES, TREE_SYMS))

filter_dict = {}
for i in TREE_SYMS:
    filter_dict[i] = eval("filter_" + i)
