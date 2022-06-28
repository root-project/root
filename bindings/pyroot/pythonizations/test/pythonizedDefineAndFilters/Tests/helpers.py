import ROOT
import numpy as np

# function to create random numbers.. gRandom did not give me signed integers
def random_long(bits, signed):
    if signed:
        low = -1*2**(bits - 1)
        high = 2**(bits - 1) -1
    else:
        low = 0
        high = 2**bits
    # print(low, high)
    return np.random.randint(low, high)
ROOT.Numba.Declare(['int', 'bool'], 'long')(random_long)

def generate_integers(N, bits, signed):
    if signed:
        low = -2**(bits - 1)
        high = 2**(bits - 1) -1
    else:
        low = 0
        high = 2**bits

    if not (bits == 64 and signed == False):
        return np.random.randint(low, high, size  = N)
    import random 
    # NUmpy cannot generate them.
    x = []
    for i in range(N):
        x.append(random.randint(0, 2**64))
    return x

def find_type(i, df):
    col_type = column_types(df)
    cols = col_type.keys()
    if isinstance(i, int):
        return 'int'
    elif isinstance(i, float):
        return 'float'
    elif isinstance(i, bool):
        return 'bool'
    elif isinstance(i, str):
        # check if it is a column name
        if i not in cols:
            return 'str'
        else:
            t = col_type[i]
            if t in col_type:
                sym = map_col_to_sym[t]
                return map_tree_to_numbadeclares[sym]
            return t
    else:
        raise TypeError("Undefine type for {}".format(i))
    
def column_types(rdf):
    name_type = {}
    col_names = [str(c) for c in rdf.GetColumnNames()]
    for col in col_names:
        type = rdf.GetColumnType(col)
        name_type[col] = type
    return name_type

col_types = ["String","Char_t", "UChar_t", "Short_t", "UShort_t", "Int_t", "UInt_t", "Float_t", "Float16_t", "Double_t",  "Double32_t", "Long64_t", "ULong64_t", "Long_t", "ULong_t", "Bool_t"]
symbols = ['C', 'B', 'b', 'S', 's', 'I', 'i', 'F', 'f', 'D', 'd', 'L', 'l', 'G', 'g', 'O']  # 16 Data Types
map_col_to_sym = dict(zip(col_types, symbols))

map_trr_to_arr = {
    'C': None, 
    'B': 'b',
    'b': 'B',
    'S': 'i',
    's': 'I',
    'I': 'l',
    'i': 'L',
    'F': 'f',
    'f': 'f',
    'D': 'd',
    'd': 'd',
    'L': 'q',
    'l': 'Q',
    'G': 'q',
    'g': 'Q',
    'O': bool,
}

map_tree_to_numbadeclares = {
    'C': 'str', 
    'B': 'int',
    'b': 'unsigned int',
    'S': 'int',
    's': 'unsigned int',
    'I': 'int',
    'i': 'unsigned int',
    'F': 'float',
    'f': 'float',
    'D': 'double',
    'd': 'double',
    'L': 'long',
    'l': 'unsigned long',
    'G': 'long',
    'g': 'unsigned long',
    'O': 'bool',
}

