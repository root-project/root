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

