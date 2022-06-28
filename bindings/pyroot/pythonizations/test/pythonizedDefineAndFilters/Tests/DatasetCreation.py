from random import uniform
import ROOT
import os
from helpers import map_col_to_sym, map_trr_to_arr, generate_integers

def CreateData():
    if not os.path.isdir("./Data"):
        os.makedirs("./Data")    
    N = 100 # df with 100 entries

    df = ROOT.RDataFrame(N)

    """
    Cannot test String, Char_t and UChar_t right now as:
    1. The generation of it (which I assumed would be equiavlent to 8bit integers is false).
    2. They cannot be converted into numpy arrays thus hindering my current "test conditions" which takes the mean of the entries.
    3. Issue in file issues.py 
    """
    # col_name = "Char_t"
    # df = df.Define(col_name, f"({col_name}) Numba::random_long(8, true)")#.Snapshot(col_name, "./Data/" + col_name + ".root") 
    # col_name = "UChar_t"
    # df = df.Define(col_name, f"({col_name}) Numba::random_long(8, false)")#.Snapshot(col_name, "./Data/" + col_name + ".root") 

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

    """
    I needed an alternative way to generate Long64 types as:
    1. COuld not use gRandom as gRandom only returns long and not long long it gave me 0s and the range was invalid
    2. Doubts as to the differences between types Long64_t and Long_t ...
    WorkAround:
    I put the numbers generated in an array and put filled the tree with that rather than use define.
    """
    # df = df.Define("Long64_t", "(Long64_t) Numba::random_long(64, true)")
    # df = df.Define("ULong64_t", "(Long64_t) Numba::random_long(64, true)")
    # df = df.Define("Long_t", "(Long_t) gRandom->Integer(pow(2,63)-1)")
    # # df = df.Define("ULong_t", "(ULong_t) gRandom->Integer(pow(2,63))") 

    col_name = "Bool_t"
    df = df.Define(col_name, f"({col_name}) gRandom->Integer(2)")#.Snapshot(col_name, "./Data/" + col_name + ".root")
    # df = df.Define(col_name, "ROOT::RVec<float>{gRandom->uniform(-1, 1),gRandom->uniform(-1, 1),gRandom->uniform(-1, 1) } ")#.Snapshot(col_name, "./Data/" + col_name + ".root")

    df.Snapshot("TestData", "./Data/TestData.root") 



    # Will use a different root file for the integer types as Fill gives weird behaviour.
    remaining_cols = ["Long64_t", "ULong64_t", "Long_t", "ULong_t"]
    from array import array
    L_arr = generate_integers(N = N, bits = 64, signed = True)
    L = array(map_trr_to_arr["L"], [L_arr[0]])

    l_arr = generate_integers(N = N, bits = 64, signed = False)
    l = array(map_trr_to_arr["l"], [l_arr[0]])

    G_arr = generate_integers(N = N, bits = 64, signed = True)
    G = array(map_trr_to_arr["G"], [G_arr[0]])

    g_arr = generate_integers(N = N, bits = 64, signed = False)
    g = array(map_trr_to_arr["g"], [g_arr[0]])

    for col_name in remaining_cols:
        f = ROOT.TFile("./Data/"+col_name + ".root", "recreate")
        T = ROOT.TTree(col_name, col_name)

        data_type = map_col_to_sym[col_name]
        arr = eval(data_type+"_arr")
        x = array(map_trr_to_arr[data_type], [arr[0]])
        T.Branch(col_name, x, col_name+"/"+data_type)
        for i in range(N):
                x[0] = arr[i]
                T.Fill()
        f.Write()
        f.Close()


def DeleteData():
    from shutil import rmtree
    rmtree("./Data", ignore_errors=True)

if __name__=="__main__":
    CreateData()


