# \file
# \ingroup tutorial_dataframe
# \notebook -nodraw
# Usage of multithreading mode with random generators.
#
# This example illustrates how to define functions that generate random numbers and use them in an RDataFrame
# computation graph in a thread-safe way.
#
# Using only one random number generator in an application running with ROOT.EnableImplicitMT() is a common pitfall.
# This pitfall creates race conditions resulting in a distorted random distribution. In the example, this issue is
# solved by creating one random number generator per RDataFrame processing slot, thus allowing for parallel and
# thread-safe access. The example also illustrates the difference between non-deterministic and deterministic random
# number generation.
#
# \macro_code
# \macro_image
# \macro_output
#
# \date February 2026
# \author Bohdan Dudar (JGU Mainz), Fernando Hueso-González (IFIC, CSIC-UV), Vincenzo Eduardo Padulano (CERN)

import os

import ROOT


def df042_ThreadSafeRNG():

    header_path = os.path.join(str(ROOT.gROOT.GetTutorialDir()), "analysis", "dataframe", "df042_ThreadSafeRNG.hxx")
    if not os.path.exists(header_path):
        raise RuntimeError(f'Could not find required header file "{header_path}".')

    # First, we declare the functions needed by the RDataFrame computation graph to the interpreter
    if not ROOT.gInterpreter.Declare(f'#include "{header_path}"'):
        raise RuntimeError("Failed to declare the functions needed by the RDataFrame computation graph.")

    myCanvas = ROOT.TCanvas("myCanvas", "myCanvas", 1000, 500)
    myCanvas.Divide(3, 1)

    nEntries = 10000000

    print("Running the first RDataFrame")
    # 1. Single thread for reference
    df1 = ROOT.RDataFrame(nEntries).Define("x", ROOT.GetNormallyDistributedNumberFromGlobalGenerator)
    h1 = df1.Histo1D(("h1", "Single thread (no MT)", 1000, -4, 4), "x")
    myCanvas.cd(1)
    h1.DrawCopy()
    print("Finished first RDataFrame")

    print("Running second RDataFrame")
    # 2. One generator per RDataFrame slot, with random_device seeding
    # Notes and Caveats:
    # - How many numbers are drawn from each generator is not deterministic
    #   and the result is not deterministic between runs.
    nSlots = max(2, os.cpu_count() // 4)
    ROOT.EnableImplicitMT(nSlots)
    # Before running the RDataFrame computation graph, we reinitialize the generators (one per slot), so they can
    # be used accordingly during the execution.
    ROOT.ReinitializeGenerators(nSlots)
    df2 = ROOT.RDataFrame(nEntries).DefineSlot("x", ROOT.GetNormallyDistributedNumberPerSlotGenerator)
    h2 = df2.Histo1D(("h2", "Thread-safe (MT, non-deterministic)", 1000, -4, 4), "x")
    myCanvas.cd(2)
    h2.DrawCopy()
    print("Finished second RDataFrame")

    # 3. One generator per RDataFrame slot, with entry seeding
    # Notes and Caveats:
    # - With RDataFrame(INTEGER_NUMBER) constructor (as in the example),
    #   the result is deterministic and identical on every run
    # - With RDataFrame(TTree) constructor, the result is not guaranteed to be deterministic.
    #   To make it deterministic, use something from the dataset to act as the event identifier
    #   instead of rdfentry_, and use it as a seed.

    print("Running third RDataFrame")
    # Before running the RDataFrame computation graph, we reinitialize the generators (one per slot), so they can
    # be used accordingly during the execution.
    ROOT.ReinitializeGenerators(nSlots)
    df3 = ROOT.RDataFrame(nEntries).DefineSlotEntry("x", ROOT.GetNormallyDistributedNumberPerSlotGeneratorForEntry)
    h3 = df3.Histo1D(("h3", "Thread-safe (MT, deterministic)", 1000, -4, 4), "x")
    myCanvas.cd(3)
    h3.DrawCopy()
    print("Finished third RDataFrame")

    print(f"{'{:<40}'.format('Final distributions')}: Mean +- StdDev")
    print(f"{'{:<40}'.format('Theoretical')}: 0.000 +- 1.000")
    print(f"{'{:<40}'.format('Single thread (no MT)')}: {h1.GetMean():.3f} +- {h1.GetStdDev():.3f}")
    print(f"{'{:<40}'.format('Thread-safe (MT, non-deterministic)')}: {h2.GetMean():.3f} +- {h2.GetStdDev():.3f}")
    print(f"{'{:<40}'.format('Thread-safe (MT, deterministic)')}: {h3.GetMean():.3f} +- {h3.GetStdDev():.3f}")

    # We draw the canvas with block=True to stop the execution before end of the
    # function and to be able to interact with the canvas until necessary
    myCanvas.Draw(block=True)


if __name__ == "__main__":
    df042_ThreadSafeRNG()
