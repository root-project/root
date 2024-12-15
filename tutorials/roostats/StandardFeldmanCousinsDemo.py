# \file
# \ingroup tutorial_roostats
# \notebook -js
# Standard demo of the Feldman-Cousins calculator
# StandardFeldmanCousinsDemo
#
# This is a standard demo that can be used with any ROOT file
# prepared in the standard way.  You specify:
#  - name for input ROOT file
#  - name of workspace inside ROOT file that holds model and data
#  - name of ModelConfig that specifies details for calculator tools
#  - name of dataset
#
# With default parameters the macro will attempt to run the
# standard hist2workspace example and read the ROOT file
# that it produces.
#
# The actual heart of the demo is only about 10 lines long.
#
# The FeldmanCousins tools is a classical frequentist calculation
# based on the Neyman Construction.  The test statistic can be
# generalized for nuisance parameters by using the profile likelihood ratio.
# But unlike the ProfileLikelihoodCalculator, this tool explicitly
# builds the sampling distribution of the test statistic via toy Monte Carlo.
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Kyle Cranmer (C++ version), and P. P. (Python translation)


import ROOT


def StandardFeldmanCousinsDemo(infile="", workspaceName="combined", modelConfigName="ModelConfig", dataName="obsData"):

    # -------------------------------------------------------
    # First part is just to access a user-defined file
    # or create the standard example file if it doesn't exist
    filename = ""
    if infile == "":
        filename = "results/example_combined_GaussExample_model.root"
        fileExist = not ROOT.gSystem.AccessPathName(filename)  # note opposite return code
        # if file does not exists generate with histfactory
        if not fileExist:
            # Normally this would be run on the command line
            print(f"will run standard hist2workspace example")
            ROOT.gROOT.ProcessLine(".! prepareHistFactory .")
            ROOT.gROOT.ProcessLine(".! hist2workspace config/example.xml")
            print(f"\n\n---------------------")
            print(f"Done creating example input")
            print(f"---------------------\n\n")

    else:
        filename = infile

    # Try to open the file
    file = ROOT.TFile.Open(filename)

    # if input file was specified but not found, quit
    if not file:
        print(f"StandardRooStatsDemoMacro: Input file {filename} is not found")
        return

    # -------------------------------------------------------
    # Tutorial starts here
    # -------------------------------------------------------

    # get the workspace out of the file
    w = file.Get(workspaceName)
    if not w:
        print(f"workspace not found")
        return

    # get the modelConfig out of the file
    mc = w.obj(modelConfigName)

    # get the modelConfig out of the file
    data = w.data(dataName)

    # make sure ingredients are found
    if not data or not mc:
        w.Print()
        print(f"data or ModelConfig was not found")
        return

    # -------------------------------------------------------
    # create and use the FeldmanCousins tool
    # to find and plot the 95% confidence interval
    # on the parameter of interest as specified
    # in the model config
    fc = ROOT.RooStats.FeldmanCousins(data, mc)
    fc.SetConfidenceLevel(0.95)  # 95% interval
    # fc.AdditionalNToysFactor(0.1); # to speed up the result
    fc.UseAdaptiveSampling(True)  # speed it up a bit
    fc.SetNBins(10)  # set how many points per parameter of interest to scan
    fc.CreateConfBelt(True)  # save the information in the belt for plotting

    # Since this tool needs to throw toy MC the PDF needs to be
    # extended or the tool needs to know how many entries in a dataset
    # per pseudo experiment.
    # In the 'number counting form' where the entries in the dataset
    # are counts, and not values of discriminating variables, the
    # datasets typically only have one entry and the PDF is not
    # extended.
    if not mc.GetPdf().canBeExtended():
        if data.numEntries() == 1:
            fc.FluctuateNumDataEntries(False)
        else:
            print(f"Not sure what to do about this model")

    # Now get the interval
    interval = fc.GetInterval()
    belt = fc.GetConfidenceBelt()

    # print out the interval on the first Parameter of Interest
    firstPOI = mc.GetParametersOfInterest().first()
    print(
        f"\n95% interval on {firstPOI.GetName()} is : [{interval.LowerLimit(firstPOI)}, ",
        interval.UpperLimit(firstPOI),
        "] ",
    )

    # ---------------------------------------------
    # No nice plots yet, so plot the belt by hand

    # Ask the calculator which points were scanned
    parameterScan = fc.GetPointsToScan()
    tmpPoint = ROOT.RooArgSet()

    # make a histogram of parameter vs. threshold
    histOfThresholds = ROOT.TH1F(
        "histOfThresholds", "", parameterScan.numEntries(), firstPOI.getMin(), firstPOI.getMax()
    )

    # loop through the points that were tested and ask confidence belt
    # what the upper/lower thresholds were.
    # For FeldmanCousins, the lower cut off is always 0
    for i in range(parameterScan.numEntries()):
        tmpPoint = parameterScan.get(i).clone("temp")
        arMax = belt.GetAcceptanceRegionMax(tmpPoint)
        arMin = belt.GetAcceptanceRegionMax(tmpPoint)
        poiVal = tmpPoint.getRealValue(firstPOI.GetName())
        histOfThresholds.Fill(poiVal, arMax)

    histOfThresholds.SetMinimum(0)
    c_belt = ROOT.TCanvas("c_belt", "c_belt")
    histOfThresholds.Draw()
    c_belt.Update()
    c_belt.Draw()
    c_belt.SaveAs("StandardFeldmanCousinsDemo.1.belt.png")


StandardFeldmanCousinsDemo(infile="", workspaceName="combined", modelConfigName="ModelConfig", dataName="obsData")
