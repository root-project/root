## \file
## \ingroup tutorial_histfactory
## A ROOT script demonstrating  an example of writing a HistFactory model using Python.
##
## \macro_image
## \macro_code
## \macro_output
##
## \author George Lewis

def main():

    try:
        import ROOT
    except:
        print("It seems that pyROOT isn't properly configured")
        return

    """
    Create a HistFactory measurement from python
    """

    InputFile = "./data/example.root"
    if (ROOT.gSystem.AccessPathName(InputFile)) :
        ROOT.Info("example.py", InputFile+" does not exist")
        exit()

    # Create the measurement
    meas = ROOT.RooStats.HistFactory.Measurement("meas", "meas")

    meas.SetOutputFilePrefix( "./results/example_UsingPy" )
    meas.SetPOI( "SigXsecOverSM" )
    meas.AddConstantParam("Lumi")
    meas.AddConstantParam("alpha_syst1")

    meas.SetLumi( 1.0 )
    meas.SetLumiRelErr( 0.10 )
    meas.SetExportOnly( True )

    # Create a channel

    chan = ROOT.RooStats.HistFactory.Channel( "channel1" )
    chan.SetData( "data", InputFile )
    chan.SetStatErrorConfig( 0.05, "Poisson" )

    # Now, create some samples

    # Create the signal sample
    signal = ROOT.RooStats.HistFactory.Sample( "signal", "signal", InputFile )
    signal.AddOverallSys( "syst1",  0.95, 1.05 )
    signal.AddNormFactor( "SigXsecOverSM", 1, 0, 3 )
    chan.AddSample( signal )


    # Background 1
    background1 = ROOT.RooStats.HistFactory.Sample( "background1", "background1", InputFile )
    background1.ActivateStatError( "background1_statUncert", InputFile )
    background1.AddOverallSys( "syst2", 0.95, 1.05  )
    chan.AddSample( background1 )


    # Background 1
    background2 = ROOT.RooStats.HistFactory.Sample( "background2", "background2", InputFile )
    background2.ActivateStatError()
    background2.AddOverallSys( "syst3", 0.95, 1.05  )
    chan.AddSample( background2 )


    # Done with this channel
    # Add it to the measurement:

    meas.AddChannel( chan )

    # Collect the histograms from their files,
    # print some output,
    meas.CollectHistograms()
    meas.PrintTree();

    # One can print XML code to an
    # output directory:
    # meas.PrintXML( "xmlFromCCode", meas.GetOutputFilePrefix() );

    meas.PrintXML( "xmlFromPy", meas.GetOutputFilePrefix() );

    # Now, do the measurement
    ws = ROOT.RooStats.HistFactory.MakeModelAndMeasurementFast(meas)

    # Retrieve the ModelConfig
    modelConfig = ws.obj("ModelConfig")

    # Extract the PDF and global observables
    pdf = modelConfig.GetPdf()
    globalObservables = ROOT.RooArgSet(modelConfig.GetGlobalObservables())

    '''
    parameters in globalObservables:

    nominalLumi
    nom_alpha_syst1 -> was set as constant in the beginning
    nom_alpha_syst2
    nom_alpha_syst3
    nom_gamma_stat_channel1_bin_0
    nom_gamma_stat_channel1_bin_1

    '''

    # Perform the fit
    result = pdf.fitTo(ws.data("obsData"), ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1), ROOT.RooFit.GlobalObservables(globalObservables))

    # Getting list of Parameters of Interest and getting first from them
    poi = modelConfig.GetParametersOfInterest().first()

    nll = pdf.createNLL(ws.data("obsData"))
    profile = nll.createProfile(poi)

    # frame for future plot
    frame = poi.frame()

    frame.SetTitle("")
    frame.GetXaxis().SetTitle("-log likelihood")
    frame.GetYaxis().SetTitle(poi.GetTitle())

    cv = ROOT.TCanvas("combined", "", 800, 600)

    xmin = poi.getMin()
    xmax = poi.getMax()

    line = ROOT.TLine(xmin, 0.5, xmax, .5)
    line.SetLineColor(ROOT.kGreen)
    line90 = ROOT.TLine(xmin, 2.71/2, xmax, 2.71/2)
    line90.SetLineColor(ROOT.kGreen)
    line95 = ROOT.TLine(xmin, 3.84/2, xmax, 3.84/2)
    line95.SetLineColor(ROOT.kGreen)

    frame.addObject(line)
    frame.addObject(line90)
    frame.addObject(line95)

    nll.plotOn(frame, ShiftToZero = True, LineColor = ROOT.kRed, LineStyle = ROOT.kDashed)
    profile.plotOn(frame)

    frame.SetMinimum(0)
    frame.SetMaximum(2)

    frame.Draw()

    # Save drawed picture as PNG file
    profilePlotName = "LikelihoodCurve.png"
    cv.SaveAs(profilePlotName)

    # Create new file to save likelihood graph and fit results
    outputFileName = meas.GetName() + "_combined.root"
    outputFile = ROOT.TFile(outputFileName, "recreate")
    internal_dir = outputFile.mkdir("FitSummary")
    internal_dir.cd()

    curve = frame.getCurve()

    # Number of points and possible X values for POI
    curve_N = curve.GetN()
    curve_x = curve.GetX()

    g = ROOT.TGraph()

    for i in range(curve_N):
        f = curve_x[i]
        poi.setVal(f)
        g.SetPoint(i, f, nll.getVal())

    # Save likelihood picture to root file
    g.SetName("FitSummary_nll")
    g.Write()

    # Save fit results to root file
    result.Write("fitResult")

    # Save file
    outputFile.Close()

    # Print fit results to console
    result.Print()

    pass


if __name__ == "__main__":
    main()
