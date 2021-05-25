## \file
## \ingroup tutorial_histfactory
## A ROOT script demonstrating  an example of writing a HistFactory model using Python.
##
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

    # Create the measurement
    meas = ROOT.RooStats.HistFactory.Measurement("meas", "meas")

    meas.SetOutputFilePrefix( "./results/example_UsingPy" )
    meas.SetPOI( "SigXsecOverSM" )
    meas.AddConstantParam("Lumi")
    meas.AddConstantParam("alpha_syst1")

    meas.SetLumi( 1.0 )
    meas.SetLumiRelErr( 0.10 )
    meas.SetExportOnly( False )

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
    ROOT.RooStats.HistFactory.MakeModelAndMeasurementFast( meas );

    pass


if __name__ == "__main__":
    main()
