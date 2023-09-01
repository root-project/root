#!/usr/bin/env python
#
# A pyROOT script that allows one to
# make quick measuremenst.
#
# This is a command-line script that
# takes in signal and background values,
# as well as potentially uncertainties on those
# values, and returns a fitted signal value
# and errors
from __future__ import print_function


def main():
    """ Create a simple model and run statistical tests  

    This script can be used to make simple statistical using histfactory.
    It takes values for signal, background, and data as input, and
    can optionally take uncertainties on signal or background.
    The model is created and saved to an output ROOT file, and
    the model can be fit if requested.

    """
    
    # Let's parse the input
    # Define the command line options of the script:
    import optparse
    desc = " ".join(main.__doc__.split())    

    vers = "0.1"
    parser = optparse.OptionParser( description = desc, version = vers, usage = "%prog [options]" )

    parser.add_option( "-s", "--signal", dest = "signal",
                       action = "store", type = "float", default=None,
                       help = "Expected Signal" )

    parser.add_option( "-b", "--background", dest = "background",
                       action = "store", type = "float", default=None,
                       help = "Expected Background" )

    parser.add_option( "-d", "--data", dest = "data",
                       action = "store", type = "float", default=None,
                       help = "Measured data" )

    parser.add_option( "--signal-uncertainty", dest = "signal_uncertainty",
                       action = "store", type = "float", default=None,
                       help = "Uncertainty on the signal rate, as a fraction. --signal-uncertainty=.05 means a 5% uncertainty." )
 
    parser.add_option( "--background-uncertainty", dest = "background_uncertainty",
                       action = "store", type = "float", default=None,
                       help = "Uncertainty on the background rate, as a fraction, not a percentage. --background-uncertainty=.05 means a 5% uncertainty." )

    parser.add_option( "--output-prefix", dest = "output_prefix",
                       action = "store", type = "string", default="Measurement",
                       help = "Prefix for output files when using export.  Can include directories (ie 'MyDirectory/MyPrefix')" )

    parser.add_option( "-e", "--export", dest = "export",
                       action = "store_true", default=False,
                       help = "Make output plots, graphs, and save the workspace." )

    # Parse the command line options:
    ( options, unknown ) = parser.parse_args()
    
    # Make a log
    # Set the format of the log messages:
    FORMAT = 'Py:%(name)-25s  %(levelname)-8s  %(message)s'
    import logging
    logging.basicConfig( format = FORMAT )
    # Create the logger object:
    logger = logging.getLogger( "makeQuickMeasurement" )
    # Set the following to DEBUG when debugging the scripts:
    logger.setLevel( logging.INFO )
    
    # So a small sanity check:
    if len( unknown ):
        logger.warning( "Options(s) not recognised: [" + ",".join( unknown ) + "]" )

    # Ensure that all necessary input has been supplied
    if options.signal == None:
        logger.error( "You have to define a value for expacted signal (use --signal)" )
        return 255

    if options.background == None:
        logger.error( "You have to define a value for expacted background (use --background)" )
        return 255

    if options.data == None:
        logger.error( "You have to define a value for measured data (use --data)" )
        return 255

    
    # Okay, if all input is acceptable, we simply pass
    # it to the MakeSimpleMeasurement function, which
    # does the real work.

    MakeSimpleMeasurement( signal_val=options.signal, background_val=options.background, data_val=options.data,
                           signal_uncertainty=options.signal_uncertainty, background_uncertainty=options.background_uncertainty,
                           Export=options.export, output_prefix=options.output_prefix)
    return


def MakeSimpleMeasurement( signal_val, background_val, data_val, signal_uncertainty=None, background_uncertainty=None, 
                           Export=False, output_prefix="Measurement"):
    """ Make a simple measurement using HistFactory
    
    Take in simple values for signal, background data, 
    and potentially uncertainty on signal and background

    """

    try:
        import ROOT
    except ImportError:
        print("It seems that pyROOT isn't properly configured")
        return

    # Create and name a measurement
    # Set the Parameter of interest, and set several
    # other parameters to be constant
    meas = ROOT.RooStats.HistFactory.Measurement("meas", "meas")
    meas.SetOutputFilePrefix( output_prefix )
    meas.SetPOI( "SigXsecOverSM" )

    # We don't include Lumi here, 
    # but since HistFactory gives it to 
    # us for free, we set it constant
    # The values are just dummies

    meas.SetLumi( 1.0 )
    meas.SetLumiRelErr( 0.10 )
    meas.AddConstantParam("Lumi")

    # We set ExportOnly to false.  This parameter
    # defines what happens when we run MakeMeasurementAndModelFast
    # If we DO run that function, we also want it to export.
    meas.SetExportOnly( False )

    # Create a channel and set
    # the measured value of data 
    # (no extenal hist necessar for cut-and-count)
    chan = ROOT.RooStats.HistFactory.Channel( "channel" )
    chan.SetData( data_val )

    # Create the signal sample and set it's value
    signal = ROOT.RooStats.HistFactory.Sample( "signal" )
    signal.SetNormalizeByTheory( False )
    signal.SetValue( signal_val )
    #signal.SetValue( 10 )

    # Add the parmaeter of interest and a systematic
    # Try to make intelligent choice of upper bound
    import math
    upper_bound = 3*math.ceil( (data_val - background_val) / signal_val )
    upper_bound = max(upper_bound, 3) 
    signal.AddNormFactor( "SigXsecOverSM", 1, 0, upper_bound )

    # If we have a signal uncertainty, add it too
    if signal_uncertainty != None:
        uncertainty_up   = 1.0 + signal_uncertainty
        uncertainty_down = 1.0 - signal_uncertainty
        signal.AddOverallSys( "signal_uncertainty",  uncertainty_down, uncertainty_up )

    # Finally, add this sample to the channel
    chan.AddSample( signal )

    # Create a background sample
    background = ROOT.RooStats.HistFactory.Sample( "background" )
    background.SetNormalizeByTheory( False )
    background.SetValue( background_val )

    # If we have a background uncertainty, add it too
    if background_uncertainty != None:
        uncertainty_up   = 1.0 + background_uncertainty
        uncertainty_down = 1.0 - background_uncertainty
        background.AddOverallSys( "background_uncertainty",  uncertainty_down, uncertainty_up )

    # Finally, add this sample to the channel
    chan.AddSample( background )

    # Add this channel to the measurement
    # There is only this one channel, after all
    meas.AddChannel( chan )

    # Now, do the measurement
    if Export: 
        workspace = ROOT.RooStats.HistFactory.MakeModelAndMeasurementFast( meas )
        return workspace

    else:
        factory = ROOT.RooStats.HistFactory.HistoToWorkspaceFactoryFast()
        workspace = factory.MakeCombinedModel( meas )
        #workspace = ROOT.RooStats.HistFactory.HistoToWorkspaceFactoryFast.MakeCombinedModel( meas )
        ROOT.RooStats.HistFactory.FitModel( workspace )

    # At this point, we are done
    return


if __name__ == "__main__":
    main()
