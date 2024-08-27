import ROOT

# Start by creating a new workspace

ws = ROOT.RooWorkspace()

# Add the variables to the workspace

ws["Lumi"] = dict({"max": 10.0, "min": 0.0, "value": 1.0})
ws["nominalLumi"] = dict({"max": 2.0, "min": 0.0, "value": 1.0})

# Add the constraint variable

ws["lumiConstraint"] = {
    "mean": "nominalLumi",
    "sigma": 0.1,
    "type": "gaussian_dist",
    "x": "Lumi",
}

# Add data channels to the workspace

ws["obsData_channel1"] = {
    "axes": [{"max": 2.0, "min": 1.0, "name": "obs_x_channel1", "nbins": 2}],
    "contents": [122, 112],
    "type": "binned",
}

ws["asimovData_channel1"] = {
    "axes": [{"max": 2.0, "min": 1.0, "name": "obs_x_channel1", "nbins": 2}],
    "contents": [120, 110],
    "type": "binned",
}

# Specify the model inside the workspace by adding variables, constraints, datasets and modifiers

ws["model_channel1"] = {
    "axes": [{"max": 2.0, "min": 1.0, "name": "obs_x_channel1", "nbins": 2}],
    "samples": [
        {
            "data": {"contents": [100, 0], "errors": [5, 0]},
            "modifiers": [
                {
                    "constraint_name": "lumiConstraint",
                    "name": "Lumi",
                    "parameter": "Lumi",
                    "type": "normfactor",
                },
                {
                    "constraint": "Gauss",
                    "data": {"hi": 1.05, "lo": 0.95},
                    "name": "syst2",
                    "parameter": "alpha_syst2",
                    "type": "normsys",
                },
                {"constraint": "Poisson", "name": "staterror", "type": "staterror"},
            ],
            "name": "background1",
        },
        {
            "data": {"contents": [0, 100], "errors": [0, 10]},
            "modifiers": [
                {
                    "constraint_name": "lumiConstraint",
                    "name": "Lumi",
                    "parameter": "Lumi",
                    "type": "normfactor",
                },
                {
                    "constraint": "Gauss",
                    "data": {"hi": 1.05, "lo": 0.95},
                    "name": "syst3",
                    "parameter": "alpha_syst3",
                    "type": "normsys",
                },
                {"constraint": "Poisson", "name": "staterror", "type": "staterror"},
            ],
            "name": "background2",
        },
        {
            "data": {"contents": [20, 10]},
            "modifiers": [
                {
                    "constraint_name": "lumiConstraint",
                    "name": "Lumi",
                    "parameter": "Lumi",
                    "type": "normfactor",
                },
                {
                    "name": "SigXsecOverSM",
                    "parameter": "SigXsecOverSM",
                    "type": "normfactor",
                },
                {
                    "constraint": "Gauss",
                    "data": {"hi": 1.05, "lo": 0.95},
                    "name": "syst1",
                    "parameter": "alpha_syst1",
                    "type": "normsys",
                },
            ],
            "name": "signal",
        },
    ],
    "type": "histfactory_dist",
}

# Configure the model by specifing the pdf and point of interest

ws["simPdf_asimovData"] = {"pdfName": "model_channel1", "poi": "SigXsecOverSM"}

# Perform the likelihood analysis and plot the graphs

ROOT.RooStats.HistFactory.FitModelAndPlot("meas", "tut", "obsData_channel1", ws)
