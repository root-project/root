## \file
## \ingroup tutorial_roofit
## \notebook
## Code HistFactory Models in JSON.
##
## With the HS3 standard, it is possible to code RooFit-Models of any kind as JSON files.
## In this tutorial, you can see how to code up a (simple) HistFactory-based model in JSON and import it into a RooWorkspace.
##
## \macro_code
##
## \date November 2021
## \author Carsten Burgard

import ROOT

# start by creating an empty workspace
ws = ROOT.RooWorkspace("workspace")

# the RooJSONFactoryWSTool is responsible for importing and exporting things to and from your workspace
tool = ROOT.RooJSONFactoryWSTool(ws)

# use it to import the information from your JSON file
tool.importJSON(ROOT.gROOT.GetTutorialDir().Data() + "/roofit/rf515_hfJSON.json")
ws.Print()

# now, you can easily use your workspace to run your fit (as you usually would)
# the model config is named after your pdf, i.e. <the pdf name>_modelConfig
model = ws["ModelConfig"]
pdf = model.GetPdf()

# we are fitting a clone of the model now, such that we are not double-fitting
# the model in the closure check
result = pdf.cloneTree().fitTo(ws["observed"], Save=True, GlobalObservables=model.GetGlobalObservables(), PrintLevel=-1)
result.Print()

# in the end, you can again write to json
# the result will be not completely identical to the JSON file you used as an input, but it will work just the same
tool.exportJSON("myWorkspace.json")

# You can again import it if you want and check for closure
ws2 = ROOT.RooWorkspace("workspace")
tool2 = ROOT.RooJSONFactoryWSTool(ws2)
tool2.importJSON("myWorkspace.json")
model2 = ws2["main_modelConfig"]
result = model.GetPdf().fitTo(ws2["observed"], Save=True, GlobalObservables=model.GetGlobalObservables(), PrintLevel=-1)
result.Print()
