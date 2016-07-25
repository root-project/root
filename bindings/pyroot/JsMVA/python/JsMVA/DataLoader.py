# -*- coding: utf-8 -*-
## @package JsMVA.DataLoader
# DataLoader module with the functions to be inserted to TMVA::DataLoader class and helper functions
# @authors Attila Bagoly <battila93@gmail.com>


from ROOT import TH1F, TMVA
import JPyInterface


## Creates the input variable histogram and perform the transformations if necessary
# @param dl DataLoader object
# @param className string Signal/Background
# @param variableName string containing the variable name
# @param numBin for creating the histogram
# @param processTrfs string containing the list of transformations to be used on input variable; eg. "I;N;D;P;U;G,D"
def GetInputVariableHist(dl, className, variableName, numBin, processTrfs=""):
    dsinfo = dl.GetDefaultDataSetInfo()
    vi = 0
    ivar = 0
    for i in range(dsinfo.GetNVariables()):
        if dsinfo.GetVariableInfo(i).GetLabel()==variableName:
            vi   = dsinfo.GetVariableInfo(i)
            ivar = i
            break
    if vi==0:
        return 0

    h = TH1F(className, str(vi.GetExpression()) + " ("+className+")", numBin, vi.GetMin(), vi.GetMax())

    clsn = dsinfo.GetClassInfo(className).GetNumber()
    ds   = dsinfo.GetDataSet()

    trfsDef = processTrfs.split(';')
    trfs    = []
    for trfDef in trfsDef:
        trfs.append(TMVA.TransformationHandler(dsinfo, "DataLoader"))
        TMVA.CreateVariableTransforms( trfDef, dsinfo, trfs[-1], dl.Log())

    inputEvents = ds.GetEventCollection()
    transformed = 0
    tmp         = 0
    for trf in trfs:
        if transformed==0:
            transformed = trf.CalcTransformations(inputEvents, 1)
        else:
            tmp = trf.CalcTransformations(transformed, 1)
            del transformed[:]
            transformed = tmp

    if transformed!=0:
        for event in transformed:
            if event.GetClass() != clsn:
                continue
            h.Fill(event.GetValue(ivar))
        del transformed
    else:
        for event in inputEvents:
            if event.GetClass() != clsn:
                continue
            h.Fill(event.GetValue(ivar))
    return (h)


## Draw correlation matrix
# This function uses the TMVA::DataLoader::GetCorrelationMatrix function added newly to root
# @param dl the object pointer
# @param className Signal/Background
def DrawCorrelationMatrix(dl, className):
    th2 = dl.GetCorrelationMatrix(className)
    th2.SetMarkerSize(1.5)
    th2.SetMarkerColor(0)
    labelSize = 0.040
    th2.GetXaxis().SetLabelSize(labelSize)
    th2.GetYaxis().SetLabelSize(labelSize)
    th2.LabelsOption("d")
    th2.SetLabelOffset(0.011)
    JPyInterface.JsDraw.Draw(th2, 'drawTH2')

## Draw input variables
# This function uses the previously defined GetInputVariableHist function to create the histograms
# @param dl The object pointer
# @param variableName string containing the variable name
# @param numBin for creating the histogram
# @param processTrfs string containing the list of transformations to be used on input variable; eg. "I;N;D;P;U;G,D"
def DrawInputVariable(dl, variableName, numBin=100, processTrfs=""):
    sig = GetInputVariableHist(dl, "Signal",     variableName, numBin, processTrfs)
    bkg = GetInputVariableHist(dl, "Background", variableName, numBin, processTrfs)
    c, l = JPyInterface.JsDraw.sbPlot(sig, bkg, {"xaxis": sig.GetTitle(),
                                    "yaxis": "Number of events",
                                    "plot": "Input variable: "+sig.GetTitle()})
    JPyInterface.JsDraw.Draw(c)

## Rewrite TMVA::DataLoader::PrepareTrainingAndTestTree
def ChangeCallOriginalPrepareTrainingAndTestTree(*args, **kwargs):
    if len(kwargs)==0:
        originalFunction, args = JPyInterface.functions.ProcessParameters(0, *args, **kwargs)
        return originalFunction(*args)
    try:
        args, kwargs = JPyInterface.functions.ConvertSpecKwargsToArgs(["SigCut", "BkgCut"], *args, **kwargs)
    except AttributeError:
        try:
            args, kwargs = JPyInterface.functions.ConvertSpecKwargsToArgs(["Cut"], *args, **kwargs)
        except AttributeError:
            raise AttributeError
    originalFunction, args = JPyInterface.functions.ProcessParameters(3, *args, **kwargs)
    return originalFunction(*args)