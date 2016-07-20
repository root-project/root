# -*- coding: utf-8 -*-
## @package JsMVA/JPyInterface
#  @authors  Attila Bagoly <battila93@gmail.com>
# This package is responsible for adding the drawing methods to TMVA
# and for creating the JavaScript outputs from objects.


from IPython.core.display import display, HTML
from string import Template
import ROOT
import DataLoader
import Factory


## Function inserter class
# This class contains the methods which are invoked by using jsmva magic, and will inject the new methods
# to TMVA::Factory, TMVA::DataLoader
class functions:

    ## Threaded functions
    ThreadedFunctions = {
        "MethodBase": ["GetInteractiveTrainingError", "ExitFromTraining", "TrainingEnded", "TrainMethod",
                       "GetMaxIter", "GetCurrentIter"]
    }

    ## The method inserter function
    # @param target which class to insert
    # @param source module which contains the methods to insert
    # @param args list of methods to insert
    @staticmethod
    def __register(target, source, *args):
        for arg in args:
            if hasattr(target, arg):
                continue
            setattr(target, arg, getattr(source, arg))

    ## This method change TMVA methods with new methods
    # @param target which class to insert
    # @param source module which contains the methods to insert
    # @param args list of methods to insert
    @staticmethod
    def __changeMethod(target, source, *args):
        def rewriter(originalFunction, newFunction):
            def wrapper(*args, **kwargs):
                kwargs["originalFunction"] = originalFunction
                return newFunction(*args, **kwargs)
            return wrapper
        for arg in args:
            if arg.find("CallOriginal")!=-1:
                originalName = arg.replace("Change", "").replace("CallOriginal", "")
                setattr(target, originalName, rewriter(getattr(target, originalName), getattr(source, arg)))
            else:
                setattr(target, arg.replace("Change", ""), getattr(source, arg))

    ## The method removes inserted functions from class
    # @param target from which class to remove functions
    # @param args list of methods to remove
    @staticmethod
    def __unregister(target, *args):
        for arg in args:
            if hasattr(target, arg):
                delattr(target, arg)

    ## Reads all methods containing a selector from specified module
    # @param module finding methods in this module
    # @param selector if method in module contains this string will be selected
    @staticmethod
    def __getMethods(module, selector):
        methods = []
        for method in dir(module):
            if method.find(selector)!=-1:
                methods.append(method)
        return methods

    ## This function will register all functions which name contains "Draw" to TMVA.DataLoader and TMVA.Factory
    # from DataLoader and Factory modules
    @staticmethod
    def register():
        functions.__register(ROOT.TMVA.DataLoader, DataLoader, *functions.__getMethods(DataLoader, "Draw"))
        functions.__register(ROOT.TMVA.Factory,    Factory,    *functions.__getMethods(Factory,    "Draw"))
        functions.__changeMethod(ROOT.TMVA.Factory,    Factory,    *functions.__getMethods(Factory,    "Change"))
        functions.__changeMethod(ROOT.TMVA.DataLoader, DataLoader, *functions.__getMethods(DataLoader, "Change"))
        for key in functions.ThreadedFunctions:
            for func in functions.ThreadedFunctions[key]:
                setattr(getattr(getattr(ROOT.TMVA, key), func), "_threaded", True)

    ## This function will remove all functions which name contains "Draw" from TMVA.DataLoader and TMVA.Factory
    # if the function was inserted from DataLoader and Factory modules
    @staticmethod
    def unregister():
        functions.__register(ROOT.TMVA.DataLoader, DataLoader, *functions.__getMethods(DataLoader, "Draw"))
        functions.__register(ROOT.TMVA.Factory,    Factory,    *functions.__getMethods(Factory,    "Draw"))


## Class for creating the output scripts and inserting them to cell output
class JsDraw:
    ## String containing the link to JavaScript files
    __jsMVASourceDir = "https://rawgit.com/qati/GSOC16/master/src/js"

    ## Drawing are sizes
    jsCanvasWidth   = 800
    jsCanvasHeight  = 450

    ## id for drawing area
    __divUID = 0

    ## Template containing HTML code with draw area and drawing JavaScript
    __jsCode = Template("""
<div id="$divid" style="width: ${width}px; height:${height}px"></div>
<script>
    require.config({
        paths: {
            'JsMVA':'$PATH/JsMVA.min'
        }
    });
    require(['JsMVA'],function(jsmva){
        jsmva.$funcName('$divid','$dat');
    });
</script>
""")

    ## Template containing data insertion JavaScript code
    __jsCodeForDataInsert = Template("""<script id="dataInserterScript">
require(['JsMVA'],function(jsmva){
jsmva.$funcName('$divid', '$dat');
var script = document.getElementById("dataInserterScript");
script.parentElement.parentElement.remove();
});
</script>""")

    ## Inserts the draw area and drawing JavaScript to output
    # @param obj ROOT object (will be converted to JSON) or JSON string containing the data to be drawed
    # @param jsDrawMethod the JsMVA JavaScrip object method name to be used for drawing
    # @param objIsJSON obj is ROOT object or JSON
    @staticmethod
    def Draw(obj, jsDrawMethod='draw', objIsJSON=False):
        if objIsJSON:
            dat = obj
        else:
            dat = ROOT.TBufferJSON.ConvertToJSON(obj)
            dat = str(dat).replace("\n","")
        JsDraw.__divUID += 1
        display(HTML(JsDraw.__jsCode.substitute({
            'funcName': jsDrawMethod,
            'divid':'jstmva_'+str(JsDraw.__divUID),
            'dat': dat,
            'PATH': JsDraw.__jsMVASourceDir,
            'width': JsDraw.jsCanvasWidth,
            'height': JsDraw.jsCanvasHeight
         })))

    ## Inserts the data inserter JavaScript code to output
    # @param obj ROOT object (will be converted to JSON) or JSON string containing the data to be inserted
    # @param jsDrawMethod the JsMVA JavaScrip object method name to be used for inserting the new data
    # @param objIsJSON obj is ROOT object or JSON
    @staticmethod
    def InsertData(obj, dataInserterMethod="updateTrainingTestingErrors", objIsJSON=False):
        if objIsJSON:
            dat = obj
        else:
            dat = ROOT.TBufferJSON.ConvertToJSON(obj)
            dat = str(dat).replace("\n", "")
        display(HTML(JsDraw.__jsCodeForDataInsert.substitute({
            'funcName': dataInserterMethod,
            'divid': 'jstmva_'+str(JsDraw.__divUID),
            'dat': dat
        })))

    ## Draws a signal and background histogram to a newly created TCanvas
    # @param sig signal histogram
    # @param bkg background histogram
    # @param title all labels
    @staticmethod
    def sbPlot(sig, bkg, title):
        canvas = ROOT.TCanvas("csbplot", title["plot"], JsDraw.jsCanvasWidth, JsDraw.jsCanvasHeight)
        sig.SetMaximum(ROOT.TMath.Max(sig.GetMaximum(),bkg.GetMaximum()*1.1))
        sig.SetTitle(sig.GetTitle().replace("(Signal)",""))
        sig.GetXaxis().SetTitle(title["xaxis"])
        sig.GetYaxis().SetTitle(title["yaxis"])
        sig.SetTitle(title["plot"])
        bkg.SetFillColorAlpha(ROOT.kRed, 0.3)
        sig.SetFillColor(ROOT.kBlue)
        bkg.SetLineColor(ROOT.kRed)
        sig.Draw("hist")
        bkg.Draw("histsame")

        legend = ROOT.TLegend(1-canvas.GetLeftMargin()-0.39, 1-canvas.GetTopMargin()-0.15,
                              1-canvas.GetLeftMargin()-0.01, 1-canvas.GetTopMargin()-0.01)
        legend.SetFillStyle(1)
        legend.AddEntry(sig, "Signal", "F")
        legend.AddEntry(bkg, "Background", "F")
        legend.SetBorderSize(1)
        legend.SetMargin(0.3)
        legend.Draw()

        return (canvas, legend)