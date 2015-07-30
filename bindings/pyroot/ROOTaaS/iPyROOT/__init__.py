from IPython import display
from IPython import get_ipython
from IPython.core.extensions import ExtensionManager
import ROOT
import utils

def iPythonize():
    utils.setStyle()
    for capture in utils.captures: capture.register()
    ExtensionManager(get_ipython()).load_extension("ROOTaaS.iPyROOT.cppmagic")
    ExtensionManager(get_ipython()).load_extension("ROOTaaS.iPyROOT.dclmagic")

    # Make sure clike JS lexer is loaded
    display.display_javascript("require(['codemirror/mode/clike/clike'], function(Clike) { console.log('ROOTaaS - C++ CodeMirror module loaded'); });", raw=True)
    # Define highlight mode for %%cpp and %%dcl magics
    display.display_javascript(utils.jsMagicHighlight.format(cppMIME = utils.cppMIME), raw=True)

    ROOT.toCpp = utils.toCpp
    ROOT.enableJSVis = utils.enableJSVis
    ROOT.disableJSVis = utils.disableJSVis
    ROOT.enableJSVisDebug = utils.enableJSVisDebug
    ROOT.disableJSVisDebug = utils.disableJSVisDebug

    #ROOT.toCpp()
    utils.welcomeMsg()


iPythonize()


