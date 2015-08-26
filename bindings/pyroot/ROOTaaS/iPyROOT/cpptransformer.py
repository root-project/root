from IPython.core.inputtransformer import InputTransformer
from IPython import get_ipython

import utils
import cppcompleter

from IPython.core import display




class CppTransformer(InputTransformer):

    def __init__(self):
        self.cell = ""
        self.mustSwitchToPython = False
        self.mustDeclare = False

    def push(self, line):
        # FIXME: must be in a single line
        fcnName="toPython()"
        if line == "%s;"%fcnName or line == fcnName:
            self.mustSwitchToPython = True
        elif line == ".dcl" and self.cell == "":
            self.mustDeclare = True
        else:
            if "" != self.cell:
               line+="\n"
            self.cell += line
        return None

    def reset(self):
        if self.cell != "":
            if self.mustDeclare:
                utils.declareCppCode(self.cell)
                self.mustDeclare = False
            else:
                utils.processCppCode(self.cell)
            self.cell = ""
        if self.mustSwitchToPython:
            ip = get_ipython()
            unload_ipython_extension(ip)
            self.mustSwitchToPython = False
            cppcompleter.unload_ipython_extension(ip)
            # Change highlight mode
            display.display_javascript(utils.jsDefaultHighlight.format(mimeType = utils.ipyMIME), raw=True)
            print "Notebook is in Python mode"

_transformer = CppTransformer()

def unload_ipython_extension(ipython):
    ipython.input_splitter.logical_line_transforms.remove(_transformer)
    ipython.input_transformer_manager.logical_line_transforms.remove(_transformer)

def load_ipython_extension(ipython):
    ipython.input_splitter.logical_line_transforms.insert(0,_transformer)
    ipython.input_transformer_manager.logical_line_transforms.insert(0,_transformer)
