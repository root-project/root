from IPython.core.inputtransformer import InputTransformer
from IPython import get_ipython

import utils
import cppcompleter

from IPython.core import display




class CppTransformer(InputTransformer):

    def __init__(self):
        self.cell = ""
        self.mustSwitchToPython = False
        self.runAsDecl = False
        self.runAsAclic = False
        self.runAsBash = False

    def push(self, line):
        '''
        >>> import ROOT
        >>> t = CppTransformer()
        >>> t.push("int i=3;")
        >>> t.reset()
        >>> ROOT.i
        3
        >>> t.push('.cpp -d')
        >>> t.push('int f(int i){return i+i;}')
        >>> t.reset()
        >>> ROOT.f(3)
        '''
        # FIXME: must be in a single line
        fcnName="toPython()"
        if line == "%s;"%fcnName or line == fcnName:
            self.mustSwitchToPython = True
        elif line == ".cpp -d" and self.cell == "":
            self.runAsDecl = True
        elif line == ".cpp -a" and self.cell == "":
            self.runAsAclic = True
        elif line == ".bash" and self.cell == "":
            self.runAsBash = True
        else:
            line+="\n"
            self.cell += line
        return None

    def reset(self):
        if self.cell != "":
            if self.runAsDecl:
                utils.declareCppCode(self.cell)
                self.runAsDecl = False
            elif self.runAsAclic:
                utils.invokeAclic(self.cell)
                self.runAsAclic = False
            elif self.runAsBash:
                utils._checkOutput(self.cell)
                self.runAsBash = False
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
        #>>> code =""".cpp -a\n
        #... class A{public: A(){cout << "A ctor\n";}};\n
        #... int i=3;\n
        #... """
        #>>> for l in code.split('\n'):
        #...     t.push(l)
        #>>> t.reset()
        #>>> ROOT.A()
        #>>> t.push('.cpp -d\n')
        #>>> t.push('int f(int i){return i+i;}\n')
        #>>> t.reset()
