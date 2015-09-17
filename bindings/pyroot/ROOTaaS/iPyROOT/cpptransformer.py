import sys
import re
from IPython.core.inputtransformer import InputTransformer
from IPython import get_ipython
from IPython.core import display
import utils


_cppDcl=re.compile("\s*\.cpp\s+-d")
_cppAclic=re.compile("\s*\.cpp\s+-a")
_bash=re.compile("\s*\.bash\d*")
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
        >>> t.push('.cpp -a')
        >>> t.push('int q(int i){return i+i;};')
        >>> t.reset()
        >>> ROOT.q(2)
        4
        >>> t.push('   .cpp      -a\t\t ')
        >>> t.push('int qq(int i){return i+i;};')
        >>> t.reset()
        >>> ROOT.qq(2)
        4
        >>> t.push('.cpp -d')
        >>> t.push('int f(int i){return i+i;}')
        >>> t.reset()
        >>> ROOT.f(3)
        6
        >>> t.push('.cpp -d')
        >>> t.push('int ff(int i){return i+i;}')
        >>> t.reset()
        >>> ROOT.ff(3)
        6
        >>> t.push('.bash echo    Hello  ')
        >>> t.reset()
        Hello
        >>> t.push(' \t .bash \t echo    Hello  ')
        >>> t.reset()
        Hello
        >>> t.push('.bash')
        >>> t.push('echo    Hello')
        >>> t.reset()
        Hello
        '''
        # FIXME: must be in a single line
        fcnName="toPython()"
        if line == "%s;"%fcnName or line == fcnName:
            self.mustSwitchToPython = True
        elif _cppDcl.match(line) and self.cell == "":
            self.runAsDecl = True
        elif _cppAclic.match(line) and self.cell == "":
            self.runAsAclic = True
        elif _bash.match(line) and self.cell == "":
            self.cell += line.replace(".bash","")+"\n"
            self.runAsBash = True
        else:
            line+="\n"
            self.cell += line
        return None

    def reset(self):
        out = None
        if self.cell != "":
            if self.runAsDecl:
                utils.declareCppCode(self.cell)
                self.runAsDecl = False
            elif self.runAsAclic:
                utils.invokeAclic(self.cell)
                self.runAsAclic = False
            elif self.runAsBash:
                cellNoEndNewLine = self.cell[:-1]
                out = utils._checkOutput(cellNoEndNewLine,"Error running shell command")
                if out: sys.stdout.write(out)
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
