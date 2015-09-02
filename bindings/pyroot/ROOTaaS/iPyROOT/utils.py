import os
import sys
import select
import time
import tempfile
import itertools
import re
import fnmatch
from hashlib import sha1
from contextlib import contextmanager
from subprocess import check_output
from IPython import get_ipython
from IPython.display import HTML
from IPython.core.extensions import ExtensionManager
import IPython.display
import ROOT
import cpptransformer
import cppcompleter


# We want iPython to take over the graphics
ROOT.gROOT.SetBatch()


cppMIME = 'text/x-c++src'
ipyMIME = 'text/x-ipython'

_jsDefaultHighlight = """
// Set default mode for code cells
IPython.CodeCell.options_default.cm_config.mode = '{mimeType}';
// Set CodeMirror's current mode
var cells = IPython.notebook.get_cells();
cells[cells.length-1].code_mirror.setOption('mode', '{mimeType}');
// Set current mode for newly created cell
cells[cells.length-1].cm_config.mode = '{mimeType}';
"""

_jsMagicHighlight = "IPython.CodeCell.config_defaults.highlight_modes['magic_{cppMIME}'] = {{'reg':[/^%%cpp/]}};"


_jsNotDrawableClassesPatterns = ["TGraph[23]D","TH3*","TGraphPolar","TProf*","TEve*","TF[23]","TGeo*","TPolyLine3D"]


_jsROOTSourceDir = "https://root.cern.ch/js/dev/"
_jsCanvasWidth = 800
_jsCanvasHeight = 600

_jsCode = """
<div id="{jsDivId}"
     style="width: {jsCanvasWidth}px; height: {jsCanvasHeight}px">
</div>

<script>
requirejs.config(
{{
  paths: {{
    'JSRootCore'    : '{jsROOTSourceDir}/scripts/JSRootCore',
    'JSRootPainter' : '{jsROOTSourceDir}/scripts/JSRootPainter',
  }}
}}
);
require(['JSRootCore', 'JSRootPainter'],
        function(Core, Painter) {{
          var obj = Core.parse('{jsonContent}');
          Painter.draw("{jsDivId}", obj, "{jsDrawOptions}");
        }}
);
</script>
"""

_enableJSVis = False
_enableJSVisDebug = False
def enableJSVis():
    global _enableJSVis
    _enableJSVis = True

def disableJSVis():
    global _enableJSVis
    _enableJSVis = False

def enableJSVisDebug():
    global _enableJSVis
    global _enableJSVisDebug
    _enableJSVis = True
    _enableJSVisDebug = True

def disableJSVisDebug():
    global _enableJSVis
    global _enableJSVisDebug
    _enableJSVis = False
    _enableJSVisDebug = False

def _getPlatform():
    return sys.platform

def _getLibExtension(thePlatform):
    '''Return appropriate file extension for a shared library
    >>> _getLibExtension('darwin')
    '.dylib'
    >>> _getLibExtension('win32')
    '.dll'
    >>> _getLibExtension('OddPlatform')
    '.so'
    '''
    pExtMap = {
        'darwin' : '.dylib',
        'win32'  : '.dll'
    }
    return pExtMap.get(thePlatform, '.so')

def welcomeMsg():
    print "Welcome to ROOTaaS %s" %ROOT.gROOT.GetVersion()

@contextmanager
def _setIgnoreLevel(level):
    originalLevel = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = level
    yield
    ROOT.gErrorIgnoreLevel = originalLevel

def commentRemover( text ):
   '''
   >>> s="// hello"
   >>> commentRemover(s)
   ''
   >>> s="int /** Test **/ main() {return 0;}"
   >>> commentRemover(s)
   'int  main() {return 0;}'
   '''
   def blotOutNonNewlines( strIn ) :  # Return a string containing only the newline chars contained in strIn
      return "" + ("\n" * strIn.count('\n'))

   def replacer( match ) :
      s = match.group(0)
      if s.startswith('/'):  # Matched string is //...EOL or /*...*/  ==> Blot out all non-newline chars
         return blotOutNonNewlines(s)
      else:                  # Matched string is '...' or "..."  ==> Keep unchanged
         return s

   pattern = re.compile(\
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE)

   return re.sub(pattern, replacer, text)


# Here functions are defined to process C++ code
def processCppCodeImpl(code):
    #code = commentRemover(code)
    ROOT.gInterpreter.ProcessLine(code)

def declareCppCodeImpl(code):
    #code = commentRemover(code)
    ROOT.gInterpreter.Declare(code)

def processCppCode(code):
    processCppCodeImpl(code)

def declareCppCode(code):
    declareCppCodeImpl(code)

def _checkOutput(command,errMsg=None):
    out = ""
    try:
        out = check_output(command.split())
    except:
        if errMsg:
            sys.stderr.write("%s (command was %s)\n" %(errMsg,command))
    return out

def _invokeAclicMac(fileName):
    '''FIXME!
    This function is a workaround. On osx, it is impossible to link against
    libzmq.so, among the others. The error is known and is
    "ld: can't link with bundle (MH_BUNDLE) only dylibs (MH_DYLIB)"
    We cannot at the moment force Aclic to change the linker command in order
    to exclude these libraries, so we launch a second root session to compile
    the library, which we then load.
    '''
    command = 'root -l -q -b -e gSystem->CompileMacro(\"%s\",\"k\")*0'%fileName
    out = _checkOutput(command, "Error ivoking ACLiC")
    libNameBase = fileName.replace(".C","_C")
    ROOT.gSystem.Load(libNameBase)

def _codeToFilename(code):
    '''Convert code to a unique file name

    >>> _codeToFilename("int f(i){return i*i;}")
    'dbf7e731.C'
    '''
    fileNameBase = sha1(code).hexdigest()[0:8]
    return fileNameBase + ".C"

def _dumpToUniqueFile(code):
    '''Dump code to file whose name is unique

    >>> _codeToFilename("int f(i){return i*i;}")
    'dbf7e731.C'
    '''
    fileName = _codeToFilename(code)
    with open (fileName,'w') as ofile:
      ofile.write(code)
    return fileName

def invokeAclic(cell):
    fileName = _dumpToUniqueFile(cell)
    if _getPlatform() == 'darwin':
        _invokeAclicMac(fileName)
    else:
        processCppCode(".L %s+" %fileName)


class StreamCapture(object):
    def __init__(self, stream, ip=get_ipython()):
        streamsFileNo={sys.stderr:2,sys.stdout:1}
        self.pipe_out = None
        self.pipe_in = None
        self.sysStreamFile = stream
        self.sysStreamFileNo = streamsFileNo[stream]
        self.shell = ip
        # Platform independent flush
        # With ctypes, the name of the libc library is not known a priori
        # We use jitted function
        flushFunctionName='_ROOTaaS_Flush'
        if (not hasattr(ROOT,flushFunctionName)):
           declareCppCode("void %s(){fflush(nullptr);};" %flushFunctionName)
        self.flush = getattr(ROOT,flushFunctionName)


    def more_data(self):
        r, _, _ = select.select([self.pipe_out], [], [], 0)
        return bool(r)

    def pre_execute(self):
        self.pipe_out, self.pipe_in = os.pipe()
        os.dup2(self.pipe_in, self.sysStreamFileNo)

    def post_execute(self):
        out = ''
        if self.pipe_out:
            while self.more_data():
                out += os.read(self.pipe_out, 1024)

        self.flush()
        self.sysStreamFile.write(out) # important to print the value printing output
        return 0

    def register(self):
        self.shell.events.register('pre_execute', self.pre_execute)
        self.shell.events.register('post_execute', self.post_execute)

class CaptureDrawnCanvases(object):
    '''
    Capture the canvas which is drawn to display it.
    '''
    def __init__(self, ip=get_ipython()):
        self.shell = ip

    def _pre_execute(self):
        pass

    def _post_execute(self):
        for can in ROOT.gROOT.GetListOfCanvases():
            if can.IsDrawn():
               can.Draw()
               can.ResetDrawn()

    def register(self):
        self.shell.events.register('pre_execute', self._pre_execute)
        self.shell.events.register('post_execute', self._post_execute)


captures = [StreamCapture(sys.stderr),
            StreamCapture(sys.stdout),
            CaptureDrawnCanvases()]

def toCpp():
    '''
    Change the mode of the notebook to CPP. It is preferred to use cell magic,
    but this option is handy to set up servers and for debugging purposes.
    '''
    ip = get_ipython()
    cpptransformer.load_ipython_extension(ip)
    # Change highlight mode
    IPython.display.display_javascript(_jsDefaultHighlight.format(mimeType = cppMIME), raw=True)
    print "Notebook is in Cpp mode"

class CanvasDrawer(object):
    '''
    Capture the canvas which is drawn and decide if it should be displayed using
    jsROOT.
    '''
    jsUID = 0

    def __init__(self, thePad):
        self.thePad = thePad

    def _getListOfPrimitivesNamesAndTypes(self):
       """
       Get the list of primitives in the pad, recursively descending into
       histograms and graphs looking for fitted functions.
       """
       primitives = self.thePad.GetListOfPrimitives()
       primitivesNames = map(lambda p: p.ClassName(), primitives)
       #primitivesWithFunctions = filter(lambda primitive: hasattr(primitive,"GetListOfFunctions"), primitives)
       #for primitiveWithFunctions in primitivesWithFunctions:
       #    for function in primitiveWithFunctions.GetListOfFunctions():
       #        primitivesNames.append(function.GetName())
       return sorted(primitivesNames)

    def _getUID(self):
        '''
        Every DIV containing a JavaScript snippet must be unique in the
        notebook. This methods provides a unique identifier.
        '''
        CanvasDrawer.jsUID += 1
        return CanvasDrawer.jsUID

    def _canJsDisplay(self):
        # to be optimised
        if not _enableJSVis: return False
        primitivesTypesNames = self._getListOfPrimitivesNamesAndTypes()
        for unsupportedPattern in _jsNotDrawableClassesPatterns:
            for primitiveTypeName in primitivesTypesNames:
                if fnmatch.fnmatch(primitiveTypeName,unsupportedPattern):
                    print >> sys.stderr, "The canvas contains an object of a type jsROOT cannot currently handle (%s). Falling back to a static png." %primitiveTypeName
                    return False
        return True

    def _jsDisplay(self):
        # Workaround to have ConvertToJSON work
        pad = ROOT.gROOT.GetListOfCanvases().FindObject(ROOT.gPad.GetName())
        json = ROOT.TBufferJSON.ConvertToJSON(pad, 3)
        #print "JSON:",json

        # Here we could optimise the string manipulation
        divId = 'root_plot_' + str(self._getUID())
        thisJsCode = _jsCode.format(jsCanvasWidth = _jsCanvasWidth,
                                    jsCanvasHeight = _jsCanvasHeight,
                                    jsROOTSourceDir = _jsROOTSourceDir,
                                    jsonContent=json.Data(),
                                    jsDrawOptions="",
                                    jsDivId = divId)

        # display is the key point of this hook
        IPython.display.display(HTML(thisJsCode))
        return 0

    def _pngDisplay(self):
        ofile = tempfile.NamedTemporaryFile(suffix=".png")
        with _setIgnoreLevel(ROOT.kError):
            self.thePad.SaveAs(ofile.name)
        img = IPython.display.Image(filename=ofile.name, format='png', embed=True)
        IPython.display.display(img)
        return 0

    def _display(self):
       if _enableJSVisDebug:
          self._pngDisplay()
          self._jsDisplay()
       else:
         if self._canJsDisplay():
            self._jsDisplay()
         else:
            self._pngDisplay()


    def Draw(self):
        self._display()
        return 0

def _PyDraw(thePad):
   """
   Invoke the draw function and intercept the graphics
   """
   drawer = CanvasDrawer(thePad)
   drawer.Draw()


def setStyle():
    style=ROOT.gStyle
    style.SetFuncWidth(3)
    style.SetHistLineWidth(3)
    style.SetMarkerStyle(8)
    style.SetMarkerSize(.5)
    style.SetMarkerColor(ROOT.kBlue)
    style.SetPalette(57)

def loadExtensionsAndCapturers():
    extNames = ["ROOTaaS.iPyROOT." + name for name in ["cppmagic"]]
    ip = get_ipython()
    extMgr = ExtensionManager(ip)
    for extName in extNames:
        extMgr.load_extension(extName)
    cppcompleter.load_ipython_extension(ip)

    for capture in captures: capture.register()

def enhanceROOTModule():
    ROOT.toCpp = toCpp
    ROOT.enableJSVis = enableJSVis
    ROOT.disableJSVis = disableJSVis
    ROOT.enableJSVisDebug = enableJSVisDebug
    ROOT.disableJSVisDebug = disableJSVisDebug
    ROOT.TCanvas.DrawCpp = ROOT.TCanvas.Draw
    ROOT.TCanvas.Draw = _PyDraw

def enableCppHighlighting():
    ipDispJs = IPython.display.display_javascript
    #Make sure clike JS lexer is loaded
    ipDispJs("require(['codemirror/mode/clike/clike'], function(Clike) { console.log('ROOTaaS - C++ CodeMirror module loaded'); });", raw=True)
    # Define highlight mode for %%cpp magic
    ipDispJs(_jsMagicHighlight.format(cppMIME = cppMIME), raw=True)

def iPythonize():
    setStyle()
    loadExtensionsAndCapturers()
    enableCppHighlighting()
    enhanceROOTModule()
    welcomeMsg()

