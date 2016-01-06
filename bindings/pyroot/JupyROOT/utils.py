# -*- coding:utf-8 -*-

#-----------------------------------------------------------------------------
#  Author: Danilo Piparo <Danilo.Piparo@cern.ch> CERN
#-----------------------------------------------------------------------------

from __future__ import print_function

import os
import sys
import select
import tempfile
import pty
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
    'JSRootCore'       : '{jsROOTSourceDir}/scripts/JSRootCore',
    'JSRootPainter'    : '{jsROOTSourceDir}/scripts/JSRootPainter',
    'JSRootGeoPainter' : '{jsROOTSourceDir}/scripts/JSRootGeoPainter',
  }}
}}
);
require(['JSRootCore', 'JSRootPainter', 'JSRootGeoPainter'],
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
    print("Welcome to JupyROOT %s" %ROOT.gROOT.GetVersion())

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

def isPlatformApple():
   return _getPlatform() == 'darwin';

def invokeAclic(cell):
    fileName = _dumpToUniqueFile(cell)
    if isPlatformApple():
        _invokeAclicMac(fileName)
    else:
        processCppCode(".L %s+" %fileName)

class StreamCapture(object):
    def __init__(self, stream, ip=get_ipython()):
        nbStreamsPyStreamsMap={sys.stderr:sys.__stderr__,sys.stdout:sys.__stdout__}
        self.shell = ip
        self.nbStream = stream
        self.pyStream = nbStreamsPyStreamsMap[stream]
        self.pipe_out, self.pipe_in = pty.openpty()
        os.dup2(self.pipe_in, self.pyStream.fileno())
        # Platform independent flush
        # With ctypes, the name of the libc library is not known a priori
        # We use jitted function
        flushFunctionName='_JupyROOT_Flush'
        if (not hasattr(ROOT,flushFunctionName)):
           declareCppCode("void %s(){fflush(nullptr);};" %flushFunctionName)
        self.flush = getattr(ROOT,flushFunctionName)

    def more_data(self):
        r, _, _ = select.select([self.pipe_out], [], [], 0)
        return bool(r)

    def post_execute(self):
        out = ''
        if self.pipe_out:
            while self.more_data():
                out += os.read(self.pipe_out, 8192)

        self.flush()
        self.nbStream.write(out) # important to print the value printing output
        return 0

    def register(self):
        self.shell.events.register('post_execute', self.post_execute)

def GetCanvasDrawers():
    lOfC = ROOT.gROOT.GetListOfCanvases()
    return [NotebookDrawer(can) for can in lOfC if can.IsDrawn()]

def GetGeometryDrawer():
    if not hasattr(ROOT,'gGeoManager'): return
    if not ROOT.gGeoManager: return
    if not ROOT.gGeoManager.GetUserPaintVolume(): return
    vol = ROOT.gGeoManager.GetTopVolume()
    if vol:
        return NotebookDrawer(vol)

def GetDrawers():
    drawers = GetCanvasDrawers()
    geometryDrawer = GetGeometryDrawer()
    if geometryDrawer: drawers.append(geometryDrawer)
    return drawers

def DrawGeometry():
    drawer = GetGeometryDrawer()
    if drawer:
        drawer.Draw()

def DrawCanvases():
    drawers = GetCanvasDrawers()
    for drawer in drawers:
        drawer.Draw()

def NotebookDraw():
    DrawGeometry()
    DrawCanvases()

class CaptureDrawnPrimitives(object):
    '''
    Capture the canvas which is drawn to display it.
    '''
    def __init__(self, ip=get_ipython()):
        self.shell = ip

    def _post_execute(self):
        NotebookDraw()

    def register(self):
        self.shell.events.register('post_execute', self._post_execute)

class NotebookDrawer(object):
    '''
    Capture the canvas which is drawn and decide if it should be displayed using
    jsROOT.
    '''
    jsUID = 0

    def __init__(self, theObject):
        self.drawableObject = theObject
        self.isCanvas = self.drawableObject.ClassName() == "TCanvas"

    def __del__(self):
       if self.isCanvas:
           self.drawableObject.ResetDrawn()
       else:
           ROOT.gGeoManager.SetUserPaintVolume(None)

    def _getListOfPrimitivesNamesAndTypes(self):
       """
       Get the list of primitives in the pad, recursively descending into
       histograms and graphs looking for fitted functions.
       """
       primitives = self.canvas.GetListOfPrimitives()
       primitivesNames = map(lambda p: p.ClassName(), primitives)
       return sorted(primitivesNames)

    def _getUID(self):
        '''
        Every DIV containing a JavaScript snippet must be unique in the
        notebook. This methods provides a unique identifier.
        '''
        NotebookDrawer.jsUID += 1
        return NotebookDrawer.jsUID

    def _canJsDisplay(self):
        if not hasattr(ROOT,"TBufferJSON"):
            print("The TBufferJSON class is necessary for JS visualisation to work and cannot be found. Did you enable the http module (-D http=ON for CMake)?", file=sys.stderr)
            return False
        if not self.isCanvas: return True
        # to be optimised
        if not _enableJSVis: return False
        primitivesTypesNames = self._getListOfPrimitivesNamesAndTypes()
        for unsupportedPattern in _jsNotDrawableClassesPatterns:
            for primitiveTypeName in primitivesTypesNames:
                if fnmatch.fnmatch(primitiveTypeName,unsupportedPattern):
                    print("The canvas contains an object of a type jsROOT cannot currently handle (%s). Falling back to a static png." %primitiveTypeName, file=sys.stderr)
                    return False
        return True


    def _getJsCode(self):
        # Workaround to have ConvertToJSON work
        json = ROOT.TBufferJSON.ConvertToJSON(self.drawableObject, 3)

        # Here we could optimise the string manipulation
        divId = 'root_plot_' + str(self._getUID())

        height = _jsCanvasHeight
        width = _jsCanvasHeight
        options = "all"

        if self.isCanvas:
            height = self.drawableObject.GetWw()
            width = self.drawableObject.GetWh()
            options = ""

        thisJsCode = _jsCode.format(jsCanvasWidth = height,
                                    jsCanvasHeight = width,
                                    jsROOTSourceDir = _jsROOTSourceDir,
                                    jsonContent = json.Data(),
                                    jsDrawOptions = options,
                                    jsDivId = divId)
        return thisJsCode

    def _getJsDiv(self):
        return HTML(self._getJsCode())

    def _jsDisplay(self):
        IPython.display.display(self._getJsDiv())
        return 0

    def _getPngImage(self):
        ofile = tempfile.NamedTemporaryFile(suffix=".png")
        with _setIgnoreLevel(ROOT.kError):
            self.drawableObject.SaveAs(ofile.name)
        img = IPython.display.Image(filename=ofile.name, format='png', embed=True)
        return img

    def _pngDisplay(self):
        img = self._getPngImage()
        IPython.display.display(img)

    def _display(self):
       if _enableJSVisDebug:
          self._pngDisplay()
          self._jsDisplay()
       else:
         if self._canJsDisplay():
            self._jsDisplay()
         else:
            self._pngDisplay()

    def GetDrawableObject(self):
        if not self.isCanvas:
           return self._getJsDiv()

        if self._canJsDisplay():
           return self._getJsDiv()
        else:
           return self._getPngImage()

    def Draw(self):
        self._display()
        return 0

def setStyle():
    style=ROOT.gStyle
    style.SetFuncWidth(3)
    style.SetHistLineWidth(3)
    style.SetMarkerStyle(8)
    style.SetMarkerSize(.5)
    style.SetMarkerColor(ROOT.kBlue)
    style.SetPalette(57)

captures = []

def loadExtensionsAndCapturers():
    global captures
    extNames = ["JupyROOT.magics." + name for name in ["cppmagic"]]
    ip = get_ipython()
    extMgr = ExtensionManager(ip)
    for extName in extNames:
        extMgr.load_extension(extName)
    cppcompleter.load_ipython_extension(ip)
    captures.append(StreamCapture(sys.stderr))
    captures.append(StreamCapture(sys.stdout))
    captures.append(CaptureDrawnPrimitives())

    for capture in captures: capture.register()

def enhanceROOTModule():
    ROOT.enableJSVis = enableJSVis
    ROOT.disableJSVis = disableJSVis
    ROOT.enableJSVisDebug = enableJSVisDebug
    ROOT.disableJSVisDebug = disableJSVisDebug

def enableCppHighlighting():
    ipDispJs = IPython.display.display_javascript
    # Define highlight mode for %%cpp magic
    ipDispJs(_jsMagicHighlight.format(cppMIME = cppMIME), raw=True)

def iPythonize():
    setStyle()
    loadExtensionsAndCapturers()
    enableCppHighlighting()
    enhanceROOTModule()
    welcomeMsg()

