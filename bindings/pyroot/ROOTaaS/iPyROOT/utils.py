import os
import sys
import select
import time
import tempfile
import itertools
import ctypes
import re
from contextlib import contextmanager
from IPython import get_ipython
from IPython.display import HTML
import IPython.display
import ROOT
import cpptransformer
import cppcompleter


# We want iPython to take over the graphics
ROOT.gROOT.SetBatch()


cppMIME = 'text/x-c++src'
ipyMIME = 'text/x-ipython'

jsDefaultHighlight = """
// Set default mode for code cells
IPython.CodeCell.options_default.cm_config.mode = '{mimeType}';
// Set CodeMirror's current mode
var cells = IPython.notebook.get_cells();
cells[cells.length-1].code_mirror.setOption('mode', '{mimeType}');
// Set current mode for newly created cell
cells[cells.length-1].cm_config.mode = '{mimeType}';
"""

jsMagicHighlight = "IPython.CodeCell.config_defaults.highlight_modes['magic_{cppMIME}'] = {{'reg':[/^%%cpp|^%%dcl/]}};"


_jsNotDrawableClassesNames = ["TGraph2D"]

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

def _loadLibrary(libName):
   """
   Dl-open a library bypassing the ROOT calling sequence
   """
   return ctypes.cdll.LoadLibrary(libName)

def welcomeMsg():
    print "Welcome to ROOTaas Beta"

@contextmanager
def _setIgnoreLevel(level):
    originalLevel = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = level
    yield
    ROOT.gErrorIgnoreLevel = originalLevel

def commentRemover( text ):
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

class StreamCapture(object):
    def __init__(self, stream, ip=get_ipython()):
        streamsFileNo={sys.stderr:2,sys.stdout:1}
        self.pipe_out = None
        self.pipe_in = None
        self.sysStreamFile = stream
        self.sysStreamFileNo = streamsFileNo[stream]
        self.shell = ip
        self.libc = _loadLibrary("libc.so.6")

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

        self.libc.fflush(None)
        self.sysStreamFile.write(out) # important to print the value printing output
        return 0

    def register(self):
        self.shell.events.register('pre_execute', self.pre_execute)
        self.shell.events.register('post_execute', self.post_execute)

class CanvasCapture(object):
    '''
    Capture the canvas which is drawn and decide if it should be displayed using
    jsROOT.
    '''
    def __init__(self, ip=get_ipython()):
        self.shell = ip
        self.canvas = None
        self.primitivesNames = []
        self.jsUID = 0

    def _hasGPad(self):
        '''
        Check if a gPad is available at all.
        '''
        if not sys.modules.has_key("ROOT"): return False
        if not ROOT.gPad: return False
        return True

    def _isCanvasEmpty(self):
        '''
        Check if the canvas contains any primitive.
        '''
        if not self._hasGPad(): return True
        return len(ROOT.gPad.GetListOfPrimitives()) == 0


    def _getListOfPrimitivesNamesAndTypes(self):
       """
       Get the list of primitives in the pad, recursively descending into
       histograms and graphs looking for fitted functions.
       """
       if not ROOT.gPad: return []
       primitives = ROOT.gPad.GetListOfPrimitives()
       primitivesNames = map(lambda p: p.GetName(), primitives)
       primitivesWithFunctions = filter(lambda primitive: hasattr(primitive,"GetListOfFunctions"), primitives)
       for primitiveWithFunctions in primitivesWithFunctions:
           for function in primitiveWithFunctions.GetListOfFunctions():
               primitivesNames.append(function.GetName())
       return sorted(primitivesNames)

    def _pre_execute(self):
        if not self._hasGPad(): return 0
        gPad = ROOT.gPad
        self.primitivesNames = self._getListOfPrimitivesNamesAndTypes()
        self.canvas = gPad

    def _hasDifferentPrimitives(self):
        '''
        Check if the graphic primitives in the canvas are different from the
        previous pass.
        TODO: names are not enough. According to the primitive, additional
        checks are to be performed such as position, number of points, mean,
        rms and so on.
        '''
        newPrimitivesNames = self._getListOfPrimitivesNamesAndTypes()
        return newPrimitivesNames != self.primitivesNames

    def _canJsDisplay(self):
        # to be optimised
        if not _enableJSVis: return False
        primitivesNames = self.primitivesNames
        for jsNotDrawClassName in _jsNotDrawableClassesNames:
            if jsNotDrawClassName in primitivesNames:
                print >> sys.stderr, "The canvas contains an object which jsROOT cannot currently handle (%s). Falling back to a static png." %jsNotDrawClassName
                return False
        return True

    def _getUID(self):
        '''
        Every DIV containing a JavaScript snippet must be unique in the
        notebook. This methods provides a unique identifier.
        '''
        self.jsUID += 1
        return self.jsUID

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
        ROOT.gPad.SaveAs(ofile.name)
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


    def _post_execute(self):
        if self._isCanvasEmpty() or not self._hasGPad(): return 0
        gPad = ROOT.gPad
        isNew = not self.canvas
        if not (isNew or self._hasDifferentPrimitives()): return 0
        gPad.Update()

        parentCanvas = gPad.GetCanvas()
        if (parentCanvas):
           ROOT.gPad=parentCanvas

        self._display()

        ROOT.gPad=gPad

        return 0

    def register(self):
        self.shell.events.register('pre_execute', self._pre_execute)
        self.shell.events.register('post_execute', self._post_execute)

    def unregister(self):
        self.shell.events.unregister('pre_execute', self._pre_execute)
        self.shell.events.unregister('post_execute', self._post_execute)

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
    cppcompleter.load_ipython_extension(ip)
    # Change highlight mode
    IPython.display.display_javascript(jsDefaultHighlight.format(mimeType = cppMIME), raw=True)
    print "Notebook is in Cpp mode"

class CanvasDrawer(object):
    '''
    Capture the canvas which is drawn and decide if it should be displayed using
    jsROOT.
    '''
    def __init__(self, thePad):
        self.thePad = thePad
        self.jsUID = 0

    def _getListOfPrimitivesNamesAndTypes(self):
       """
       Get the list of primitives in the pad, recursively descending into
       histograms and graphs looking for fitted functions.
       """
       primitives = self.thePad.GetListOfPrimitives()
       primitivesNames = map(lambda p: p.GetName(), primitives)
       #primitivesWithFunctions = filter(lambda primitive: hasattr(primitive,"GetListOfFunctions"), primitives)
       #for primitiveWithFunctions in primitivesWithFunctions:
       #    for function in primitiveWithFunctions.GetListOfFunctions():
       #        primitivesNames.append(function.GetName())
       return sorted(primitivesNames)

    def _canJsDisplay(self):
        # to be optimised
        if not _enableJSVis: return False
        primitivesNamesAndTypes = self._getListOfPrimitivesNamesAndTypes()
        for jsNotDrawClassName in _jsNotDrawableClassesNames:
            if jsNotDrawClassName in primitivesNamesAndTypes:
                print >> sys.stderr, "The canvas contains an object which jsROOT cannot currently handle (%s). Falling back to a static png." %jsNotDrawClassName
                return False
        return True

    def _getUID(self):
        '''
        Every DIV containing a JavaScript snippet must be unique in the
        notebook. This methods provides a unique identifier.
        '''
        self.jsUID += 1
        return self.jsUID

    def _jsDisplay(self):
        # Workaround to have ConvertToJSON work
        pad = ROOT.gROOT.GetListOfCanvases().FindObject(self.thePad.GetName())
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

# Here functions are defined to process C++ code
def processCppCodeImpl(cell):
    cell = commentRemover(cell)
    ROOT.gInterpreter.ProcessLine(cell)

def declareCppCodeImpl(cell):
    cell = commentRemover(cell)
    ROOT.gInterpreter.Declare(cell)

def processCppCode(cell):
    processCppCodeImpl(cell)

def declareCppCode(cell):
    declareCppCodeImpl(cell)
