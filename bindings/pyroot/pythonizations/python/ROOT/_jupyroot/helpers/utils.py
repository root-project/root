# -*- coding:utf-8 -*-

# -----------------------------------------------------------------------------
#  Author: Danilo Piparo <Danilo.Piparo@cern.ch> CERN
# -----------------------------------------------------------------------------

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


from __future__ import print_function

import fnmatch
import os
import re
import sys
import tempfile
import time
import ctypes
from contextlib import contextmanager
from datetime import datetime
from hashlib import sha1
from subprocess import check_output

from IPython import display, get_ipython
from IPython.core.extensions import ExtensionManager

import ROOT
from ROOT._jupyroot.helpers import handlers

# We want iPython to take over the graphics
ROOT.gROOT.SetBatch()

cppMIME = "text/x-c++src"

_enableJSVis = True

# Keep display handle for canvases to be able update them
_canvasHandles = {}

_visualObjects = []

_jsMagicHighlight = """
Jupyter.CodeCell.options_default.highlight_modes['magic_{cppMIME}'] = {{'reg':[/^%%cpp/]}};
console.log("JupyROOT - %%cpp magic configured");
"""

_jsNotDrawableClassesPatterns = ["TEve*"]

_jsCanvasWidth = 800
_jsCanvasHeight = 600

_jsFixedSizeDiv = """
<div id="{jsDivId}" style="width: {jsCanvasWidth}px; height: {jsCanvasHeight}px; position: relative">
</div>
"""

_jsFullWidthDiv = """
<div style="width: 100%; height: {jsCanvasHeight}px; position: relative">
   <div id="{jsDivId}">
   </div>
</div>
"""

_jsDrawJsonCode = """
Core.unzipJSON({jsonLength},'{jsonZip}').then(json => {{
   const obj = Core.parse(json);
   Core.draw('{jsDivId}', obj, '{jsDrawOptions}');
}});
"""

_jsBrowseFileCode = """
const binaryString = atob('{fileBase64}');
const bytes = new Uint8Array(binaryString.length);
for (let i = 0; i < binaryString.length; i++)
   bytes[i] = binaryString.charCodeAt(i);
Core.buildGUI('{jsDivId}','notebook').then(h => h.openRootFile(bytes.buffer));
"""

_jsCode = """
{jsDivHtml}
</div>
<script>
   function process_{jsDivId}() {{
      function execCode(Core) {{
         Core.settings.HandleKeys = false;
         {jsDrawCode}
      }}
      const servers = ['/static/', 'https://jsroot.gsi.de/dev/', 'https://root.cern/js/dev/'],
            path = 'build/jsroot';
      if (typeof JSROOT !== 'undefined')
         execCode(JSROOT);
      else if (typeof requirejs !== 'undefined') {{
         servers.forEach((s,i) => {{ servers[i] = s + path; }});
         requirejs.config({{ paths: {{ 'jsroot' : servers }} }})(['jsroot'],  execCode);
      }} else {{
         const config = document.getElementById('jupyter-config-data');
         if (config)
            servers[0] = (JSON.parse(config.innerHTML || '{{}}')?.baseUrl || '/') + 'static/';
         else
            servers.shift();
         function loadJsroot() {{
            return !servers.length ? 0 : import(servers.shift() + path + '.js').catch(loadJsroot).then(() => execCode(JSROOT));
         }}
         loadJsroot();
      }}
   }}
   process_{jsDivId}();
</script>
"""


_jsFileCode = """
<div style="width: 100%; height: {jsCanvasHeight}px; position: relative">
   <div id="{jsDivId}">
   </div>
</div>
<script>
   function process_{jsDivId}() {{
      function showFile(Core) {{
         Core.settings.HandleKeys = false;
         const binaryString = atob('{fileBase64}');
         const bytes = new Uint8Array(binaryString.length);
         for (let i = 0; i < binaryString.length; i++)
            bytes[i] = binaryString.charCodeAt(i);
         Core.buildGUI('{jsDivId}','notebook').then(h => h.openRootFile(bytes.buffer));
      }}
      const servers = ['/static/', 'https://jsroot.gsi.de/dev/', 'https://root.cern/js/dev/'],
            path = 'build/jsroot';
      if (typeof JSROOT !== 'undefined')
         showFile(JSROOT);
      else if (typeof requirejs !== 'undefined') {{
         servers.forEach((s,i) => {{ servers[i] = s + path; }});
         requirejs.config({{ paths: {{ 'jsroot' : servers }} }})(['jsroot'],  showFile);
      }} else {{
         const config = document.getElementById('jupyter-config-data');
         if (config)
            servers[0] = (JSON.parse(config.innerHTML || '{{}}')?.baseUrl || '/') + 'static/';
         else
            servers.shift();
         function loadJsroot() {{
            return !servers.length ? 0 : import(servers.shift() + path + '.js').catch(loadJsroot).then(() => showFile(JSROOT));
         }}
         loadJsroot();
      }}
   }}
   process_{jsDivId}();
</script>
"""



TBufferJSONErrorMessage = "The TBufferJSON class is necessary for JS visualisation to work and cannot be found. Did you enable the http module (-D http=ON for CMake)?"


def TBufferJSONAvailable():
    if not hasattr(ROOT, "TBufferJSON"):
        return False
    if not hasattr(ROOT.TBufferJSON, "ConvertToJSON"):
        print(TBufferJSONErrorMessage, file=sys.stderr)
        return False
    return True


def TWebCanvasAvailable():
    if not hasattr(ROOT, "TWebCanvas"):
        return False
    if not hasattr(ROOT.TWebCanvas, "CreateCanvasJSON"):
        return False
    return True


def RCanvasAvailable():
    if not hasattr(ROOT, "Experimental"):
        return False
    if not hasattr(ROOT.Experimental, "RCanvas"):
        return False
    return True


def initializeJSVis():
    global _enableJSVis
    jupyter_jsroot = ROOT.gEnv.GetValue("Jupyter.JSRoot", "on").lower()
    if jupyter_jsroot not in {"on", "off"}:
        print(f"Invalid Jupyter.JSRoot value '{jupyter_jsroot}' in .rootrc. Using default 'on'.")
        jupyter_jsroot = "on"
    _enableJSVis = jupyter_jsroot == "on"


def enableJSVis(flag=True):
    global _enableJSVis
    _enableJSVis = flag and TBufferJSONAvailable()


def addVisualObject(object, kind="none", option=""):
    global _visualObjects
    if kind == "tfile":
        _visualObjects.append(NotebookDrawerFile(object, option))
    else:
        _visualObjects.append(NotebookDrawerJson(object))


def _getPlatform():
    return sys.platform


def _getUniqueDivId():
    """
    Every DIV containing a JavaScript snippet must be unique in the
    notebook. This method provides a unique identifier.
    With the introduction of JupyterLab, multiple Notebooks can exist
    simultaneously on the same HTML page. In order to ensure a unique
    identifier with the UID throughout all open Notebooks the UID is
    generated as a timestamp.
    """
    return "root_plot_" + str(int(round(time.time() * 1000)))



def _getLibExtension(thePlatform):
    """Return appropriate file extension for a shared library
    >>> _getLibExtension('darwin')
    '.dylib'
    >>> _getLibExtension('win32')
    '.dll'
    >>> _getLibExtension('OddPlatform')
    '.so'
    """
    pExtMap = {"darwin": ".dylib", "win32": ".dll"}
    return pExtMap.get(thePlatform, ".so")


@contextmanager
def _setIgnoreLevel(level):
    originalLevel = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = level
    yield
    ROOT.gErrorIgnoreLevel = originalLevel


def commentRemover(text):
    """
    >>> s="// hello"
    >>> commentRemover(s)
    ''
    >>> s="int /** Test **/ main() {return 0;}"
    >>> commentRemover(s)
    'int  main() {return 0;}'
    """

    def blotOutNonNewlines(strIn):  # Return a string containing only the newline chars contained in strIn
        return "" + ("\n" * strIn.count("\n"))

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):  # Matched string is //...EOL or /*...*/  ==> Blot out all non-newline chars
            return blotOutNonNewlines(s)
        else:  # Matched string is '...' or "..."  ==> Keep unchanged
            return s

    pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', re.DOTALL | re.MULTILINE)

    return re.sub(pattern, replacer, text)


# Here functions are defined to process C++ code
def processCppCodeImpl(code):
    # code = commentRemover(code)
    ROOT.gInterpreter.ProcessLine(code)


def processMagicCppCodeImpl(code):
    err = ROOT.ProcessLineWrapper(code)
    if err == ROOT.TInterpreter.kProcessing:
        ROOT.gInterpreter.ProcessLine(".@")
        ROOT.gInterpreter.ProcessLine('cerr << "Unbalanced braces. This cell was not processed." << endl;')


def declareCppCodeImpl(code):
    # code = commentRemover(code)
    ROOT.gInterpreter.Declare(code)


def processCppCode(code):
    processCppCodeImpl(code)


def processMagicCppCode(code):
    processMagicCppCodeImpl(code)


def declareCppCode(code):
    declareCppCodeImpl(code)


def _checkOutput(command, errMsg=None):
    out = ""
    try:
        out = check_output(command.split())
    except Exception:
        if errMsg:
            sys.stderr.write("%s (command was %s)\n" % (errMsg, command))
    return out


def _invokeAclicMac(fileName):
    """FIXME!
    This function is a workaround. On osx, it is impossible to link against
    libzmq.so, among the others. The error is known and is
    "ld: can't link with bundle (MH_BUNDLE) only dylibs (MH_DYLIB)"
    We cannot at the moment force Aclic to change the linker command in order
    to exclude these libraries, so we launch a second root session to compile
    the library, which we then load.
    """
    command = 'root -q -b -e gSystem->CompileMacro("%s","k")*0' % fileName
    out = _checkOutput(command, "Error ivoking ACLiC")  # noqa: F841
    libNameBase = fileName.replace(".C", "_C")
    ROOT.gSystem.Load(libNameBase)


def _codeToFilename(code):
    """Convert code to a unique file name

    >>> code = "int f(i){return i*i;}"
    >>> _codeToFilename(code)[0:9]
    'dbf7e731_'
    >>> _codeToFilename(code)[9:-2].isdigit()
    True
    >>> _codeToFilename(code)[-2:]
    '.C'
    """
    code_enc = code if type(code) is bytes else code.encode("utf-8")
    fileNameBase = sha1(code_enc).hexdigest()[0:8]
    timestamp = datetime.now().strftime("%H%M%S%f")
    return fileNameBase + "_" + timestamp + ".C"


def _dumpToUniqueFile(code):
    """Dump code to file whose name is unique

    >>> code = "int f(i){return i*i;}"
    >>> _dumpToUniqueFile(code)[0:9]
    'dbf7e731_'
    >>> _dumpToUniqueFile(code)[9:-2].isdigit()
    True
    >>> _dumpToUniqueFile(code)[-2:]
    '.C'
    """
    fileName = _codeToFilename(code)
    with open(fileName, "w") as ofile:
        code_dec = code if type(code) is not bytes else code.decode("utf-8")
        ofile.write(code_dec)
    return fileName


def isPlatformApple():
    return _getPlatform() == "darwin"


def invokeAclic(cell):
    fileName = _dumpToUniqueFile(cell)
    if isPlatformApple():
        _invokeAclicMac(fileName)
    else:
        processCppCode(".L %s+" % fileName)


transformers = []


class StreamCapture(object):
    def __init__(self, ip=get_ipython()):
        # For the registration
        self.shell = ip

        self.ioHandler = handlers.IOHandler()
        self.flag = True
        self.outString = ""
        self.errString = ""

        self.poller = handlers.Poller()
        self.poller.start()
        self.asyncCapturer = handlers.Runner(self.syncCapture, self.poller)

        self.isFirstPreExecute = True
        self.isFirstPostExecute = True

    def syncCapture(self, defout=""):
        self.outString = defout
        self.errString = defout
        waitTimes = [0.01, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
        lenWaitTimes = 7

        iterIndex = 0
        while self.flag:
            self.ioHandler.Poll()
            if not self.flag:
                return
            waitTime = 0.1 if iterIndex >= lenWaitTimes else waitTimes[iterIndex]
            time.sleep(waitTime)

    def pre_execute(self):
        if self.isFirstPreExecute:
            self.isFirstPreExecute = False
            return 0

        self.flag = True
        self.ioHandler.Clear()
        self.ioHandler.InitCapture()
        self.asyncCapturer.AsyncRun("")

    def post_execute(self):
        if self.isFirstPostExecute:
            self.isFirstPostExecute = False
            self.isFirstPreExecute = False
            return 0
        self.flag = False
        self.asyncCapturer.Wait()
        self.ioHandler.Poll()
        self.ioHandler.EndCapture()

        # Print for the notebook
        out = self.ioHandler.GetStdout()
        err = self.ioHandler.GetStderr()
        if not transformers:
            sys.stdout.write(out)
            sys.stderr.write(err)
        else:
            for t in transformers:
                (out, err, otype) = t(out, err)
                if otype == "html":
                    display.display(display.HTML(out))
                    display.display(display.HTML(err))
        return 0

    def register(self):
        self.shell.events.register("pre_execute", self.pre_execute)
        self.shell.events.register("post_execute", self.post_execute)

    def __del__(self):
        self.poller.Stop()


def GetCanvasDrawers():
    lOfC = ROOT.gROOT.GetListOfCanvases()
    return [NotebookDrawerTCanvas(can) for can in lOfC if can.IsDrawn() or can.IsUpdated()]


def GetRCanvasDrawers():
    if not RCanvasAvailable():
        return []
    lOfC = ROOT.Experimental.RCanvas.GetCanvases()
    return [NotebookDrawerRCanvas(can.__smartptr__().get()) for can in lOfC if can.IsShown() or can.IsUpdated()]

def GetVisualDrawers():
    global _visualObjects
    res = []
    for obj in _visualObjects:
        res.append(obj)
    _visualObjects.clear()
    return res

def GetGeometryDrawers():
    if not hasattr(ROOT, "gGeoManager"):
        return []
    if not ROOT.gGeoManager:
        return []
    vol = ROOT.gGeoManager.GetUserPaintVolume()
    if not vol:
        return []
    return [NotebookDrawerGeometry(vol)]


def GetDrawers():
    return GetCanvasDrawers() + GetRCanvasDrawers() + GetVisualDrawers() + GetGeometryDrawers()


def NotebookDraw():
    drawers = GetDrawers()
    for drawer in drawers:
        drawer.Draw(display.display)


class CaptureDrawnPrimitives(object):
    """
    Capture the canvas which is drawn to display it.
    """

    def __init__(self, ip=get_ipython()):
        self.shell = ip

    def _post_execute(self):
        NotebookDraw()

    def register(self):
        self.shell.events.register("post_execute", self._post_execute)




class NotebookDrawerFile(object):
    """
    Special drawer for TFile - shows files hierarchy with possibility to draw objects
    """

    def __init__(self, theObject, theOption=""):
       self.drawObject = theObject
       self.drawOption = theOption

    def _getFileJsCode(self):
        sz = self.drawObject.GetSize()
        if sz > 10000000 and self.drawOption != "force":
            return f"File size {sz} is too large for JSROOT display. Use 'force' draw option to show file nevertheless"

        # create plain buffer and get pointer on it
        u_buffer = (ctypes.c_ubyte * sz)(*range(sz))
        addrc = ctypes.cast(ctypes.pointer(u_buffer), ctypes.c_char_p)

        if self.drawObject.ReadBuffer(addrc, 0, sz):
           return f"Fail to read file {self.drawObject.GetName()} buffer of size {sz}"

        base64 = ROOT.TBase64.Encode(addrc, sz)

        id = _getUniqueDivId()

        drawHtml = _jsFullWidthDiv.format(
            jsDivId=id,
            jsCanvasHeight=_jsCanvasHeight
        )

        browseFileCode = _jsBrowseFileCode.format(
            jsDivId=id,
            fileBase64=base64
        )

        thisJsCode = _jsCode.format(
            jsDivId=id,
            jsDivHtml=drawHtml,
            jsDrawCode=browseFileCode
        )

        return thisJsCode

    def Draw(self, displayFunction):
        code = self._getFileJsCode()
        displayFunction(display.HTML(code))



class NotebookDrawerJson(object):
    """
    Generic class to create JSROOT drawing for the arbitrary object based on json conversion.
    """

    def __init__(self, theObject):
       self.drawObject = theObject

    def _canJsDisplay(self):
        return TBufferJSONAvailable()

    def _canPngDisplay(self):
        return True

    def _getWidth(self):
        return _jsCanvasWidth

    def _getHeight(self):
        return _jsCanvasHeight

    def _getJson(self):
        return ROOT.TBufferJSON.ConvertToJSON(self.drawObject, 23).Data()

    def _getJsOptions(self):
        return ""

    def _getJsCode(self):
        width = self._getWidth()
        height = self._getHeight()
        json = self._getJson()
        options = self._getJsOptions()

        if not json:
            return f"Class {self.drawObject.ClassName()} not supported yet"

        zip = ROOT.TBufferJSON.zipJSON(json)

        id = _getUniqueDivId()

        drawHtml = _jsFixedSizeDiv.format(
            jsDivId=id,
            jsCanvasWidth=width,
            jsCanvasHeight=height
        )

        drawJsonCode = _jsDrawJsonCode.format(
            jsDivId=id,
            jsonLength=len(json),
            jsonZip=zip,
            jsDrawOptions=options
        )

        thisJsCode = _jsCode.format(
            jsDivId=id,
            jsDivHtml=drawHtml,
            jsDrawCode=drawJsonCode
        )
        return thisJsCode

    def _getCanvas(self):
        c1 = ROOT.TCanvas("__tmp_draw_image_canvas__", "", self._getWidth(), self._getHeight())
        c1.Add(self.drawObject)
        return c1

    def _getPngImage(self):
        ofile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        canv = self._getCanvas()
        with _setIgnoreLevel(ROOT.kError):
            canv.SaveAs(ofile.name)
        img = display.Image(filename=ofile.name, format="png", embed=True)
        ofile.close()
        os.unlink(ofile.name)
        return img


    def Draw(self, displayFunction):
        global _enableJSVis
        if _enableJSVis and self._canJsDisplay():
            code = self._getJsCode()
            displayFunction(display.HTML(code))
        elif self._canPngDisplay():
            displayFunction(self._getPngImage())
        else:
            displayFunction(display.HTML(f"Neither JSROOT nor plain drawing of {self.drawObject.ClassName()} class is not implemented"))



class NotebookDrawerGeometry(NotebookDrawerJson):
    """
    Drawer for geometry - png is not works with it.
    """

    def __del__(self):
        self.drawObject.GetGeoManager().SetUserPaintVolume(ROOT.nullptr)

    def _getJsOptions(self):
        return "all"

    def _canPngDisplay(self):
        return False



class NotebookDrawerCanvBase(NotebookDrawerJson):
    """
    Base class for TCanvas/RCanvas drawing.
    Implementst specific Draw function where canvas update is handled
    """

    def _getCanvasId(self):
        return ""

    def _getUpdated(self):
        return False

    def _getCanvas(self):
        return self.drawObject

    def Draw(self, displayFunction):
        global _enableJSVis, _canvasHandles
        code = ""
        if _enableJSVis and self._canJsDisplay():
            code = display.HTML(self._getJsCode())
        elif self._canPngDisplay():
            code = self._getPngImage()
        else:
            code = display.HTML(f"Neither JSROOT nor plain drawing of {self.drawObject.ClassName()} is implemented")

        name = self._getCanvasId()
        updated = self._getUpdated()
        if updated and name and (name in _canvasHandles):
            _canvasHandles[name].update(code)
        elif name:
            _canvasHandles[name] = displayFunction(code, display_id=True)
        else:
            displayFunction(code)



class NotebookDrawerTCanvas(NotebookDrawerCanvBase):
    """
    Drawer of TCanvas.
    """

    def __del__(self):
        self.drawObject.ResetDrawn()
        self.drawObject.ResetUpdated()

    def _getCanvasId(self):
        return self.drawObject.GetName() + str(ROOT.AddressOf(self.drawObject)[0])

    def _getUpdated(self):
       return self.drawObject.IsUpdated()

    def _canJsDisplay(self):
        if not TBufferJSONAvailable():
            return False
        if TWebCanvasAvailable():
            return True
        primitives = self.drawObject.GetListOfPrimitives()
        primitivesTypesNames = sorted(map(lambda p: p.ClassName(), primitives))
        for unsupportedPattern in _jsNotDrawableClassesPatterns:
            for primitiveTypeName in primitivesTypesNames:
                if fnmatch.fnmatch(primitiveTypeName, unsupportedPattern):
                    print(
                        "The canvas contains an object of a type jsROOT cannot currently handle (%s). Falling back to a static png."
                        % primitiveTypeName,
                        file=sys.stderr,
                    )
                    return False
        return True

    def _getWidth(self):
        if self.drawObject.GetWindowWidth() > 0:
            return self.drawObject.GetWindowWidth()
        return _jsCanvasWidth

    def _getHeight(self):
        if self.drawObject.GetWindowHeight() > 0:
            return self.drawObject.GetWindowHeight()
        return _jsCanvasHeight

    def _getJson(self):
        if self.drawObject.IsUpdated() and not self.drawObject.IsDrawn():
            self.drawObject.Draw()

        if TWebCanvasAvailable():
            return ROOT.TWebCanvas.CreateCanvasJSON(self.drawObject, 23, True).Data()

        # Add extra primitives to canvas with custom colors, palette, gStyle

        prim = self.drawObject.GetListOfPrimitives()

        style = ROOT.gStyle
        colors = ROOT.gROOT.GetListOfColors()
        palette = None

        # always provide gStyle object
        if prim.FindObject(style):
            style = None
        else:
            prim.Add(style)

        cnt = 0
        for n in range(colors.GetLast() + 1):
            if colors.At(n):
                cnt = cnt + 1

        # add all colors if there are more than 598 colors defined
        if cnt < 599 or prim.FindObject(colors):
            colors = None
        else:
            prim.Add(colors)

        if colors:
            pal = ROOT.TColor.GetPalette()
            palette = ROOT.TObjArray()
            palette.SetName("CurrentColorPalette")
            for i in range(pal.GetSize()):
                palette.Add(colors.At(pal[i]))
            prim.Add(palette)

        ROOT.TColor.DefinedColors()

        canvas_json = ROOT.TBufferJSON.ConvertToJSON(self.drawObject, 23)

        # Cleanup primitives after conversion
        if style is not None:
            prim.Remove(style)
        if colors is not None:
            prim.Remove(colors)
        if palette is not None:
            prim.Remove(palette)

        return canvas_json.Data()


class NotebookDrawerRCanvas(NotebookDrawerCanvBase):
    """
    Drawer of RCanvas.
    """

    def __del__(self):
        self.drawObject.ClearShown()
        self.drawObject.ClearUpdated()

    def _getCanvasId(self):
        return self.drawObject.GetUID()

    def _getUpdated(self):
        return self.drawObject.IsUpdated()

    def _getWidth(self):
        if self.drawObject.GetWidth() > 0:
            return self.drawObject.GetWidth()
        return _jsCanvasWidth

    def _getHeight(self):
        if self.drawObject.GetHeight() > 0:
            return self.drawObject.GetHeight()
        return _jsCanvasHeight

    def _getJson(self):
        return self.drawObject.CreateJSON()



def setStyle():
    style = ROOT.gStyle
    style.SetFuncWidth(2)


captures = []


def loadMagicsAndCapturers():
    global captures
    extNames = ["ROOT._jupyroot.magics." + name for name in ["cppmagic", "jsrootmagic"]]
    ip = get_ipython()
    extMgr = ExtensionManager(ip)
    for extName in extNames:
        extMgr.load_extension(extName)
    captures.append(StreamCapture())
    captures.append(CaptureDrawnPrimitives())

    for capture in captures:
        capture.register()


def declareProcessLineWrapper():
    ROOT.gInterpreter.Declare("""
TInterpreter::EErrorCode ProcessLineWrapper(const char* line) {
    TInterpreter::EErrorCode err;
    gInterpreter->ProcessLine(line, &err);
    return err;
}
""")


def enhanceROOTModule():
    ROOT.enableJSVis = enableJSVis


def enableCppHighlighting():
    ipDispJs = display.display_javascript
    # Define highlight mode for %%cpp magic
    ipDispJs(_jsMagicHighlight.format(cppMIME=cppMIME), raw=True)


def iPythonize():
    setStyle()
    initializeJSVis()
    loadMagicsAndCapturers()
    declareProcessLineWrapper()
    # enableCppHighlighting()
    enhanceROOTModule()
