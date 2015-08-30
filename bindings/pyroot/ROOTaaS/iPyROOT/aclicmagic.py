import IPython.core.magic as ipym
import utils
from hashlib import sha1

def _codeToFilename(code):
    fileNameBase = sha1(code).hexdigest()[0:8]
    return fileNameBase + ".C"

def _dumpToUniqueFile(code):
    fileName = _codeToFilename(code)
    ofile = open (fileName,'w')
    ofile.write(code)
    ofile.close()
    return fileName

@ipym.magics_class
class AclicMagics(ipym.Magics):
    @ipym.line_cell_magic
    def aclic(self, line, cell=None):
        """Compile code with aclic"""
        code = cell if cell else line
        fileName = _dumpToUniqueFile(code)
        utils.invokeAclic(fileName)



def load_ipython_extension(ipython):
    ipython.register_magics(AclicMagics)
