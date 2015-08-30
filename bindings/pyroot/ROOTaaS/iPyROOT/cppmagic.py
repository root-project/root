import IPython.core.magic as ipym
import utils

@ipym.magics_class
class CppMagics(ipym.Magics):
    @ipym.line_cell_magic
    def cpp(self, line, cell=None):
        """Inject into root."""
        code = cell if cell else line
        utils.processCppCode(code)

def load_ipython_extension(ipython):
    ipython.register_magics(CppMagics)
