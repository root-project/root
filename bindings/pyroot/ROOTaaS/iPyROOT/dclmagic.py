import IPython.core.magic as ipym
import utils

@ipym.magics_class
class DeclareMagics(ipym.Magics):
    @ipym.line_cell_magic
    def dcl(self, line, cell=None):
        """Inject into root."""
        code = cell if cell else line
        utils.declareCppCode(code)

def load_ipython_extension(ipython):
    ipython.register_magics(DeclareMagics)
