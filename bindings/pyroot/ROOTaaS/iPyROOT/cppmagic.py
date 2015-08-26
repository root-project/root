import IPython.core.magic as ipym
import utils

@ipym.magics_class
class CppMagics(ipym.Magics):
    @ipym.line_cell_magic
    def cpp(self, line, cell=None):
        """Inject into root."""
        if cell is None: # this is a line magic
            utils.processCppCode(line)
        else:
            utils.processCppCode(cell)

def load_ipython_extension(ipython):
    ipython.register_magics(CppMagics)
