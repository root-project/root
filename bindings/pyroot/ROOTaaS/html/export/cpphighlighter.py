"""This preprocessor detect cells using C++:
- Through `%%cpp` or `%%dcl` magic extensions
- As a result of applying ROOT.toCpp()
Cell's metadata is marked so that the appropriate highlighter
can be used in the `highlight` filter.
"""

from IPython.nbconvert.preprocessors.base import Preprocessor
import re


class CppHighlighter(Preprocessor):
    """
    Detects and tags code cells that use the C++ language.
    """

    magics = [ '%%cpp', '%%dcl' ]
    cpp = 'cpp'
    python = 'python'

    def __init__(self, config=None, **kw):
        """Public constructor"""

        super(CppHighlighter, self).__init__(config=config, **kw)

        # Build regular expressions to catch language extensions or switches and choose
        # an adequate pygments lexer
        any_magic = "|".join(self.magics)
        self.re_magic_language = re.compile(r"^\s*({0}).*".format(any_magic), re.DOTALL)
        self.re_to_cpp = re.compile(r".*ROOT\.toCpp\(\).*", re.DOTALL)
        self.re_to_python = re.compile(r".*toPython\(\).*", re.DOTALL)

        self.current_language = self.python

    def matches(self, source, reg_expr):
        """
        When the cell matches the regular expression, this function returns True.
        Otherwise, it returns False.

        Parameters
        ----------
        source: str
            Source code of the cell to highlight
        reg_exp: SRE_Pattern 
            Regular expression to match
        """

        m = reg_expr.match(source)
        if m:
            return True
        else:
            return False

    def preprocess_cell(self, cell, resources, cell_index):
        """
        Tags cells using a magic extension language

        Parameters
        ----------
        cell : NotebookNode cell
            Notebook cell being processed
        resources : dictionary
            Additional resources used in the conversion process.  Allows
            preprocessors to pass variables into the Jinja engine.
        cell_index : int
            Index of the cell being processed (see base.py)
        """

        # Only tag code cells
        if cell.cell_type == "code":
            if self.matches(cell.source, self.re_magic_language):
                cell['metadata']['magics_language'] = self.cpp
            elif self.current_language == self.cpp:
                cell['metadata']['magics_language'] = self.cpp
                if self.matches(cell.source, self.re_to_python):
                    self.current_language = self.python
            elif self.matches(cell.source, self.re_to_cpp):
		self.current_language = self.cpp 
		
        return cell, resources
