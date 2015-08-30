from IPython import get_ipython
from IPython.core.extensions import ExtensionManager
import utils

def iPythonize():
    utils.setStyle()
    for capture in utils.captures: capture.register()
    
    extMgr = ExtensionManager(get_ipython())
    for extName in utils.extNames:
        extMgr.load_extension(extName)

    utils.enableCppHighlighting()
    utils.enhanceROOTModule()
    utils.welcomeMsg()


iPythonize()


