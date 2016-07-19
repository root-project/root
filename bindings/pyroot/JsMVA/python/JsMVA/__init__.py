from IPython.core.extensions import ExtensionManager

def loadExtensions():
    ip     = get_ipython()
    extMgr = ExtensionManager(ip)
    extMgr.load_extension("JsMVA.JsMVAMagic")


loadExtensions();