# ROOTaaS sample configuration file for Jupyter notebooks

c = get_config()

# Basic notebook configuration
c.NotebookApp.password = u'sha1:7ebf0f845169:c71e54322fc8455ec8e3ac82d758853a08a3a9c2'
c.NotebookApp.certfile = u'/path/to/cert/mycert.pem'
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 9999
c.NotebookApp.base_url = u'/pyroot/'
c.NotebookApp.notebook_dir = u'/path/to/notebooks/'

# Automatic import
c.InteractiveShellApp.exec_lines = ["from ROOTaaS.iPyROOT import ROOT"]

# Automatic extension loading
#c.InteractiveShellApp.extensions = [ "ROOTaaS.iPyROOT.cppmagic" ]

# For exporting HTML from the notebook
c.Exporter.preprocessors = [ 'ROOTaaS.html.export.cpphighlighter.CppHighlighter' ]
c.HTMLExporter.template_file = 'root_notebook.tpl'
c.HTMLExporter.template_path = ['.', '/path/to/ROOTaaS/html/static/templates']

