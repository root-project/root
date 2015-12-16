# JupyROOT sample configuration file for nbconvert

c = get_config()

# Custom C++ highlighter
c.Exporter.preprocessors = [ 'JupyROOT.html.cpphighlighter.CppHighlighter' ]

# Custom Jinja template
c.HTMLExporter.template_path = [ '/path/to/JupyROOT/html/templates' ]
c.HTMLExporter.template_file = 'root_notebook.tpl'
