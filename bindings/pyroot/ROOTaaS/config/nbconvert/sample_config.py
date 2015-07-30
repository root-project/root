# ROOTaaS sample configuration file for nbconvert

c = get_config()

# Custom C++ highlighter
c.Exporter.preprocessors = [ 'ROOTaaS.html.export.cpphighlighter.CppHighlighter' ]

# Custom Jinja template
c.HTMLExporter.template_path = [ '/path/to/ROOTaaS/html/static/templates' ]

