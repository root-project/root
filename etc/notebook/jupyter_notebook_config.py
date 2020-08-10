import os
if 'ROOTSYS' in os.environ:
    c.NotebookApp.extra_static_paths.append(os.path.join(os.environ['ROOTSYS'], 'js/'))
