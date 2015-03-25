
## Networking Libraries

### HTTP Server

##### Command Interface
One can now register an arbitrary command to the server, which become visible in the web browser. Then, when the item is clicked by the user, the command ends-up in a gROOT->ProcessLineSync() call.

##### Custom Properties 
Custom properties can be configured for any item in the server. For example, one could configure an icon for each item visible in the browser. Or one could 'hide' any item from the user (but keep access with normal http requests). With such properties one could specify which item is drawn when web page is loaded, or configure monitoring. See tutorials/http/httpcontrol.C macro for more details.

##### Method Calls
Implement exe.json requests to be able to execute any method of registered objects. This request is used to provide remote TTree::Draw() functionality.

##### Misc
Correctly set 'Cache-Control' headers when replying to http requests.
Better support of STL containers when converting objects into json with TBufferJSON class.


## JavaScript ROOT

- Several files can now be loaded simultaneously
- Use d3.time.scale to display time scales
- Implemented drag and drop to superimpose histograms or graphs
- Allow selection of drawing option via context menu
- Better support of touch devices
- Provide simple layout, making it default
- Allow to open ROOT files in online session (via url parameter)
- One could monitor simultaneously objects from server and root files
- Implement 'autocol' draw option  - when superimposing histograms,
   their line colors will be automatically assigned
- Implement 'nostat' draw option - disabled stat drawing
- Using '_same_' identifier in item name, one can easily draw or superimpose
   similar items from different files. Could be used in URL like:
     `...&files=[file1.root,file2.root]&items=[file1.root/hpx, file2.root/_same_]`
     `...&files=[file1.root,file2.root]&item=file1.root/hpx+file2.root/_same_`
   Main limitation - file names should have similar length.
- When 'autozoom' specified in draw options, histogram zoomed into
  non-empty content. Same command available via context menu.
- Item of 'Text' kind can be created. It is displayed as
  lain text in the browser. If property 'mathjax' specified,
  MathJax.js library will be loaded and used for rendering.
  See tutorials/http/httpcontrol.C macro for example.
- When using foreignObject, provide workaround for absolute positioning
  problem in Chrome/Safari, see <http://bit.ly/1wjqCQ9>
- Support usage of minimized versions of .js and .css files.
  Minimized scripts used by default on web servers.
- Implement JSROOT.extend instead of jQuery.extend, reduce
  usage of jquery.js in core JSROOT classes
- Implement main graphics without jquery at all,
  such mode used in `nobrowser` mode.
- Provide optional latex drawing with MathJax SVG.
  TMathText always drawn with MathJax,
  other classes require `mathjax` option in URL
- Improve drawing of different text classes, correctly handle
  their alignment and scaling, special handling for IE

