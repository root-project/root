
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


