sap.ui.define(['sap/ui/core/Component',
               'sap/ui/core/mvc/Controller',
               'sap/ui/core/Control',
               'sap/ui/core/Icon',
               'sap/ui/core/mvc/XMLView',
               'sap/m/Text',
               'sap/m/CheckBox',
               'sap/m/MessageBox',
               'sap/m/MessageToast',
               'sap/m/TabContainerItem',
               'sap/ui/layout/Splitter',
               "sap/ui/core/ResizeHandler",
               "sap/ui/layout/HorizontalLayout",
               "sap/ui/table/Column",
               "sap/ui/core/util/File",
               "sap/ui/model/json/JSONModel",
               "rootui5/browser/model/BrowserModel",
],function(Component, Controller, CoreControl, CoreIcon, XMLView, mText, mCheckBox, MessageBox, MessageToast, TabContainerItem,
           Splitter, ResizeHandler, HorizontalLayout, tableColumn, File, JSONModel, BrowserModel) {

   "use strict";

   /** Central ROOT RBrowser controller
    * All Browser functionality is loaded after main ui5 rendering is performed */

   return Controller.extend("rootui5.browser.controller.Browser", {
      onInit: function () {

         this.websocket = this.getView().getViewData().conn_handle;

         // this is code for the Components.js
         // this.websocket = Component.getOwnerComponentFor(this.getView()).getComponentData().conn_handle;

         this.websocket.SetReceiver(this);
         this.websocket.Connect();

         this.queue = []; // received draw messages

         // if true, most operations are performed locally without involving server
         this.standalone = this.websocket.kind == "file";

/*         if (JSROOT.GetUrlOption('nobrowser') !== null) {
            // remove complete area
            this.getView().byId("mainSplitter").removeAllContentAreas();
         } else {
*/
            // create model only for browser - no need for anybody else
            this.model = new BrowserModel();

            // copy extra attributes from element to node in the browser
            // later can be done automatically
            this.model.addNodeAttributes = function(node, elem) {
               node.icon = elem.icon;
               node.fsize = elem.fsize;
               node.mtime = elem.mtime;
               node.ftype = elem.ftype;
               node.fuid = elem.fuid;
               node.fgid = elem.fgid;
            }

            var t = this.getView().byId("treeTable");

            t.setModel(this.model);

            this.model.assignTreeTable(t);
            t.addColumn(new tableColumn({
               label: "Name",
               autoResizable: true,
               visible: true,
               template: new HorizontalLayout({
                  content: [
                     new CoreIcon({src:"{icon}"}),
                     new mText({text:" {name}", renderWhitespace: true, wrapping: false })
                  ]
               })
            }));
            t.addColumn(new tableColumn({
               label: "Size",
               autoResizable: true,
               visible: true,
               template: new HorizontalLayout({
                  content: [
                     new mText({text:"{fsize}", wrapping: false })
                  ]
               })
            }));
            t.addColumn(new tableColumn({
               label: "Time",
               autoResizable: true,
               visible: false,
               template: new HorizontalLayout({
                  content: [
                     new mText({text:"{mtime}", wrapping: false })
                  ]
               })
            }));
            t.addColumn(new tableColumn({
               label: "Type",
               autoResizable: true,
               visible: false,
               template: new HorizontalLayout({
                  content: [
                     new mText({text:"{ftype}", wrapping: false })
                  ]
               })
            }));
            t.addColumn(new tableColumn({
               label: "UID",
               autoResizable: true,
               visible: false,
               template: new HorizontalLayout({
                  content: [
                     new mText({text:"{fuid}", wrapping: false })
                  ]
               })
            }));
            t.addColumn(new tableColumn({
               label: "GID",
               autoResizable: true,
               visible: false,
               template: new HorizontalLayout({
                  content: [
                     new mText({text:"{fgid}", wrapping: false })
                  ]
               })
            }));

            // catch re-rendering of the table to assign handlers
            t.addEventDelegate({
               onAfterRendering: function() { this.assignRowHandlers(); }
            }, this);

            this.getView().byId("aCodeEditor").setModel(new JSONModel({
               code: ""
            }));
            
            /* this.getView().byId("aRootCanvas1").setModel(new JSONModel({
               rootCanvas: ""
            }));
            */
     //    }
            
          this.addNewButtonPressHandler(); // always create new canvas in the beginning   
      },

      /** @brief Extract the file name and extension
      * @desc Used to set the editor's model properties and display the file name on the tab element  */
     setFileNameType: function(filename) {
         var oEditor = this.getView().byId("aCodeEditor");
         var oModel = oEditor.getModel();
         var oTabElement = oEditor.getParent().getParent();
         var ext = "txt";
         if (filename.lastIndexOf('.') > 0)
            ext = filename.substr(filename.lastIndexOf('.') + 1);
         switch(ext.toLowerCase()) {
            case "c":
            case "cc":
            case "cpp":
            case "cxx":
            case "h":
            case "hh":
            case "hxx":
               oEditor.setType('c_cpp');
               break;
            case "f":
               oEditor.setType('fortran');
               break;
            case "htm":
            case "html":
               oEditor.setType('html');
               break;
            case "js":
               oEditor.setType('javascript');
               break;
            case "json":
               oEditor.setType('json');
               break;
            case "md":
               oEditor.setType('markdown');
               break;
            case "py":
               oEditor.setType('python');
               break;
            case "tex":
               oEditor.setType('latex');
               break;
            case "cmake":
            case "log":
            case "txt":
               oEditor.setType('plain_text');
               break;
            case "xml":
               oEditor.setType('xml');
               break;
            default: // unsupported type
               if (filename.lastIndexOf('README') >= 0)
                  oEditor.setType('plain_text');
               else
                  return false;
               break;
         }
         oTabElement.setAdditionalText(filename);
         if (filename.lastIndexOf('.') > 0)
            filename = filename.substr(0, filename.lastIndexOf('.'));
         oModel.setProperty("/filename", filename);
         oModel.setProperty("/ext", ext);
         return true;
      },

      /** @brief Handle the "Browse..." button press event */
      onChangeFile: function(oEvent) {
         var oEditor = this.getView().byId("aCodeEditor");
         var oModel = oEditor.getModel();
         var oReader = new FileReader();
         oReader.onload = function() {
            oModel.setProperty("/code", oReader.result);
         }
         var file = oEvent.getParameter("files")[0];
         if (this.setFileNameType(file.name))
            oReader.readAsText(file);
      },

      /** @brief Handle the "Save As..." button press event */
      onSaveAs: function() {
         var oEditor = this.getView().byId("aCodeEditor");
         var oModel = oEditor.getModel();
         var sText = oModel.getProperty("/code");
         var filename = oModel.getProperty("/filename");
         var ext = oModel.getProperty("/ext");
         if (filename == undefined) filename = "untitled";
         if (ext == undefined) ext = "txt";
         File.save(sText, filename, ext);
      },

      /** @brief Assign the "double click" event handler to each row */
      assignRowHandlers: function() {
         var rows = this.byId("treeTable").getRows();
         for (var k=0;k<rows.length;++k) {
            rows[k].$().dblclick(this.onRowDblClick.bind(this, rows[k]));
         }
      },

      /** @brief Send RBrowserRequest to the browser */
      sendBrowserRequest: function(_oper, args) {
         var req = { path: "", first: 0, number: 0, sort: _oper };
         JSROOT.extend(req, args);
         this.websocket.Send("BRREQ:" + JSON.stringify(req));
      },

      /** @brief Double-click event handler */
      onRowDblClick: function(row) {
         if (row._bHasChildren) // ignore directories for now
            return;
         var fullpath = "";
         var ctxt = row.getBindingContext(),
             prop = ctxt ? ctxt.getProperty(ctxt.getPath()) : null;
         if (prop && prop.fullpath) {
            fullpath = prop.fullpath.substr(1, prop.fullpath.length-2);
            var dirname = fullpath.substr(0, fullpath.lastIndexOf('/'));
            if (dirname.endsWith(".root"))
               return this.websocket.Send("DBLCLK:" + fullpath);
         }
         var oEditor = this.getView().byId("aCodeEditor");
         var oModel = oEditor.getModel();
         var filename = fullpath.substr(fullpath.lastIndexOf('/') + 1);
         if (this.setFileNameType(filename))
            return this.websocket.Send("DBLCLK:" + fullpath);
       },

      OnWebsocketOpened: function(handle) {
         this.isConnected = true;
         
         // this is required when canvas created faster then connection is established
         if (this.newCanvasId) {
            this.websocket.Send("CANVAS:" + this.newCanvasId);
            delete this.newCanvasId;
         }

         if (this.model)
            this.model.sendFirstRequest(this.websocket);

         // when connection established, checked if we can submit requested
         this.checkRequestMsg();

      },

      OnWebsocketClosed: function() {
         // when connection closed, close panel as well
         console.log('CLOSE WINDOW WHEN CONNECTION CLOSED');

         if (window) window.close();

         this.isConnected = false;
      },

      /** Entry point for all data from server */
      OnWebsocketMsg: function(handle, msg, offset) {

         if (typeof msg != "string")
            return console.error("Browser do not uses binary messages len = " + mgs.byteLength);

         var mhdr = msg.substr(0,6);
         msg = msg.substr(6);

         // console.log(mhdr, msg.length, msg.substr(0,70), "...");

         switch (mhdr) {
         case "DESCR:":  // browser hierarchy
            this.parseDescription(msg, true);
            break;
         case "FESCR:":  // searching hierarchy
            this.parseDescription(msg, false);
            break;
         case "FREAD:":  // file read
            this.getView().byId("aCodeEditor").getModel().setProperty("/code", msg);
            break;
         case "CANVS:":  // canvas created by server, need to establish connection
            this.connectCanvas(msg);
            break;
         case "FROOT:": // Root file
           let selecedTabID = this.getSelectedtabFromtabContainer("myTabContainer"); // The ID of the selected tab in the TabContainer

           let jsonAnswer = JSON.parse(msg); // message received from the server to JSON

           let rootFileArray = jsonAnswer.path.split("/"); // spliting the path on /
           let rootFileRelativePath = ""; // Declaration of the var to open the good file

           let i = 0; // Iterator
           while (rootFileArray[i].slice(-5) !== ".root") { // Iterating over the splited path until it find the .root file
             rootFileRelativePath += "/" + rootFileArray[i];
             i++;
           }
           rootFileRelativePath += "/" + rootFileArray[i]; // Adding the last bit (the wanted graphic) to the relative path

           let oCanvas = this.getView().byId("aRootCanvas" + selecedTabID); // Get the drawing place object

           if (oCanvas === undefined || oCanvas === null) { // If the selected tabs it not a Root canvas then display and error message
             MessageToast.show("Please, select a Root Canvas tab", {duration: 1500});
             return;
           }

           let oTabElement = oCanvas.getParent(); // Get the tab from the drawing place
           let rootFileDisplayName = rootFileArray[i] + "/" + rootFileArray[i + 1]; // Creating a simple nameOfTheFile.root/graphic;1 to display on the tab

           document.getElementById("TopBrowserId--aRootCanvas" + selecedTabID).innerHTML = ""; // Clearing the canvas
           oTabElement.setAdditionalText(rootFileDisplayName); // Setting the tab file name
           let finalJsonRoot = JSROOT.JSONR_unref(jsonAnswer.data); // Creating the graphic from the json
           JSROOT.draw("TopBrowserId--aRootCanvas" + selecedTabID, finalJsonRoot, "colz"); // Drawing the graphic into the selected tab canvas

           break;
         case "BREPL:":   // browser reply
            if (this.model) {
               var bresp = JSON.parse(msg);

               this.model.processResponse(bresp);

               if (bresp.path === '/') {
                  var tt = this.getView().byId("treeTable");
                  var cols = tt.getColumns();
                  tt.autoResizeColumn(2);
                  tt.autoResizeColumn(1);
                  // for (var k=0;k<cols.length;++k)
                  //    tt.autoResizeColumn(k);
               }
            }
            break;
         default:
            console.error('Non recognized msg ' + mhdr + ' len=' + msg.length);
         }
      },

      /** Get the ID of the currently selected tab of given tab container */
      getSelectedtabFromtabContainer: function(divid) {
         let tabContainer = this.getView().byId('myTabContainer').getSelectedItem()
         return tabContainer.slice(6, tabContainer.length);
      },

      /** Show special message instead of nodes hierarchy */
      showTextInBrowser: function(text) {
         var br = this.byId("treeTable");
         br.collapseAll();
         if (!text || (text === "RESET")) {
            br.setNoData("");
            br.setShowNoData(false);

            this.model.setNoData(false);
            this.model.refresh();

         } else {
            br.setNoData(text);
            br.setShowNoData(true);
            this.model.setNoData(true);
            this.model.refresh();
         }
      },

      omBeforeRendering: function() {
         this.renderingDone = false;
      },

      onAfterRendering: function() {
         this.renderingDone = true;

         this.checkRequestMsg();
      },

      checkRequestMsg: function() {
         if (this.isConnected && this.renderingDone) {

            if (this.creator && !this.ask_getdraw) {
               this.websocket.Send("GETDRAW");
               this.ask_getdraw = true;
            }
         }
      },

      /** Reload (refresh) file tree browser */
      onRealoadPress: function (oEvent) {
         this.doReload(true);
      },

      doReload: function(force) {
         if (this.standalone) {
            this.showTextInBrowser();
            this.paintFoundNodes(null);
            this.model.setFullModel(this.fullModel);
         } else {
            this.model.reloadMainModel(force);
         }
      },

      /** Quit ROOT session */
      onQuitRootPress: function(oEvent) {
         this.websocket.Send("QUIT_ROOT");
      },

      /** @brief Add Tab event handler */
      addNewButtonPressHandler: function(oEvent) {
         if (this.newCanvasId) return;
         
         var oTabContainer = this.byId("myTabContainer");
         var oTabContainerItem = new TabContainerItem({
            name: "ROOT Canvas",
            icon: "sap-icon://column-chart-dual-axis"
         });

         oTabContainer.addItem(oTabContainerItem);

         if (this.isConnected)
            this.websocket.Send("CANVAS:" + oTabContainerItem.getId());
         else
            this.newCanvasId = oTabContainerItem.getId();

         this.last_created_item = oTabContainerItem; // FIXME, how to find item by ID!!!
         
         /* let ID = oTabContainerItem.sId.slice(6, oTabContainerItem.sId.length);

         let html = new sap.ui.core.HTML("TopBrowserId--aRootCanvas" + ID, {
            content: "<div style=\"height:100%\">{/rootCanvas}</div>"
         });
         oTabContainerItem.addContent(html);

         oTabContainer.setSelectedItem(oTabContainerItem);

         this.getView().byId("aRootCanvas" + ID).setModel(new JSONModel({
            rootCanvas: ""
         }));
         */
      },
      
      /** process reply from server when canvas created */
      connectCanvas: function(msg) {
         var arr = JSROOT.parse(msg);
         if (!arr || (arr.length != 3)) return;
         
         var tabItem = this.last_created_item;
         delete this.last_created_item;
         if (!tabItem || tabItem.getId() != arr[0]) return;
         
         tabItem.canvasName = arr[2]; // name can be used to set active canvas or close canvas
         
         var conn = new JSROOT.WebWindowHandle(this.websocket.kind);
         
         // this is producing 
         var addr = this.websocket.href, relative_path = arr[1];
         if (relative_path.indexOf("../")==0) {
            var ddd = addr.lastIndexOf("/",addr.length-2);
            addr = addr.substr(0,ddd) + relative_path.substr(2);
         } else {
            addr += relative_path;
         }
         
         console.log("connecting with", addr)
         // establish connection
         // conn.Connect(addr);
         
         var painter = new JSROOT.TCanvasPainter(null);
         painter.online_canvas = true;
         painter.use_openui = true; 
         painter.batch_mode = false;
         painter._window_handle = conn;
         painter._window_handle_href = addr; // argument for connect
         
         XMLView.create({
            viewName: "rootui5.canv.view.Canvas",
            viewData: { canvas_painter: painter },
            height: "100%"
         }).then(function(oView) {
            tabItem.addContent(oView);
            // JSROOT.CallBack(call_back, true);
         });
         
      },

      /** @brief Close Tab event handler */
      tabCloseHandler: function(oEvent) {
         // prevent the tab being closed by default
         oEvent.preventDefault();

         var oTabContainer = this.byId("myTabContainer");
         var oItemToClose = oEvent.getParameter('item');
         // prevent closing the Code Editor
         if (oItemToClose.getName() == "Code Editor") {
            MessageToast.show("Sorry, you cannot close the Code Editor", {duration: 1500});
            return;
         }

         MessageBox.confirm('Do you really want to close the "' + oItemToClose.getName() + '" tab?', {
            onClose: function (oAction) {
               if (oAction === MessageBox.Action.OK) {
                  oTabContainer.removeItem(oItemToClose);
                  MessageToast.show('Closed the "' + oItemToClose.getName() + '" tab', {duration: 1500});
               }
            }
         });
      }
   });

});
