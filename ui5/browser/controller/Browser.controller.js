sap.ui.define(['sap/ui/core/Component',
               'sap/ui/core/mvc/Controller',
               'sap/ui/core/Control',
               'sap/ui/core/Icon',
               'sap/m/Text',
					'sap/m/CheckBox',
					'sap/m/MessageBox',
					'sap/m/MessageToast',
               'sap/ui/layout/Splitter',
               "sap/ui/core/ResizeHandler",
               "sap/ui/layout/HorizontalLayout",
               "sap/ui/table/Column",
					"sap/ui/core/util/File",
					"sap/ui/model/json/JSONModel",
               "rootui5/browser/model/BrowserModel"
],function(Component, Controller, CoreControl, CoreIcon, mText, mCheckBox, MessageBox, MessageToast, Splitter,
	        ResizeHandler, HorizontalLayout, tableColumn, File, JSONModel, BrowserModel) {

   "use strict";

   /** Central ROOT Browser contoller
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

         if (JSROOT.GetUrlOption('nobrowser') !== null) {
            // remove complete area
            this.getView().byId("mainSplitter").removeAllContentAreas();
         } else {

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
			}
      },

		onChangeFile: function(oEvent) {
        var oModel = this.getView().byId("aCodeEditor").getModel();
        var oReader = new FileReader();
        oReader.onload = function() {
          oModel.setProperty("/code", oReader.result);
		  }
		  var file = oEvent.getParameter("files")[0];
		  var filename = file.name;
		  if (filename.endsWith(".C") || filename.endsWith(".c") || filename.endsWith(".cc") ||
		      filename.endsWith(".cpp") || filename.endsWith(".cxx") || filename.endsWith(".h") ||
		      filename.endsWith(".hh") || filename.endsWith(".hxx"))
          this.getView().byId("aCodeEditor").setType('c_cpp');
        else if (filename.endsWith(".py"))
          this.getView().byId("aCodeEditor").setType('python');
        else if (filename.endsWith(".js"))
          this.getView().byId("aCodeEditor").setType('javascript');
        else if (filename.endsWith(".htm") || filename.endsWith(".html"))
          this.getView().byId("aCodeEditor").setType('html');
        else
          this.getView().byId("aCodeEditor").setType('text');
		  oReader.readAsText(file);
		},

      assignRowHandlers: function() {
			/*
			var rows = this.getView().byId("treeTable").getRows();
         for (var k=0;k<rows.length;++k) {
            rows[k].$().hover(this.onRowHover.bind(this, rows[k], true), this.onRowHover.bind(this, rows[k], false));
			}
         */
      },

      OnWebsocketOpened: function(handle) {
         this.isConnected = true;

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

      /** Show special message insted of nodes hierarchy */
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

      /** Reload geometry description and base drawing, normally not required */
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

		tabCloseHandler: function(oEvent) {
			// prevent the tab being closed by default
			oEvent.preventDefault();

			var oTabContainer = this.byId("myTabContainer");
			var oItemToClose = oEvent.getParameter('item');

			MessageBox.confirm("Do you want to close the tab '" + oItemToClose.getName() + "'?", {
				onClose: function (oAction) {
					if (oAction === MessageBox.Action.OK) {
						oTabContainer.removeItem(oItemToClose);
						MessageToast.show("Item closed: " + oItemToClose.getName(), {duration: 500});
					} else {
						MessageToast.show("Item close canceled: " + oItemToClose.getName(), {duration: 500});
					}
				}
			});
		}
	});

});
