sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/m/Link',
               'sap/ui/core/Fragment',
               'rootui5/browser/model/BrowserModel',
               'sap/ui/model/json/JSONModel',
               'sap/ui/core/util/File',
               'sap/ui/table/Column',
               'sap/ui/layout/HorizontalLayout',
               'sap/m/TabContainerItem',
               'sap/m/MessageToast',
               'sap/m/MessageBox',
               'sap/m/Text',
               'sap/ui/core/mvc/XMLView',
               'sap/ui/core/Icon',
               'sap/ui/layout/Splitter',
               'sap/m/Toolbar',
               'sap/ui/unified/FileUploader',
               'sap/m/Button',
               'sap/ui/layout/SplitterLayoutData',
               'sap/ui/codeeditor/CodeEditor',
               'sap/m/HBox',
               'sap/m/Image',
               'sap/m/Dialog',
               'rootui5/browser/controller/FileDialog.controller'
],function(Controller,
           Link,
           Fragment,
           BrowserModel,
           JSONModel,
           File,
           tableColumn,
           HorizontalLayout,
           TabContainerItem,
           MessageToast,
           MessageBox,
           mText,
           XMLView,
           CoreIcon,
           Splitter,
           Toolbar,
           FileUploader,
           Button,
           SplitterLayoutData,
           CodeEditor,
           HBox,
           Image,
           Dialog,
           FileDialogController) {

   "use strict";

   /** Central ROOT RBrowser controller
    * All Browser functionality is loaded after main ui5 rendering is performed */

   return Controller.extend("rootui5.browser.controller.Browser", {
      onInit: async function () {

        this.globalId = 1;
        this.nextElem = "";

         this.websocket = this.getView().getViewData().conn_handle;

         // this is code for the Components.js
         // this.websocket = Component.getOwnerComponentFor(this.getView()).getComponentData().conn_handle;

         this.websocket.SetReceiver(this);
         this.websocket.Connect();

         // if true, most operations are performed locally without involving server
         this.standalone = this.websocket.kind == "file";

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
            node.className = elem.className
         };

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
         t.addColumn(new tableColumn({
            label: "ClassName",
            autoResizable: true,
            visible: false,
            template: new HorizontalLayout({
               content: [
                         new mText({text:"{className}", wrapping: false })
                         ]
            })
         }));

         // catch re-rendering of the table to assign handlers
         t.addEventDelegate({
            onAfterRendering: function() { this.assignRowHandlers(); }
         }, this);

         this.newCodeEditor();

         this.drawingOptions = { TH1: 'hist', TH2: 'COL', TProfile: 'E0'};
      },

      /* ========================================================= */
      /* =============== Generic factory functions =============== */
      /* ========================================================= */

      getElementFromCurrentTab: function (element) {
         const currentTabID = this.getView().byId("myTabContainer").getSelectedItem();
         return sap.ui.getCore().byId(currentTabID + element);
      },

      /* ========================================================= */
      /* =============== Generic factory functions =============== */
      /* ========================================================= */

      /* =========================================== */
      /* =============== Code Editor =============== */
      /* =========================================== */

      newCodeEditor: async function () {
         const oTabContainer = this.getView().byId("myTabContainer");

         const ID = "CodeEditor" + this.globalId;
         this.globalId++;

         const oTabContainerItem = new TabContainerItem(ID, {
            icon: "sap-icon://write-new-document",
            name: "Code Editor",
            additionalText: "untitled",
            content: this.newCodeEditorFragment(ID)
         });

         oTabContainer.addItem(oTabContainerItem);
         oTabContainer.setSelectedItem(oTabContainerItem);
      },

      newCodeEditorFragment: function (ID) {
         return new Splitter({
            orientation: "Vertical",
            contentAreas: [
               new Toolbar({
                  content: [
                     new FileUploader({
                        change: [this.onChangeFile, this]
                     }),
                     new Button(ID + "SaveAs", {
                        text: "Save as...",
                        tooltip: "Save current file as...",
                        press: [this.onSaveAs, this]
                     }),
                     new Button(ID + "Save", {
                        text: "Save",
                        tooltip: "Save current file",
                        press: [this.onSaveFile, this]
                     }),
                     new Button(ID + "Run", {
                        text: "Run",
                        tooltip: "Run Current Macro",
                        icon: "sap-icon://play",
                        enabled: false,
                        press: [this.onRunMacro, this]
                     }),
                  ],
                  layoutData: new SplitterLayoutData({
                     size: "35px",
                     resizable: false
                  })
               }),
               new CodeEditor(ID + "Editor", {
                  height: "100%",
                  colorTheme: "default",
                  type: "c_cpp",
                  value: "{/code}",
                  change: function () {
                     this.getModel().setProperty("/modified", true);
                  }
               }).setModel(new JSONModel({
                  code: "",
                  ext: "",
                  filename: "",
                  fullpath: "",
                  modified: false
               }))
            ]
         });
      },

      /** @brief Invoke dialog with server side code */
      onSaveAs: function() {

         const oEditor = this.getSelectedCodeEditor();
         const oModel = oEditor.getModel();
         const sText = oModel.getProperty("/code");
         let filename = oModel.getProperty("/fullfilename");

         var newconn = this.websocket.CreateChannel();

         this.saveAsController = new FileDialogController;

         this.saveAsController.initDialog(newconn, filename, this.dialogCompletionHandler.bind(this));

         this.websocket.Send("SAVEAS:" + JSON.stringify([ filename || "untiled",  newconn.getChannelId().toString() ]));
      },

      dialogCompletionHandler: function(on) {
         if (!this.saveAsController)
            return;

         if (on) {
            var fullname = this.saveAsController.getFullFileName();
            console.log('Save AS', fullname);

            const oEditor = this.getSelectedCodeEditor();
            const oModel = oEditor.getModel();
            const sText = oModel.getProperty("/code");

            fullname.push(sText);

            this.websocket.Send("DOSAVE:" + JSON.stringify(fullname));
         }

         delete this.saveAsController;

         this.websocket.Send("CLOSESAVEAS");
      },

      /** @brief Handle the "Save As..." button press event */
      onSaveAsOld: function () {
         const oEditor = this.getSelectedCodeEditor();
         const oModel = oEditor.getModel();
         const sText = oModel.getProperty("/code");
         let filename = oModel.getProperty("/filename");
         let ext = oModel.getProperty("/ext");
         if (filename === undefined) filename = "untitled";
         if (ext === undefined) ext = "txt";
         File.save(sText, filename, ext);
         oModel().setProperty("/modified", false);
      },

      /** @brief Handle the "Save" button press event */
      onSaveFile: function () {
         const oEditor = this.getSelectedCodeEditor();
         const oModel = oEditor.getModel();
         const sText = oModel.getProperty("/code");
         const fullpath = oModel.getProperty("/fullpath");
         if (fullpath === undefined) {
            return onSaveAs();
         }
         oModel.setProperty("/modified", false);
         return this.websocket.Send("SAVEFILE:" + fullpath + ":" + sText);
      },

      reallyRunMacro: function () {
         const oEditor = this.getSelectedCodeEditor();
         const oModel = oEditor.getModel();
         const fullpath = oModel.getProperty("/fullpath");
         if (fullpath === undefined)
            return this.onSaveAs();
         return this.websocket.Send("RUNMACRO:" + fullpath);
      },

      /** @brief Handle the "Run" button press event */
      onRunMacro: function () {
         this.saveCheck(this.reallyRunMacro.bind(this));
      },

      saveCheck: function(functionToRunAfter) {
         const oEditor = this.getSelectedCodeEditor();
         const oModel = oEditor.getModel();
         if (oModel.getProperty("/modified") === true) {
            MessageBox.confirm('The text has been modified! Do you want to save it?', {
               title: 'Unsaved file',
               icon: sap.m.MessageBox.Icon.QUESTION,
               onClose: (oAction) => {
                  if (oAction === MessageBox.Action.YES) {
                     this.onSaveFile();
                  } else if (oAction === MessageBox.Action.CANCEL) {
                     return;
                  }
                  return functionToRunAfter();
               },
               actions: [sap.m.MessageBox.Action.YES, sap.m.MessageBox.Action.NO, sap.m.MessageBox.Action.CANCEL]
            });
         } else {
            return functionToRunAfter();
         }
      },

      getSelectedCodeEditor: function (no_warning) {
         let oTabItemString = this.getView().byId("myTabContainer").getSelectedItem();

         if (oTabItemString.indexOf("CodeEditor") !== -1) {
            return sap.ui.getCore().byId(oTabItemString + "Editor");
         } else {
            if (!no_warning) MessageToast.show("Sorry, you need to select a code editor tab", {duration: 1500});
            return -1;
         }
      },

      /** @brief Extract the file name and extension
       * @desc Used to set the editor's model properties and display the file name on the tab element  */
      setFileNameType: function (filename) {
         let oEditor = this.getSelectedCodeEditor();
         let oModel = oEditor.getModel();
         let oTabElement = oEditor.getParent().getParent();
         let ext = "txt";
         let runButton = this.getElementFromCurrentTab("Run");
         runButton.setEnabled(false);
         if (filename.lastIndexOf('.') > 0)
            ext = filename.substr(filename.lastIndexOf('.') + 1);

         switch (ext.toLowerCase()) {
            case "c":
            case "cc":
            case "cpp":
            case "cxx":
               runButton.setEnabled(true);
               oEditor.setType('c_cpp');
               break;
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
            case "css":
               oEditor.setType('css');
               break;
            case "csh":
            case "sh":
               oEditor.setType('sh');
               break;
            case "md":
               oEditor.setType('markdown');
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
         oModel.setProperty("/fullfilename", filename);
         if (filename.lastIndexOf('.') > 0)
            filename = filename.substr(0, filename.lastIndexOf('.'));
         oModel.setProperty("/filename", filename);
         oModel.setProperty("/ext", ext);
         return true;
      },

      /** @brief Handle the "Browse..." button press event */
      onChangeFile: function (oEvent) {
         let oEditor = this.getSelectedCodeEditor();
         let oModel = oEditor.getModel();
         let oReader = new FileReader();
         oReader.onload = function () {
            oModel.setProperty("/code", oReader.result);
         };
         let file = oEvent.getParameter("files")[0];
         if (this.setFileNameType(file.name))
            oReader.readAsText(file);
      },

      /* =========================================== */
      /* =============== Code Editor =============== */
      /* =========================================== */

      /* ============================================ */
      /* =============== Image viewer =============== */
      /* ============================================ */

      newImageViewerFragment: function (ID) {
         return new HBox({
            alignContent: "Center",
            alignItems: "Center",
            justifyContent: "Center",
            height: "100%",
            width: "100%",
            items: new Image(ID + "Image", {
               src: "",
               densityAware: false
            })
         })
      },

      newImageViewer: async function () {
         let oTabContainer = this.getView().byId("myTabContainer");

         const ID = "ImageViewer" + this.globalId;
         this.globalId++;

         let tabContainerItem = new TabContainerItem(ID, {
            icon: "sap-icon://background",
            name: "Image Viewer",
            additionalText: "untitled",
            content: this.newImageViewerFragment(ID)
         });

         oTabContainer.addItem(tabContainerItem);
         oTabContainer.setSelectedItem(tabContainerItem);
      },

      getSelectedImageViewer: function (no_warning) {
         let oTabItemString = this.getView().byId("myTabContainer").getSelectedItem();


         if (oTabItemString.indexOf("ImageViewer") !== -1) {
            return sap.ui.getCore().byId(oTabItemString + "Image");
         }

         if (!no_warning) MessageToast.show("Sorry, you need to select an image viewer tab", {duration: 1500});
         return -1;
      },

      /* ============================================ */
      /* =============== Image viewer =============== */
      /* ============================================ */

      /* ============================================= */
      /* =============== Settings menu =============== */
      /* ============================================= */

      _getSettingsMenu: async function () {
         if (!this._oSettingsMenu) {
            let fragment;
            await Fragment.load({name: "rootui5.browser.view.settingsmenu", controller: this}).then(function (oSettingsMenu) {
               fragment = oSettingsMenu;
            });
            if (fragment) {
               let oModel = new JSONModel({
                  "TH1": [
                     {"name": "hist"},
                     {"name": "P"},
                     {"name": "P0"},
                     {"name": "E"},
                     {"name": "E1"},
                     {"name": "E2"},
                     {"name": "E3"},
                     {"name": "E4"},
                     {"name": "E1X0"},
                     {"name": "L"},
                     {"name": "LF2"},
                     {"name": "B"},
                     {"name": "B1"},
                     {"name": "A"},
                     {"name": "TEXT"},
                     {"name": "LEGO"},
                     {"name": "same"}
                  ],
                  "TH2": [
                     {"name": "COL"},
                     {"name": "COLZ"},
                     {"name": "COL0"},
                     {"name": "COL1"},
                     {"name": "COL0Z"},
                     {"name": "COL1Z"},
                     {"name": "COLA"},
                     {"name": "BOX"},
                     {"name": "BOX1"},
                     {"name": "PROJ"},
                     {"name": "PROJX1"},
                     {"name": "PROJX2"},
                     {"name": "PROJX3"},
                     {"name": "PROJY1"},
                     {"name": "PROJY2"},
                     {"name": "PROJY3"},
                     {"name": "SCAT"},
                     {"name": "TEXT"},
                     {"name": "TEXTE"},
                     {"name": "TEXTE0"},
                     {"name": "CONT"},
                     {"name": "CONT1"},
                     {"name": "CONT2"},
                     {"name": "CONT3"},
                     {"name": "CONT4"},
                     {"name": "ARR"},
                     {"name": "SURF"},
                     {"name": "SURF1"},
                     {"name": "SURF2"},
                     {"name": "SURF4"},
                     {"name": "SURF6"},
                     {"name": "E"},
                     {"name": "A"},
                     {"name": "LEGO"},
                     {"name": "LEGO0"},
                     {"name": "LEGO1"},
                     {"name": "LEGO2"},
                     {"name": "LEGO3"},
                     {"name": "LEGO4"},
                     {"name": "same"}
                  ],
                  "TProfile": [
                     {"name": "E0"},
                     {"name": "E1"},
                     {"name": "E2"},
                     {"name": "p"},
                     {"name": "AH"},
                     {"name": "hist"}
                  ]
               });
               fragment.setModel(oModel);
               this.getView().addDependent(fragment);
               this._oSettingsMenu = fragment;
            }
         }
         return this._oSettingsMenu;
      },

      onSettingPress: async function () {
         await this._getSettingsMenu();
         this._oSettingsMenu.open();
      },

      handleSettingsChange: function (oEvent) {
         let graphType = oEvent.getSource().sId.split("-")[1];
         this.drawingOptions[graphType] = oEvent.getSource().mProperties.value;
      },

      /* ============================================= */
      /* =============== Settings menu =============== */
      /* ============================================= */

      /* ========================================= */
      /* =============== Tabs menu =============== */
      /* ========================================= */

      /** @brief Add Tab event handler */
      addNewButtonPressHandler: async function (oEvent) {
         //TODO: Change to some UI5 function (unknown for now)

         let oButton = oEvent.getSource().mAggregations._tabStrip.mAggregations.addButton;

         // create action sheet only once
         if (!this._tabMenu) {
            let fragment;
            await Fragment.load({name: "rootui5.browser.view.tabsmenu", controller: this}).then(function (oFragment) {
               fragment = oFragment;
            });
            if (fragment) {
               this.getView().addDependent(fragment);
               this._tabMenu = fragment;
            }
         }
         this._tabMenu.openBy(oButton);
      },

      newRootXCanvas: function (oEvent) {
         let msg;
         if (oEvent.getSource().getText().indexOf("6") !== -1) {
            msg = "NEWTCANVAS";
         } else {
            msg = "NEWRCANVAS";
         }
         if (this.isConnected) {
            this.websocket.Send(msg);
         }
      },

      /* ========================================= */
      /* =============== Tabs menu =============== */
      /* ========================================= */

      /* =========================================== */
      /* =============== Breadcrumbs =============== */
      /* =========================================== */

      updateBReadcrumbs: function(split) {
         // already array with all items inside
         let oBreadcrumbs = this.getView().byId("breadcrumbs");
         oBreadcrumbs.removeAllLinks();
         for (let i=-1; i<split.length; i++) {
            let txt = i<0 ? "/": split[i];
            if (i === split.length-1) {
               oBreadcrumbs.setCurrentLocationText(txt);
            } else {
               let link = new Link({text: txt});
               link.attachPress(this, this.onBreadcrumbsPress, this);
               oBreadcrumbs.addLink(link);
            }
         }
      },

      onBreadcrumbsPress: function(oEvent) {
         let sId = oEvent.getSource().getId();
         let oBreadcrumbs = oEvent.getSource().getParent();
         let oLinks = oBreadcrumbs.getLinks();
         let path = [];
         for (let i = 0; i < oLinks.length; i++) {
            if (i>0) path.push(oLinks[i].getText());
            if (oLinks[i].getId() === sId ) break;
         }
         this.websocket.Send('CHPATH:' + JSON.stringify(path));
         this.doReload(true);
      },

      /* =========================================== */
      /* =============== Breadcrumbs =============== */
      /* =========================================== */

      /* ============================================ */
      /* =============== TabContainer =============== */
      /* ============================================ */

      tabSelectItem: function(oEvent) {
         var oItemSelected = oEvent.getParameter('item');

         if (oItemSelected.getName() !== "ROOT Canvas") return;

         console.log("Canvas selected:", oItemSelected.getAdditionalText());

         this.websocket.Send("SELECT_CANVAS:" + oItemSelected.getAdditionalText());

      },

      /** @brief Close Tab event handler */
      tabCloseHandler: function(oEvent) {
         // prevent the tab being closed by default
         oEvent.preventDefault();

         let oTabContainer = this.byId("myTabContainer");
         let oItemToClose = oEvent.getParameter('item');


         if (oItemToClose.getName() === "Code Editor") {

            let count = 0;
            const items = oTabContainer.getItems();
            for (let i=0; i< items.length; i++) {
               if (items[i].getId().indexOf("CodeEditor") !== -1) {
                  count++
               }
            }
            if (count <= 1) {
               MessageToast.show("Sorry, you cannot close the Code Editor", {duration: 1500});
            } else {
               this.saveCheck(function ()  {oTabContainer.removeItem(oItemToClose);});
            }
         } else {
            let pthis = this;
            MessageBox.confirm('Do you really want to close the "' + oItemToClose.getName() + '" tab?', {
               onClose: function (oAction) {
                  if (oAction === MessageBox.Action.OK) {
                     if (oItemToClose.getName() === "ROOT Canvas")
                        pthis.websocket.Send("CLOSE_CANVAS:" + oItemToClose.getAdditionalText());

                     oTabContainer.removeItem(oItemToClose);

                     MessageToast.show('Closed the "' + oItemToClose.getName() + '" tab', {duration: 1500});
                  }
               }
            });

         }
      },

      /* ============================================ */
      /* =============== TabContainer =============== */
      /* ============================================ */

      /* ======================================== */
      /* =============== Terminal =============== */
      /* ======================================== */

      onTerminalSubmit: function(oEvent) {
         let command = oEvent.getSource().getValue();
         let url = '/ProcessLine/cmd.json?arg1="' + command + '"';
         console.log(command);
         this.websocket.Send("CMD:" + command);
         oEvent.getSource().setValue("");
         this.requestRootHist();
         this.requestLogs();
      },

      requestRootHist: function() {
         return this.websocket.Send("ROOTHIST:");
      },

      updateRootHist: function (hist) {
         let pos = hist.lastIndexOf(',');
         hist = hist.substring(0,pos) + "" + hist.substring(pos+1);
         hist = hist.split(",");
         let json = {hist:[]};

         for(let i=0; i<hist.length; i++) {
            json.hist.push({name: hist[i] });

         }
         this.getView().byId("terminal-input").setModel(new JSONModel(json));
      },

      requestLogs: function() {
         return this.websocket.Send("LOGS:");
      },

      updateLogs: function(logs) {
         this.getView().byId("output_log").setValue(logs);
      },

      /* ======================================== */
      /* =============== Terminal =============== */
      /* ======================================== */

      /** @brief Assign the "double click" event handler to each row */
      assignRowHandlers: function () {
         var rows = this.byId("treeTable").getRows();
         for (var k = 0; k < rows.length; ++k) {
            rows[k].$().dblclick(this.onRowDblClick.bind(this, rows[k]));
         }
      },

      sendDblClick: function (fullpath, opt) {
         this.websocket.Send('DBLCLK: ["' + fullpath + '","' + (opt || "") + '"]');
      },

      /** @brief Double-click event handler */
      onRowDblClick: function (row) {
         let ctxt = row.getBindingContext(),
            prop = ctxt ? ctxt.getProperty(ctxt.getPath()) : null,
            fullpath = (prop && prop.fullpath) ? prop.fullpath.substr(1, prop.fullpath.length - 2) : "";

         if (!fullpath) return;

         // do not use row._bHasChildren while it is not documented member of m.Row object
         if (!prop.isLeaf) {
            if (!prop.fullpath.endsWith(".root/")) {

               let oBreadcrumbs = this.getView().byId("breadcrumbs");
               let links = oBreadcrumbs.getLinks();
               let currentText = oBreadcrumbs.getCurrentLocationText();

               let path = "";
               if ((currentText == "/") || (links.length < 1)) {
                  path = prop.fullpath;
               } else {
                  path = "/";
                  for (let i = 1; i < links.length; i++)
                     path += links[i].getText() + "/";
                  path += currentText + prop.fullpath;
               }

               // TODO: use plain array also here to avoid any possible confusion
               this.websocket.Send('CHDIR:' + path);
               return this.doReload(true);
            }
         }

         // first try to activate editor
         let codeEditor = this.getSelectedCodeEditor(true);
         if (codeEditor !== -1) {
            this.nextElem = { fullpath };
            let filename = fullpath.substr(fullpath.lastIndexOf('/') + 1);
            if (this.setFileNameType(filename))
               return this.sendDblClick(fullpath, "$$$editor$$$");
         }

         let viewerTab = this.getSelectedImageViewer(true);
         if (viewerTab !== -1) {
            this.nextElem = { fullpath };
            return this.sendDblClick(fullpath, "$$$image$$$");
         }

         let className = this.getBaseClass(prop ? prop.className : "");
         let drawingOptions = "";
         if (className && this.drawingOptions[className])
            drawingOptions = this.drawingOptions[className];

         return this.sendDblClick(fullpath, drawingOptions);
      },

      getBaseClass: function(className) {
         if (typeof className !== 'string')
            className = "";
         if (className.match(/^TH1/)) {
            return "TH1";
         } else if (className.match(/^TH2/)) {
            return "TH2";
         }
         return className;
      },

      OnWebsocketOpened: function(handle) {
         this.isConnected = true;

         if (this.model)
            this.model.sendFirstRequest(this.websocket);

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

         let mhdr = msg.split(":")[0];
         msg = msg.substr(mhdr.length+1);

         switch (mhdr) {
         case "INMSG":
            this.processInitMsg(msg);
            break;
         case "FREAD":  // file read
            let result = this.getSelectedCodeEditor();
            if (result !== -1) {
               result.getModel().setProperty("/code", msg);
               this.getElementFromCurrentTab("Save").setEnabled(true);
               result.getModel().setProperty("/fullpath", this.nextElem.fullpath);
            }
            break;
         case "FIMG":  // image file read
            const image = this.getSelectedImageViewer(true);
            if(image !== -1) {
               image.getParent().getParent().setAdditionalText(this.nextElem.fullpath);
               image.setSrc(msg);
            }
            break;
         case "CANVS":  // canvas created by server, need to establish connection
            var arr = JSON.parse(msg);
            this.createCanvas(arr[0], arr[1], arr[2]);
            break;
         case "WORKPATH":
            this.updateBReadcrumbs(JSON.parse(msg));
            break;
         case "SLCTCANV": // Selected the back selected canvas
           let oTabContainer = this.byId("myTabContainer");
           let oTabContainerItems = oTabContainer.getItems();
           for(let i=0; i<oTabContainerItems.length; i++) {
             if (oTabContainerItems[i].getAdditionalText() === msg) {
               oTabContainer.setSelectedItem(oTabContainerItems[i]);
               break;
             }
           }
           break;
         case "BREPL":   // browser reply
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
            case "HIST":
               this.updateRootHist(msg);
               break;
            case "LOGS":
               this.updateLogs(msg);
               break;
         default:
            console.error('Non recognized msg ' + mhdr + ' len=' + msg.length);
         }
      },

      /** Get the ID of the currently selected tab of given tab container */
      getSelectedtabFromtabContainer: function(divid) {
         var  tabContainer = this.getView().byId('myTabContainer').getSelectedItem();
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

      onBeforeRendering: function() {
         this.renderingDone = false;
      },

      onAfterRendering: function() {
         this.renderingDone = true;

         // this is how master width can be changed, may be extra control can be provided
         // var oSplitApp = this.getView().byId("SplitAppBrowser");
         // oSplitApp.getAggregation("_navMaster").$().css("width", "400px");
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

      onSearch : function(oEvt) {
         this.changeItemsFilter(oEvt.getSource().getValue());
      },

      /** Submit node search query to server, ignore in offline case */
      changeItemsFilter: function(query, from_handler) {

         if (!from_handler) {
            // do not submit immediately, but after very short timeout
            // if user types very fast - only last selection will be shown
            if (this.search_handler) clearTimeout(this.search_handler);
            this.search_handler = setTimeout(this.changeItemsFilter.bind(this, query, true), 1000);
            return;
         }

         delete this.search_handler;

         this.model.changeItemsFilter(query);
      },

      /** process initial message, now it is list of existing canvases */
      processInitMsg: function(msg) {
         var arr = JSROOT.parse(msg);
         if (!arr) return;

         this.updateBReadcrumbs(arr[0]);
         this.requestRootHist();
         this.requestLogs();

         for (var k=1; k<arr.length; ++k)
            this.createCanvas(arr[k][0], arr[k][1], arr[k][2]);
      },

      createCanvas: function(kind, url, name) {
         console.log("Create canvas ", url, name);
         if (!url || !name) return;

         var oTabContainer = this.byId("myTabContainer");
         var oTabContainerItem = new TabContainerItem({
            name: "ROOT Canvas",
            icon: "sap-icon://column-chart-dual-axis"
         });

         oTabContainerItem.setAdditionalText(name); // name can be used to set active canvas or close canvas

         oTabContainer.addItem(oTabContainerItem);

         // Change the selected tabs, only if it is new one, not the basic one
         if(name !== "rcanv1") {
           oTabContainer.setSelectedItem(oTabContainerItem);
         }

         var conn = new JSROOT.WebWindowHandle(this.websocket.kind);

         // this is producing
         var addr = this.websocket.href, relative_path = url;
         if (relative_path.indexOf("../")==0) {
            var ddd = addr.lastIndexOf("/",addr.length-2);
            addr = addr.substr(0,ddd) + relative_path.substr(2);
         } else {
            addr += relative_path;
         }

         var painter = null;

         if (kind == "root7") {
            painter = new JSROOT.v7.TCanvasPainter(null);
         } else {
            painter = new JSROOT.TCanvasPainter(null);
         }

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
            oTabContainerItem.addContent(oView);
            // JSROOT.CallBack(call_back, true);
         });
      },
   });

});
