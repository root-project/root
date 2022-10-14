sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/m/Link',
               'sap/ui/core/Fragment',
               'rootui5/browser/model/BrowserModel',
               'sap/ui/Device',
               'sap/ui/model/json/JSONModel',
               'sap/ui/table/Column',
               'sap/ui/layout/HorizontalLayout',
               'sap/m/TabContainerItem',
               'sap/m/MessageToast',
               'sap/m/MessageBox',
               'sap/m/Text',
               'sap/m/Page',
               'sap/ui/core/mvc/XMLView',
               'sap/ui/core/Icon',
               'sap/m/Button',
               'sap/m/ButtonType',
               'sap/ui/core/ValueState',
               'sap/m/Dialog',
               'sap/m/DialogType',
               'sap/ui/codeeditor/CodeEditor',
               'sap/m/Image',
               'sap/tnt/ToolHeader',
               'sap/m/ToolbarSpacer',
               'sap/m/OverflowToolbarLayoutData',
               'rootui5/browser/controller/FileDialog.controller'
],function(Controller,
           Link,
           Fragment,
           BrowserModel,
           uiDevice,
           JSONModel,
           tableColumn,
           HorizontalLayout,
           TabContainerItem,
           MessageToast,
           MessageBox,
           mText,
           mPage,
           XMLView,
           CoreIcon,
           Button,
           ButtonType,
           ValueState,
           Dialog,
           DialogType,
           CodeEditor,
           Image,
           ToolHeader,
           ToolbarSpacer,
           OverflowToolbarLayoutData,
           FileDialogController) {

   "use strict";

   /** @summary Central ROOT RBrowser controller
     * @desc All Browser functionality is loaded after main ui5 rendering is performed */

   return Controller.extend("rootui5.browser.controller.Browser", {
      onInit: function () {

        uiDevice.orientation.attachHandler(mParams => this.handleChangeOrientation(mParams.landscape));

        this.handleChangeOrientation(uiDevice.orientation.landscape);

        this._oSettingsModel = new JSONModel({
            SortMethods: [
               { name: 'name', value: 'name' },
               { name: 'size', value: 'size' },
               { name: 'none', value: '' }
            ],
            SortMethod: 'name',
            ReverseOrder: false,
            ShowHiddenFiles: false,
            AppendToCanvas: false,
            DBLCLKRun: false,
            optTH1: 'hist',
            TH1: [
               {name: 'hist'},
               {name: 'p'},
               {name: 'p0'},
               {name: 'e'},
               {name: 'e1'},
               {name: 'e2'},
               {name: 'e3'},
               {name: 'e4'},
               {name: 'e1x0'},
               {name: 'l'},
               {name: 'lf2'},
               {name: 'b'},
               {name: 'b1'},
               {name: 'A'},
               {name: 'text'},
               {name: 'lego'},
               {name: 'same'}
            ],
            optTH2: 'col',
            TH2: [
               {name: 'col'},
               {name: 'colz'},
               {name: 'col0'},
               {name: 'col1'},
               {name: 'col0z'},
               {name: 'col1z'},
               {name: 'colA'},
               {name: 'box'},
               {name: 'box1'},
               {name: 'proj'},
               {name: 'projx1'},
               {name: 'projx2'},
               {name: 'projx3'},
               {name: 'projy1'},
               {name: 'projy2'},
               {name: 'projy3'},
               {name: 'scat'},
               {name: 'text'},
               {name: 'texte'},
               {name: 'texte0'},
               {name: 'cont'},
               {name: 'cont1'},
               {name: 'cont2'},
               {name: 'cont3'},
               {name: 'cont4'},
               {name: 'arr'},
               {name: 'surf'},
               {name: 'surf1'},
               {name: 'surf2'},
               {name: 'surf4'},
               {name: 'surf6'},
               {name: 'e'},
               {name: 'A'},
               {name: 'lego'},
               {name: 'lego0'},
               {name: 'lego1'},
               {name: 'lego2'},
               {name: 'lego3'},
               {name: 'lego4'},
               {name: 'same'}
            ],
            optTProfile: 'e0',
            TProfile: [
               {name: 'e0'},
               {name: 'e1'},
               {name: 'e2'},
               {name: 'p'},
               {name: 'Ah'},
               {name: 'hist'}
            ]
         });

        let data = this.getView().getViewData();
        this.websocket = data?.conn_handle;
        this.jsroot = data?.jsroot;

         // this is code for the Components.js
         // this.websocket = Component.getOwnerComponentFor(this.getView()).getComponentData().conn_handle;

         this.websocket.setReceiver(this);
         this.websocket.connect();

         // if true, most operations are performed locally without involving server
         this.standalone = (this.websocket.kind == 'file');

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
            node.className = elem.className;
            node.title = elem.title;
         };

         let t = this.getView().byId('treeTable');
         t.setModel(this.model);

         this.model.assignTreeTable(t);
         t.addColumn(new tableColumn({
            label: 'Name',
            autoResizable: true,
            visible: true,
            template: new HorizontalLayout({
               content: [
                         new CoreIcon({src:'{icon}', tooltip: '{className}' }),
                         new mText({text:' {name}', tooltip: '{title}', renderWhitespace: true, wrapping: false })
                         ]
            })
         }));
         t.addColumn(new tableColumn({
            label: 'Size',
            autoResizable: true,
            visible: true,
            template: new HorizontalLayout({
               content: [ new mText({text:'{fsize}', wrapping: false }) ]
            })
         }));
         t.addColumn(new tableColumn({
            label: 'Time',
            autoResizable: true,
            visible: false,
            template: new HorizontalLayout({
               content: [ new mText({text:'{mtime}', wrapping: false }) ]
            })
         }));
         t.addColumn(new tableColumn({
            label: 'Type',
            autoResizable: true,
            visible: false,
            template: new HorizontalLayout({
               content: [ new mText({text:'{ftype}', wrapping: false }) ]
            })
         }));
         t.addColumn(new tableColumn({
            label: 'UID',
            autoResizable: true,
            visible: false,
            template: new HorizontalLayout({
               content: [ new mText({text:'{fuid}', wrapping: false }) ]
            })
         }));
         t.addColumn(new tableColumn({
            label: 'GID',
            autoResizable: true,
            visible: false,
            template: new HorizontalLayout({
               content: [ new mText({text:'{fgid}', wrapping: false }) ]
            })
         }));
         t.addColumn(new tableColumn({
            label: 'ClassName',
            autoResizable: true,
            visible: false,
            template: new HorizontalLayout({
               content: [ new mText({text:'{className}', wrapping: false }) ]
            })
         }));

         // catch re-rendering of the table to assign handlers
         t.addEventDelegate({
            onAfterRendering: function() { this.assignRowHandlers(); }
         }, this);
      },

      createImageViewer: function (dummy_url, name, title, tooltip) {
         let oTabContainer = this.getView().byId("tabContainer"),
             image = new Image({ src: "", densityAware: false });

         image.addStyleClass("imageViewer");

         let item = new TabContainerItem(name, {
            icon: "sap-icon://background",
            name: "Image Viewer",
            key: name,
            additionalText: title,
            tooltip: tooltip || '',
            content: new mPage({
               showNavButton: false,
               showFooter: false,
               showSubHeader: false,
               showHeader: false,
               content: image
            })
         });

         item.setModel(new JSONModel({
            can_close: true  // always can close image viewer
         }));

         oTabContainer.addItem(item);

         return item;
      },

      /* =========================================== */
      /* =============== Tree Viewer =============== */
      /* =========================================== */

      createTreeViewer: function(url, name, title, tooltip) {
         const oTabContainer = this.getView().byId("tabContainer");

         let item = new TabContainerItem(name, {
            icon: "sap-icon://tree",
            name: "Tree Viewer",
            key: name,
            additionalText: title,
            tooltip: tooltip || ''
         });

         oTabContainer.addItem(item);

         this.jsroot.connectWebWindow({
            kind: this.websocket.kind,
            href: this.websocket.getHRef(url),
            user_args: { nobrowser: true }
         }).then(handle => XMLView.create({
            viewName: "rootui5.tree.view.TreeViewer",
            viewData: { conn_handle: handle, embeded: true, jsroot: this.jsroot }
         })).then(oView => item.addContent(oView));

         return item;
      },

      /* =========================================== */
      /* =============== Code Editor =============== */
      /* =========================================== */

      createCodeEditor: function(dummy_url, name, editor_title, tooltip) {
         const oTabContainer = this.getView().byId("tabContainer");

         let item = new TabContainerItem(name, {
            icon: "sap-icon://write-new-document",
            name: "Code Editor",
            key: name,
            additionalText: "{/title}",
            tooltip: tooltip || ''
         });

         item.addContent(new ToolHeader({
            height: "40px",
            content: [
               new Button({
                  text: "Run",
                  tooltip: "Run Current Macro",
                  icon: "sap-icon://play",
                  type: "Transparent",
                  enabled: "{/runEnabled}",
                  press: () => this.onRunMacro(item)
               }),
               new ToolbarSpacer({
                  layoutData: new OverflowToolbarLayoutData({
                     priority:"NeverOverflow",
                     minWidth: "16px"
                  })
               }),
               new Button({
                  text: "Sync",
                  tooltip: "Sync editor content on server side",
                  type: "Transparent",
                  enabled: "{/modified}",
                  press: () => this.syncEditor(item)
               }),
               new Button({
                  text: "Save as...",
                  tooltip: "Save current file as...",
                  type: "Transparent",
                  press: () => this.onSaveAsFile(item)
               }),
               new Button({
                  text: "Save",
                  tooltip: "Save current file",
                  type: "Transparent",
                  enabled: "{/saveEnabled}",
                  press: () => this.onSaveFile(item)
               })
            ]
         }));
         item.addContent( new CodeEditor({
            // height: 'auto',
            colorTheme: "default",
            type: "c_cpp",
            value: "{/code}",
            height: "calc(100% - 40px)",
            liveChange: function() {
               const model = this.getModel();
               if (model.getProperty("/first_change")) {
                  model.setProperty("/first_change", false);
               } else {
                  model.setProperty("/modified", true);
                  model.setProperty("/can_close", false);
               }
            }
         }));

         item.setModel(new JSONModel({
            code: "",
            ext: "",
            title: editor_title,
            filename: "",  // only set when really exists
            modified: false, // if content modified compared to server side
            can_close: true,  // if file is stored, one can close without confirmation
            runEnabled: false,
            saveEnabled: false
         }));

         oTabContainer.addItem(item);

         return item;
      },

      /** @brief Invoke dialog with server side code */
      onSaveAsFile: function(tab) {

         const oModel = tab.getModel();
         FileDialogController.SaveAs({
            websocket: this.websocket,
            filename: oModel.getProperty("/filename") || oModel.getProperty("/title"),
            title: "Select file name to save",
            filter: "Any files",
            filters: ["Text files (*.txt)", "C++ files (*.cxx *.cpp *.c)", "Any files (*)"],
            // working_path: "/Home",
            onOk: fname => {
               let p = Math.max(fname.lastIndexOf("/"), fname.lastIndexOf("\\"));
               let title = (p > 0) ? fname.substr(p+1) : fname;
               this.setEditorFileKind(tab, title);
               oModel.setProperty("/title", title);
               oModel.setProperty("/filename", fname);
               this.syncEditor(tab, "SAVE");
               this.doReload(true); // while new file appears, one should reload items on server
            },
            onCancel: () => { },
            onFailure: () => { }
         });
      },

      /** @summary send editor content to server (if was changed) */
      syncEditor: function(tab, cmd) {
         const oModel = tab.getModel();
         let modified = oModel.getProperty("/modified");
         if ((modified === false) && !cmd) return;
         let data = [ tab.getKey(),
                      oModel.getProperty("/title") || "",
                      oModel.getProperty("/filename") || "",
                      modified ? "changed" : "",
                      modified ? oModel.getProperty("/code") : ""];
         if (cmd) data.push(cmd);
         oModel.setProperty("/modified", false);
         if (cmd) oModel.setProperty("/can_close", true); // any command means file will be stored
         return this.websocket.send("SYNCEDITOR:" + JSON.stringify(data));
      },

      /** @brief Handle the "Save" button press event */
      onSaveFile: function (tab) {
         if (!tab.getModel().getProperty("/filename"))
            return this.onSaveAsFile(tab);
         this.syncEditor(tab, "SAVE");
      },

      /** @brief Handle the "Run" button press event */
      onRunMacro: function (tab) {
         if (!tab.getModel().getProperty("/filename"))
            return this.onSaveAsFile(tab);
         this.syncEditor(tab, "RUN");
      },

      /** @summary Search TabContainerItem by key value */
      findTab: function(name, set_active) {
         let oTabContainer = this.byId("tabContainer"),
             items = oTabContainer.getItems();
         for(let i = 0; i< items.length; i++)
            if (items[i].getKey() === name) {
               if (set_active) oTabContainer.setSelectedItem(items[i]);
               return items[i];
            }
      },

      /** @summary Retuns current selected tab, instance of TabContainerItem */
      getSelectedTab: function() {
         let oTabContainer = this.byId("tabContainer");
         let items = oTabContainer.getItems();
         for(let i = 0; i< items.length; i++)
            if (items[i].getId() === oTabContainer.getSelectedItem())
               return items[i];
      },

      /** @summary Retuns code editor from the tab */
      getCodeEditor: function(tab) {
         let items = tab ? tab.getContent() : [];
         for (let n = 0; n < items.length; ++n)
            if (items[n].isA("sap.ui.codeeditor.CodeEditor"))
               return items[n];
      },

      /** @summary Extract the file name and extension
        * @desc Used to set the editor's model properties and display the file name on the tab element */
      setEditorFileKind: function (oTabElement, title) {
         let oEditor = this.getCodeEditor(oTabElement);
         if (!oEditor) return;
         let oModel = oTabElement.getModel();
         let ext = "txt";

         oModel.setProperty("/runEnabled", false);
         oModel.setProperty("/saveEnabled", true);

         if (title.lastIndexOf('.') > 0)
            ext = title.substr(title.lastIndexOf('.') + 1);

         switch (ext.toLowerCase()) {
            case "c":
            case "cc":
            case "cpp":
            case "cxx":
               oModel.setProperty("/runEnabled", true);
               // runButton.setEnabled(true);
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
               oModel.setProperty("/runEnabled", true);
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
            case "xml":
               oEditor.setType('xml');
               break;
            default: // unsupported type
               if (title.indexOf('README') >= 0)
                  oEditor.setType('plain_text');
               else
                  return false;
               break;

         }
         oModel.setProperty("/ext", ext);
         return true;
      },

      /* ============================================= */
      /* =============== Settings menu =============== */
      /* ============================================= */

      onSettingPress: function () {
         this._oSettingsModel.setProperty("/AppendToCanvas", this.model.isAppendToCanvas());
         this._oSettingsModel.setProperty("/OnlyLastCycle", (this.model.getOnlyLastCycle() > 0));
         this._oSettingsModel.setProperty("/ShowHiddenFiles", this.model.isShowHidden());
         this._oSettingsModel.setProperty("/SortMethod", this.model.getSortMethod());
         this._oSettingsModel.setProperty("/ReverseOrder", this.model.isReverseOrder());

         console.log('show TH2 draw option = ', this._oSettingsModel.getProperty("/optTH2"));

         if (!this._oSettingsMenu)
            this._oSettingsMenu = Fragment.load({
               name: "rootui5.browser.view.settingsmenu",
               controller: this
            }).then(menu => {
               menu.setModel(this._oSettingsModel);
               this.getView().addDependent(menu);
               return menu;
            });

         this._oSettingsMenu.then(menu => menu.open());
      },

      handleSeetingsConfirm: function() {
         let append = this._oSettingsModel.getProperty("/AppendToCanvas"),
             lastcycle = this._oSettingsModel.getProperty("/OnlyLastCycle"),
             hidden = this._oSettingsModel.getProperty("/ShowHiddenFiles"),
             sort = this._oSettingsModel.getProperty("/SortMethod"),
             reverse = this._oSettingsModel.getProperty("/ReverseOrder"),
             changed = false;

         if (append != this.model.isAppendToCanvas())
            this.model.setAppendToCanvas(append);

         if (lastcycle != (this.model.getOnlyLastCycle() > 0)) {
            changed = true;
            this.model.setOnlyLastCycle(lastcycle ? 1 : -1);
         }

         if (hidden != this.model.isShowHidden()) {
            changed = true;
            this.model.setShowHidden(hidden);
         }

         if (reverse != this.model.isReverseOrder()) {
            changed = true;
            this.model.setReverseOrder(reverse);
         }

         if (sort != this.model.getSortMethod()) {
            changed = true;
            this.model.setSortMethod(sort);
         }

         let optmsg = this.getOptionsMessage();
         if (optmsg != this.lastOptMessage) {
            this.lastOptMessage = optmsg;
            this.websocket.send(optmsg);
         }

         if (changed)
            this.doReload();
      },

      /** @summary Return message need to be send to server to change options */
      getOptionsMessage: function() {
         let arr = [this._oSettingsModel.getProperty('/optTH1') || '',
                    this._oSettingsModel.getProperty('/optTH2') || '',
                    this._oSettingsModel.getProperty('/optTProfile') || '' ];
         return 'OPTIONS:' + JSON.stringify(arr);
      },

      /** @summary Add Tab event handler */
      handlePressAddTab: function (oEvent) {
         //TODO: Change to some UI5 function (unknown for now), not know how to get
         let oButton = oEvent.getSource().mAggregations._tabStrip.mAggregations.addButton;

         // create action sheet only once
         if (!this._tabMenu)
            this._tabMenu = Fragment.load({
               name: "rootui5.browser.view.tabsmenu",
               controller: this
            }).then(menu => {
               this.getView().addDependent(menu);
               return menu;
            });

         this._tabMenu.then(menu => menu.openBy(oButton));
      },

      /** @summary handle creation of new tab */
      handleNewTab: function (oEvent) {
         let msg, txt = oEvent.getSource().getText();

         if (txt.indexOf('editor') >= 0)
            msg = "NEWWIDGET:editor";
         else if (txt.indexOf('Image') >= 0)
            msg = "NEWWIDGET:image";
         else if (txt.indexOf('Geometry') >= 0)
            msg = "NEWWIDGET:geom";
         else if (txt.indexOf('Root 6') >= 0)
            msg = "NEWWIDGET:tcanvas";
         else if (txt.indexOf('Root 7') >= 0)
            msg = "NEWWIDGET:rcanvas";

         if (this.isConnected && msg)
            this.websocket.send(msg);
      },

      /* =========================================== */
      /* =============== Breadcrumbs =============== */
      /* =========================================== */

      updateBReadcrumbs: function(split) {
         // already array with all items inside
         let oBreadcrumbs = this.getView().byId("breadcrumbs");
         oBreadcrumbs.removeAllLinks();
         for (let i = -1; i < split.length; i++) {
            let txt = i < 0 ? '/' : split[i];
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
            if (i > 0) path.push(oLinks[i].getText());
            if (oLinks[i].getId() === sId ) break;
         }
         // after CHPATH will be replied, client also start reload
         this.websocket.send('CHPATH:' + JSON.stringify(path));
      },

      tabSelectItem: function(oEvent) {
         let item = oEvent.getParameter('item');
         if (item && item.getKey())
            this.websocket.send("WIDGET_SELECTED:" + item.getKey());
      },

      doCloseTabItem: function(item) {
         let oTabContainer = this.byId("tabContainer");
         if (item.getKey())
            this.websocket.send("CLOSE_TAB:" + item.getKey());
         oTabContainer.removeItem(item);
      },

      /** @brief Close Tab event handler */
      handleTabClose: function(oEvent) {
         // prevent the tab being closed by default
         oEvent.preventDefault();

         let oItemToClose = oEvent.getParameter('item'),
             oModel = oItemToClose.getModel();

         if (oModel && oModel.getProperty("/can_close"))
            return this.doCloseTabItem(oItemToClose);

         MessageBox.confirm('Do you really want to close the "' + oItemToClose.getAdditionalText() + '" tab?', {
            onClose: oAction => {
               if (oAction === MessageBox.Action.OK) {
                   this.doCloseTabItem(oItemToClose);
                   MessageToast.show('Closed the "' + oItemToClose.getName() + '" tab', { duration: 1500 });
                }
            }
         });
      },

      /* ======================================== */
      /* =============== Terminal =============== */
      /* ======================================== */

      onTerminalSubmit: function(oEvent) {
         let command = oEvent.getSource().getValue();
         this.websocket.send("CMD:" + command);
         oEvent.getSource().setValue("");
         this.requestRootHist();
         this.requestLogs();
      },

      requestRootHist: function() {
         return this.websocket.send("GETHISTORY:");
      },

      updateRootHist: function(entries) {
         let json = { hist:[] };
         entries.forEach(entry => json.hist.push({ name: entry }));
         this.getView().byId("terminal-input").setModel(new JSONModel(json));
      },

      requestLogs: function() {
         return this.websocket.send("GETLOGS:");
      },

      formatLine: function(line) {
         let res = "", p = line.indexOf("\u001b");

         while (p >= 0) {
            if (p > 0)
               res += line.slice(0, p);
            line = line.slice(p+1);
            // cut of colors codes
            if (line[0] == '[') {
               p = 0;
               while ((p < line.length) && (line[p] != 'm')) p++;
               line = line.slice(p+1);
            }
            p = line.indexOf("\u001b");
         }

         return res + line;
      },

      updateLogs: function(logs) {
         let str = "";
         logs.forEach(line => str += this.formatLine(line)+"\n");
         this.getView().byId("output_log").setValue(str);
      },

      onFullScreen: function() {
         let splitApp = this.getView().byId("SplitAppBrowser");
         if (uiDevice.orientation.landscape) {
            if(splitApp.getMode() === "ShowHideMode") {
               splitApp.setMode("HideMode");
            } else {
               splitApp.setMode("ShowHideMode");
            }
         } else {
            if(splitApp.isMasterShown()) {
               splitApp.hideMaster();
            } else {
               splitApp.showMaster();
            }
         }
      },

      handleChangeOrientation: function(is_landscape) {
         let btn = this.getView().byId('expandMaster');
         btn.setVisible(is_landscape);
         btn.setIcon("sap-icon://open-command-field");
         this.getView().byId('masterPage').getParent().removeStyleClass('masterExpanded');
      },

      onExpandMaster: function () {
         const master = this.getView().byId('masterPage').getParent();
         master.toggleStyleClass('masterExpanded');
         const expanded = master.hasStyleClass('masterExpanded');
         const btn = this.getView().byId('expandMaster');
         btn.setIcon(expanded ? "sap-icon://close-command-field" : "sap-icon://open-command-field");
      },

      /* ========================================== */
      /* =============== ToolHeader =============== */
      /* ========================================== */

      /** @summary Assign the "double click" event handler to each row */
      assignRowHandlers: function () {
         let rows = this.byId("treeTable").getRows();
         for (let k = 0; k < rows.length; ++k) {
            rows[k].$().dblclick(this.onRowDblClick.bind(this, rows[k]));
         }
      },

      /** @summary Double-click event handler */
      onRowDblClick: function (row) {
         let ctxt = row.getBindingContext(),
             prop = ctxt ? ctxt.getProperty(ctxt.getPath()) : null;

         if (!prop || !prop.path) return;

         let opt = "<dflt>", exec = "";

         if (this.model.isAppendToCanvas())
            opt = "<append>" + opt;

         if (this._oSettingsModel.getProperty("/DBLCLKRun"))
            exec = "exec";

         let args = prop.path.slice(); // make copy of array
         args.push(opt, exec);

         this.websocket.send("DBLCLK:" + JSON.stringify(args));

         this.invokeWarning("Processing double click on: " + prop.name, 500);
      },

      invokeWarning: function(msg, tmout) {
         this.cancelWarning();

         this.warn_timeout = setTimeout(() => {
            if (!this.warn_timeout) return;
            delete this.warn_timeout;

            this.oWarningDialog = new Dialog({
               type: DialogType.Message,
               title: "Warning",
               state: ValueState.Warning,
               content: new mText({ text: msg }),
               beginButton: new Button({
                  type: ButtonType.Emphasized,
                  text: "OK",
                  press: () => this.cancelWarning()
               })
            });

            this.oWarningDialog.open();

         }, tmout);

      },

      cancelWarning: function() {
         if (this.warn_timeout) {
            clearTimeout(this.warn_timeout);
            delete this.warn_timeout;
         }
         if (this.oWarningDialog) {
            this.oWarningDialog.close();
            delete this.oWarningDialog;
         }

      },

      onWebsocketOpened: function(/*handle*/) {
         this.isConnected = true;

         if (this.model)
            this.model.sendFirstRequest(this.websocket);
      },

      onWebsocketClosed: function() {
         // when connection closed, close panel as well
         console.log('Close RBrowser window when connection closed');

         if (window) window.close();

         this.isConnected = false;
      },

     /** @summary Entry point for all data from server */
     onWebsocketMsg: function(handle, msg, offset) {

         // any message from server clear all warnings
         this.cancelWarning();

         if (typeof msg != "string")
            return console.error("Browser do not uses binary messages len = " + mgs.byteLength);

         // console.log('MSG', msg.substr(0,20));

         let p = msg.indexOf(':'), mhdr = '';
         if (p > 0) {
            mhdr = msg.slice(0, p);
            msg = msg.slice(p+1);
         } else {
            mhdr = msg;
            msg = '';
         }

         switch (mhdr) {
         case "INMSG":
            this.processInitMsg(msg);
            break;
         case "NOPE":
            break;
         case "EDITOR": { // update code editor
            let arr = JSON.parse(msg),
                tab = this.findTab(arr[0]);

            if (tab) {
               this.setEditorFileKind(tab, arr[1]);
               tab.getModel().setProperty("/title", arr[1]);
               tab.getModel().setProperty("/filename", arr[2]);
               tab.getModel().setProperty("/code", arr[3]);
               tab.getModel().setProperty("/modified", false);
               tab.getModel().setProperty("/can_close", true);
               tab.getModel().setProperty("/first_change", true);
            }
            break;
         }
         case "IMAGE": { // update image viewer
            let arr = JSON.parse(msg);
            let tab = this.findTab(arr[0]);

            if (tab) {
               tab.setAdditionalText(arr[1]);
               // let filename = arr[2];
               let oViewer = tab.getContent()[0].getContent()[0];
               oViewer.setSrc(arr[3]);
            }
            break;
         }
         case "NEWWIDGET": {  // widget created by server, need to establish connection
            let arr = JSON.parse(msg);
            this.createElement(arr[0], arr[1], arr[2], arr[3], arr[4]);
            this.findTab(arr[2], true); // set active
            break;
         }
         case "SET_TITLE": {
            let arr = JSON.parse(msg),
                tab = this.findTab(arr[0]);
            tab?.setAdditionalText(arr[1]);
            tab?.setTooltip(arr[2] || '');
            break;
         }
         case "WORKPATH":
            this.updateBReadcrumbs(JSON.parse(msg));
            this.doReload();
            break;
         case "SELECT_WIDGET":
           this.findTab(msg, true); // set active
           break;
         case "BREPL":   // browser reply
            if (this.model) {
               let bresp = JSON.parse(msg);
               this.model.processResponse(bresp);

               if (bresp.path.length == 0) {
                  let tt = this.getView().byId("treeTable");
                  tt.autoResizeColumn(2);
                  tt.autoResizeColumn(1);
               }
            }
            break;
         case "HISTORY":
            this.updateRootHist(JSON.parse(msg));
            break;
         case "LOGS":
            this.updateLogs(JSON.parse(msg));
            break;
         default:
            console.error(`Not recognized msg ${mhdr} len= ${msg.length}`);
         }
      },

      /** @summary Show special message instead of nodes hierarchy */
      showTextInBrowser: function(text) {
         let br = this.byId("treeTable");
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
         // let oSplitApp = this.getView().byId("SplitAppBrowser");
         // oSplitApp.getAggregation("_navMaster").$().css("width", "400px");
      },

      /** @summary Reload (refresh) file tree browser */
      onRealoadPress: function() {
         this.doReload(true); // force also update of items on server
      },

      onWorkingDirPress: function() {
         this.websocket.send("CDWORKDIR");
      },

      doReload: function(force_reload) {
         if (this.standalone) {
            this.showTextInBrowser();
            this.paintFoundNodes(null);
            this.model.setFullModel(this.fullModel);
         } else {
            this.model.reloadMainModel(true, force_reload);
         }
      },

      /** @summary Quit ROOT session */
      onQuitRootPress: function() {
         this.websocket.send("QUIT_ROOT");
         setTimeout(() => { if (window) window.close(); }, 2000);
      },

      onSearch: function(oEvt) {
         this.changeItemsFilter(oEvt.getSource().getValue());
      },

      /** @summary Submit node search query to server, ignore in offline case */
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

      /** @summary process initial message, now it is list of existing canvases */
      processInitMsg: function(msg) {
         let arr = this.jsroot.parse(msg);
         if (!arr) return;

         this.updateBReadcrumbs(arr[0]);

         for (let k = 1; k < arr.length; ++k) {
            let kind = arr[k][0];
            if (kind == "active") {
               this.findTab(arr[k][1], true); // set active
            } else if (kind == "history") {
               arr[k].shift();
               this.updateRootHist(arr[k]);
            } else if (kind == "logs") {
               arr[k].shift();
               this.updateLogs(arr[k]);
            } else if (kind == "drawoptions") {
               this._oSettingsModel.setProperty('/optTH1', arr[k][1]);
               this._oSettingsModel.setProperty('/optTH2', arr[k][2]);
               this._oSettingsModel.setProperty('/optTProfile', arr[k][3]);
            } else {
               this.createElement(kind, arr[k][1], arr[k][2], arr[k][3], arr[k][4]);
            }
         }

         this.lastOptMessage = this.getOptionsMessage();
      },

      createElement: function(kind, par1, par2, par3, tooltip) {
         let tabItem;
         switch(kind) {
            case "editor": tabItem = this.createCodeEditor(par1, par2, par3, tooltip); break;
            case "image": tabItem = this.createImageViewer(par1, par2, par3, tooltip); break;
            case "tree": tabItem = this.createTreeViewer(par1, par2, par3, tooltip); break;
            case "geom": tabItem = this.createGeomViewer(par1, par2, par3, tooltip); break;
            case "catched": tabItem = this.createCatchedWidget(par1, par2, par3, tooltip); break;
            default: tabItem = this.createCanvas(kind, par1, par2, par3, tooltip);
         }
         return tabItem;
      },

      createCatchedWidget: function(url, name, catched_kind, tooltip) {
         switch(catched_kind) {
            case "rcanvas": return this.createCanvas("rcanvas", url, name, "Catched RCanvas", tooltip);
            case "tcanvas": return this.createCanvas("tcanvas", url, name, "Catched TCanvas", tooltip);
            case "tree": return this.createTreeViewer(url, name, "Catched tree viewer", tooltip);
            case "geom": return this.createGeomViewer(url, name, "Catched geom viewer", tooltip);
            default: console.error("Not supported cacthed kind", catched_kind);
         }
      },

      createGeomViewer: function(url, name , _title, tooltip) {
         let oTabContainer = this.byId("tabContainer");
         let item = new TabContainerItem({
            name: "Geom viewer",
            key: name,
            additionalText: name,
            icon: "sap-icon://column-chart-dual-axis",
            tooltip: tooltip || ''
         });

         oTabContainer.addItem(item);
         // oTabContainer.setSelectedItem(item);

         this.jsroot.connectWebWindow({
            kind: this.websocket.kind,
            href: this.websocket.getHRef(url),
            user_args: { nobrowser: true }
         }).then(handle => XMLView.create({
            viewName: "rootui5.eve7.view.GeomViewer",
            viewData: { conn_handle: handle, embeded: true, jsroot: this.jsroot }
         })).then(oView => item.addContent(oView));

         return item;
      },

      createCanvas: function(kind, url, name, title, tooltip) {
         if (!url || !name || (kind != "tcanvas" && kind != "rcanvas")) return;

         let item = new TabContainerItem({
            name: (kind == "rcanvas") ? "RCanvas" : "TCanvas",
            key: name,
            additionalText: title || name,
            icon: "sap-icon://column-chart-dual-axis",
            tooltip: tooltip || ''
         });

         this.byId("tabContainer").addItem(item);

         let conn = new this.jsroot.WebWindowHandle(this.websocket.kind);
         conn.setHRef(this.websocket.getHRef(url)); // argument for connect, makes relative path

         import(this.jsroot.source_dir + 'modules/draw.mjs').then(draw => {
            if (kind == "rcanvas")
               return import(this.jsroot.source_dir + 'modules/gpad/RCanvasPainter.mjs').then(h => {
                   draw.assignPadPainterDraw(h.RPadPainter);
                   return new h.RCanvasPainter(null, null);
                });

            return import(this.jsroot.source_dir + 'modules/gpad/TCanvasPainter.mjs').then(h => {
               draw.assignPadPainterDraw(h.TPadPainter);
               return new h.TCanvasPainter(null, null);
            });
         }).then(painter => {
            painter.online_canvas = true; // indicates that canvas gets data from running server
            painter.embed_canvas = true;  // use to indicate that canvas ui should not close complete window when closing
            painter.use_openui = true;
            painter.batch_mode = false;
            painter._window_handle = conn;

            return XMLView.create({
               viewName: "rootui5.canv.view.Canvas",
               viewData: { canvas_painter: painter },
               height: "100%"
            });
         }).then(oView => {
            item.addContent(oView);
            let ctrl = oView.getController();
            ctrl.onCloseCanvasPress = this.doCloseTabItem.bind(this, item);
         });

         return item;
      }

   });

});
