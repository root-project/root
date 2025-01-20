sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/m/Link',
               'sap/ui/core/Fragment',
               'rootui5/browser/model/BrowserModel',
               'sap/ui/Device',
               'sap/ui/model/json/JSONModel',
               'sap/ui/table/Column',
               'sap/ui/table/TreeTable',
               'sap/ui/layout/HorizontalLayout',
               'sap/m/TabContainerItem',
               'sap/m/MessageToast',
               'sap/m/MessageBox',
               'sap/m/Text',
               'sap/m/VBox',
               'sap/m/ProgressIndicator',
               'sap/m/Page',
               'sap/ui/core/mvc/XMLView',
               'sap/ui/core/Icon',
               'sap/m/Button',
               'sap/m/library',
               'sap/ui/core/library',
               'sap/m/Dialog',
               'sap/ui/codeeditor/CodeEditor',
               'sap/m/Image',
               'sap/tnt/ToolHeader',
               'sap/m/ToolbarSpacer',
               'sap/m/OverflowToolbarLayoutData',
               'sap/m/ScrollContainer',
               'rootui5/browser/controller/FileDialog.controller'
],function(Controller,
           Link,
           Fragment,
           BrowserModel,
           uiDevice,
           JSONModel,
           tableColumn,
           TreeTable,
           HorizontalLayout,
           TabContainerItem,
           MessageToast,
           MessageBox,
           mText,
           mVBox,
           mProgressIndicator,
           mPage,
           XMLView,
           CoreIcon,
           Button,
           mLibrary,
           uiCoreLibrary,
           Dialog,
           CodeEditor,
           Image,
           ToolHeader,
           ToolbarSpacer,
           OverflowToolbarLayoutData,
           ScrollContainer,
           FileDialogController) {

   "use strict";

   /** @summary Central ROOT RBrowser controller
     * @desc All Browser functionality is loaded after main ui5 rendering is performed */

   return Controller.extend("rootui5.browser.controller.Browser", {
      onInit() {

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
            optTH1: '<dflt>',
            TH1: [
               {name: '<dflt>'},
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
               {name: '<dflt>'},
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
            optTProfile: '<dflt>',
            TProfile: [
               {name: '<dflt>'},
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

         // add reload handler
         if (!this.standalone && this.websocket.addReloadKeyHandler)
            this.websocket.addReloadKeyHandler();

         // create model only for browser - no need for anybody else
         this.model = new BrowserModel();

         // copy extra attributes from element to node in the browser
         // later can be done automatically
         this.model.addNodeAttributes = function(node, elem) {
            node.icon = elem.icon || 'sap-icon://document';
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

         // ignore first resize
         this._columnResized = -1;

         // catch re-rendering of the table to assign handlers
         t.addEventDelegate({
            onAfterRendering() {
               this.assignRowHandlers();
               if (this._columnResized < 1) return;
               this._columnResized = 0;
               let fullsz = 4;

               t.getColumns().forEach(col => {
                  if (col.getVisible()) fullsz += 4 + col.$().width();
               });
               // this.getView().byId('masterPage').getParent().removeStyleClass('masterExpanded');
               this.getView().byId('SplitAppBrowser').getAggregation('_navMaster').setWidth(fullsz + 'px');
            }
         }, this);

         t.attachEvent("columnResize", {}, evnt => {
            this._columnResized++;
         }, this);
      },

      /* =========================================== */
      /* =============== Image Viewer ============== */
      /* =========================================== */

      createImageViewer(dummy_url, name, title, tooltip) {
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

      createTreeViewer(url, name, title, tooltip) {
         const oTabContainer = this.getView().byId("tabContainer");

         let item = new TabContainerItem(name, {
            icon: "sap-icon://tree",
            name: "Tree Viewer",
            key: name,
            additionalText: title,
            tooltip: tooltip || ''
         });

         oTabContainer.addItem(item);

         // with non empty url creates independent connection
         const handle = this.websocket.createChannel(url);
         handle.setUserArgs({ nobrowser: true });
         item._jsroot_conn = handle; // keep to be able disconnect

         return XMLView.create({
            viewName: "rootui5.tree.view.TreeViewer",
            viewData: { conn_handle: handle, embeded: true, jsroot: this.jsroot }
         }).then(oView => {
            item.addContent(oView);
            return item;
         });
      },

      /* =========================================== */
      /* =============== Info Viewer =============== */
      /* =========================================== */

      createInfoViewer(dummy_url, name, title, tooltip) {
         const oTabContainer = this.getView().byId("tabContainer");

         let item = new TabContainerItem(name, {
            icon: "sap-icon://hint",
            name: "Globals",
            key: name,
            additionalText: "{/title}",
            tooltip: tooltip || ''
         });

         let table = new TreeTable({
            width: "100%",
            editable: false,
            columnHeaderVisible : true,
            visibleRowCountMode: 'Auto',
            rows: "{path:'/items', parameters: {arrayNames:['sub']}}",
            selectionMode: 'None',
            enableSelectAll: false,
            ariaLabelledBy: "title"
         });

         let column1 = new tableColumn({
            label: 'Source',
            width: '15rem',
            autoResizable: true,
            visible: true,
            template: new HorizontalLayout({
               content: [ new mText({ text:'{name}', wrapping: false }) ]
            })
         });

         let column2 = new tableColumn({
            label: 'Information',
            autoResizable: true,
            visible: true,
            template: new HorizontalLayout({
               content: [ new mText({ text:'{info}', wrapping: false }) ]
            })
         });

         table.addColumn(column1);

         table.addColumn(column2);

         item.addContent(new ToolHeader({
            height: "40px",
            content: [
               new ToolbarSpacer({
                  layoutData: new OverflowToolbarLayoutData({
                     priority:"NeverOverflow",
                     minWidth: "16px"
                  })
               }),
               new Button({
                  text: "Expand all",
                  tooltip: "Get cling variables info again",
                  type: "Transparent",
                  press: () => table.expandToLevel(1)
               }),
               new Button({
                  text: "Callapse all",
                  tooltip: "Collapse all",
                  type: "Transparent",
                  press: () => table.collapseAll()
               }),
               new Button({
                  text: "Refresh",
                  tooltip: "Get cling variables info again",
                  type: "Transparent",
                  press: () => this.onRefreshInfo(item)
               })
            ]
         }));

         let cont = new ScrollContainer({
            height: 'calc(100% - 48px)'
         });

         cont.addContent(table);

         item.addContent(cont);

         item.setModel(new JSONModel({
            title,
            info: '---',
            items: {}
         }));

         oTabContainer.addItem(item);

         return item;
      },

      /** @brief Handle the "Refresh" button press event */
      onRefreshInfo(tab) {
         this.websocket.send("GETINFO:" + tab.getKey());
      },

      updateInfo(tab, content) {

         let arr = content.split('\n');

         let items = { sub: [] }, cache = {};

         arr.forEach(line => {
            let name = '', p = line.indexOf(' ');
            if (p == 0)
               name = '<default>';
            else
               name = line.slice(0, p).trim();

            if (!name) return;
            if (name == '<command')
               name = '<command line>';

            p = line.indexOf('(address: NA)');

            if (p < 0) return;

            if (name.indexOf('ROOT_cli') == 0)
               name = 'ROOT_cli';

            let info = line.slice(p + 14),
                item = cache[name];

            if (!item) {
               item = cache[name] = { name, sub: [] };
               items.sub.push(item);
            }
            item.sub.push({ name: "", info });
         });

         items.sub.sort((a,b) => a.name > b.name);

         // single elements not create hierarchy
         items.sub.forEach(item => {
            if (item.sub.length == 1) {
               item.info = item.sub[0].info;
               delete item.sub;
            }
         });

         tab.getModel().setProperty("/items", items);
      },


      /* =========================================== */
      /* =============== Code Editor =============== */
      /* =========================================== */

      createCodeEditor(dummy_url, name, editor_title, tooltip) {
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
            liveChange() {
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
      onSaveAsFile(tab) {

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
      syncEditor(tab, cmd) {
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
      onSaveFile(tab) {
         if (!tab.getModel().getProperty("/filename"))
            return this.onSaveAsFile(tab);
         this.syncEditor(tab, "SAVE");
      },

      /** @brief Handle the "Run" button press event */
      onRunMacro(tab) {
         if (!tab.getModel().getProperty("/filename"))
            return this.onSaveAsFile(tab);
         this.syncEditor(tab, "RUN");
      },

      /** @summary Search TabContainerItem by key value */
      findTab(name, set_active) {
         const oTabContainer = this.byId("tabContainer"),
               items = oTabContainer.getItems();
         for(let i = 0; i < items.length; i++)
            if (items[i].getKey() === name) {
               if (set_active)
                  oTabContainer.setSelectedItem(items[i]);
               return items[i];
            }
      },

      /** @summary Returns current selected tab, instance of TabContainerItem */
      getSelectedTab() {
         let oTabContainer = this.byId("tabContainer");
         let items = oTabContainer.getItems();
         for(let i = 0; i< items.length; i++)
            if (items[i].getId() === oTabContainer.getSelectedItem())
               return items[i];
      },

      /** @summary Returns code editor from the tab */
      getCodeEditor(tab) {
         let items = tab ? tab.getContent() : [];
         for (let n = 0; n < items.length; ++n)
            if (items[n].isA("sap.ui.codeeditor.CodeEditor"))
               return items[n];
      },

      /** @summary Extract the file name and extension
        * @desc Used to set the editor's model properties and display the file name on the tab element */
      setEditorFileKind(oTabElement, title) {
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

      onSettingsPress() {
         this._oSettingsModel.setProperty("/AppendToCanvas", this.model.isAppendToCanvas());
         this._oSettingsModel.setProperty("/OnlyLastCycle", (this.model.getOnlyLastCycle() > 0));
         this._oSettingsModel.setProperty("/ShowHiddenFiles", this.model.isShowHidden());
         this._oSettingsModel.setProperty("/SortMethod", this.model.getSortMethod());
         this._oSettingsModel.setProperty("/ReverseOrder", this.model.isReverseOrder());

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

      handleSettingsReset() {
         this._oSettingsModel.setProperty('/optTH1', '<dflt>');
         this._oSettingsModel.setProperty('/optTH2', 'col');
         this._oSettingsModel.setProperty('/optTProfile', '<dflt>')
      },

      handleSeetingsConfirm() {
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
      getOptionsMessage() {
         let arr = [ this._oSettingsModel.getProperty('/optTH1') || '',
                     this._oSettingsModel.getProperty('/optTH2') || '',
                     this._oSettingsModel.getProperty('/optTProfile') || '' ];
         for (let n = 0; n < arr.length; ++n)
            if (arr[n] == '<dflt>')
               arr[n] = '';
         return 'OPTIONS:' + JSON.stringify(arr);
      },

      /** @summary Add Tab event handler */
      handlePressAddTab(oEvent) {
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
      handleNewTab(oEvent) {
         let msg, txt = oEvent.getSource().getText();

         if (txt.indexOf('editor') >= 0)
            msg = "NEWWIDGET:editor";
         if (txt.indexOf('info') >= 0)
            msg = "NEWWIDGET:info";
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

      updateBReadcrumbs(split) {
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

      onBreadcrumbsPress(oEvent) {
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

      tabSelectItem(oEvent) {
         let item = oEvent.getParameter('item');
         if (item && item.getKey())
            this.websocket.send("WIDGET_SELECTED:" + item.getKey());
      },

      doCloseTabItem(item, skip_send) {
         let oTabContainer = this.byId("tabContainer");
         if (item.getKey() && !skip_send)
            this.websocket.send("CLOSE_TAB:" + item.getKey());
         // force connection to close
         item._jsroot_conn?.close(true);
         item._jsroot_painter?.cleanup();
         oTabContainer.removeItem(item);
      },

      /** @brief Close Tab event handler */
      handleTabClose(oEvent) {
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

      onTerminalSubmit(oEvent) {
         let command = oEvent.getSource().getValue();
         this.websocket.send("CMD:" + command);
         oEvent.getSource().setValue("");
         this.requestRootHist();
         this.requestLogs();
      },

      requestRootHist() {
         return this.websocket.send("GETHISTORY:");
      },

      updateRootHist(entries) {
         let json = { hist:[] };
         entries.forEach(entry => json.hist.push({ name: entry }));
         this.getView().byId("terminal-input").setModel(new JSONModel(json));
      },

      requestLogs() {
         return this.websocket.send("GETLOGS:");
      },

      formatLine(line) {
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

      updateLogs(logs) {
         let str = "";
         logs.forEach(line => str += this.formatLine(line)+"\n");
         this.getView().byId("output_log").setValue(str);
      },

      onFullScreen() {
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

      handleChangeOrientation(is_landscape) {
         let btn = this.getView().byId('expandMaster');
         btn.setVisible(is_landscape);
         btn.setIcon("sap-icon://open-command-field");
         this.getView().byId('masterPage').getParent().removeStyleClass('masterExpanded');
      },

      onExpandMaster() {
         // when button pressed - remove exact width value to let rule it via the style
         this.getView().byId('SplitAppBrowser').getAggregation('_navMaster').setWidth('');
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
      assignRowHandlers() {
         let rows = this.byId("treeTable").getRows();
         for (let k = 0; k < rows.length; ++k) {
            rows[k].$().dblclick(this.onRowDblClick.bind(this, rows[k]));
         }
      },

      /** @summary Double-click event handler */
      onRowDblClick(row) {
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

      invokeWarning(msg, tmout) {
         this.cancelWarning();

         this.warn_timeout = setTimeout(() => {
            if (!this.warn_timeout) return;
            delete this.warn_timeout;

            let content = new mVBox;
            content.addItem(new mText({ text: msg }));
            this.oWarningProgress = new mProgressIndicator({ percentValue: 0, displayValue: '0', showValue: true, visible: false });
            content.addItem(this.oWarningProgress);

            this.oWarningDialog = new Dialog({
               type: mLibrary.DialogType.Message,
               title: "Warning",
               state: uiCoreLibrary.ValueState.Warning,
               content,
               beginButton: new Button({
                  type: mLibrary.ButtonType.Emphasized,
                  text: 'OK',
                  press: () => this.cancelWarning()
               })
            });

            this.oWarningDialog.open();

         }, tmout);

      },

      cancelWarning() {
         if (this.warn_timeout) {
            clearTimeout(this.warn_timeout);
            delete this.warn_timeout;
         }
         if (this.oWarningDialog) {
            this.oWarningDialog.close();
            delete this.oWarningDialog;
            delete this.oWarningProgress;
         }
      },

      sendNewChannel(tabname, chid) {
         this.websocket.send(`NEWCHANNEL:["${tabname}","${chid}"]`);
      },

      onWebsocketOpened(/*handle*/) {
         this.isConnected = true;

         if (this.model)
            this.model.sendFirstRequest(this.websocket);
      },

      onWebsocketClosed() {
         // when connection closed, close panel as well
         console.log('Close RBrowser window when connection closed');

         if (window) window.close();

         this.isConnected = false;
      },

     /** @summary Entry point for all data from server */
     onWebsocketMsg(handle, msg, offset) {

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

         // any message from server clear all warnings
         if (mhdr === 'PROGRESS') {
            let progr = Number.parseFloat(msg);
            if (this.oWarningProgress && Number.isFinite(progr)) {
               this.oWarningProgress.setVisible(true);
               this.oWarningProgress.setPercentValue(progr*100);
               this.oWarningProgress.setDisplayValue((progr*100).toFixed(1) + '%');
            }
            return;
         }

         this.cancelWarning();

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
            let arr = JSON.parse(msg),
                tab = this.findTab(arr[0]);

            if (tab) {
               tab.setAdditionalText(arr[1]);
               // let filename = arr[2];
               let oViewer = tab.getContent()[0].getContent()[0];
               oViewer.setSrc(arr[3]);
            }
            break;
         }
         case "INFO": {
            let arr = JSON.parse(msg),
                tab = this.findTab(arr[0]);
            if (tab)
               this.updateInfo(tab, arr[2]);
            break;
         }
         case "NEWWIDGET": {  // widget created by server, need to establish connection
            const arr = JSON.parse(msg);
            const pr = this.createElement(arr[0], arr[1], arr[2], arr[3], arr[4]);

            this.findTab(arr[2], true); // set active
            Promise.resolve(pr).then(tab => {
               if (tab?._jsroot_conn?.isChannel())
                  this.sendNewChannel(arr[2], tab._jsroot_conn.getChannelId());
            });
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
         case "CLOSE_WIDGETS":
            JSON.parse(msg).forEach(name => {
               let tab = this.findTab(name);
               if (tab) this.doCloseTabItem(tab, true);
            });
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
      showTextInBrowser(text) {
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

      onBeforeRendering() {
         this.renderingDone = false;
      },

      onAfterRendering() {
         this.renderingDone = true;

         // this is how master width can be changed, may be extra control can be provided
         // let oSplitApp = this.getView().byId("SplitAppBrowser");
         // oSplitApp.getAggregation("_navMaster").$().css("width", "400px");
      },

      /** @summary Reload (refresh) file tree browser */
      onRealoadPress() {
         this.doReload(true); // force also update of items on server
      },

      onWorkingDirPress() {
         this.websocket.send("CDWORKDIR");
      },

      doReload(force_reload) {
         if (this.standalone) {
            this.showTextInBrowser();
            this.paintFoundNodes(null);
            this.model.setFullModel(this.fullModel);
         } else {
            this.model.reloadMainModel(true, force_reload);
         }
      },

      /** @summary Quit ROOT session */
      onQuitRootPress() {
         this.websocket.send("QUIT_ROOT");
         setTimeout(() => { if (window) window.close(); }, 2000);
      },

      /** @summary Start reload sequence with server */
      onReloadPress() {
         this.websocket?.askReload();
      },

      onSearch(oEvt) {
         this.changeItemsFilter(oEvt.getSource().getValue());
      },

      /** @summary Submit node search query to server, ignore in offline case */
      changeItemsFilter(query, from_handler) {

         if (from_handler) {
            delete this.search_handler;
            this.model.setItemsFilter(query);
         } else {
            // do not submit immediately, but after short timeout
            // if user types very fast - only last selection will be shown
            if (this.search_handler) clearTimeout(this.search_handler);
            this.search_handler = setTimeout(() => this.changeItemsFilter(query, true), 1000);
         }
      },

      /** @summary process initial message, now it is list of existing canvases */
      processInitMsg(msg) {
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
               this._oSettingsModel.setProperty('/optTH1', arr[k][1] || '<dflt>');
               this._oSettingsModel.setProperty('/optTH2', arr[k][2]|| '<dflt>');
               this._oSettingsModel.setProperty('/optTProfile', arr[k][3]|| '<dflt>');
            } else {
               const pr = this.createElement(kind, arr[k][1], arr[k][2], arr[k][3], arr[k][4]);
               Promise.resolve(pr).then(tab => {
                  if (tab?._jsroot_conn?.isChannel())
                     this.sendNewChannel(arr[k][2], tab._jsroot_conn.getChannelId());
               });
            }
         }

         this.lastOptMessage = this.getOptionsMessage();
      },

      createElement(kind, par1, par2, par3, tooltip) {
         let tabItem;
         switch(kind) {
            case "editor": tabItem = this.createCodeEditor(par1, par2, par3, tooltip); break;
            case "image": tabItem = this.createImageViewer(par1, par2, par3, tooltip); break;
            case "info": tabItem = this.createInfoViewer(par1, par2, par3, tooltip); break;
            case "tree": tabItem = this.createTreeViewer(par1, par2, par3, tooltip); break;
            case "geom": tabItem = this.createGeomViewer(par1, par2, par3, tooltip); break;
            case "catched": tabItem = this.createCatchedWidget(par1, par2, par3, tooltip); break;
            default: tabItem = this.createCanvas(kind, par1, par2, par3, tooltip);
         }
         return tabItem;
      },

      createCatchedWidget(url, name, catched_kind, tooltip) {
         switch(catched_kind) {
            case "rcanvas": return this.createCanvas("rcanvas", url, name, "Catched RCanvas", tooltip);
            case "tcanvas": return this.createCanvas("tcanvas", url, name, "Catched TCanvas", tooltip);
            case "tree": return this.createTreeViewer(url, name, "Catched tree viewer", tooltip);
            case "geom": return this.createGeomViewer(url, name, "Catched geom viewer", tooltip);
            default: console.error("Not supported cacthed kind", catched_kind);
         }
      },

      createGeomViewer(url, name , _title, tooltip) {
         let oTabContainer = this.byId("tabContainer");
         let item = new TabContainerItem({
            name: "Geom viewer",
            key: name,
            additionalText: name,
            icon: "sap-icon://column-chart-dual-axis",
            tooltip: tooltip || ''
         });

         oTabContainer.addItem(item);

         // with non empty url creates independent connection
         const handle = this.websocket.createChannel(url);
         handle.setUserArgs({ nobrowser: true });
         item._jsroot_conn = handle; // keep to be able disconnect

         return XMLView.create({
            viewName: 'rootui5.geom.view.GeomViewer',
            viewData: { conn_handle: handle, embeded: true, jsroot: this.jsroot }
         }).then(oView => {
            item.addContent(oView);
            return item;

         });
      },

      createCanvas(kind, url, name, title, tooltip) {
         if (!name || (kind != "tcanvas" && kind != "rcanvas"))
            return null;

         let item = new TabContainerItem({
            name: (kind == "rcanvas") ? "RCanvas" : "TCanvas",
            key: name,
            additionalText: title || name,
            icon: "sap-icon://column-chart-dual-axis",
            tooltip: tooltip || ''
         });

         this.byId("tabContainer").addItem(item);

         // with non empty url creates independent connection
         const conn = this.websocket.createChannel(url);
         item._jsroot_conn = conn; // keep to be able disconnect

         return import('jsroot/draw').then(draw => {
            if (kind == "rcanvas")
               return import('jsrootsys/modules/gpad/RCanvasPainter.mjs').then(h => {
                   draw.assignPadPainterDraw(h.RPadPainter);
                   return new h.RCanvasPainter(null, null);
                });

            return import('jsrootsys/modules/gpad/TCanvasPainter.mjs').then(h => {
               draw.assignPadPainterDraw(h.TPadPainter);
               return new h.TCanvasPainter(null, null);
            });
         }).then(painter => {
            item._jsroot_painter = painter;

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
            if (!item._jsroot_painter._window_handle)
               return item;
            // wait until painter is ready and fully configured to send message to server with newly created channel
            return new Promise(resolveFunc => {
               item._jsroot_painter._window_resolve = resolveFunc;
            }).then(() => item);
         });
      }

   });

});
