sap.ui.define(['sap/ui/core/Component',
               'sap/ui/core/UIComponent',
               'sap/ui/core/mvc/Controller',
               'sap/ui/layout/Splitter',
               'sap/ui/layout/SplitterLayoutData',
               'sap/m/library',
               'sap/m/Button',
               'sap/m/MenuItem',
               'sap/m/MessageBox',
               'rootui5/eve7/lib/EveManager',
               "sap/ui/core/mvc/XMLView",
               'sap/ui/model/json/JSONModel'
], function(Component, UIComponent, Controller, Splitter, SplitterLayoutData, MobileLibrary, mButton, mMenuItem, MessageBox, EveManager, XMLView, JSONModel) {

   "use strict";

   return Controller.extend("rootui5.eve7.controller.Main", {
      onInit: function () {
         this.mgr = new EveManager();
         // this.initClientLog();

         let conn_handle = Component.getOwnerComponentFor(this.getView()).getComponentData().conn_handle;
         this.mgr.UseConnection(conn_handle);
         // this.mgr.UseConnection(this.getView().getViewData().conn_handle);

         this.mgr.RegisterController(this);
         // method to found summary controller by ID and set manager to it
         let elem = this.byId("Summary");
         let ctrl = elem.getController();
         ctrl.SetMgr(this.mgr);

         this.primarySplitter = this.getView().byId("MainAreaSplitter");
         this.primarySplitter.secondary = null;
      },

      onDisconnect : function() {
         let t = this.byId("centerTitle");
         t.setHtmlText("<strong style=\"color: red;\">Client Disconnected !</strong>");
      },

      /** called when relative number of possible send operation below or over the threshold  */
      onSendThresholdChanged: function(below, value) {
         let t = this.byId("centerTitle");
         t.$().css('color', below ? 'yellow' : '')
      },

      initClientLog: function() {
         let consoleObj = {};
         consoleObj.data = [];

         consoleObj.model = new JSONModel();
         consoleObj.model.setData(consoleObj.data);

         consoleObj.cntInfo = 0;
         consoleObj.cntWarn = 0;
         consoleObj.cntErr = 0;

         consoleObj.stdlog = console.log.bind(console);
         console.log = function ()
         {
            consoleObj.data.push({ type: "Information", title: Array.from(arguments), counter: ++consoleObj.cntInfo });
            consoleObj.stdlog.apply(console, arguments);
            consoleObj.model.setData(consoleObj.data);
            consoleObj.model.refresh(true);
         };

         consoleObj.stdwarn = console.warn.bind(console);
         console.warn = function ()
         {
            consoleObj.data.push({ type: "Warning", title: Array.from(arguments), counter: ++consoleObj.cntWarn });
            consoleObj.stdwarn.apply(console, arguments);
            consoleObj.model.setData(consoleObj.data);
            consoleObj.model.refresh(true);
         };

         consoleObj.stderror = console.error.bind(console);
         console.error = function ()
         {
            consoleObj.data.push({ type: "Error", title: Array.from(arguments), counter: ++consoleObj.cntErr });
            consoleObj.stderror.apply(console, arguments);
            consoleObj.model.setData(consoleObj.data);
            consoleObj.model.refresh(true);
         };

         // create GUI ClientLog
         let pthis = this;
         XMLView.create({
            viewName: "rootui5.eve7.view.ClientLog",
         }).then(function (oView)
         {
            oView.setModel(consoleObj.model);
            oView.getController().oDialog.setModel(consoleObj.model);
            let logCtrl = oView.getController();
            let toolbar = pthis.byId("otb1");
            toolbar.addContentRight(logCtrl.getButton());
         });
      },

      UpdateCommandsButtons: function(cmds) {
         if (!cmds || this.commands) return;

         let toolbar = this.byId("otb1");

         this.commands = cmds;
         for (let k = cmds.length-1; k>=0; --k) {
            let btn = new mButton({
               icon: cmds[k].icon,
               text: cmds[k].name,
               press: this.executeCommand.bind(this, cmds[k])
            });
            toolbar.insertContentLeft(btn, 0);
         }
      },

      viewItemPressed: function (elem, oEvent) {
         let item = oEvent.getSource();
         // console.log('item pressed', item.getText(), elem);

         let name = item.getText();
         if (name.indexOf(" ") > 0) name = name.substr(0, name.indexOf(" "));
         // FIXME: one need better way to deliver parameters to the selected view
         EVE.$eve7tmp = { mgr: this.mgr, eveViewerId: elem.fElementId};
         let oRouter = UIComponent.getRouterFor(this);
         if (name == "Table")
            oRouter.navTo("Table", { viewName: name });
         else if (name == "Lego")
            oRouter.navTo("Lego", { viewName: name });
         else
            oRouter.navTo("View", { viewName: name });
      },

      updateViewers: function(loading_done) {
         let viewers = this.mgr.FindViewers();

         // first check number of views to create
         let staged = [];
         for (let n=0;n<viewers.length;++n) {
            let el = viewers[n];
            // at startup show only mandatory views
            if (typeof el.subscribed == 'undefined')
               el.subscribed = el.Mandatory;

            if (!el.$view_created && el.fRnrSelf) staged.push(el);
         }
         if (staged.length == 0) return;

         let vMenu = this.getView().byId("menuViewId");

         for (let n=0;n<staged.length;++n) {
            let eveView = staged[n];

            // add menu item
            let vi = new mMenuItem({ text: staged[n].fName, press: this.subscribeView.bind(this, staged[n]) });
            vi.setEnabled(!eveView.subscribed);
            vi.eveView = eveView;
            vMenu.addItem(vi);

            eveView.$view_created = true;
            if(eveView.subscribed) this.makeEveViewController(eveView);
         }

         if (staged.length === 1) {
            let eveView = staged[0];
            let t = eveView.ca.byId("tbar");
            t.getContent()[2].setEnabled(false);
         }
      },

      makeEveViewController: function(elem)
      {
         let viewid = "EveViewer" + elem.fElementId;
         let main = this;
         // create missing view
         console.log("Creating view", viewid);
 
         // TODO: Generalize instantiation without  the if/else statements
         let vtype = "rootui5.eve7.view.GL";
         if (elem.fName === "Table")
            vtype = "rootui5.eve7.view.EveTable";
         else if (elem.fName === "Lego")
            vtype = "rootui5.eve7.view.Lego";
         else if (elem.fName === "GeoTable")
               vtype = "rootui5.eve7.view.GeoTable";

         let oOwnerComponent = Component.getOwnerComponentFor(this.getView());
         let view = oOwnerComponent.runAsOwner(function() {
            return new sap.ui.xmlview({
               id: viewid,
               viewName: vtype,
               viewData: { mgr: main.mgr, eveViewerId: elem.fElementId },
            });
         });

         if (elem.fRnrSelf) {
            if (this.primarySplitter.getContentAreas().length == 1) {
               this.primarySplitter.addContentArea(view);
            }
            else {
               if (!this.primarySplitter.secondary) {
                  let vv = new Splitter("SecondaryViewSplitter", { orientation: "Vertical" });
                  vv.setLayoutData(new SplitterLayoutData({ resizable: true, size: "25%" }));
                  this.primarySplitter.addContentArea(vv);
                  this.primarySplitter.secondary = vv;
               }
               this.primarySplitter.secondary.addContentArea(view);
               this.setToolbarSwapIcon(view, "arrow-left");
            }
         }
         elem.ca = view;

         // reset flag needed by UT_PostStream callback
         delete elem.pendInstance;
      },

      subscribeView: function(viewer, e)
      {
         let vMenu = this.getView().byId("menuViewId");
         viewer.subscribed = true;
         viewer.pendInstance = true;
         e.getSource().setEnabled(false);

         this.mgr.SendMIR("ConnectClient()", viewer.fElementId, "ROOT::Experimental::REveViewer");
      },

      switchSingle: function (elem, oEvent) {
         let viewer = this.mgr.GetElement(elem.fElementId);
         // console.log('item pressed', item.getText(), elem);

         let name = viewer.fName;
        // if (name.indexOf(" ") > 0) name = name.substr(0, name.indexOf(" "));
         // FIXME: one need better way to deliver parameters to the selected view
         EVE.$eve7tmp = { mgr: this.mgr, eveViewerId: elem.fElementId};

         let oRouter = UIComponent.getRouterFor(this);
         if (name == "Table") {
            oRouter.navTo("Table", { viewName: name });
         }
         else if (name == "Lego")
         {
            oRouter.navTo("Lego", { viewName: name });
         }
         else {
            oRouter.navTo("View", { viewName: name });
         }
      },

      removeView: function(viewer) {
         let primary = this.getView().byId("MainAreaSplitter");
         let secondary;
         if (primary.getContentAreas().length == 3)
            secondary = primary.getContentAreas()[2];

         if (viewer.fRnrSelf) {
            let pa = primary.getContentAreas()[1];
            if (viewer.fElementId == pa.oViewData.eveViewerId) {
               //  viewer.ca = pa;
               ca.destroy();
               primary.removeContentArea(pa);
               if (secondary) {
                  let ss = secondary.getContentAreas();
                  let ssf = ss[0];
                  secondary.removeContentArea(ssf);
                  primary.removeContentArea(secondary);
                  primary.addContentArea(ssf);
                  primary.addContentArea(secondary);
               }
            }
            else {
               secondary.getContentAreas().forEach(ca => {
                  if (viewer.fElementId == ca.oViewData.eveViewerId) {
                     // viewer.ca = ca;
                     secondary.removeContentArea(ca);
                     ca.destroy();
                     return false;
                  }
               });
            }
            viewer.subscribed = false;


            let vMenu = this.getView().byId("menuViewId");
            vMenu.getItems().forEach(c => { if (c.eveView == viewer) c.setEnabled(true); });

            let siList = viewer.childs;
            for (let i = 0; i < siList.length; ++i)
            {
               let scene = this.mgr.GetElement(siList[i].fSceneId);
               this.mgr.recursiveDestroy(scene);
            }

            let mir = "DisconnectClient()";
            this.mgr.SendMIR(mir, viewer.fElementId, "ROOT::Experimental::REveViewer");
         }
      },

      switchViewSides: function (viewer) {
         let primary = this.primarySplitter;
         let secondary = primary.secondary;
         let pa = this.primarySplitter.getContentAreas()[1];
         this.setToolbarSwapIcon(pa, "arrow-left");

         if (viewer.fElementId == pa.oViewData.eveViewerId)
         {
            let sa = secondary.getContentAreas()[0];
            primary.removeContentArea(pa);
            secondary.removeContentArea(sa);
            primary.insertContentArea(sa, 1);
            secondary.insertContentArea(pa, 0);
            this.setToolbarSwapIcon(sa, "arrow-right");
         }
         else {
            let idx = secondary.indexOfContentArea(viewer.ca);
            primary.removeContentArea(pa);
            secondary.removeContentArea(viewer.ca);
            primary.insertContentArea(viewer.ca, 1);
            secondary.insertContentArea(pa, idx);
            this.setToolbarSwapIcon(viewer.ca, "arrow-right");
         }
         secondary.resetContentAreasSizes();
      },

      setToolbarSwapIcon(va, iName)
      {
         let t = va.byId("tbar");
         let sBtn = t.getContent()[2];
         sBtn.setIcon("sap-icon://" + iName);
      },

      setToolbarExpandedAction(va) {
         let bar = va.byId("tbar");
         let ca = bar.getContent();
         while (bar.getContent().length > 1)
            bar.removeContent(bar.getContent().length - 1);

         var bb = new sap.m.Button({
            type: MobileLibrary.ButtonType.Default,
            text: "Back",
            enabled: true,
            press: function () {
               window.history.go(-1)
            }
         });
         bar.addContent(bb);
      },

      onEveManagerInit: function() {
         // console.log("manager updated");
         this.UpdateCommandsButtons(this.mgr.commands);
         this.updateViewers();
      },

      /*
       * processWaitingMsg: function() { for ( let i = 0; i <
       * msgToWait.length; ++i ) {
       * this.onWebsocketMsg(handleToWait, msgToWait[i]); }
       * handleToWait = 0; msgToWait = []; },
       */
      event: function() {
         // this._event = lst;

      },

      setMainVerticalSplitterHeight: function(){
         let mainViewHeight = document.body.clientHeight;
         let mainToolbarHeight = 49;
         let height = mainViewHeight - mainToolbarHeight;
         let splitter =  this.getView().byId("MainAreaSplitter");
         if (splitter) {
            // console.log("set splitter height >>> " , height);
            splitter.setHeight(height + "px");
         }
      },

      onAfterRendering: function(){
         let me = this;
         setTimeout(
            function(){
               $(window).on("resize", function() {
                  me.setMainVerticalSplitterHeight();
               });
               me.setMainVerticalSplitterHeight();
            }, 100);
      },

      onToolsMenuAction : function (oEvent) {

         let item = oEvent.getParameter("item");

         switch (item.getText()) {
            case "GED Editor": this.getView().byId("Summary").getController().toggleEditor(); break;
          //  case "Event statusbar": this.toggleShowStatus(); break;
          //  case "Toolbar": this.toggleToolBar(); break;
          //  case "Tooltip info": this.toggleToolTip(); break;
         }
      },

      showHelp : function(oEvent) {
         alert("User support: root-webgui@cern.ch");
      },

      showUserURL : function(oEvent) {
         MobileLibrary.URLHelper.redirect("https://github.com/alja/jsroot/blob/dev/eve7.md", true);
      },

      executeCommand : function(cmd) {
         if (!cmd || !this.mgr.handle)
            return;

         // idealy toolbar shoulf be unactive, but thus is the role af web application
         if (this.mgr.busyProcessingChanges) {
            return;
         }

         this.mgr.SendMIR(cmd.func, cmd.elementid, cmd.elementclass);

         if ((cmd.name == "QuitRoot") && window) {
             window.close();
         }
      }

   });
});
