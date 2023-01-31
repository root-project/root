sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/core/Component',
   'sap/ui/model/json/JSONModel',
   'sap/ui/core/mvc/XMLView',
   'sap/m/MessageToast',
   'sap/m/Dialog',
   'sap/m/List',
   'sap/m/InputListItem',
   'sap/m/Input',
   'sap/m/Button',
   'sap/m/ButtonType',
   'sap/ui/layout/Splitter',
   'sap/ui/layout/SplitterLayoutData'
], function (Controller, Component, JSONModel, XMLView, MessageToast, Dialog, List, InputListItem, Input, Button, ButtonType, Splitter, SplitterLayoutData) {
   "use strict";

   function chk_icon(flag) {
      return flag ? "sap-icon://accept" : "sap-icon://decline";
   }

   let CController = Controller.extend("rootui5.canv.controller.Canvas", {

      onInit() {
         this._Page = this.getView().byId("CanvasMainPage");

         this.bottomVisible = false;

         let model = new JSONModel({ MenuBarIcon: chk_icon(true),
                                     GedIcon: chk_icon(false),
                                     StatusIcon: chk_icon(false),
                                     ToolbarIcon: chk_icon(false),
                                     TooltipIcon: chk_icon(true),
                                     StatusLbl1: "", StatusLbl2: "", StatusLbl3: "", StatusLbl4: "",
                                     Standalone: true, isRoot6: true });
         this.getView().setModel(model);

         let cp = this.getView().getViewData()?.canvas_painter;

         if (!cp) cp = Component.getOwnerComponentFor(this.getView()).getComponentData().canvas_painter;

         if (cp) {

            if (cp.embed_canvas) model.setProperty("/Standalone", false);

            this.getView().byId("MainPanel").getController().setPainter(cp);

            cp.executeObjectMethod = this.executeObjectMethod.bind(this);

            // overwriting method of canvas with standalone handling of GED
            cp.activateGed = this.activateGed.bind(this);
            cp.removeGed = this.cleanupIfGed.bind(this);
            cp.hasGed = this.isGedEditor.bind(this);

            cp.hasMenuBar = this.isMenuBarShow.bind(this);
            cp.actiavteMenuBar = this.toggleMenuBar.bind(this);
            cp.hasEventStatus = this.isStatusShown.bind(this);
            cp.activateStatusBar = this.toggleShowStatus.bind(this);
            cp.showCanvasStatus = this.showCanvasStatus.bind(this); // used only for UI5, otherwise global func
            cp.showMessage = this.showMessage.bind(this);
            cp.showSection = this.showSection.bind(this);

            cp.showUI5ProjectionArea = this.showProjectionArea.bind(this);
            cp.drawInUI5ProjectionArea = this.drawInProjectionArea.bind(this);

            cp.showUI5Panel = this.showLeftArea.bind(this);

            if (cp.v7canvas) model.setProperty("/isRoot6", false);

            let ws = cp._websocket || cp._window_handle;
            if (!cp.embed_canvas && ws?.addReloadKeyHandler)
               ws.addReloadKeyHandler();
         }
      },

      isv7() {
         let cp = this.getCanvasPainter();
         return cp?.v7canvas;
      },

      executeObjectMethod(painter, method, menu_obj_id) {

         if (method.fArgs!==undefined) {
            this.showMethodsDialog(painter, method, menu_obj_id);
            return true;
         }

         if (method.fName == "Inspect") {
            painter.showInspector();
            return true;
         }

         if (method.fName == "FitPanel") {
            this.showLeftArea("FitPanel");
            return true;
         }

         if (method.fName == "Editor") {
            this.activateGed(painter);
            return true;
         }

         return false; // not processed

      },

      /** @summary function used to activate GED in full canvas */
      activateGed(painter, kind, mode) {

         let canvp = this.getCanvasPainter();

         return this.showGed(true).then(() => {
            canvp.selectObjectPainter(painter);

            if (typeof canvp.processChanges == 'function')
               canvp.processChanges("sbits", canvp);

            return true;
         });
      },

      /** @desc Provide canvas painter */
      getCanvasPainter(also_without_websocket) {
         let p = this.getView().byId("MainPanel")?.getController().getPainter();

         return (p && (p._websocket || also_without_websocket)) ? p : null;
      },

      closeMethodDialog(painter, method, menu_obj_id) {

         let args = "";

         if (method) {
            let cont = this.methodDialog.getContent();

            let items = cont[0].getItems();

            if (method.fArgs.length !== items.length)
               alert(`Length mismatch between method description ${method.fArgs.length} and args list ${items.length} in dialog`);

            for (let k = 0; k < method.fArgs.length; ++k) {
               let arg = method.fArgs[k],
                   value = items[k].getContent()[0].getValue();

               if (value === "") value = arg.fDefault;

               if ((arg.fTitle=="Option_t*") || (arg.fTitle=="const char*")) {
                  // check quotes,
                  // TODO: need to make more precise checking of escape characters
                  if (!value) value = '""';
                  if (value[0] != '"') value = '"' + value;
                  if (value[value.length-1] != '"') value += '"';
               }

               args += (k > 0 ? "," : "") + value;
            }
         }

         this.methodDialog.close();
         this.methodDialog.destroy();

         if (painter && method && args) {

            if (painter.executeMenuCommand(method, args)) return;
            let exec = method.fExec;
            if (args) exec = exec.substr(0,exec.length-1) + args + ')';
            // invoked only when user press Ok button
            console.log(`execute method for object ${menu_obj_id} exec ${exec}`);

            let canvp = this.getCanvasPainter(),
                p = menu_obj_id.indexOf("#");

            if (canvp?.v7canvas)
               canvp.submitExec(painter, exec, (p > 0) ? menu_obj_id.slice(p+1) : '');
            else if (canvp)
               canvp.sendWebsocket(`OBJEXEC:${menu_obj_id}:${exec}`);
         }
      },

      showMethodsDialog(painter, method, menu_obj_id) {

         // TODO: deliver class name together with menu items
         method.fClassName = painter.getClassName();
         if ((menu_obj_id.indexOf("#x") > 0) || (menu_obj_id.indexOf("#y") > 0) || (menu_obj_id.indexOf("#z") > 0))
            method.fClassName = "TAxis";

         let items = [];

         for (let n = 0; n < method.fArgs.length; ++n) {
            let arg = method.fArgs[n];
            arg.fValue = arg.fDefault;
            if (arg.fValue == '""') arg.fValue = "";
            let item = new InputListItem({
               label: arg.fName + ' (' +arg.fTitle + ')',
               content: new Input({ placeholder: arg.fName, value: arg.fValue })
            });
            items.push(item);
         }

         this.methodDialog = new Dialog({
            title: method.fClassName + '::' + method.fName,
            content: new List({ items }),
             beginButton: new Button({
               text: 'Cancel',
               press: this.closeMethodDialog.bind(this)
             }),
             endButton: new Button({
               text: 'Ok',
               press: this.closeMethodDialog.bind(this, painter, method, menu_obj_id)
             })
         });

         // this.getView().getModel().setProperty("/Method", method);
         //to get access to the global model
         // this.getView().addDependent(this.methodDialog);

         this.methodDialog.addStyleClass("sapUiSizeCompact");

         this.methodDialog.open();
      },

      onFileMenuAction(oEvent) {
         //let oItem = oEvent.getParameter("item"),
         //    sItemPath = "";
         //while (oItem instanceof sap.m.MenuItem) {
         //   sItemPath = oItem.getText() + " > " + sItemPath;
         //   oItem = oItem.getParent();
         //}
         //sItemPath = sItemPath.substr(0, sItemPath.lastIndexOf(" > "));

         let p = this.getCanvasPainter();
         if (!p) return;

         let name = oEvent.getParameter('item').getText();

         switch (name) {
            case "Close canvas":
               this.onCloseCanvasPress();
               break;
            case "Interrupt":
               p.sendWebsocket("INTERRUPT");
               break;
            case "Reload":
               if (typeof p._websocket?.askReload == 'function')
                  p._websocket.askReload();
               break;
            case "Quit ROOT":
               p.sendWebsocket("QUIT");
               break;
            case "Canvas.png":
            case "Canvas.jpeg":
            case "Canvas.svg":
               p.saveCanvasAsFile(name);
               break;
            case "Canvas.root":
            case "Canvas.pdf":
            case "Canvas.ps":
            case "Canvas.C":
               p.sendSaveCommand(name);
               break;
         }

         MessageToast.show(`Action triggered on item: ${name}`);
      },

      onCloseCanvasPress() {
         let p = this.getCanvasPainter();
         if (p) {
            p.onWebsocketClosed();
            p.closeWebsocket(true);
         }
      },

      onInterruptPress() {
         let p = this.getCanvasPainter();
         if (p) p.sendWebsocket("INTERRUPT");
      },

      onQuitRootPress() {
         let p = this.getCanvasPainter();
         if (p) p.sendWebsocket("QUIT");
      },

      onReloadPress() {
         let p = this.getCanvasPainter();
         if (p) p.sendWebsocket("RELOAD");
      },

      isGedEditor() {
         return this.getView().getModel().getProperty("/LeftArea") == "Ged";
      },

      showGed(new_state) {
         return this.showLeftArea(new_state ? "Ged" : "");
      },

      cleanupIfGed() {
         let ged = this.getLeftController("Ged"),
             p = this.getCanvasPainter();
         if (p) p.registerForPadEvents(null);
         if (ged) ged.cleanupGed();
         if (typeof p?.processChanges == 'function')
            p.processChanges('sbits', p);
      },

      getLeftController(name) {
         if (this.getView().getModel().getProperty("/LeftArea") != name) return null;
         let split = this.getView().byId("MainAreaSplitter");
         return split ? split.getContentAreas()[0].getController() : null;
      },

      toggleGedEditor() {
         if (this.isGedEditor())
            this.showLeftArea("");
         else
            this.activateGed(this.getCanvasPainter());
      },

      /** @summary Load custom panel in canvas lef area */
      showLeftArea(panel_name, panel_handle) {
         let split = this.getView().byId('MainAreaSplitter'),
             model = this.getView().getModel(),
             curr = model.getProperty('/LeftArea');

         if (!split || (curr === panel_name))
            return Promise.resolve(null);

         model.setProperty("/LeftArea", panel_name);
         model.setProperty("/GedIcon", chk_icon(panel_name == 'Ged'));

         // first need to remove existing
         if (curr) {
            this.cleanupIfGed();
            split.removeContentArea(split.getContentAreas()[0]);
         }

         if (!panel_name)
            return Promise.resolve(null);

         let canvp = this.getCanvasPainter();

         let viewName = panel_name;

         if (panel_name == "FitPanel")
            viewName = "rootui5.fitpanel.view.FitPanel";
         else if (panel_name.indexOf(".") < 0)
            viewName = "rootui5.canv.view." + panel_name;

         let viewData = canvp.getUi5PanelData(panel_name);
         viewData.masterPanel = this;
         viewData.handle = panel_handle;

         let can_elem = this.getView().byId('MainPanel');

         return XMLView.create({
             viewName,
             viewData,
             layoutData: new SplitterLayoutData({ resizable: true, size: '250px' }),
             height: (panel_name == 'Panel') ? '100%' : undefined
         }).then(oView => {

            // workaround, while CanvasPanel.onBeforeRendering called too late
            can_elem.getController().preserveCanvasContent();

            split.insertContentArea(oView, 0);

            if (panel_name === "Ged") {
               let ged = oView.getController();
               if (ged && (typeof canvp?.registerForPadEvents == 'function')) {
                  canvp.registerForPadEvents(ged.padEventsReceiver.bind(ged));
                  canvp.selectObjectPainter(canvp);
               }
            }

            return oView.getController();
         });
      },

      getBottomController() {
         if (!this.bottomVisible) return null;
         let split = this.getView().byId("MainAreaSplitter"),
             cont = split.getContentAreas(),
             vsplit = cont[cont.length-1],
             vcont = vsplit.getContentAreas(),
             bottom = vcont[vcont.length-1];
         return bottom ? bottom.getController() : null;
      },

      drawInProjectionArea(obj, opt) {
         let cp = this.getCanvasPainter();
         if (typeof cp?.drawObject != 'function')
            return Promise.resolve(null);

         let ctrl = this.getBottomController();
         if (!ctrl) ctrl = this.getLeftController('Panel');

         return ctrl.getRenderPromise().then(dom => {
            dom.style.overflow = "hidden";
            return cp.drawObject(dom, obj, opt);
         }).then(painter => {
            ctrl.setObjectPainter(painter);
            return painter;
         });
      },

      showProjectionArea(kind) {
         let bottom = null;
         return this.showBottomArea(kind == "X")
             .then(area => { bottom = area; return this.showLeftArea(kind == "Y" ? "Panel" : ""); })
             .then(left => {

               let ctrl = bottom || left;

               if (!ctrl || ctrl.getView().getDomRef())
                  return Promise.resolve(!!ctrl);

               return ctrl.getRenderPromise();
            });
      },

      showBottomArea(is_on) {

         if (this.bottomVisible == is_on)
            return Promise.resolve(this.getBottomController());

         let split = this.getView().byId("MainAreaSplitter");
         if (!split) return Promise.resolve(null);

         let cont = split.getContentAreas();

         this.bottomVisible = !this.bottomVisible;

         if (!this.bottomVisible) {
            // vertical splitter exists - toggle it
            let vsplit = cont[cont.length-1],
                main = vsplit.removeContentArea(0);

            vsplit.destroyContentAreas();
            split.removeContentArea(vsplit);
            split.addContentArea(main);
            return Promise.resolve(null);
         }

         // remove panel with normal drawing
         split.removeContentArea(cont[cont.length-1]);

         let vsplit = new Splitter({ orientation: "Vertical" });

         split.addContentArea(vsplit);

         vsplit.addContentArea(cont[cont.length-1]);

         return XMLView.create({
            viewName: 'rootui5.canv.view.Panel',
            layoutData: new SplitterLayoutData({ resizable: true, size: "200px" }),
            height: "100%"
         }).then(oView => {
            vsplit.addContentArea(oView);
            return oView.getController();
         });
      },

      showCanvasStatus(text1, text2, text3, text4) {
         let model = this.getView().getModel();
         model.setProperty("/StatusLbl1", text1);
         model.setProperty("/StatusLbl2", text2);
         model.setProperty("/StatusLbl3", text3);
         model.setProperty("/StatusLbl4", text4);
      },

      isStatusShown() {
         return this._Page.getShowFooter();
      },

      toggleShowStatus(new_state) {
         if ((new_state === undefined) || (new_state == "toggle"))
            new_state = !this.isStatusShown();

         this._Page.setShowFooter(new_state);
         this.getView().getModel().setProperty("/StatusIcon", chk_icon(new_state));

         let canvp = this.getCanvasPainter();
         if (canvp) canvp.processChanges("sbits", canvp);
      },

      toggleToolBar(new_state) {
         if (new_state === undefined) new_state = !this.getView().getModel().getProperty("/ToolbarIcon");

         this._Page.setShowSubHeader(new_state);

         this.getView().getModel().setProperty("/ToolbarIcon", chk_icon(new_state));
      },

      toggleToolTip(new_state) {
         let p = this.getCanvasPainter(true);

         if (new_state === undefined)
            new_state = p ? !p.isTooltipAllowed() : true;

         this.getView().getModel().setProperty("/TooltipIcon", chk_icon(new_state));

         if (p) p.setTooltipAllowed(new_state);
      },

      isMenuBarShow() {
         return this._Page.getShowHeader();
      },

      toggleMenuBar(new_state) {
         if ((new_state === undefined) || (new_state == "toggle"))
            new_state = !this._Page.getShowHeader();
         this.getView().getModel().setProperty("/MenuBarIcon", chk_icon(new_state));
         this._Page.setShowHeader(new_state);
      },

      onDivideDialog() {
         if (!this.oDivideDialog) {
            this.oDivideDialog = new Dialog({
               title: "Divide canvas",
               content: new Input({ placeholder: 'input N or NxM', value: '{/divideArg}' }),
               beginButton: new Button({
                  type: ButtonType.Emphasized,
                  text: "OK",
                  press: () => {
                     let arg = this.getView().getModel().getProperty('/divideArg');
                     this.oDivideDialog.close();
                     let cp = this.getCanvasPainter();
                     if (arg && cp)
                        cp.sendWebsocket('DIVIDE:' + JSON.stringify([(cp.findActivePad() || cp).snapid, arg]));
                  }
               }),
               endButton: new Button({
                  text: "Close",
                  press: () => {
                     this.oDivideDialog.close();
                  }
               })
            });

            // to get access to the controller's model
            this.getView().addDependent(this.oDivideDialog);
         }

         this.oDivideDialog.open();
      },

      onEditMenuAction(oEvent) {
         let cp = this.getCanvasPainter();
         if (!cp) return;

         let name = oEvent.getParameter('item').getText();
         switch (name) {
            case 'Divide':
               this.onDivideDialog();
               break;
            case 'Clear pad':
               cp.sendWebsocket('CLEAR:' + (cp.findActivePad() || cp).snapid);
               break;
            case 'Clear canvas':
               cp.sendWebsocket('CLEAR:' + cp.snapid);
               break;
         }
      },

      onViewMenuAction(oEvent) {
         let item = oEvent.getParameter("item");

         switch (item.getText()) {
            case "Menu": this.toggleMenuBar(); break;
            case "Editor": this.toggleGedEditor(); break;
            case "Event statusbar": this.toggleShowStatus(); break;
            case "Toolbar": this.toggleToolBar(); break;
            case "Tooltip info": this.toggleToolTip(); break;
         }
      },

      onOptionsMenuAction(oEvent) {
         let cp = this.getCanvasPainter();
         if (!cp) return;

         let item = oEvent.getParameter('item');
         if (item.getText() == 'Interrupt')
            cp.sendWebsocket('INTERRUPT');
      },

      onToolsMenuAction(oEvent) {
         let item = oEvent.getParameter("item"),
             name = item.getText();

         if (name != "Fit panel") return;

         let curr = this.getView().getModel().getProperty("/LeftArea");

         this.showLeftArea(curr == "FitPanel" ? "" : "FitPanel");
      },

      showMessage(msg) {
         MessageToast.show(msg);
      },

      /** @summary this function call when section state changed from server side */
      showSection(that, on) {
         switch(that) {
            case "Menu": this.toggleMenuBar(on); break;
            case "StatusBar": this.toggleShowStatus(on); break;
            case "Editor": return this.showGed(on);
            case "ToolBar": this.toggleToolBar(on); break;
            case "ToolTips": this.toggleToolTip(on); break;
         }
         return Promise.resolve(true);
      }
   });

   return CController;

});
