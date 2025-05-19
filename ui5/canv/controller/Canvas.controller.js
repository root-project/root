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
   'sap/m/Text',
   'sap/m/Button',
   'sap/m/library',
   'sap/ui/layout/SplitterLayoutData',
   'rootui5/browser/controller/FileDialog.controller'
], function (Controller,
             Component,
             JSONModel,
             XMLView,
             MessageToast,
             Dialog,
             List,
             InputListItem,
             Input,
             Text,
             Button,
             mLibrary,
             SplitterLayoutData,
             FileDialogController) {
   "use strict";

   function chk_icon(flag) {
      return flag ? 'sap-icon://accept' : 'sap-icon://decline';
   }

   return Controller.extend('rootui5.canv.controller.Canvas', {

      onInit() {
         this._Page = this.getView().byId('CanvasMainPage');

         this.bottomVisible = false;

         let model = new JSONModel({ MenuBarIcon: chk_icon(true),
                                     GedIcon: chk_icon(false),
                                     StatusIcon: chk_icon(false),
                                     ToolbarIcon: chk_icon(false),
                                     TooltipIcon: chk_icon(true),
                                     AutoResizeIcon: chk_icon(true),
                                     HighlightPadIcon: chk_icon(false),
                                     CanvasName: 'c1',
                                     StatusLbl1: '', StatusLbl2: '', StatusLbl3: '', StatusLbl4: '',
                                     Standalone: true, isRoot6: true, canResize: true, FixedSize: false,
                                     ErrorVisible: false, ErrorTooltip: 'Error in JavaScript code' });
         this.getView().setModel(model);

         let cp = this.getView().getViewData()?.canvas_painter;

         if (!cp) cp = Component.getOwnerComponentFor(this.getView()).getComponentData().canvas_painter;

         if (cp) {

            if (cp.embed_canvas) model.setProperty('/Standalone', false);

            this.getView().byId('MainPanel').getController().setPainter(cp);

            cp.setFixedCanvasSize = this.setFixedCanvasSize.bind(this);

            cp.executeObjectMethod = this.executeObjectMethod.bind(this);

            // overwriting method of canvas with standalone handling of GED
            cp.activateGed = this.activateGed.bind(this);
            cp.removeGed = this.cleanupIfGed.bind(this);
            cp.hasGed = this.isGedEditor.bind(this);

            cp.onCanvasUpdated = this.onCanvasUpdated.bind(this);
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

            cp.showConsoleError = this.showConsoleError.bind(this);

            if (cp.v7canvas) model.setProperty('/isRoot6', false);

            cp.highlight_gpad = false;

            model.setProperty('/canResize', !cp.embed_canvas && cp.online_canvas);

            let ws = cp.getWebsocket() || cp._window_handle;
            if (!cp.embed_canvas && ws?.addReloadKeyHandler)
               ws.addReloadKeyHandler();
         }
      },

      onAfterRendering() {
      },

      isv7() {
         return this.getCanvasPainter()?.v7canvas;
      },

      onCanvasUpdated() {
         const canv_name = this.getCanvasPainter()?.getCanvasName();
         if (canv_name)
            this.getView().getModel().setProperty('/CanvasName', canv_name);
      },

      executeObjectMethod(painter, method, menu_obj_id) {

         if (method.fArgs!==undefined) {
            this.showMethodsDialog(painter, method, menu_obj_id);
            return true;
         }

         if (method.fName == 'Inspect') {
            painter.showInspector();
            return true;
         }

         if (method.fName == 'FitPanel') {
            const canvp = this.getCanvasPainter();
            if (canvp?.startFitPanel)
               canvp.startFitPanel(true);
            return true;
         }

         if (method.fName == 'Editor') {
            this.activateGed(painter);
            return true;
         }

         return false; // not processed
      },

      /** @summary function used to activate GED in full canvas */
      activateGed(painter /*, kind, mode */) {

         let canvp = this.getCanvasPainter();

         return this.showGed(true).then(() => {
            canvp.selectObjectPainter(painter);

            if (typeof canvp.processChanges == 'function')
               canvp.processChanges('sbits', canvp);

            return true;
         });
      },

      /** @desc Provide canvas painter */
      getCanvasPainter(also_without_websocket) {
         let p = this.getView().byId('MainPanel')?.getController().getPainter();

         return (p && (p.getWebsocket() || also_without_websocket)) ? p : null;
      },

      setFixedCanvasSize(cw, ch, fixed) {
         let ctrl = this.getView().byId('MainPanel')?.getController(),
             is_fixed = ctrl?.setFixedSize(cw, ch, fixed) ?? false;

         this.getView().getModel().setProperty('/FixedSize', is_fixed);
         this.getView().getModel().setProperty('/AutoResizeIcon', chk_icon(!is_fixed));

         return is_fixed;
      },

      closeMethodDialog(painter, method, menu_obj_id) {

         let args = '';

         if (method) {
            let cont = this.methodDialog.getContent();

            let items = cont[0].getItems();

            if (method.fArgs.length !== items.length)
               alert(`Length mismatch between method description ${method.fArgs.length} and args list ${items.length} in dialog`);

            for (let k = 0; k < method.fArgs.length; ++k) {
               let arg = method.fArgs[k],
                   value = items[k].getContent()[0].getValue();

               if (value === '') value = arg.fDefault;

               if ((arg.fTitle=='Option_t*') || (arg.fTitle=='const char*')) {
                  // check quotes,
                  // TODO: need to make more precise checking of escape characters
                  if (!value) value = '""';
                  if (value[0] != '"') value = '"' + value;
                  if (value[value.length-1] != '"') value += '"';
               }

               args += (k > 0 ? ',' : '') + value;
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
                p = menu_obj_id.indexOf('#');

            if (canvp?.v7canvas)
               canvp.submitExec(painter, exec, (p > 0) ? menu_obj_id.slice(p+1) : '');
            else if (canvp)
               canvp.sendWebsocket(`OBJEXEC:${menu_obj_id}:${exec}`);
         }
      },

      showMethodsDialog(painter, method, menu_obj_id) {

         // TODO: deliver class name together with menu items
         method.fClassName = painter.getClassName();
         if ((menu_obj_id.indexOf('#x') > 0) || (menu_obj_id.indexOf('#y') > 0) || (menu_obj_id.indexOf('#z') > 0))
            method.fClassName = 'TAxis';

         let items = [];

         for (let n = 0; n < method.fArgs.length; ++n) {
            let arg = method.fArgs[n];
            arg.fValue = arg.fDefault;
            if (arg.fValue == '""') arg.fValue = '';
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

         // this.getView().getModel().setProperty('/Method', method);
         //to get access to the global model
         // this.getView().addDependent(this.methodDialog);

         this.methodDialog.addStyleClass('sapUiSizeCompact');

         this.methodDialog.open();
      },

      onFileMenuAction(oEvent) {
         //let oItem = oEvent.getParameter('item'),
         //    sItemPath = '';
         //while (oItem instanceof sap.m.MenuItem) {
         //   sItemPath = oItem.getText() + ' > ' + sItemPath;
         //   oItem = oItem.getParent();
         //}
         //sItemPath = sItemPath.substr(0, sItemPath.lastIndexOf(' > '));

         let p = this.getCanvasPainter();
         if (!p) return;

         let name = oEvent.getParameter('item').getText(),
             canname = this.getView().getModel().getProperty('/CanvasName');

         switch (name) {
            case 'Close canvas':
               this.onCloseCanvasPress();
               break;
            case 'Interrupt':
               p.sendWebsocket('INTERRUPT');
               break;
            case 'Reload':
               if (typeof p.getWebsocket()?.askReload == 'function')
                  p.getWebsocket().askReload();
               break;
            case 'Quit ROOT':
               p.sendWebsocket('QUIT');
               break;
            case canname + '.png':
            case canname + '.jpeg':
            case canname + '.svg':
            case canname + '.webp':
            case canname + '.pdf':
                  p.saveCanvasAsFile(name);
               break;
            case canname + '.root':
            case canname + '.C':
               p.sendSaveCommand(name);
               break;
            case 'Save as ...': {
               let filters = ['Png files (*.png)', 'Jpeg files (*.jpeg)', 'SVG files (*.svg)', 'PDF files (*.pdf)', 'WEBP files (*.webp)', 'ROOT files (*.root)' ];
               if (!p?.v7canvas)
                  filters.push('C++ (*.cxx *.cpp *.c)');

               FileDialogController.SaveAs({
                  websocket: p.getWebsocket(),
                  filename: canname + '.png',
                  title: 'Select file name to save canvas',
                  filter: 'Png files',
                  filters,
                  // working_path: '/Home',
                  onOk: fname => {
                     if (fname.endsWith('.png') || fname.endsWith('.jpeg') || fname.endsWith('.svg') || fname.endsWith('.pdf') || fname.endsWith('.webp'))
                         p.saveCanvasAsFile(fname);
                     else
                         p.sendSaveCommand(fname);
                  },
                  onCancel: () => {},
                  onFailure: () => {}
               });

               break;
           }
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
         this.getCanvasPainter()?.sendWebsocket('INTERRUPT');
      },

      onQuitRootPress() {
         this.getCanvasPainter()?.sendWebsocket('QUIT');
      },

      onReloadPress() {
         this.getCanvasPainter()?.sendWebsocket('RELOAD');
      },

      isGedEditor() {
         return this.getView().getModel().getProperty('/LeftArea') == 'Ged';
      },

      showGed(new_state) {
         return this.showLeftArea(new_state ? 'Ged' : '');
      },

      cleanupIfGed() {
         let ged = this.getLeftController('Ged'),
             p = this.getCanvasPainter();
         p?.registerForPadEvents(null);
         ged?.cleanupGed();
         if (typeof p?.processChanges == 'function')
            p.processChanges('sbits', p);
      },

      getLeftController(name) {
         if (this.getView().getModel().getProperty('/LeftArea') != name)
            return null;
         let split = this.getView().byId('MainAreaSplitter'),
             cont = split ? split.getContentAreas() : [];
         return cont && cont[0] && cont[0].getController ? cont[0].getController() : null;
      },

      toggleGedEditor() {
         if (this.isGedEditor())
            this.showLeftArea('');
         else
            this.activateGed(this.getCanvasPainter());
      },

      /** @summary Load custom panel in canvas left area */
      showLeftArea(panel_name, panel_handle) {
         let split = this.getView().byId('MainAreaSplitter'),
             model = this.getView().getModel(),
             curr = model.getProperty('/LeftArea'),
             is_fit = (panel_name === 'FitPanel'),
             is_ged = (panel_name === 'Ged'),
             is_flex = is_fit || is_ged ? 1 : 0,
             was_fit = (curr === 'FitPanel'),
             was_ged = (curr === 'Ged'),
             was_flex = was_fit || was_ged ? 1 : 0,
             canvp = this.getCanvasPainter(),
             can_elem = this.getView().byId('MainPanel');

         if (is_fit && !canvp?.startFitPanel) {
            panel_name = '';
            is_fit = false;
         }

         if (!split || (!curr && !panel_name) || (curr === panel_name))
            return Promise.resolve(null);

         const adjust_window_width = (is_flex ^ was_flex) && !canvp?._ignore_section_resize;

         if (adjust_window_width)
            can_elem?.getController().rememberAreaSize();

         model.setProperty('/LeftArea', panel_name);
         model.setProperty('/GedIcon', chk_icon(is_ged));

         // first need to remove existing
         if (curr) {
            this.cleanupIfGed();
            let area = split.getContentAreas()[0];
            if (adjust_window_width && !panel_name)
               can_elem?.getController().resizeBrowser(-1*(area.$().width() + 16), 0);
            split.removeContentArea(area);
         }

         if (!panel_name)
            return Promise.resolve(null);

         let viewName = panel_name;

         if (is_fit) {
            viewName = 'rootui5.fitpanel.view.FitPanel';
            // send panel command and return new channel
            panel_handle = canvp.startFitPanel();
         } else if (panel_name.indexOf('.') < 0)
            viewName = 'rootui5.canv.view.' + panel_name;

         let panel_width = Math.max(250, Math.round(0.25 * this.getView().$().width()));

         if (adjust_window_width)
            can_elem?.getController().resizeBrowser(panel_width + 16, 0);

         let viewData = canvp.getUi5PanelData(panel_name);
         viewData.masterPanel = this;
         viewData.handle = panel_handle;

         return XMLView.create({
             viewName,
             viewData,
             layoutData: new SplitterLayoutData({ resizable: true, size: panel_width + 'px' }),
             height: (panel_name == 'Panel') ? '100%' : undefined
         }).then(oView => {
            // workaround, while CanvasPanel.onBeforeRendering called too late
            can_elem.getController().preserveCanvasContent();

            split.insertContentArea(oView, 0);

            if (is_ged) {
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
         let split = this.getView().byId('BottomAreaSplitter'),
             cont = split.getContentAreas(),
             bottom = cont[cont.length-1];
         return bottom?.getController();
      },

      drawInProjectionArea(obj, opt, kind) {
         let cp = this.getCanvasPainter(),
             ctrl = (kind == 'X') ? this.getBottomController() : this.getLeftController('Panel');

         if (!ctrl || (typeof cp?.drawObject != 'function'))
            return Promise.resolve(null);

         if (typeof ctrl?.cleanupPainter == 'function')
            ctrl.cleanupPainter();

         return ctrl.getRenderPromise().then(dom => {
            dom.innerHTML = ''; // delete everything
            dom.style.overflow = 'hidden';
            return cp.drawObject(dom, obj, opt);
         }).then(painter => {
            ctrl.setObjectPainter(painter);
            return painter;
         });
      },

      showProjectionArea(kind) {
         let bottom = null, is_xy = kind == 'XY';
         return this.showBottomArea((kind == 'X') || is_xy, is_xy)
             .then(area => { bottom = area; return this.showLeftArea((kind == 'Y') || is_xy ? 'Panel' : ''); })
             .then(left => {

               let ctrl = bottom || left;

               if (!ctrl || ctrl.getView().getDomRef())
                  return Promise.resolve(!!ctrl);

               return ctrl.getRenderPromise();
            });
      },

      handleBottomResize(evnt) {
         let sz = evnt.getParameters().newSizes;
         if (!sz) return;

         let ctrl = this.getLeftController('Panel');
         if (!ctrl) return;

         let fullHeight = this.getView().$().height();
         if (fullHeight && sz[0]) {
            // ctrl.getView().setHeight(Math.round(sz[0]/fullHeight) + '%');
            ctrl.getView().$().height(sz[0] + 'px');
            if (typeof ctrl.invokeResizeTimeout == 'function')
               ctrl.invokeResizeTimeout(10);
         }
      },

      showBottomArea(is_on, with_handler) {

         if (this.bottomVisible == is_on)
            return Promise.resolve(this.getBottomController());

         let split = this.getView().byId('BottomAreaSplitter');
         if (!split) return Promise.resolve(null);

         let cont = split.getContentAreas();

         this.bottomVisible = !this.bottomVisible;

         if (!this.bottomVisible) {
            // just remove bottom controller
            split.removeContentArea(cont.length-1);
            return Promise.resolve(null);
         }

         let h = this.getView().$().height();

         return XMLView.create({
            viewData: {},
            viewName: 'rootui5.canv.view.Panel',
            layoutData: new SplitterLayoutData({ resizable: true, size: Math.round(h*0.25) + 'px'}),
            height: '100%'
         }).then(oView => {
            split.addContentArea(oView);
            if (with_handler)
               split.attachResize(null, this.handleBottomResize, this);
            return oView.getController();
         });
      },

      showCanvasStatus(text1, text2, text3, text4) {
         let model = this.getView().getModel();
         model.setProperty('/StatusLbl1', text1);
         model.setProperty('/StatusLbl2', text2);
         model.setProperty('/StatusLbl3', text3);
         model.setProperty('/StatusLbl4', text4);
      },

      isStatusShown() {
         return this._Page.getShowFooter();
      },

      toggleShowStatus(new_state) {
         if ((new_state === undefined) || (new_state == 'toggle'))
            new_state = !this.isStatusShown();

         this.getView().getModel().setProperty('/StatusIcon', chk_icon(new_state));

         if (this.isStatusShown() != new_state) {
            const canvp = this.getCanvasPainter();
            // restore canvas size after getting next resize event
            if (!canvp?._ignore_section_resize)
               this.getView().byId('MainPanel')?.getController().rememberAreaSize();
            this._Page.setShowFooter(new_state);
            canvp?.processChanges('sbits', canvp);
         }
      },

      isToolBarShown() {
         return this._Page.getShowSubHeader();
      },

      toggleToolBar(new_state) {
         const old_state = this.isToolBarShown();
         if (new_state === undefined)
            new_state = !old_state;

         if (new_state !== old_state) {
            // restore canvas size after getting next resize event
            if (!this.getCanvasPainter()?._ignore_section_resize)
               this.getView().byId('MainPanel')?.getController().rememberAreaSize();
            this._Page.setShowSubHeader(new_state);
            this.getView().getModel().setProperty('/ToolbarIcon', chk_icon(new_state));
         }
      },

      toggleToolTip(new_state) {
         let p = this.getCanvasPainter(true);

         if (new_state === undefined)
            new_state = p ? !p.isTooltipAllowed() : true;

         this.getView().getModel().setProperty('/TooltipIcon', chk_icon(new_state));

         if (p) {
            p.setTooltipAllowed(new_state);
            p.processChanges('sbits', p);
         }
      },

      isMenuBarShow() {
         return this._Page.getShowHeader();
      },

      toggleMenuBar(new_state) {
         if ((new_state === undefined) || (new_state == 'toggle'))
            new_state = !this._Page.getShowHeader();
         this.getView().getModel().setProperty('/MenuBarIcon', chk_icon(new_state));

         if (this.isMenuBarShow() != new_state)
            this._Page.setShowHeader(new_state);
      },

      onDivideDialog() {
         if (!this.oDivideDialog) {
            this.oDivideDialog = new Dialog({
               title: 'Divide canvas',
               content: new Input({ placeholder: 'input N or NxM', value: '{/divideArg}' }),
               beginButton: new Button({
                  type: mLibrary.ButtonType.Emphasized,
                  text: 'OK',
                  press: () => {
                     let arg = this.getView().getModel().getProperty('/divideArg');
                     this.oDivideDialog.close();
                     let cp = this.getCanvasPainter();
                     if (arg && cp)
                        cp.sendWebsocket('DIVIDE:' + JSON.stringify([(cp.findActivePad() || cp).getSnapId(), arg]));
                  }
               }),
               endButton: new Button({
                  text: 'Close',
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
               cp.sendWebsocket('CLEAR:' + (cp.findActivePad() || cp).getSnapId());
               break;
            case 'Clear canvas':
               cp.sendWebsocket('CLEAR:' + cp.getSnapId());
               break;
            default:
               let sz = name.split('x');
               if (cp.resizeBrowser && (sz?.length == 2))
                  cp.resizeBrowser(Number.parseInt(sz[0]), Number.parseInt(sz[1]));
               break;
         }
      },

      onViewMenuAction(oEvent) {
         let item = oEvent.getParameter('item');

         switch (item.getText()) {
            case 'Menu': this.toggleMenuBar(); break;
            case 'Editor': this.toggleGedEditor(); break;
            case 'Event statusbar': this.toggleShowStatus(); break;
            case 'Toolbar': this.toggleToolBar(); break;
            case 'Tooltip info': this.toggleToolTip(); break;
         }
      },

      onOptionsMenuAction(oEvent) {
         let cp = this.getCanvasPainter();
         if (!cp) return;

         let item = oEvent.getParameter('item');
         if (item.getText() == 'Interrupt') {
            cp.sendWebsocket('INTERRUPT');
         } else if (item.getText() == 'Auto resize') {
             let was_fixed = this.getView().getModel().getProperty('/FixedSize'),
                 cp = this.getCanvasPainter(), w = 0, h = 0;
             if (!was_fixed) {
                w = cp?.getPadWidth();
                h = cp?.getPadHeight();
             }
             let is_fixed = this.setFixedCanvasSize(w, h, !was_fixed);
             if ((is_fixed != was_fixed) && cp) {
                console.log('Changed fix state - inform server!!!');
                cp._online_fixed_size = is_fixed;
                cp.sendResized(true);
             }
         } else if (item.getText() == 'Highlight gPad') {
            cp = this.getCanvasPainter();
            if (cp) {
               cp.highlight_gpad = !cp.highlight_gpad;
               this.getView().getModel().setProperty('/HighlightPadIcon', chk_icon(cp.highlight_gpad));
               cp.redraw();
            }
         }
      },

      onToolsMenuAction(oEvent) {
         let item = oEvent.getParameter('item'),
             name = item.getText();

         if (name == 'Fit panel') {
            let curr = this.getView().getModel().getProperty('/LeftArea');
            this.showLeftArea(curr == 'FitPanel' ? '' : 'FitPanel');
         } else if (name == 'Start browser') {
            this.getCanvasPainter()?.sendWebsocket('START_BROWSER');
         }
      },

      onHelpMenuAction(oEvent) {
         const item = oEvent.getParameter('item'),
               name = item.getText();
         let url = '';
         if (name === 'ROOT')
            url = 'https://root.cern';
         else if (name === 'Canvas')
            url = this.isv7() ? 'https://root.cern/doc/master/classROOT_1_1Experimental_1_1RCanvas.html' : 'https://root.cern/doc/master/classTCanvas.html';
         else if (name === 'Tutorials')
            url = this.isv7() ? 'https://root.cern/doc/master/group__tutorial__rcanvas.html' : 'https://root.cern/doc/master/group__tutorial__graphics.html';
         else if (name === 'About')
            url = 'https://root.cern/about/';
         if (url)
            this.getCanvasPainter()?.sendWebsocket('SHOWURL:' + url);
      },

      showConsoleError(err) {
         this.getView().getModel().setProperty('/ErrorVisible', true);
         this.err_message = err?.message ?? '';
         this.getView().getModel().setProperty('/ErrorTooltip', 'Err: ' + (this.err_message || 'unknown'));
         this.err_stack = err?.stack ?? err ?? 'Abstract stack';
      },

      onShowConsoleErrors() {
         const errDialog = new Dialog({
            title: 'JavaScript Error: ' + this.err_message,
            type: 'Message',
            state: 'Warning',
            content: new Text({ text: this.err_stack }),
            endButton: new Button({
              text: 'Ok',
              press: () => { errDialog.close(); errDialog.destroy(); }
            })
         });

         errDialog.open();
      },

      showMessage(msg) {
         MessageToast.show(msg);
      },

      /** @summary this function call when section state changed from server side */
      showSection(that, on) {
         switch(that) {
            case 'Menu': this.toggleMenuBar(on); break;
            case 'StatusBar': this.toggleShowStatus(on); break;
            case 'Editor': return this.showGed(on);
            case 'ToolBar': this.toggleToolBar(on); break;
            case 'ToolTips': this.toggleToolTip(on); break;
         }
         return Promise.resolve(true);
      }
   });

});
