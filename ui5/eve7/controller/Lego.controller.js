sap.ui.define([
    'sap/ui/core/mvc/Controller',
    "sap/ui/core/ResizeHandler",
    'sap/ui/core/UIComponent',
    'rootui5/eve7/lib/EveManager'
], function (Controller, ResizeHandler, UIComponent, EveManager) {

    "use strict";

    return Controller.extend("rootui5.eve7.controller.Lego", {

        onInit: function () {
            // disable narrowing axis range
            EVE.JSR.settings.Zooming = false;

            let data = this.getView().getViewData();
            if (data) {
                this.setupManagerAndViewType(data.eveViewerId, data.mgr);
            }
            else {
                UIComponent.getRouterFor(this).getRoute("Lego").attachPatternMatched(this.onViewObjectMatched, this);
            }

            ResizeHandler.register(this.getView(), this.onResize.bind(this));
        },

        onViewObjectMatched: function (oEvent) {
            let args = oEvent.getParameter("arguments");
            // Main's switchSingle() passes the manager in EVE.$eve7tmp. Without it
            // (browser forward, or this address opened directly) take it from the
            // main page; if there is none yet, go there: this view has no
            // standalone mode.
            let moredata = EVE.$eve7tmp || EveManager.FindMainPageViewer(this.getView(), args.viewName);
            delete EVE.$eve7tmp;
            if (!moredata) {
                UIComponent.getRouterFor(this).navTo("Default", {}, true);
                return;
            }
            this.setupManagerAndViewType(moredata.eveViewerId, moredata.mgr);
        },

        setupManagerAndViewType: function (eveViewerId, mgr) {
            this.eveViewerId = eveViewerId;
            this.mgr       = mgr;

            let eviewer = this.mgr.GetElement(this.eveViewerId);
            let sceneInfo = eviewer.childs[0];
            let sceneId = sceneInfo.fSceneId;
            this.mgr.RegisterController(this);
            this.mgr.RegisterSceneReceiver(sceneId, this);

            let scene = this.mgr.GetElement(sceneId);
            let chld = scene.childs[0];

            this.eve_lego = chld;
            this.canvas_json = EVE.JSR.parse(atob(chld.fTitle));
        },
        onResize : function() {
            // use timeout
            if (this.resize_tmout) clearTimeout(this.resize_tmout);
            if (this.canvas_painter) this.canvas_painter.CheckCanvasResize();
            this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), 250); // minimal latency
        },
        drawHist: function () {
            if (!this.controllerIsMapped()) return;

            let domref = this.byId("legoPlotPlace").getDomRef();
            if (!this.jst_ptr) {
                this.jst_ptr = 1;
                EVE.JSR.draw(domref, this.canvas_json).then(hist_painter => {
                    this.canv_painter = hist_painter.getCanvPainter();
                });
            }
            else {
                // if completely different, called cleanup(dom) first.
                EVE.JSR.redraw(domref, this.canvas_json).then(hist_painter => {
                    this.canv_painter = hist_painter.getCanvPainter();
                });
            }
        },
        controllerIsMapped() {
            // a router page that was redirected to the main page has no manager
            if (!this.mgr) return false;
            let ev = this.mgr.GetElement(this.eveViewerId);
            return ev.fRnrSelf;
        },
        onResizeTimeout: function () {
            this.drawHist(); // AMT when to draw hist ??
            delete this.resize_tmout;
        },
        onSceneCreate: function (element, id) {
            // console.log("LEGO onSceneCreate", id);
        },

        sceneElementChange: function (el) {
            //console.log("LEGO element changed");
        },

        endChanges: function (oEvent) {
            if (!this.controllerIsMapped()) return;

            let domref = this.byId("legoPlotPlace").getDomRef();
            this.canvas_json = EVE.JSR.parse( atob(this.eve_lego.fTitle) );
            EVE.JSR.redraw(domref, this.canvas_json);
        },

        elementRemoved: function (elId) {
        },

        SelectElement: function (selection_obj, element_id, sec_idcs) {
           // console.log("LEGO element selected", element_id);
        },

        UnselectElement: function (selection_obj, element_id) {
        },
        switchSingle: function () {
            let oRouter = UIComponent.getRouterFor(this);
            EVE.$eve7tmp = { mgr: this.mgr, eveViewerId: this.eveViewerId };

            oRouter.navTo("Lego", { viewName: this.mgr.GetElement(this.eveViewerId).fName });
        },

        swap: function () {
            this.mgr.controllers[0].switchViewSides(this.mgr.GetElement(this.eveViewerId));
        },

        detachViewer: function () {
            this.mgr.controllers[0].removeView(this.mgr.GetElement(this.eveViewerId));
            this.destroy();
        }
    });
});
