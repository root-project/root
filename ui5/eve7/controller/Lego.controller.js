sap.ui.define([
    'sap/ui/core/mvc/Controller',
    'sap/ui/core/Component',
    "sap/ui/core/ResizeHandler",
    'sap/ui/core/UIComponent',
    'sap/ui/model/json/JSONModel',
    'sap/ui/model/Sorter',
    'sap/m/Column',
    'sap/m/ColumnListItem',
    'sap/m/Input',
    'sap/m/Label',
    'sap/m/Button',
    "sap/m/FormattedText",
    "sap/ui/layout/VerticalLayout",
    "sap/ui/layout/HorizontalLayout",
    "sap/ui/table/Column",
    "sap/m/MessageBox"
], function (Controller, Component, ResizeHandler, UIComponent, JSONModel, Sorter,
    mColumn, mColumnListItem, mInput, mLabel, mButton,
    FormattedText, VerticalLayout, HorizontalLayout, tableColumn, MessageBox) {

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
            this.setupManagerAndViewType(EVE.$eve7tmp.eveViewerId, EVE.$eve7tmp.mgr);
            delete EVE.$eve7tmp;
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
            let element = this.byId("legoX");
            element.setHtmlText("Pointset infected by TCanvas / Lego Stack");

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
        }
    });
});
