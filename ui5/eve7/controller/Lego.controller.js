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
            let data = this.getView().getViewData();
            this.mgr       = data.mgr;
            this.eveViewerId = data.eveViewerId;

            let eviewer = this.mgr.GetElement(this.eveViewerId);
            let sceneInfo = eviewer.childs[0];
            let sceneId = sceneInfo.fSceneId;
            this.mgr.RegisterController(this);
            this.mgr.RegisterSceneReceiver(sceneId, this);

            let scene = this.mgr.GetElement(sceneId);

            let chld = scene.childs[0];
            let element = this.byId("legoX");
            element.setHtmlText("Pointset infected by TCanvas / Lego Stack");

            ResizeHandler.register(this.getView(), this.onResize.bind(this));

            this.eve_lego = chld;
            this.canvas_json = JSROOT.parse( atob(chld.fTitle) );
            // console.log(JSON.stringify(this.canvas_json));
        },

        onResize()
        {
            let domref = this.byId("legoPlotPlace").getDomRef();

            if ( ! this.jst_ptr)
            {
                this.jst_ptr = 1;
                JSROOT.draw(domref, this.canvas_json);
            }
            else
            {
                // if completely different, call JSROOT.cleanup(dom) first.
                JSROOT.redraw(domref, this.canvas_json);
            }
        },

        onSceneCreate: function (element, id) {
            //console.log("LEGO onSceneCreate", id);
        },

        sceneElementChange: function (el) {
            //console.log("LEGO element changed");
        },

        endChanges: function (oEvent) {
            let domref = this.byId("legoPlotPlace").getDomRef();
            this.canvas_json = JSROOT.parse( atob(this.eve_lego.fTitle) );
            JSROOT.redraw(domref, this.canvas_json);
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
