sap.ui.define([
    'sap/ui/core/mvc/Controller',
    'sap/ui/core/Component',
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
], function (Controller, Component, UIComponent, JSONModel, Sorter,
    mColumn, mColumnListItem, mInput, mLabel, mButton,
    FormattedText, VerticalLayout, HorizontalLayout, tableColumn, MessageBox) {

    "use strict";

    return Controller.extend("rootui5.eve7.controller.Lego", {

        onInit: function () {
            let data = this.getView().getViewData();

            console.log("LEGO onInit ", data);

            this.mgr       = data.mgr;
            this.eveViewerId = data.eveViewerId;

            let eviewer = this.mgr.GetElement(this.eveViewerId);
            console.log("viewer", eviewer);
            let sceneInfo = eviewer.childs[0];
            let sceneId = sceneInfo.fSceneId;
            
            this.mgr.RegisterController(this);
            this.mgr.RegisterSceneReceiver(sceneId, this);

            let scene = this.mgr.GetElement(sceneId);
            console.log("Lego scene 2", scene);

            let dump = JSON.stringify(scene.childs[0]);
            var element = this.byId("legoX");
            element.setHtmlText(dump);

        },



        onSceneCreate: function (element, id) {
            console.log("LEGO onSceneCreate", id);
        },

        sceneElementChange: function (el) {
            console.log("LEGO element changed");
        },

        endChanges: function (oEvent) {
        },

        elementRemoved: function (elId) {
        },

        SelectElement: function (selection_obj, element_id, sec_idcs) {
            console.log("LEGO element selected", element_id);
        },


        UnselectElement: function (selection_obj, element_id) {
        }
    });
});
