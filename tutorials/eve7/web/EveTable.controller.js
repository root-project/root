sap.ui.define([
    'sap/ui/core/mvc/Controller',
    'sap/ui/model/json/JSONModel',
    "sap/ui/core/ResizeHandler"
], function (Controller, JSONModel, ResizeHandler) {
   
    "use strict";
    
    return Controller.extend("eve.EveTable", {
       
       onInit : function() {
           var id = this.getView().getId();
           console.log("eve.GL.onInit id = ", id );

           var data = this.getView().getViewData();
           console.log("VIEW DATA", data);
           
           this.mgr = data.mgr;
           this.elementid = data.elementid;
           this.kind = data.kind;
           
           this._load_scripts = true;
           this._render_html = false;

           console.log("TABLE VIEW CREATED");
           
           var oData = [{
              width: "auto",
              header: "Product Name",
              demandPopin: false,
              minScreenWidth: "",
              styleClass: "cellBorderLeft cellBorderRight"
           }, {
              width: "20%",
              header: "Supplier Name",
              demandPopin: false,
              minScreenWidth: "",
              styleClass: "cellBorderRight"
           }, {
              width: "50%",
              header: "Description",
              demandPopin: true,
              minScreenWidth: "Tablet",
              styleClass: "cellBorderRight"
           }];
        
           var oData2 = [ {
              Name: "abc1",
              SupplierName: "abc1 title",
              Description: "abc1 description"
           }, {
              Name: "abc1",
              SupplierName: "abc1 title",
              Description: "abc1 description",
              highlight: "Information"
           }, {
              Name: "abc2",
              SupplierName: "abc2 title",
              Description: "abc2 description"
           }];
        
           this.oColumnModel = new JSONModel();
           this.oColumnModel.setData(oData);
           this.getView().setModel(this.oColumnModel, "columns");

           this.oProductsModel = new JSONModel();
           this.oProductsModel.setData(oData2);
           this.getView().setModel(this.oProductsModel, "products");

           // JSROOT.AssertPrerequisites("geom;user:evedir/EveElements.js", this.onLoadScripts.bind(this));
       },
       
       onLoadScripts: function() {
          this._load_scripts = true;
          this.checkScences();
       },

       // function called from GuiPanelController
       onExit : function() {
          if (this.mgr) this.mgr.Unregister(this);
       },
       
       onElementChanged: function(id, element) {
          console.log("!!!CHANGED", id);
          
          this.checkScences();
       },
       
       onAfterRendering: function() {
          this._render_html = true;
          
          // this.getView().$().css("overflow", "hidden");
          
          // only when rendering completed - register for modify events
          var element = this.mgr.GetElement(this.elementid);
          
          this.checkScences();
       },
       
       checkScences: function() {
       }

   });
    
});