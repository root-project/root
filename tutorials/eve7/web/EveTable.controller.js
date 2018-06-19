sap.ui.define([
    'sap/ui/core/mvc/Controller',
    'sap/ui/model/json/JSONModel',
        'sap/ui/commons/CheckBox',
    'sap/ui/commons/CheckBox',
	'sap/ui/table/Column',
    "sap/ui/core/ResizeHandler"
], function (Controller, JSONModel, ResizeHandler) {
   
    "use strict";
    
    return Controller.extend("eve.EveTable", {
       
        onInit : function() {

           var data = this.getView().getViewData();
           console.log("VIEW DATA", data);
           
           var id = this.getView().getId();
           console.log("eve.GL.onInit id = ", id );

           
           this._load_scripts = true;
           this._render_html = false;

           console.log("TABLE VIEW CREATED");
            
           this.mgr = data.mgr;
           this.elementid = data.elementid;
           this.kind = data.kind;
         
            var element = this.mgr.GetElement(this.elementid);
            // loop over scene and add dependency
            for (var k=0;k<element.childs.length;++k) {
               var scene = element.childs[k];
               console.log("FOUND scene", scene.fSceneId);
               
               this.mgr.Register(scene.fSceneId, this, "onElementChanged")
            }
            
            /*
           
           var oData = [{
              width: "auto",
              header: "Product Name",
              demandPopin: false,
              minScreenWidth: "",
              styleClass: "cellBorderLeft cellBorderRight"
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

           */
        },
        setEveData: function() {
            var mgr = this.mgr;
            var element = mgr.GetElement(this.elementid);
            for (var k=0;k<element.childs.length;++k) {
                var sceneInfo = element.childs[k];
                var abc = mgr.GetElement(sceneInfo.fSceneId);
                for (var c = 0; c < abc.childs.length; ++c) {
                    this.setupTable(abc.childs[c]);
                    return;
                }
            }
        },
        setupTable: function(eveData) {
            console.log("set table ", eveData );

            var oTable = this.getView().byId("table");
            console.log(oTable);
            
            var columnData = [];
            columnData.push({columnName:"Name"});
            columnData.push({columnName:"Visible"});
           
            for (var i = 0; i < eveData.childs.length; i++)
            {                
                columnData.push({columnName:eveData.childs[i].fName});
            }

            var rowData = eveData.body;

            var collection = this.mgr.GetElement(eveData.fCollectionId);

            
            for (var i = 0; i < collection.childs.length; i++)
            {
                rowData[i].Name =  collection.childs[i].fName;
                console.log("rnr selg ",  collection.childs[i].fFiltered );
                rowData[i].Visible =  collection.childs[i].fFiltered === true ? "*" : "";
            }
            
            console.log("collection ",collection );
            console.log("rowData ", rowData );
            
            console.log("columnData", columnData);
            var oModel = new sap.ui.model.json.JSONModel();
	    oModel.setData({
		rows: rowData,
		columns: columnData
	    });
	    oTable.setModel(oModel);

            

            oTable.bindColumns("/columns", function(sId, oContext) {
		var columnName = oContext.getObject().columnName;
		return new sap.ui.table.Column({
		    label: columnName,
		    template: columnName,
                    sortProperty: columnName
		});
	    });
            
	    oTable.bindRows("/rows");
            
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
          
          this.setEveData();
       },
       
        UpdateMgr : function(mgr) {
            var elem = mgr.map[this.elementid];
            var scene = mgr.map[ elem.fMotherId];
            console.log("Table update mgr", this, elem);
            console.log("Table ", scene);
            this.mgr = mgr;
        },
        
       onAfterRendering: function() {
          this._render_html = true;
          
          // this.getView().$().css("overflow", "hidden");
          
          // this.getView().$().parent().css("overflow", "hidden");
          
          // only when rendering completed - register for modify events
          var element = this.mgr.GetElement(this.elementid);
          
          this.checkScences();
       },
       
       checkScences: function() {
       }

   });
    
});
