sap.m.StandardTreeItem.extend('MySuperDuperTreeItem', {
   metadata: {
      properties: {
	 myStuff: 'string',
	 status: 'string'
      }
   },
   onAfterRendering: function() {
      return;
      if (sap.m.StandardTreeItem.prototype.onAfterRendering) {
	 sap.m.StandardTreeItem.prototype.onAfterRendering.apply(this, arguments);
      }


      var oi = this.getMetadata();
      console.log("superduper metadata ", oi);
      console.log("superduper this ", this);
      console.log("superduper this query ", this.$());

      //   this.$().css({ "font-size": "0.95rem"});
      //    $(".sapMTreeItemBase").css({ "font-size": "0.65rem"});

      this.$().removeClass("sapMTreeItemBase");
      this.$().addClass("eveTreeItem");
   },
   renderer:{}
});


sap.ui.define([
   'sap/ui/core/mvc/Controller',
   "sap/ui/model/json/JSONModel",
   "sap/m/Button",
   "sap/m/ButtonRenderer",
   "sap/m/ColorPalettePopover",
], function(Controller, JSONModel, Button, ButtonRenderer, ColorPalettePopover) {
   "use strict";
   
   var currentColor = "rgb(100, 0, 0)"
  // var currentColorId =;

       
   var EVEColorButton = Button.extend("sap.ui.jsroot.EVEColorButton", {
      renderer: ButtonRenderer.render,
      
      init: function() {
         // svg images are always loaded without @2
         this.addEventDelegate({
            onAfterRendering: function() {
               this._setColor();
               //$("span").children().css('color', currentColor);
            }
         }, this);
      }

   });

   EVEColorButton.prototype._setColor = function() {
      this.$().children().css('background-color', this.data("attrcolor"));
   }

   
   return Controller.extend("eve.Summary", {


        

      onInit: function () {
         /*
$.getScript("jsrootsys/openui5/ColorButton.js", function() {
//   alert("Script loaded but not necessarily executed.");
});*/


         var data = [{ fName: "Event" }];

         var oTree = this.getView().byId("tree");
         oTree.setMode(sap.m.ListMode.Single);
         oTree.setIncludeItemInSelection(true);

         if (false) {
            var oModel = new sap.ui.model.json.JSONModel();
            oModel.setData([]);
            oModel.setSizeLimit(10000);
            this.getView().setModel(oModel, "treeModel");

         } else {
            // old code, keep for history

            var oModel = new sap.ui.model.json.JSONModel();
            oModel.setData([]);
            oModel.setSizeLimit(10000);
            this.getView().setModel(oModel, "treeModel");

            var oStandardTreeItemTemplate = new MySuperDuperTreeItem({
               title: "{treeModel>fName}",
               visible: "{treeModel>fVisible}",
               type: "{treeModel>fType}",
               // highlight: "{treeModel>fHighlight}"
            });
            oStandardTreeItemTemplate.attachDetailPress({}, this.onDetailPress, this);
            oStandardTreeItemTemplate.attachBrowserEvent("mouseenter", this.onMouseEnter, this);
            oStandardTreeItemTemplate.attachBrowserEvent("mouseleave", this.onMouseLeave, this);
            /*
              var oDataTemplate = new sap.ui.core.CustomData({
              key:"eveElement"
              });
              oDataTemplate.bindProperty("value", "answer");
            */
            oTree.bindItems("treeModel>/", oStandardTreeItemTemplate);
         }

         this.oModelGED = new JSONModel({ "widgetlist" : []});
         sap.ui.getCore().setModel(this.oModelGED, "ged");
         

         this.oGuiClassDef = {
            "TEveElement" : [{
               name : "RnrSelf",
               _type   : "Bool"
            }, {
               name : "RnrChildren",
               _type   : "Bool"
            }, {
               name : "MainColor",
               member: "fMainColor",
               srv  : "SetMainColorRGB",
               _type   : "Color"
            }],
            "TEveElementList" : [ {sub: ["TEveElement"]}],
            "TEvePointSet" : [
            {
                  sub: ["TEveElement"]
            }, {
               name : "MarkerSize",
               _type   : "Number"
            }],
            "TEveJetCone" : [{sub: ["TEveElement"]},{
               name : "NDiv",
               _type   : "Number"
            }],
            "TEveTrack" : [ {sub: ["TEveElement"]},{
               name : "LineWidth",
               _type   : "Number"
            }],
            "TEveDataCollection" : [{
               name : "Filter",
               _type   : "String"
            }],
            "TEveDataItem" : [{
               name : "Filtered",
               _type   : "Bool"
            }]
         };

      },

      UpdateMgr : function(mgr) {

         var model = this.getView().getModel("treeModel");
         model.setData(mgr.CreateSummaryModel());
         model.refresh(true);

         this.mgr = mgr;

         console.log("!!!!! CALL REGISTER", this);
         this.mgr.RegisterHighlight(this, "onElementHighlight1");


         var oTree = this.getView().byId("tree");
         oTree.expandToLevel(4);
         // console.log('Update summary model');

         // console.log('Update summary model');
      },

      addNodesToTreeItemModel: function(el, model) {
         console.log("FILL el ", el.fName)
         model.fName = el.fName;
         model.guid = el.guid;
         if (el.arr) {
            model.arr = new Array(el.arr.length);
            for (var n=0; n< el.arr.length; ++n) {

               //  console.log("child  ", el.arr[n]);
               model.arr[n]= {"fName" : "unset"};

               this.addNodesToTreeItemModel(el.arr[n], model.arr[n]);
            }
         }

         /*
           for (var n=0; n< lst.arr.length; ++n)
           {
           var el = lst.arr[n];
           var node = {
           "fName" : el.fName,
           "guid" : el.guid
           };

           model.arr.push(node);
           if (el.arr) {
           node.arr = [];
           this.addNodesToTreeItemModel(el, node);
           }
           }
	 */
      },

      addNodesToCustomModel:function(lst, model) {/*
						    for ((var n=0; n< lst.arr.length; ++n))
						    {
						    var el = lst.arr[n];
						    var node = {fName : el.fName , guid : el.guid};
						    model.push(node);
						    if (el.arr) {
						    node.arr = [];
						    addNodesToTreeItemModel(el, node);
						    }
						    }
						  */
      },
      event: function(lst) {
         this._event = lst;
         // console.log("summary event lst \n", lst);

         var oTreeData = {fName: "unset"}

         oTreeData.arr = [];
         this.addNodesToTreeItemModel(lst, oTreeData);
         // console.log("event model ", { "top" : oTreeData});

         this.model.setData({ "fName" : "Top", "arr" : oTreeData }); // ??? is this necessary

         // console.log("tree ", this.tree.getItems());
         this.model.refresh(true);
         this.tree.expandToLevel(2);
         sap.ui.getCore().setModel(this.model, "treeModel");


         this.oProductModel = new sap.ui.model.json.JSONModel();
         this.oProductModel.setData([this._event]);
         sap.ui.getCore().setModel(this.oProductModel, "event");
      },
      makeDataForGED : function (element) {
         // remove ROOT::Experimental::
         var shtype = element._typename.substring(20);
         var cgd = this.oGuiClassDef[shtype];
         var arrw = [];
         var modelw = [];

         this.maxLabelLength = 0;
         var off = 0;
         
         // sub editors
         var subEds= [];
         if (cgd[0].sub) {
            off = 1;
            var sarr = cgd[0].sub;
            for (var i = 0; i< sarr.length; ++i) {
               var x = this.oGuiClassDef[sarr[i]];
               for (var j=0; j < x.length; j++)
               {
                  arrw.push(x[j]);
               }
            }
         }

         for (var i = off; i < cgd.length; ++i)
         {
            arrw.push(cgd[i]);
         }
         
         for (var i=0; i< arrw.length; ++i) {
            var parName = arrw[i].name;
            
            if (!arrw[i].member) {
               arrw[i].member = "f" + parName;
            }

            if (!arrw[i].srv) {
               arrw[i].srv = "Set" + parName;
            }

            var v  = element[arrw[i].member];
            if (arrw[i]._type == "Color") {
               v = JSROOT.Painter.root_colors[v];
            }
            var labeledInput = {
               "value" : v,
               "name"  : arrw[i].name,
               "data"  : arrw[i]
            };

            modelw.push({"value" : v, "name" : arrw[i].name, "data" : arrw[i]});
            
            if (this.maxLabelLength < arrw[i].name.length) this.maxLabelLength = arrw[i].name.length;
          }
         
         this.getView().getModel("ged").setData({"widgetlist":modelw});
      },

      onMouseEnter: function(oEvent) {
         var items = this.getView().byId("tree").getItems(), item = null;
         for (var n = 0; n<items.length;++n)
            if (items[n].getId() == oEvent.target.id)
         { item = items[n]; break; }

         // var item = this.getView().byId(oEvent.target.id).getControl();

         if (!item) return;

         var path = item.getBindingContext("treeModel").getPath();

         var ttt = item.getBindingContext("treeModel").getProperty(path);

         var masterid = this.mgr.GetMasterId(ttt.id);

         this.mgr.ProcessHighlight(this, masterid);
      },

      onMouseLeave: function(oEvent) {
         // actual call will be performed 100ms later and can be overwritten
         this.mgr.ProcessHighlight(this, 0, 100);
      },

      onElementHighlight1: function(masterid) {
         var items = this.getView().byId("tree").getItems();
         for (var n = 0; n<items.length;++n) {
            var item = items[n],
                ctxt = item.getBindingContext("treeModel"),
                path = ctxt.getPath(),
                ttt = item.getBindingContext("treeModel").getProperty(path);

            var col = masterid && (masterid == ttt.masterid) ? "yellow" : "";

            item.$().css("background-color", col);
         }
      },
      getGed: function()
      {
	 if (this.ged) {
	    if (!this.ged.visible) {
	       this.toggleEditor();
	    }
	    return;
	 }
	 var pp = this.byId("sumSplitter");
	 console.log("parent", pp);
	 var panel = new sap.m.Panel("productDetailsPanel", {class:"sapUiSizeCompact",  height: "100%" ,width : "97%"});
	 panel.setHeaderText("ElementGED");

	 panel.setLayoutData(new sap.ui.layout.SplitterLayoutData("sld", {size : "30%"}));
	 pp.addContentArea(panel);
	 /*
	 var box = new sap.m.VBox();
	 panel.addContent(box);
	 box.addItem(vert);
	 */
	 var vert = new sap.ui.layout.VerticalLayout("GED",  {class:"sapUiSizeCompact"});

	 vert.addStyleClass("eveTreeItem");
	 vert.addStyleClass("sapUiNoMarginTop");
	 vert.addStyleClass("sapUiNoMarginBottom");
	 
	 panel.addContent(vert);
	 this.ged = panel;
	 this.gedVert = vert;
	 this.ged.visible = true;
      },
      toggleEditor: function()
      {
	 console.log("toggle ");
	 if (!this.ged) {
	    this.getGed();
	 }
	 else {
	 var pp = this.byId("sumSplitter");
	    if (this.ged.visible) {
               console.log("remove ged");
	       pp.removeContentArea(this.ged);
	       this.ged.visible = false;

	    }
	    else {

	       pp.addContentArea(this.ged);
	       this.ged.visible = true;
	    }
	 }
	 
      },
      onDetailPress: function(oEvent) {
         // when edit button pressed
	 this.getGed();
         var item = oEvent.getSource();

         var path =  item.getBindingContext("treeModel").getPath();
         // console.log("path XXX ", oEvent.getParameter("listItem").getBindingContext("treeModel").getProperty(path) );
         var ttt = item.getBindingContext("treeModel").getProperty(path);

         console.log('path', path, ttt);

         if (!ttt) return;

         this.editorElement = this.mgr.GetElement(ttt.id);

         console.log('path', path, 'ttt', this.editorElement._typename);
	 var oProductDetailPanel = this.ged;
        // var oProductDetailPanel = this.byId("productDetailsPanel");
         var title =   this.editorElement.fName + " (" +  this.editorElement._typename.substring(20) + " )" ;
         oProductDetailPanel.setHeaderText(title);

        
         //var oProductDetailPanel = this.byId("productDetailsPanel");
//         console.log("event path ", eventPath);
	  var eventPath = item.getBindingContext("treeModel").getPath();
        oProductDetailPanel.bindElement({ path: eventPath, model: "event" });

         var gedFrame =  this.gedVert;//this.getView().byId("GED");
         gedFrame.unbindElement();
         gedFrame.destroyContent();
         this.makeDataForGED(this.editorElement);
         // console.log("going to bind >>> ", this.getView().getModel("ged"));
         var hl = this.gedFactory;
         gedFrame.bindAggregation("content", "ged>/widgetlist"  , hl );
      },

      onItemPressed: function(oEvent)
      {
         var path =  oEvent.getParameter("listItem").getBindingContext("treeModel").getPath();
         // console.log("path XXX ", oEvent.getParameter("listItem").getBindingContext("treeModel").getProperty(path) );
         var ttt = oEvent.getParameter("listItem").getBindingContext("treeModel").getProperty(path);

         console.log('path', path, ttt, oEvent);

         if (!ttt) return;

         var obj = this.mgr.GetElement(ttt.id);

         console.log('Press', obj);
      },

      gedFactory:function(sId, oContext)
      {
         // console.log("factory id ",sId);
         var base = "/widgetlist/";
         var path = oContext.getPath();
         var idx = path.substring(base.length);
         var customData =  oContext.oModel.oData["widgetlist"][idx].data;
         //console.log("model ",  oContext.oModel);
         //console.log("custom data ", customData);
         var controller =  sap.ui.getCore().byId("TopEveId--Summary").getController();
         var widget;
         switch (customData._type) {

         case "Number":
            var widget = new sap.m.Input(sId, {
               value: {
                  path: "ged>value"
               },
               change: function(event) {
                  controller.sendMethodInvocationRequest(event.getParameter("value"), event);
               }
            });
            widget.setType(sap.m.InputType.Number);
            break;

         case "String":
            var widget = new sap.m.Input(sId, {
               value: {
                  path: "ged>value"
               },
               change: function(event) {
                  controller.sendMethodInvocationRequest(event.getParameter("value"), event);
               }

            });
            widget.setType(sap.m.InputType.String);
            widget.setWidth("250px"); // AMT this should be handled differently
            break;
         case "Bool":
            widget = new sap.m.CheckBox(sId, {
               selected: {
                  path: "ged>value",
               },
               select: function(event) {
                  controller.sendMethodInvocationRequest(event.getSource().getSelected(), event);
               }
            });
            break;

         case "Color":
            var colVal = oContext.oModel.oData["widgetlist"][idx].value;
            currentColor=colVal;
            var model = controller.getView().getModel("colors");
            //   model["mainColor"] = colVal; 
            //  console.log("col value ", colVal, JSROOT.Painter.root_colors[colVal]);
            widget = new sap.ui.jsroot.EVEColorButton(sId, {
               //  text:"x",
               icon: "sap-icon://palette",
               attrcolorXXX:  colVal,
               
	       press: function () {
                  
		     var oCPPop = new ColorPalettePopover( {
		        defaultColor: "cyan",
                        colors: ['gold','darkorange', 'indianred','rgb(102,51,0)', 'cyan',// 'magenta'
                                 'blue', 'lime', 'gray','slategray','rgb(204, 198, 170)',
                                 'white', 'black','red' , 'rgb(102,154,51)', 'rgb(200, 0, 200)'],
                                 
		        colorSelect: controller.handleColorSelect,
                        
		     });

		     oCPPop.openBy(this);
                     oCPPop.data("controller", controller);
                  this.palette =  oCPPop;
	       }
            });
            widget.data("attrcolor", colVal);
            console.log("test ", widget.useColorPalette);
//            model.attachPropertyChange({ "bla": "ddd"}, controller.colorChange, controller);
            break;

         }         
         widget.data("myData", customData);

         var label = new sap.m.Text(sId + "label", { text:{ path: "ged>name"}});
         var ll =  controller.maxLabelLength;
         label.setWidth(ll +"ex");
         label.addStyleClass("sapUiTinyMargin");
         var HL= new sap.ui.layout.HorizontalLayout({
            content : [label, widget]
         });

         return HL;
      },
      handleColorSelect:function(event, data)
      {
         console.log("COLOR CHANGE event !", data, this.data("controller"));
         var val = event.getParameters().value;
         var controller = this.data("controller");
         console.log("parameters" , val,  event.getParameters());
         console.log("val red = ", val.r);


         var rgb;
            var regex = /rgb\((\d+)\,\s?(\d+)\,\s?(\d+)\)/;
         var found = val.match(regex);
         if (found) {
            console.log("match color ", found);
            /*
            rgb.r = found[1];
            rgb.g = found[2];
            rgb.b = found[3];
            */
            rgb = { r: found[1],g:found[2], b:found[3]};
         }
         else {
         var hex = UI5PopupColors[val];

         // Expand shorthand form (e.g. "03F") to full form (e.g. "0033FF")
         var shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
         
         hex = hex.replace(shorthandRegex, function(m, r, g, b) {
            return r + r + g + g + b + b;
         });

         var rgb = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
         rgb = rgb ? {
            r: parseInt(rgb[1], 16),
            g: parseInt(rgb[2], 16),
            b: parseInt(rgb[3], 16)
         } : null;
         }

     
         console.log("COLOR CHANGE data !", sap.ui.getCore().byId("Summary"));
         console.log("rgb", rgb);
   //      var mir =  "SetMainColorRGB(" + rgb.r + ", " + rgb.g +  ", " + rgb.b + ")";
         var mir =  "SetMainColorRGB((UChar_t)" + rgb.r + ", (UChar_t)" + rgb.g +  ", (UChar_t)" + rgb.b + ")";
      //   var mir =  "SetMainColorRGB(" + String.fromCharCode(97 + rgb.r) + ", " + String.fromCharCode(97 + rgb.g) +  ", " + String.fromCharCode(97 + rgb.b) + ")";
         var obj = {"mir" : mir, "fElementId" : controller.editorElement.fElementId, "class" : controller.editorElement._typename};
         console.log("MIR color ", obj);
         sap.ui.getCore().byId("TopEveId").getController().handle.Send(JSON.stringify(obj));
         delete this.palette;

      },
      sendMethodInvocationRequest: function(value, event) {
         // console.log("on change !!!!!!", event.getSource().data("myData"));
         var mir =  event.getSource().data("myData").srv + "( " + value + " )";
         // console.log("=====> ", mir);
         var obj = {"mir" : mir, "fElementId" : this.editorElement.fElementId, "class" : this.editorElement._typename};

         sap.ui.getCore().byId("TopEveId").getController().handle.Send(JSON.stringify(obj));
      },
      changeNumPoints:function()
      {
         var myJSON = "changeNumPoints(" +  this.editorElement.guid + ", "  + this.editorElement.fN +  ")";
         sap.ui.getCore().byId("TopEveId").getController().getHandle().Send(myJSON);
      },
      printEvent: function(event)
      {
         var propertyPath = event.getSource().getBinding("value").getPath();
         // console.log("property path ", propertyPath);
         var bindingContext = event.getSource().getBindingContext("event");

         var path =  bindingContext.getPath(propertyPath);
         var object =  bindingContext.getObject(propertyPath);
         // console.log("obj ",object );

         this.changeNumPoints();
      },
      changeRnrSelf: function(event) {
         console.log("change Rnr ", event.getParameters());

         var myJSON = "changeRnrSelf(" +  this.editorElement.guid + ", "  + event.getParameters().selected +  ")";
         sap.ui.getCore().byId("TopEveId").getController().getHandle().Send(myJSON);
      },
      changeRnrChld: function(event) {
         console.log("change Rnr ", event, " source ", event.getSource());
      },
      updateGED : function (elementId) {
         if (this.editorElement.fElementId == elementId) {
            var gedFrame =  this.gedVert;
            gedFrame.unbindElement();
            gedFrame.destroyContent();
            this.makeDataForGED(this.editorElement);
            // console.log("going to bind >>> ", this.getView().getModel("ged"));
            var hl = this.gedFactory;
            gedFrame.bindAggregation("content", "ged>/widgetlist"  , hl );
         }
      }
   });
});
