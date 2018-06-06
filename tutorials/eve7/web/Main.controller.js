sap.ui.define(['sap/ui/core/mvc/Controller' ], function(Controller) {
   "use strict";

    return Controller.extend("eve.Main", {
      onInit: function () {
           this.handle = this.getView().getViewData().conn_handle;
           this.handle.SetReceiver(this);
           this.handle.Connect();

           this.mgr = new JSROOT.EVE.EveManager();

           
           // this.getView().byId("Summary").SetMgr(this.mgr);
        },
            
        getHandle: function () {
           return this.handle;
        },
            
        OnWebsocketMsg: function(handle, msg, offset) {
                          
           if (typeof msg != "string") {
              // console.log('ArrayBuffer size ',
              // msg.byteLength, 'offset', offset);
              this.mgr.UpdateBinary(msg, offset);
              
              this.mgr.ProcessModified();
              
              return;
           }

           console.log("txt:", msg);
           var resp = JSON.parse(msg);

           if (resp && resp[0] && resp[0].content == "TEveScene::StreamElements") {
                             
              this.mgr.Update(resp);
                             // console.log('element',
                              // this.getView().byId("Summary").getController());

              this.getView().byId("Summary").getController().UpdateMgr(this.mgr);
                            
              var viewers = this.mgr.FindViewers();
                             
              console.log("FOUND viewers", viewers.length);
              
             // first check number of views to create
              var count = 0;
              for (var n=0;n<viewers.length;++n) {
                 if (!viewers[n].$view_created) count++;
              }
              if (count == 0) return;
              
              var main = this, vv = null, sv = this.getView().byId("ViewAreaSplitter");
                    
              JSROOT.AssertPrerequisites("geom;user:evedir/EveElements.js", function() {
                 
                 count = 0;
                 for (var n=0;n<viewers.length;++n) {
                    var elem = viewers[n];
                    var viewid = "EveViewer" + elem.fElementId;
                    if (elem.$view_created) continue; // this.getView().byId(viewid)
                       
                    // create missing view
                    elem.$view_created = true;
                    console.log("Creating view", viewid);
                    
                    var view = new JSROOT.sap.ui.xmlview({
                       id: viewid,
                       viewName: "eve.GL",
                       viewData: { mgr: main.mgr, elementid: elem.fElementId }
                    });
                    
                    count++;
                       
                    if (count == 1) { 
                        sv.addContentArea(view);
                        continue;
                     } 

                    if (!vv) {
                        vv = new JSROOT.sap.ui.layout.Splitter("SecondaryViewSplitter", {
                            splitterOrientation : "Vertical", 
                            splitterPosition : "50%",
                            minSizeFirstPane : "0%",
                            minSizeSecondPane : "0%"
                        });
                        sv.addContentArea(vv);
                     }
                    
                     if (count>2) {
                        vv.addContentArea(new JSROOT.sap.ui.layout.Splitter("ThirdLevelSplitter" + count, {
                             splitterOrientation : "Horizontal"}));
                     }
                    
                     vv.addContentArea(view);
                  }
              });


           } else if (resp.function === "geometry")
                          {
                              console.log("GEO");
                              viewManager.setGeometry( resp);
                          }
                      
                          else if (resp.function === "event")
                          {
                              console.log("EVE ", resp);
                              this._event = resp.args[0];
                              this.event();
                          }
                          else if (resp.function === "replaceElement")
                          {
                              var oldEl = this.findElementWithId(resp.guid, this._event);
                              var newEl = resp;
                              viewManager.replace(oldEl, newEl);

                              this.event(); 
                          }
                          else if (resp.function === "endChanges") {
                              this.endChanges = resp.val;
                              if (resp.val)
                              {
                   /*
                      * var ele = this.getView().byId("GL"); var cont =
                      * ele.getController(); cont.endChanges(resp.val);
                      */
                   viewManager.envokeViewFunc("endChanges", resp.val);
                              }
                          }
                      },
                           /*
                            * processWaitingMsg: function() { for ( var i = 0; i <
                            * msgToWait.length; ++i ) {
                            * this.OnWebsocketMsg(handleToWait, msgToWait[i]); }
                            * handleToWait = 0; msgToWait = []; },
                            */
                      event: function() {
                          // this._event = lst;
               
                      },
                      
            setMainVerticalSplitterHeight: function(){
                var mainViewHeight = document.body.clientHeight;
                var mainToolbarHeight = 49;
                var height = mainViewHeight - mainToolbarHeight;    
                var splitter =  this.getView().byId("MainAreaSplitter");
                if (splitter) {
                    // console.log("set splitter height >>> " , height);
                     splitter.setHeight(height + "px");
                 }
            },
            onAfterRendering: function(){
               var me = this;
               setTimeout(
                     function(){
                        $(window).on("resize", function(){
                           me.setMainVerticalSplitterHeight();
                        });
                        me.setMainVerticalSplitterHeight();
                     }, 100);
            },
            
            findElementWithId: function(valueToSearch, el) {
                 if (!el) {
                     el = this._event;
                  }
                          // console.log("serach ",valueToSearch, "in", el )
                 if (el.guid == valueToSearch) {
                    // console.log("found it findElementWithId ", el)
                    return el;
                 }
                 if ( el.arr) {
                    for (var i = 0; i < el.arr.length; i++) {
                       var x = this.findElementWithId(valueToSearch, el.arr[i]);
                       if (x) return x; 
                    }
                 }
                 return 0;
            }
             });

       }
);
