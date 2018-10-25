#include <memory>

#include <ZeroMQSvc.h>

ZeroMQSvc& zmqSvc() {
   static std::unique_ptr<ZeroMQSvc> svc;
   if (!svc) {
      svc = std::make_unique<ZeroMQSvc>();
   }
   return *svc;
}
