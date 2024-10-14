#include "flag.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"

using namespace nndeploy;

class MyNode : public nndeploy::dag::Node {
public:
    MyNode(const std::string& name, dag::Edge* in, dag::Edge* out) : nndeploy::dag::Node(name, in, out) {}

    virtual base::Status run() override {
        std::cout << "Executing node: " << getName() << std::endl;
        return base::kStatusCodeOk;
    }
};

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (demo::FLAGS_usage) {
    demo::showUsage();
    return -1;
  }

  dag::Edge graph_in("graph_in");
  dag::Edge graph_out("graph_out");
  dag::Graph *graph = new dag::Graph("demo", &graph_in, &graph_out);
  auto* node1 = new MyNode("Node1", nullptr, nullptr);
  graph->addNode(node1);

  auto* node2 = graph->createNode<MyNode>("Node2", &graph_in, &graph_out);

  auto status = graph->init();
  graph->dump();
  graph->run();
  graph->deinit();

  NNDEPLOY_LOGI("hello world!\n");

  delete graph;
  delete node1;
  return 0;
}
