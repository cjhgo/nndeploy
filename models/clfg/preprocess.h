#ifndef NNDEPLOY_PREPROCESS_PREPROCESS_NODE_H
#define NNDEPLOY_PREPROCESS_PREPROCESS_NODE_H

#include "nndeploy/dag/node.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/base/common.h"

using namespace nndeploy;

namespace models {
namespace clfg {

class PreprocessNode : public dag::Node {
 public:
  PreprocessNode(const std::string& name, dag::Edge* input, dag::Edge* output);
  virtual ~PreprocessNode();
  virtual base::Status run() override;
 private:
  dag::Edge* input_;
  dag::Edge* output_;
  int index_ = 0;
};

}  // namespace clfg
}  // namespace models
#endif  // NNDEPLOY_PREPROCESS_PREPROCESS_NODE_H