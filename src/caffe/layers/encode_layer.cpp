#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/encode_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EncodeLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LossLayer<Dtype>::LayerSetUp(bottom, top);
  num_output_ = 10;
  this->label_count = bottom[0]->count(); //batch_size
  //this->outer_num_ = blabel_count;
}

template <typename Dtype>
void EncodeLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LossLayer<Dtype>::LayerSetUp(bottom, top);
  vector<int> top_shape(2);
  top_shape[0] = label_count;
  top_shape[1] = num_output_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void EncodeLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* label = bottom[0]->cpu_data();
  //int count = bottom[0]->cpu_data()->count();
  int count = 0;

  for (int i = 0; i < label_count; ++i) {
    const int label_value = static_cast<int>(label[i]);

    for (int j = 0; j < num_output_; j++) {

          if(label_value == j) {
              top_data[i * num_output_ + j] = 1;
          }
          else {
              top_data[i * num_output_ + j] = 0;
          }
    }
    count++;
  }

}

INSTANTIATE_CLASS(EncodeLayer);
REGISTER_LAYER_CLASS(Encode);

}  // namespace caffe
