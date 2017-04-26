#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/multicore_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MulticoreLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const int num_core = this->layer_param_.multicore_param().core();
	N_ = num_core;
    // Initialize the weights
	this->blobs_.resize(1);
    vector<int> core_shape(N_,1);
    this->blobs_[0].reset(new Blob<Dtype>(core_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
		  this->layer_param_.multicore_param().bias_filler()));
      bias_filler->Fill(this->blobs_[0].get());
    this->param_propagate_down_.resize(this->blobs_.size(), true);
	//CHECK_EQ(bottom[0]->channels(), this->blobs_[0]->count());
}

template <typename Dtype>
void MulticoreLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
	batch_size_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
  vector<int> top_shape = bottom[0]->shape();
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MulticoreLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* core = this->blobs_[0]->cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
	  for (int c = 0; c < channels_; ++c) {
		  top_data[c] = bottom_data[c] - core[c];
	  }
	  bottom_data += bottom[0]->offset(1);
	  top_data += top[0]->offset(1);
  }
  for (int c = 0; c < channels_; ++c){
	  LOG(INFO) << "debug info here---:core: " << c<<" :"<<core[c];
  }
}

template <typename Dtype>
void MulticoreLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype* core_diff = this->blobs_[0]->mutable_cpu_diff();
	caffe_set(this->blobs_[0]->count(), Dtype(0), core_diff);
	for (int n = 0; n < bottom[0]->num(); ++n){
		for (int c = 0; c < channels_; ++c){
			bottom_diff[c] = top_diff[c];
			core_diff[c] += core_diff[c] + (-1)*top_diff[c];
		}
		bottom_diff += bottom[0]->offset(1);
		top_diff += top[0]->offset(1);
	}
	for (int c = 0; c < channels_; ++c){
		LOG(INFO) << "debug info here---:core_diff: " << c << " :" << core_diff[c];
	}
}

#ifdef CPU_ONLY
STUB_GPU(MulticoreLayer);
#endif

INSTANTIATE_CLASS(MulticoreLayer);
REGISTER_LAYER_CLASS(Multicore);

}  // namespace caffe
