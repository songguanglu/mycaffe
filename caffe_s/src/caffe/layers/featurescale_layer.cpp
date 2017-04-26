#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/featurescale_layer.hpp"
#include "caffe/util/math_functions.hpp"
//----------------------------
// add by songgl
// for cal feature* score
// time:2016/1/11
//----------------------------
namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void FeaturescaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(top.size(), 1);
}

template <typename Dtype>
void FeaturescaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  batch_size_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->height(), 1) << "Feature height error!";
  CHECK_EQ(bottom[0]->width(), 1) << "Feature width error!";
  CHECK_EQ(bottom[1]->height(), 1) << "Score height must be 1";
  CHECK_EQ(bottom[1]->width(), 1) << "Score width must be 1";
  CHECK_EQ(bottom[1]->channels(), 1) << "Score channel must be 1";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "Feature num and Score num mismatch!";
  std::vector<int> shape = bottom[0]->shape();
  top[0]->Reshape(shape);
}

template <typename Dtype>
void FeaturescaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_data_score = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  int slice_cnt = top[0]->count();
  int count = 0;
  for (int n = 0; n < bottom[0]->num(); ++n) {
	  for (int c = 0; c < channels_; ++c) {
		  top_data[c] = bottom_data[c] * bottom_data_score[n];
	  }
	  top_data += top[0]->offset(1);
	  bottom_data += bottom[0]->offset(1);
  }
}

template <typename Dtype>
void FeaturescaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const  Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_data_score = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_score_diff = bottom[1]->mutable_cpu_diff();
  caffe_set(bottom[1]->count(), Dtype(0), bottom_score_diff);
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // cal bottom[0] diff
  channels_ = bottom[0]->channels();
  const int score_num = bottom[1]->channels();
  for (int n = 0; n < bottom[0]->num(); ++n){
	  for (int c = 0; c < bottom[0]->channels(); ++c){
			  bottom_diff[c] = top_diff[c] * bottom_data_score[n];
	  }
	  bottom_diff += bottom[0]->offset(1);
	  top_diff += top[0]->offset(1);
  }
  //cal bottom[1] diff

  top_diff = top[0]->cpu_diff();
  for (int n = 0; n < bottom[0]->num(); ++n){
	  for (int c = 0; c < bottom[0]->channels(); ++c){
		  bottom_score_diff[n] += top_diff[c] * bottom_data[c];
	  }
	  bottom_data += bottom[0]->offset(1);
	  top_diff += top[0]->offset(1);
	}
}
#ifdef CPU_ONLY
STUB_GPU(FeaturescaleLayer);
#endif

INSTANTIATE_CLASS(FeaturescaleLayer);
REGISTER_LAYER_CLASS(Featurescale);

}  // namespace caffe
