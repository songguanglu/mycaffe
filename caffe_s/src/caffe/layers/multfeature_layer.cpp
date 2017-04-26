#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/multfeature_layer.hpp"
#include "caffe/util/math_functions.hpp"
//----------------------------
// add by songgl
// for cal feature* score
// time:2016/1/4
//----------------------------
namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void MultfeatureLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(top.size(), 1);
}

template <typename Dtype>
void MultfeatureLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  batch_size_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->height(), bottom[1]->height()) << "Feature height and score mismatch!";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width()) << "Feature width and score mismatch!";
  std::vector<int> shape = bottom[0]->shape();
  shape[1] = channels_*bottom[1]->channels();
  shape[2] = 1;
  shape[3] = 1;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void MultfeatureLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_data_score = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  const int score_num = bottom[1]->channels();
  int slice_cnt = top[0]->count();
  int count = 0;
  for (int n = 0; n < bottom[0]->num(); ++n) {
	  for (int m = 0; m < score_num; ++m){
		  for (int c = 0; c < channels_; ++c) {
			  top_data[count] = bottom_data[c] * bottom_data_score[m];
			  count++;
		  }
	  }
	  top_data += top[0]->offset(1);
	  bottom_data += bottom[0]->offset(1);
	  bottom_data_score += bottom[1]->offset(1);
	  count = 0;
  }
}

template <typename Dtype>
void MultfeatureLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
		  for (int m = 0; m < bottom[1]->channels(); ++m){
			  bottom_diff[c] += top_diff[c + m*channels_] * bottom_data_score[m];
		  }
	  }
	  bottom_data_score += bottom[1]->offset(1);
	  bottom_diff += bottom[0]->offset(1);
	  top_diff += top[0]->offset(1);
  }
  //cal bottom[1] diff

  top_diff = top[0]->cpu_diff();
  for (int n = 0; n < bottom[0]->num(); ++n){
	  for (int c = 0; c < bottom[1]->channels(); ++c){
		  for (int m = 0; m < bottom[0]->channels(); ++m){
			  bottom_score_diff[c] += top_diff[m + c*channels_] * bottom_data[m];
		  }
		}
	  bottom_data += bottom[0]->offset(1);
	  bottom_score_diff += bottom[1]->offset(1);
	  top_diff += top[0]->offset(1);
	}
}
#ifdef CPU_ONLY
STUB_GPU(MultfeatureLayer);
#endif

INSTANTIATE_CLASS(MultfeatureLayer);
REGISTER_LAYER_CLASS(Multfeature);

}  // namespace caffe
