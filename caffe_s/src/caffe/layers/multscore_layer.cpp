#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/multscore_layer.hpp"
#include "caffe/util/math_functions.hpp"
//----------------------------
// add by songgl
// for cal feature_map* score_map
// time:2016/12/10
//----------------------------
namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void MultscoreLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(top.size(), 1);
}

template <typename Dtype>
void MultscoreLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  batch_size_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->height(), bottom[1]->height()) << "Feature height and score mismatch!";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width()) << "Feature width and score mismatch!";
  std::vector<int> shape = bottom[0]->shape();
  shape[2] = 1;
  shape[3] = 1;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void MultscoreLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_data_score = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  int slice_cnt = top[0]->count();
  for (int n = 0; n < bottom[0]->num(); ++n) {
	  for (int c = 0; c < channels_; ++c) {
		  for (int ph = 0; ph < height_; ++ph) {
			  for (int pw = 0; pw < width_; ++pw) {
				  top_data[c] += bottom_data[ph*width_ + pw] * bottom_data_score[ph*width_ + pw];
				  // top_data[ph*width_ + pw] = bottom_data[ph*width_ + pw] * bottom_data_score[ph*width_ + pw];
			  }
		  }
		  bottom_data += bottom[0]->offset(0, 1);
		//  top_data += top[0]->offset(0,1);
	  }
	  top_data += top[0]->offset(1);
	  bottom_data_score += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void MultscoreLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_data_score = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_score_diff = bottom[1]->mutable_cpu_diff();
  caffe_set(bottom[1]->count(), Dtype(0), bottom_score_diff);
  // cal bottom[0] diff
  for (int n = 0; n < bottom[0]->num(); ++n) {
	  for (int c = 0; c < channels_; ++c) {
		  for (int ph = 0; ph < height_; ++ph) {
			  for (int pw = 0; pw < width_; ++pw) {
				 //bottom_diff[ph*width_ + pw] = top_diff[ph*width_ + pw] * bottom_data_score[ph*width_ + pw];
				  bottom_diff[ph*width_ + pw] = top_diff[c] * bottom_data_score[ph*width_ + pw];
			  }
			  }
		  bottom_diff += bottom[0]->offset(0, 1);
		  //top_diff += top[0]->offset(0,1);
		  }
	  top_diff += top[0]->offset(1);
	  bottom_data_score += bottom[1]->offset(1);
  }
  top_diff = top[0]->cpu_diff();
  // cal bottom[1] score diff
  for (int n = 0; n < bottom[0]->num(); ++n) {
	  for (int c = 0; c < channels_; ++c) {
		  for (int ph = 0; ph < height_; ++ph) {
			  for (int pw = 0; pw < width_; ++pw) {
				//  bottom_score_diff[ph*width_ + pw] = bottom_score_diff[ph*width_+pw]+top_diff[ph*width_ + pw] * bottom_data[ph*width_ + pw];
				  bottom_score_diff[ph*width_ + pw] = bottom_score_diff[ph*width_ + pw] + top_diff[c] * bottom_data[ph*width_ + pw];
			  }
			  }
		  bottom_data += bottom[0]->offset(0, 1);
		 //top_diff += top[0]->offset(0,1);
		}
	  top_diff += top[0]->offset(1);
	  bottom_score_diff += bottom[1]->offset(1);
  }
}


#ifdef CPU_ONLY
STUB_GPU(MultscoreLayer);
#endif

INSTANTIATE_CLASS(MultscoreLayer);
REGISTER_LAYER_CLASS(Multscore);

}  // namespace caffe
