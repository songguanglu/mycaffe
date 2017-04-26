

/*
* three_loss_layer.cpp
*/

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/three_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void ThreeLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    CHECK_EQ(bottom[0]->num(), bottom[1]->num());
    CHECK_EQ(bottom[1]->num(), bottom[2]->num());
    CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
    CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), 1);
    CHECK_EQ(bottom[0]->width(), 1);
    CHECK_EQ(bottom[1]->height(), 1);
    CHECK_EQ(bottom[1]->width(), 1);
    CHECK_EQ(bottom[2]->height(), 1);
    CHECK_EQ(bottom[2]->width(), 1);

    /* bottom[3] is for sample's weight, decarded here */
    //CHECK_EQ(bottom[3]->channels(),1);
    //CHECK_EQ(bottom[3]->height(), 1);
    //CHECK_EQ(bottom[3]->width(), 1);

    diff_ap_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    diff_an_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    diff_pn_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);

    diff_sq_ap_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    diff_sq_an_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
	diff_sq_pn_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);

    dist_sq_ap_.Reshape(bottom[0]->num(), 1, 1, 1);
    dist_sq_an_.Reshape(bottom[0]->num(), 1, 1, 1);
	dist_sq_pn_.Reshape(bottom[0]->num(), 1, 1, 1);

	// vector of ones used to sum along channels
    summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
    for (int i = 0; i < bottom[0]->channels(); ++i)
      summer_vec_.mutable_cpu_data()[i] = Dtype(1);
    dist_binary_.Reshape(bottom[0]->num(), 1, 1, 1);
    for (int i = 0; i < bottom[0]->num(); ++i)
      dist_binary_.mutable_cpu_data()[i] = Dtype(1);
  }

  template <typename Dtype>
  void ThreeLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();

    //const Dtype* sampleW = bottom[3]->cpu_data();
    const Dtype sampleW = Dtype(1);
    caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // p
      diff_ap_.mutable_cpu_data());  // a_i-p_i
    caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[2]->cpu_data(),  // n
      diff_an_.mutable_cpu_data());  // a_i-n_i
    caffe_sub(
      count,
      bottom[1]->cpu_data(),  // p
      bottom[2]->cpu_data(),  // n
      diff_pn_.mutable_cpu_data());  // p_i-n_i
    const int channels = bottom[0]->channels();
    Dtype margin = this->layer_param_.three_loss_param().margin();

    Dtype loss(0.0);
	loss1_ = 0.0;
	loss2_ = 0.0;
	loss3_ = 0.0;
    for (int i = 0; i < bottom[0]->num(); ++i) {
      dist_sq_ap_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
                                                        diff_ap_.cpu_data() + (i*channels), diff_ap_.cpu_data() + (i*channels));
      dist_sq_an_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
                                                        diff_an_.cpu_data() + (i*channels), diff_an_.cpu_data() + (i*channels));
	  dist_sq_pn_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
														diff_pn_.cpu_data() + (i*channels), diff_pn_.cpu_data() + (i*channels));
      //Dtype mdist = sampleW*std::max(margin + dist_sq_ap_.cpu_data()[i] - dist_sq_an_.cpu_data()[i], Dtype(0.0));
	  Dtype mdist_1 = sampleW*std::max(margin - dist_sq_ap_.cpu_data()[i], Dtype(0.0));
	  Dtype mdist_2 = sampleW*std::max(margin - dist_sq_an_.cpu_data()[i], Dtype(0.0));
	  Dtype mdist_3 = sampleW*std::max(margin - dist_sq_pn_.cpu_data()[i], Dtype(0.0));
	  loss += mdist_1+mdist_2+mdist_3;
	  loss1_ += mdist_1;
	  loss2_ += mdist_2;
	  loss3_ += mdist_3;
	//  LOG(INFO) << "debug info here---:loss " << loss << "mdist: " << mdist << "minus: " << dist_sq_ap_.cpu_data()[i] - dist_sq_an_.cpu_data()[i];
      if (mdist_1 < Dtype(1e-9)) {
        //dist_binary_.mutable_cpu_data()[i] = Dtype(0);
        //prepare for backward pass
		 caffe_set(channels, Dtype(0), diff_ap_.mutable_cpu_data() + (i*channels));
       // caffe_set(channels, Dtype(0), diff_an_.mutable_cpu_data() + (i*channels));
       // caffe_set(channels, Dtype(0), diff_pn_.mutable_cpu_data() + (i*channels));
      }
	  if (mdist_2 < Dtype(1e-9)){
		  caffe_set(channels, Dtype(0), diff_an_.mutable_cpu_data() + (i*channels));
	  }
	  if (mdist_3 < Dtype(1e-9)){
		  caffe_set(channels, Dtype(0), diff_pn_.mutable_cpu_data() + (i*channels));
	  }
    }
    loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
	loss1_ = loss1_ / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
	loss2_ = loss2_ / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
	loss3_ = loss3_ / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
	LOG(INFO) << "debug info here---:final loss " << loss;
	LOG(INFO) << "debug info here---:final loss_1 " << loss1_;
	LOG(INFO) << "debug info here---:final loss_2 " << loss2_;
	LOG(INFO) << "debug info here---:final loss_3 " << loss3_;
    top[0]->mutable_cpu_data()[0] = loss;
  }

  template <typename Dtype>
  void ThreeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    //Dtype margin = this->layer_param_.contrastive_loss_param().margin();
    //const Dtype* sampleW = bottom[3]->cpu_data();
    const Dtype sampleW = Dtype(1);

    for (int i = 0; i < 3; ++i) {
      if (propagate_down[i]) {
        const Dtype sign = (i < 2) ? 1 : 1;
       // const Dtype alpha = sign * top[0]->cpu_diff()[0] /
        //  static_cast<Dtype>(bottom[i]->num());
		Dtype alpha_1 = 0.0;
		Dtype alpha_2 = 0.0;
		Dtype alpha_3 = 0.0;
		if (loss1_ > 0.0){
			alpha_1 = sign*top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[i]->num());
		}
		else{
			alpha_1 = sign*loss1_ / static_cast<Dtype>(bottom[i]->num());
		}
		if (loss2_ > 0.0){
			alpha_2 = sign*top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[i]->num());
		}
		else{
			alpha_2 = sign*loss2_ / static_cast<Dtype>(bottom[i]->num());
		}
		if (loss3_ > 0.0){
			alpha_3 = sign*top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[i]->num());
		}
		else{
			alpha_3 = sign*loss3_ / static_cast<Dtype>(bottom[i]->num());
		}
	//	const  Dtype alpha_3 = sign*loss3_ / static_cast<Dtype>(bottom[i]->num());
        int num = bottom[i]->num();
        int channels = bottom[i]->channels();
        for (int j = 0; j < num; ++j) {
          Dtype* bout = bottom[i]->mutable_cpu_diff();
          if (i == 0) {  // a
            //if(dist_binary_.cpu_data()[j]>Dtype(0)){
            caffe_cpu_axpby(
              channels,
              -alpha_1*sampleW,
              diff_ap_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
			caffe_cpu_axpby(
				channels,
				-alpha_2*sampleW,
				diff_an_.cpu_data() + (j*channels),
				Dtype(1.0),
				bout + (j*channels));
            //}else{
            //  caffe_set(channels, Dtype(0), bout + (j*channels));
            //}
          }
          else if (i == 1) {  // p
            //if(dist_binary_.cpu_data()[j]>Dtype(0)){
            caffe_cpu_axpby(
              channels,
              alpha_1*sampleW,
              diff_ap_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
			caffe_cpu_axpby(
				channels,
				-alpha_3*sampleW,
				diff_pn_.cpu_data() + (j*channels),
				Dtype(1.0),
				bout + (j*channels));
			//}else{
            //	  caffe_set(channels, Dtype(0), bout + (j*channels));
            //}
          }
          else if (i == 2) {  // n
            //if(dist_binary_.cpu_data()[j]>Dtype(0)){
            caffe_cpu_axpby(
              channels,
              alpha_2*sampleW,
              diff_an_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
			caffe_cpu_axpby(
				channels,
				alpha_3*sampleW,
				diff_pn_.cpu_data() + (j*channels),
				Dtype(1.0),
				bout + (j*channels));
            //}else{
            //   caffe_set(channels, Dtype(0), bout + (j*channels));
            //}
          }
        } // for num
      } //if propagate_down[i]
    } //for i
  }

#ifdef CPU_ONLY
  STUB_GPU(TripletLossLayer);
#endif

  INSTANTIATE_CLASS(ThreeLossLayer);
  REGISTER_LAYER_CLASS(ThreeLoss);

}  // namespace caffe

