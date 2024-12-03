#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 196
#define N_INPUT_2_1 3
#define N_INPUT_1_1 196
#define N_INPUT_2_1 3
#define N_OUTPUTS_3 192
#define N_FILT_3 48
#define N_OUTPUTS_5 188
#define N_FILT_5 64
#define N_OUTPUTS_7 186
#define N_FILT_7 96
#define N_TIME_STEPS_9 186
#define N_OUT_9 128
#define N_OUT_10 128
#define N_LAYER_11 512
#define N_LAYER_13 50
#define N_LAYER_13 50

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<16,6> batch_normalization_2_scale_t;
typedef ap_fixed<16,6> batch_normalization_2_bias_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<16,6> conv1d_6_weight_t;
typedef ap_fixed<16,6> conv1d_6_bias_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<16,6> conv1d_7_weight_t;
typedef ap_fixed<16,6> conv1d_7_bias_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> conv1d_8_weight_t;
typedef ap_fixed<16,6> conv1d_8_bias_t;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<16,6> lstm_4_weight_t;
typedef ap_fixed<16,6> lstm_4_recurrent_weight_t;
typedef ap_fixed<16,6> lstm_4_bias_t;
typedef ap_fixed<16,6> lstm_4_recurrent_bias_t;
typedef ap_fixed<18,8> lstm_4_table_t;
typedef ap_uint<1> layer9_index;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<16,6> lstm_5_weight_t;
typedef ap_fixed<16,6> lstm_5_recurrent_weight_t;
typedef ap_fixed<16,6> lstm_5_bias_t;
typedef ap_fixed<16,6> lstm_5_recurrent_bias_t;
typedef ap_fixed<18,8> lstm_5_table_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<16,6> dense_4_weight_t;
typedef ap_fixed<16,6> dense_4_bias_t;
typedef ap_uint<1> layer11_index;
typedef ap_fixed<16,6> layer13_t;
typedef ap_fixed<16,6> dense_5_weight_t;
typedef ap_fixed<16,6> dense_5_bias_t;
typedef ap_uint<1> layer13_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> dense_5_softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT> dense_5_softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT> dense_5_softmax_inv_table_t;

#endif
