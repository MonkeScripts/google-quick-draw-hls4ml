#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_conv1d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_recurrent.h"

// hls-fpga-machine-learning insert weights
#include "weights/s2.h"
#include "weights/b2.h"
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/w5.h"
#include "weights/b5.h"
#include "weights/w7.h"
#include "weights/b7.h"
#include "weights/w9.h"
#include "weights/wr9.h"
#include "weights/b9.h"
#include "weights/br9.h"
#include "weights/w10.h"
#include "weights/wr10.h"
#include "weights/b10.h"
#include "weights/br10.h"
#include "weights/w11.h"
#include "weights/b11.h"
#include "weights/w13.h"
#include "weights/b13.h"

// hls-fpga-machine-learning insert layer-config
// batch_normalization_2
struct config2 : nnet::batchnorm_config {
    static const unsigned n_in = N_INPUT_1_1*N_INPUT_2_1;
    static const unsigned n_filt = 3;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_2_bias_t bias_t;
    typedef batch_normalization_2_scale_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// conv1d_6
struct config3_mult : nnet::dense_config {
    static const unsigned n_in = 15;
    static const unsigned n_out = 48;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef conv1d_6_bias_t bias_t;
    typedef conv1d_6_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config3 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 196;
    static const unsigned n_chan = 3;
    static const unsigned filt_width = 5;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 48;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 192;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 196;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 192;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_3<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef conv1d_6_bias_t bias_t;
    typedef conv1d_6_weight_t weight_t;
    typedef config3_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
};
const ap_uint<config3::filt_width> config3::pixels[] = {0};

// conv1d_7
struct config5_mult : nnet::dense_config {
    static const unsigned n_in = 240;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef conv1d_7_bias_t bias_t;
    typedef conv1d_7_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config5 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 192;
    static const unsigned n_chan = 48;
    static const unsigned filt_width = 5;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 188;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 192;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 188;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_5<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef conv1d_7_bias_t bias_t;
    typedef conv1d_7_weight_t weight_t;
    typedef config5_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
};
const ap_uint<config5::filt_width> config5::pixels[] = {0};

// conv1d_8
struct config7_mult : nnet::dense_config {
    static const unsigned n_in = 192;
    static const unsigned n_out = 96;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef conv1d_8_bias_t bias_t;
    typedef conv1d_8_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config7 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 188;
    static const unsigned n_chan = 64;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 96;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 186;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 188;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 186;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_7<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef conv1d_8_bias_t bias_t;
    typedef conv1d_8_weight_t weight_t;
    typedef config7_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
};
const ap_uint<config7::filt_width> config7::pixels[] = {0};

// lstm_4
struct config9_1 : nnet::dense_config {
    static const unsigned n_in = N_FILT_7;
    static const unsigned n_out = N_OUT_9 * 4;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 49152;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef lstm_4_bias_t bias_t;
    typedef lstm_4_weight_t weight_t;
    typedef layer9_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config9_2 : nnet::dense_config {
    static const unsigned n_in = N_OUT_9;
    static const unsigned n_out = N_OUT_9 * 4;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 65536;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef lstm_4_bias_t bias_t;
    typedef lstm_4_weight_t weight_t;
    typedef layer9_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct sigmoid_config9_recr : nnet::activ_config {
    static const unsigned n_in = N_OUT_9 * 3;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef lstm_4_table_t table_t;
};

struct tanh_config9 : nnet::activ_config {
    static const unsigned n_in = N_OUT_9;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef lstm_4_table_t table_t;
};

struct config9 : nnet::lstm_config {
    typedef model_default_t accum_t;
    typedef lstm_4_weight_t weight_t;  // Matrix
    typedef lstm_4_bias_t bias_t;  // Vector
    typedef config9_1 mult_config1;
    typedef config9_2 mult_config2;
    typedef sigmoid_config9_recr ACT_CONFIG_LSTM;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::sigmoid<x_T, y_T, config_T>;
    typedef tanh_config9 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = N_FILT_7;
    static const unsigned n_out = N_OUT_9;
    static const unsigned n_state = N_OUT_9;
    static const unsigned n_sequence = N_OUTPUTS_7;
    static const unsigned n_sequence_out = N_TIME_STEPS_9;
    static const unsigned io_type = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
};

// lstm_5
struct config10_1 : nnet::dense_config {
    static const unsigned n_in = N_OUT_9;
    static const unsigned n_out = N_OUT_10 * 4;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 65536;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef lstm_5_bias_t bias_t;
    typedef lstm_5_weight_t weight_t;
    typedef layer10_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config10_2 : nnet::dense_config {
    static const unsigned n_in = N_OUT_10;
    static const unsigned n_out = N_OUT_10 * 4;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 65536;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef lstm_5_bias_t bias_t;
    typedef lstm_5_weight_t weight_t;
    typedef layer10_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct sigmoid_config10_recr : nnet::activ_config {
    static const unsigned n_in = N_OUT_10 * 3;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef lstm_5_table_t table_t;
};

struct tanh_config10 : nnet::activ_config {
    static const unsigned n_in = N_OUT_10;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef lstm_5_table_t table_t;
};

struct config10 : nnet::lstm_config {
    typedef model_default_t accum_t;
    typedef lstm_5_weight_t weight_t;  // Matrix
    typedef lstm_5_bias_t bias_t;  // Vector
    typedef config10_1 mult_config1;
    typedef config10_2 mult_config2;
    typedef sigmoid_config10_recr ACT_CONFIG_LSTM;
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::sigmoid<x_T, y_T, config_T>;
    typedef tanh_config10 ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::tanh<x_T, y_T, config_T>;
    static const unsigned n_in  = N_OUT_9;
    static const unsigned n_out = N_OUT_10;
    static const unsigned n_state = N_OUT_10;
    static const unsigned n_sequence = N_TIME_STEPS_9;
    static const unsigned n_sequence_out = 1;
    static const unsigned io_type = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
};

// dense_4
struct config11 : nnet::dense_config {
    static const unsigned n_in = 128;
    static const unsigned n_out = 512;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 65536;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef dense_4_bias_t bias_t;
    typedef dense_4_weight_t weight_t;
    typedef layer11_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense_5
struct config13 : nnet::dense_config {
    static const unsigned n_in = 512;
    static const unsigned n_out = 50;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 25600;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef dense_5_bias_t bias_t;
    typedef dense_5_weight_t weight_t;
    typedef layer13_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// dense_5_softmax
struct softmax_config14 : nnet::activ_config {
    static const unsigned n_in = 50;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned axis = -1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    typedef dense_5_softmax_exp_table_t exp_table_t;
    typedef dense_5_softmax_inv_table_t inv_table_t;
};


#endif
