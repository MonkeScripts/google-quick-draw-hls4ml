#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t input_layer_2[N_INPUT_1_1*N_INPUT_2_1],
    result_t layer14_out[N_LAYER_13]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_layer_2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_layer_2,layer14_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<batch_normalization_2_scale_t, 3>(s2, "s2.txt");
        nnet::load_weights_from_txt<batch_normalization_2_bias_t, 3>(b2, "b2.txt");
        nnet::load_weights_from_txt<conv1d_6_weight_t, 720>(w3, "w3.txt");
        nnet::load_weights_from_txt<conv1d_6_bias_t, 48>(b3, "b3.txt");
        nnet::load_weights_from_txt<conv1d_7_weight_t, 15360>(w5, "w5.txt");
        nnet::load_weights_from_txt<conv1d_7_bias_t, 64>(b5, "b5.txt");
        nnet::load_weights_from_txt<conv1d_8_weight_t, 18432>(w7, "w7.txt");
        nnet::load_weights_from_txt<conv1d_8_bias_t, 96>(b7, "b7.txt");
        nnet::load_weights_from_txt<lstm_4_weight_t, 49152>(w9, "w9.txt");
        nnet::load_weights_from_txt<lstm_4_recurrent_weight_t, 65536>(wr9, "wr9.txt");
        nnet::load_weights_from_txt<lstm_4_bias_t, 512>(b9, "b9.txt");
        nnet::load_weights_from_txt<lstm_4_recurrent_bias_t, 512>(br9, "br9.txt");
        nnet::load_weights_from_txt<lstm_5_weight_t, 65536>(w10, "w10.txt");
        nnet::load_weights_from_txt<lstm_5_recurrent_weight_t, 65536>(wr10, "wr10.txt");
        nnet::load_weights_from_txt<lstm_5_bias_t, 512>(b10, "b10.txt");
        nnet::load_weights_from_txt<lstm_5_recurrent_bias_t, 512>(br10, "br10.txt");
        nnet::load_weights_from_txt<dense_4_weight_t, 65536>(w11, "w11.txt");
        nnet::load_weights_from_txt<dense_4_bias_t, 512>(b11, "b11.txt");
        nnet::load_weights_from_txt<dense_5_weight_t, 25600>(w13, "w13.txt");
        nnet::load_weights_from_txt<dense_5_bias_t, 50>(b13, "b13.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1*N_INPUT_2_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::normalize<input_t, layer2_t, config2>(input_layer_2, layer2_out, s2, b2); // batch_normalization_2

    layer3_t layer3_out[N_OUTPUTS_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::conv_1d_cl<layer2_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3); // conv1d_6

    layer5_t layer5_out[N_OUTPUTS_5*N_FILT_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::conv_1d_cl<layer3_t, layer5_t, config5>(layer3_out, layer5_out, w5, b5); // conv1d_7

    layer7_t layer7_out[N_OUTPUTS_7*N_FILT_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::conv_1d_cl<layer5_t, layer7_t, config7>(layer5_out, layer7_out, w7, b7); // conv1d_8

    layer9_t layer9_out[N_TIME_STEPS_9*N_OUT_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::lstm_stack<layer7_t, layer9_t, config9>(layer7_out, layer9_out, w9, wr9, b9, br9); // lstm_4

    layer10_t layer10_out[N_OUT_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::lstm_stack<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, wr10, b10, br10); // lstm_5

    layer11_t layer11_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::dense<layer10_t, layer11_t, config11>(layer10_out, layer11_out, w11, b11); // dense_4

    layer13_t layer13_out[N_LAYER_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::dense<layer11_t, layer13_t, config13>(layer11_out, layer13_out, w13, b13); // dense_5

    nnet::softmax<layer13_t, result_t, softmax_config14>(layer13_out, layer14_out); // dense_5_softmax

}
