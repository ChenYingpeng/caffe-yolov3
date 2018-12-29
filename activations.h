/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2018/06/04
 */

#ifndef __ACTIVATIONS_H_
#define __ACTIVATIONS_H_

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
} ACTIVATION;

void activate_array_gpu(float* x,int n,ACTIVATION a);

#endif
