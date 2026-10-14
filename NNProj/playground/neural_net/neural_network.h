#ifndef __MY_NEURAL_NET__
#define __MY_NEURAL_NET__

typedef struct Neuron {
	float weight;
	float bias;
} Neuron;

typedef struct Dense {
	Neuron* neurons;
	int size;
} Dense;

typedef struct Activation {
	enum ActType {
		sigmoid = 0,
		relu = 1,
		leakyrelu = 2,
		tanh = 3,
		sofmax = 4
	};

	ActType type;
} Activation;

typedef struct NeuralNet {
	enum LayerType {
		dense = 0,
		activation = 1,
	};

	size_t layer_count;
	LayerType* layout;
	Dense* dense_layers;
	Activation* act_layers;
} NeuralNet;



// Initialize a Neural Net
int init_network();

// Define the architecture
/*
	All the enums are just int.
	When defining:
		Ex.
			
*/
int define_network_arch(int* arch, int size);









#endif
