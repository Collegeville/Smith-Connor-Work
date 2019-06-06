
#include <stdio.h>

#include "fann.h"

int FANN_API_test_callback(struct fann *ann, struct fann_train_data *train,
	unsigned int max_epochs, unsigned int epochs_between_reports,
	float desired_error, unsigned int epochs)
{
 	printf("Epochs    %8d. MSE: %.5f. Desired-MSE: %.5f\n", epochs, fann_get_MSE(ann), desired_error);
	return 0;
}

int main()
{
	fann_type *calc_out;
	const unsigned int num_input = 13;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 50;
	const float desired_error = (const float) 0;
	const unsigned int max_epochs = 10000;
	const unsigned int epochs_between_reports = 100;
	struct fann *ann;
	struct fann_train_data *data;

	unsigned int i = 0;
	unsigned int decimal_point;

	float accuracy;
	unsigned int correct_counter = 0;
	int rounded_result;

	printf("Creating network.\n");
	ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);
	
	data = fann_read_train_from_file("new_wine.data");

	fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);

	fann_set_activation_function_hidden(ann, FANN_COS_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_COS_SYMMETRIC);

	fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.01f);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

	fann_init_weights(ann, data);

	printf("Training network.\n");
	fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);

	printf("Testing network. %f\n", fann_test_data(ann,data));

	for(i = 0; i < fann_length_train_data(data); i++)
	{
		calc_out = fann_run(ann, data->input[i]);
		printf("Wine test (%f,%f) -> %f, should be %f, difference=%f\n",
			data->input[i][0], data->input[i][1], calc_out[0], data->output[i][0],
			fann_abs(calc_out[0] - data->output[i][0]));
		rounded_result = round(calc_out[0]);
		if (rounded_result == data->output[i][0])
			correct_counter++;
	}

	accuracy = correct_counter / (float)fann_length_train_data(data);

	printf("Accuracy: %f\n", accuracy);

	printf("Cleaning up.\n");
	fann_destroy_train(data);
	fann_destroy(ann);

	return 0;

}

