#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#define HISTORY_LENGTH 12

typedef struct Perceptron Perceptron;

Perceptron* create_perceptron();

int predict(Perceptron* self, int* branch_prediction_table);

int sign(int x);

void train(Perceptron* self, int* branch_history_table, int computed_y, int branch_outcome, int theta);

void destroy_perceptron(Perceptron* p);

#endif