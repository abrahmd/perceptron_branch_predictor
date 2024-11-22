#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "perceptron.h"

struct Perceptron {
    int w0;
    int weights[HISTORY_LENGTH];

    int (*predict)(int* branch_history_table);
    void (*train)(int computed_y, int branch_outcome);
};

Perceptron* create_perceptron() {
    Perceptron* p = (Perceptron*)malloc(sizeof(Perceptron));

    p->w0 = 1; 
    memset(p->weights, 0, sizeof(p->weights)); //initialize weights to zero

    //p->predict = predict;
    //p->train = train;

    return p;
}


int predict(Perceptron* self, int* branch_prediction_table) { // Not sure about the type for the bpt here or below
    int result = self->w0;
    for (int i = 0; i < HISTORY_LENGTH; i++) {
        result += branch_prediction_table[i] * self->weights[i];
    }
    return result;
}

int sign(int x) {
    return (x > 0) - (x < 0);
}

void train(Perceptron* self, int* branch_history_table, int computed_y, int branch_outcome, int theta) {
    if (sign(computed_y) != branch_outcome || abs(computed_y) <= theta) {
        for (int i = 0; i < HISTORY_LENGTH; i++) {
            self->weights[i] = self->weights[i] + branch_outcome*branch_history_table[i];
        }
    }
}

void destroy_perceptron(Perceptron* p) {
    free(p);
}

int main() {
    Perceptron* p = create_perceptron();

    

    destroy_perceptron(p);
}

