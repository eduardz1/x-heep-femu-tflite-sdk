#ifndef LENET_5_TEST_H
#define LENET_5_TEST_H

int init_tflite();
int infer(const char *data, size_t len, int8_t **out, size_t *out_len);

#endif