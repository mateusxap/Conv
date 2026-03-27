#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>
#include <cmath>
#include <numeric>
#include <string>
#include <immintrin.h>
#include <array>

class Tensor4D
{
protected:
	std::vector<float> data;
	int B_dim, C_dim, H_dim, W_dim;

public:
	Tensor4D(int b = 0, int c = 0, int h = 0, int w = 0)
		: B_dim(b), C_dim(c), H_dim(h), W_dim(w), data(b * c * h * w, 0.0f) {}
	virtual ~Tensor4D() = default;
	int getB() const { return B_dim; }
	int getC() const { return C_dim; }
	int getH() const { return H_dim; }
	int getW() const { return W_dim; }
	virtual inline float &operator()(int n, int c, int h, int w) = 0;
	virtual inline const float &operator()(int n, int c, int h, int w) const = 0;
	void setRandom(std::mt19937 &rng, std::uniform_real_distribution<float> &dist)
	{
		for (float &val : data)
		{
			val = dist(rng);
		}
	}
	int size() const { return (int)data.size(); }
	void fill(float val) { std::fill(data.begin(), data.end(), val); }
	double check_difference(const Tensor4D &t2);
};

class Tensor4D_NCHW : public Tensor4D
{
public:
	Tensor4D_NCHW(int b = 0, int c = 0, int h = 0, int w = 0) : Tensor4D(b, c, h, w) {}
	inline float &operator()(int n, int c, int h, int w) override
	{
		return data[n * (C_dim * H_dim * W_dim) + c * (H_dim * W_dim) + h * W_dim + w];
	}
	inline const float &operator()(int n, int c, int h, int w) const override
	{
		return data[n * (C_dim * H_dim * W_dim) + c * (H_dim * W_dim) + h * W_dim + w];
	}
	Tensor4D_NCHW pad_same_3x3() const;
	Tensor4D_NCHW concatenateChannels(const Tensor4D &t2) const;
};

static std::vector<float> pad_nhwc(const std::vector<float> &input,
                                   int N, int H, int W, int C, int pad);

void convolve_fused_1x1_3x3_param(
	const ConvParams &p,
	const float *padded_input,
	const float *kernel_1x1_HWIO,
	const float *kernel_3x3_HWIO,
	float *output,
	int C_out_1x1,
	int C_out_3x3);

std::vector<float> convolve_fused_1x1_3x3(
    const std::vector<float> &padded_input_NHWC,
    const std::vector<float> &kernel_1x1_HWIO,
    const std::vector<float> &kernel_3x3_HWIO,
    std::vector<float>       &output,
    int N, int H_in, int W_in, int C_in,
    int C_out_1x1, int C_out_3x3);


std::vector<float> convolve_basic(const std::vector<float> &input_NHWC,
								  const std::vector<float> &kernel_HWIO,
								  std::vector<float> &output,
								  int C_out_curr);


const int N = 1;
const int C_in = 256;
const int H_in = 28;
const int W_in = 28;
const int C_out = 128 + 96 + 32; // 384 + 192 + 48
const int N_ITERATIONS = 1000;
const int WARMUP_ITERATIONS = 10;

const int KH = 1;
const int KW = 1;

const int H_out = H_in - KH + 1;
const int W_out = W_in - KW + 1;
const int BLOCK_SIZE = 8;
const int C_out_block = C_out / BLOCK_SIZE;
const int C_in_block = C_in / BLOCK_SIZE;

std::vector<float> conv_optimized_w_c(const std::vector<float> &input_NCHWc, const std::vector<float> &kernel_OIHWio, std::vector<float> &output, int C_out_curr = C_out);
std::vector<float> conv_optimized_c_w(const std::vector<float> &input_NCHWc, const std::vector<float> &kernel_OIHWio, std::vector<float> &output, int C_out_curr = C_out);
std::vector<float> conv_optimized_v3(const std::vector<float> &input_NCHWc, const std::vector<float> &kernel_OIHWio, std::vector<float> &output, int C_out_curr = C_out);

// ---- Parameterized convolution support ----
struct ConvParams {
    int N   = 1;
    int C_in;
    int H_in;
    int W_in;
    int KH  = 1;
    int KW  = 1;

    int H_out()      const { return H_in - KH + 1; }
    int W_out()      const { return W_in - KW + 1; }
    int C_in_block() const { return C_in / BLOCK_SIZE; }
};

void conv_param_w_c(const ConvParams &p, const float *input, const float *kernel,
                    float *output, int C_out_curr);
void conv_param_c_w(const ConvParams &p, const float *input, const float *kernel,
                    float *output, int C_out_curr);
void conv_param_v3 (const ConvParams &p, const float *input, const float *kernel,
                    float *output, int C_out_curr);

void convolve_basic_param(const ConvParams &p, const float *input, const float *kernel,
						  float *output, int C_out_curr);
