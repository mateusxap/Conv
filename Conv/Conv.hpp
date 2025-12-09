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
	size_t B_dim, C_dim, H_dim, W_dim;

public:
	Tensor4D(size_t b = 0, size_t c = 0, size_t h = 0, size_t w = 0)
		: B_dim(b), C_dim(c), H_dim(h), W_dim(w), data(b * c * h * w, 0.0f) {}
	virtual ~Tensor4D() = default;
	size_t getB() const { return B_dim; }
	size_t getC() const { return C_dim; }
	size_t getH() const { return H_dim; }
	size_t getW() const { return W_dim; }
	virtual inline float &operator()(size_t n, size_t c, size_t h, size_t w) = 0;
	virtual inline const float &operator()(size_t n, size_t c, size_t h, size_t w) const = 0;
	void setRandom(std::mt19937 &rng, std::uniform_real_distribution<float> &dist)
	{
		for (float &val : data)
		{
			val = dist(rng);
		}
	}
	size_t size() const { return data.size(); }
	void fill(float val) { std::fill(data.begin(), data.end(), val); }
	double check_difference(const Tensor4D &t2);
};

class Tensor4D_NCHW : public Tensor4D
{
public:
	Tensor4D_NCHW(size_t b = 0, size_t c = 0, size_t h = 0, size_t w = 0) : Tensor4D(b, c, h, w) {}
	inline float &operator()(size_t n, size_t c, size_t h, size_t w) override;
	inline const float &operator()(size_t n, size_t c, size_t h, size_t w) const override;
	Tensor4D_NCHW pad_same_3x3() const;
	Tensor4D_NCHW concatenateChannels(const Tensor4D &t2) const;
};

class Tensor4D_NHWC : public Tensor4D
{
public:
	Tensor4D_NHWC(size_t b = 0, size_t h = 0, size_t w = 0, size_t c = 0) : Tensor4D(b, c, h, w) {}
	inline float &operator()(size_t n, size_t h, size_t w, size_t c) override;
	inline const float &operator()(size_t n, size_t h, size_t w, size_t c) const override;
	Tensor4D_NHWC pad_same_3x3() const;
	Tensor4D_NHWC concatenateChannels(const Tensor4D &t2) const;
};

class Tensor4D_HWIO : public Tensor4D
{
public:
	Tensor4D_HWIO(size_t h = 0, size_t w = 0, size_t i = 0, size_t o = 0) : Tensor4D(o, i, h, w) {}
	inline float &operator()(size_t h, size_t w, size_t i, size_t o) override
	{
		return data[h * W_dim * C_dim * B_dim + w * C_dim * B_dim + i * B_dim + o];
	}
	inline const float &operator()(size_t h, size_t w, size_t i, size_t o) const override
	{
		return data[h * W_dim * C_dim * B_dim + w * C_dim * B_dim + i * B_dim + o];
	}
};

Tensor4D_NCHW convolve_basic(const Tensor4D_NCHW &input_maybe_padded, const Tensor4D_NCHW &kernel);

Tensor4D_NCHW convolve_fused_1x1_3x3_with_if(const Tensor4D_NCHW &input,
											 const Tensor4D_NCHW &kernel_1x1,
											 const Tensor4D_NCHW &kernel_3x3);

Tensor4D_NCHW convolve_fused_1x1_3x3_no_if(const Tensor4D_NCHW &input,
										   const Tensor4D_NCHW &kernel_1x1,
										   const Tensor4D_NCHW &kernel_3x3);

Tensor4D_NHWC convolve_basic(const Tensor4D_NHWC &input, const Tensor4D_HWIO &kernel);

Tensor4D_NHWC convolve_fused_1x1_3x3_no_if(const Tensor4D_NHWC &input,
										   const Tensor4D_NHWC &kernel_1x1,
										   const Tensor4D_NHWC &kernel_3x3);

const size_t N = 1;
const size_t C_in = 832;
const size_t H_in = 7;
const size_t W_in = 7;
const size_t C_out = 624; // 384 + 192 + 48
const int N_ITERATIONS = 1000;
const int WARMUP_ITERATIONS = 10;

const size_t KH = 1;
const size_t KW = 1;

const size_t H_out = H_in - KH + 1;
const size_t W_out = W_in - KW + 1;
const size_t BLOCK_SIZE = 8;
const size_t C_out_block = C_out / BLOCK_SIZE;
const size_t C_in_block = C_in / BLOCK_SIZE;

std::vector<float> conv_optimized(const std::vector<float> &input_NCHWc, const std::vector<float> &kernel_OIHWio, std::vector<float> &output, size_t C_out_curr = C_out);