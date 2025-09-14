#pragma once



struct MultPolicy {
	template <typename T>
	static T multiply_accumulate(const T& a, const T& b, T& sum) {
		sum += a * b;
	}

	template <typename T>
	static T multiply(const T& a, const T& b) {
		return a * b;
	}
};