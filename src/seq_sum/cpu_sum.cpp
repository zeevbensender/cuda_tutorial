float cpu_sum(const float *a, int n) {
    float total = 0.0f;
    for (int i = 0; i < n; ++i)
        total += a[i];
    return total;
}
