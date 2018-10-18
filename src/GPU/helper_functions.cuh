#pragma once
inline __device__ float distance(float x1, float y1, float x2, float y2) {
  auto dx = x1 - x2;
  auto dy = y1 - y2;
  return sqrtf(dx * dx + dy * dy);
}

bool __device__ __host__ is_close(int delta, int range) {
  int abs_delta = abs(delta);
  return (abs_delta < range || range > 2048 - range);
  // return (delta + range + 2047) % 2048 < 2 * range - 1;
}
