/// Utility functions for index manipulation in tensor operations.
///
/// These functions are commonly used for padding operations with various
/// boundary modes (reflect, replicate, circular).
library;

/// Reflects an index at the boundary.
///
/// For indices outside [0, size), reflects them back into the valid range.
/// Used by reflect padding mode.
///
/// Example for size=5:
/// - idx=-1 → 0 (reflects at left boundary)
/// - idx=5 → 4 (reflects at right boundary)
/// - idx=6 → 3 (continues reflecting)
int reflectIndex(int idx, int size) {
  if (idx < 0) {
    return -idx - 1;
  } else if (idx >= size) {
    return 2 * size - idx - 1;
  }
  return idx;
}

/// Clamps an index to the valid range [0, size-1].
///
/// Indices below 0 become 0, indices >= size become size-1.
/// Used by replicate (edge) padding mode.
///
/// Example for size=5:
/// - idx=-1 → 0
/// - idx=5 → 4
/// - idx=10 → 4
int replicateIndex(int idx, int size) {
  return idx.clamp(0, size - 1);
}

/// Wraps an index circularly within [0, size).
///
/// Uses modulo arithmetic to wrap around.
/// Used by circular (wrap) padding mode.
///
/// Example for size=5:
/// - idx=-1 → 4
/// - idx=5 → 0
/// - idx=7 → 2
int circularIndex(int idx, int size) {
  return ((idx % size) + size) % size;
}
