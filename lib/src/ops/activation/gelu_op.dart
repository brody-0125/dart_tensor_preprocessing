import 'dart:math' as math;
import 'dart:typed_data';

import '../../core/tensor_buffer.dart';
import '../../exceptions/tensor_exceptions.dart';
import '../../utils/dtype_dispatcher.dart';
import '../transform_op.dart';

/// Gaussian Error Linear Unit (GELU) activation function.
///
/// Standard activation in Transformers (BERT, GPT, ViT).
/// Equivalent to `F.gelu()` in PyTorch.
///
/// Supports two modes:
/// - `approximate: 'none'` (default): Exact GELU using error function
/// - `approximate: 'tanh'`: Fast approximation (PyTorch default)
///
/// ## Formulas
///
/// - **Exact**: `x * Φ(x)` where Φ is the standard normal CDF
/// - **Tanh approx**: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// ```dart
/// final result = GELUOp()(tensor);  // Exact
/// final result = GELUOp(approximate: 'tanh')(tensor);  // Fast
/// ```
class GELUOp extends TransformOp with InPlaceTransform, RequiresContiguous {
  /// Approximation method: 'none' (exact) or 'tanh' (fast).
  final String approximate;

  /// Creates a GELU operation.
  ///
  /// [approximate] can be 'none' for exact computation or 'tanh' for
  /// the fast approximation used by PyTorch.
  GELUOp({this.approximate = 'none'}) {
    if (approximate != 'none' && approximate != 'tanh') {
      throw InvalidParameterException(
        'approximate',
        approximate,
        "must be 'none' or 'tanh'",
      );
    }
  }

  @override
  String get name =>
      approximate == 'none' ? 'GELU' : 'GELU(approximate=$approximate)';

  @override
  OperationCapabilities get capabilities => const OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
      );

  @override
  TensorBuffer apply(TensorBuffer input) {
    final output = cloneForModification(input);
    _gelu(output);
    return output;
  }

  @override
  void applyInPlace(TensorBuffer input) {
    if (!input.isContiguous) {
      throw const NonContiguousException('GELUOp.applyInPlace');
    }
    _gelu(input);
  }

  // Constants for GELU computation
  static const double _sqrt2 = 1.4142135623730951;
  static const double _sqrt2OverPi = 0.7978845608028654; // sqrt(2/π)
  static const double _tanhCoeff = 0.044715;

  void _gelu(TensorBuffer tensor) {
    final useTanh = approximate == 'tanh';

    DTypeDispatcher.dispatchVoid(
      tensor,
      onFloat32: (list, numel) {
        if (useTanh) {
          _geluTanhFloat32(list, numel);
        } else {
          _geluExactFloat32(list, numel);
        }
      },
      onFloat64: (list, numel) {
        if (useTanh) {
          _geluTanhFloat64(list, numel);
        } else {
          _geluExactFloat64(list, numel);
        }
      },
      fallback: (t) {
        final n = t.numel;
        if (useTanh) {
          for (int i = 0; i < n; i++) {
            final x = t.storage.getAsDouble(i);
            final inner = _sqrt2OverPi * (x + _tanhCoeff * x * x * x);
            final tanhVal = _tanh(inner);
            t.storage.setFromDouble(i, 0.5 * x * (1.0 + tanhVal));
          }
        } else {
          for (int i = 0; i < n; i++) {
            final x = t.storage.getAsDouble(i);
            t.storage.setFromDouble(i, x * 0.5 * (1.0 + _erf(x / _sqrt2)));
          }
        }
      },
    );
  }

  void _geluExactFloat32(Float32List list, int numel) {
    for (int i = 0; i < numel; i++) {
      final x = list[i];
      list[i] = x * 0.5 * (1.0 + _erf(x / _sqrt2));
    }
  }

  void _geluExactFloat64(Float64List list, int numel) {
    for (int i = 0; i < numel; i++) {
      final x = list[i];
      list[i] = x * 0.5 * (1.0 + _erf(x / _sqrt2));
    }
  }

  void _geluTanhFloat32(Float32List list, int numel) {
    for (int i = 0; i < numel; i++) {
      final x = list[i];
      final inner = _sqrt2OverPi * (x + _tanhCoeff * x * x * x);
      list[i] = 0.5 * x * (1.0 + _tanh(inner));
    }
  }

  void _geluTanhFloat64(Float64List list, int numel) {
    for (int i = 0; i < numel; i++) {
      final x = list[i];
      final inner = _sqrt2OverPi * (x + _tanhCoeff * x * x * x);
      list[i] = 0.5 * x * (1.0 + _tanh(inner));
    }
  }

  /// Approximation of the error function using Abramowitz and Stegun formula.
  static double _erf(double x) {
    // Constants for approximation
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    final sign = x < 0 ? -1.0 : 1.0;
    final absX = x.abs();

    final t = 1.0 / (1.0 + p * absX);
    final t2 = t * t;
    final t3 = t2 * t;
    final t4 = t3 * t;
    final t5 = t4 * t;

    final y =
        1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * math.exp(-absX * absX);

    return sign * y;
  }

  /// Fast tanh implementation.
  static double _tanh(double x) {
    if (x > 20) return 1.0;
    if (x < -20) return -1.0;
    final exp2x = math.exp(2 * x);
    return (exp2x - 1) / (exp2x + 1);
  }

  @override
  List<int> computeOutputShape(List<int> inputShape) => inputShape;
}
