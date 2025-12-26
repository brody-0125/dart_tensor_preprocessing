/// DType and type conversion tests based on PyTorch test_type_promotion.py patterns
///
/// Tests cover:
/// - DType properties and buffer creation
/// - Type casting operations
/// - Overflow/underflow handling
/// - Type inference from TypedData
import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('DType Properties', () {
    test('all dtypes have correct byte sizes', () {
      expect(DType.float32.byteSize, equals(4));
      expect(DType.float64.byteSize, equals(8));
      expect(DType.int8.byteSize, equals(1));
      expect(DType.int16.byteSize, equals(2));
      expect(DType.int32.byteSize, equals(4));
      expect(DType.int64.byteSize, equals(8));
      expect(DType.uint8.byteSize, equals(1));
      expect(DType.uint16.byteSize, equals(2));
      expect(DType.uint32.byteSize, equals(4));
      expect(DType.uint64.byteSize, equals(8));
    });

    test('all dtypes have unique ONNX IDs', () {
      final ids = DType.values.map((d) => d.onnxId).toSet();
      expect(ids.length, equals(DType.values.length));
    });

    test('isFloatingPoint correctly identifies float types', () {
      expect(DType.float32.isFloatingPoint, isTrue);
      expect(DType.float64.isFloatingPoint, isTrue);

      expect(DType.int8.isFloatingPoint, isFalse);
      expect(DType.int16.isFloatingPoint, isFalse);
      expect(DType.int32.isFloatingPoint, isFalse);
      expect(DType.int64.isFloatingPoint, isFalse);
      expect(DType.uint8.isFloatingPoint, isFalse);
      expect(DType.uint16.isFloatingPoint, isFalse);
      expect(DType.uint32.isFloatingPoint, isFalse);
      expect(DType.uint64.isFloatingPoint, isFalse);
    });

    test('isInteger correctly identifies integer types', () {
      expect(DType.float32.isInteger, isFalse);
      expect(DType.float64.isInteger, isFalse);

      expect(DType.int8.isInteger, isTrue);
      expect(DType.int16.isInteger, isTrue);
      expect(DType.int32.isInteger, isTrue);
      expect(DType.int64.isInteger, isTrue);
      expect(DType.uint8.isInteger, isTrue);
      expect(DType.uint16.isInteger, isTrue);
      expect(DType.uint32.isInteger, isTrue);
      expect(DType.uint64.isInteger, isTrue);
    });

    test('isSigned correctly identifies signed types', () {
      expect(DType.float32.isSigned, isTrue);
      expect(DType.float64.isSigned, isTrue);
      expect(DType.int8.isSigned, isTrue);
      expect(DType.int16.isSigned, isTrue);
      expect(DType.int32.isSigned, isTrue);
      expect(DType.int64.isSigned, isTrue);

      expect(DType.uint8.isSigned, isFalse);
      expect(DType.uint16.isSigned, isFalse);
      expect(DType.uint32.isSigned, isFalse);
      expect(DType.uint64.isSigned, isFalse);
    });
  });

  group('DType Buffer Creation', () {
    test('createBuffer returns correct TypedData type', () {
      expect(DType.float32.createBuffer(10), isA<Float32List>());
      expect(DType.float64.createBuffer(10), isA<Float64List>());
      expect(DType.int8.createBuffer(10), isA<Int8List>());
      expect(DType.int16.createBuffer(10), isA<Int16List>());
      expect(DType.int32.createBuffer(10), isA<Int32List>());
      expect(DType.int64.createBuffer(10), isA<Int64List>());
      expect(DType.uint8.createBuffer(10), isA<Uint8List>());
      expect(DType.uint16.createBuffer(10), isA<Uint16List>());
      expect(DType.uint32.createBuffer(10), isA<Uint32List>());
      expect(DType.uint64.createBuffer(10), isA<Uint64List>());
    });

    test('createBuffer creates correct length', () {
      for (final dtype in DType.values) {
        final buffer = dtype.createBuffer(100);
        expect(_getTypedDataLength(buffer), equals(100));
      }
    });

    test('createBuffer initializes to zero', () {
      for (final dtype in DType.values) {
        final buffer = dtype.createBuffer(10);
        for (int i = 0; i < 10; i++) {
          expect(_getTypedDataValue(buffer, i), equals(0));
        }
      }
    });
  });

  group('DType Inference', () {
    test('fromTypedData correctly infers all types', () {
      expect(DType.fromTypedData(Float32List(1)), equals(DType.float32));
      expect(DType.fromTypedData(Float64List(1)), equals(DType.float64));
      expect(DType.fromTypedData(Int8List(1)), equals(DType.int8));
      expect(DType.fromTypedData(Int16List(1)), equals(DType.int16));
      expect(DType.fromTypedData(Int32List(1)), equals(DType.int32));
      expect(DType.fromTypedData(Int64List(1)), equals(DType.int64));
      expect(DType.fromTypedData(Uint8List(1)), equals(DType.uint8));
      expect(DType.fromTypedData(Uint16List(1)), equals(DType.uint16));
      expect(DType.fromTypedData(Uint32List(1)), equals(DType.uint32));
      expect(DType.fromTypedData(Uint64List(1)), equals(DType.uint64));
    });

    test('fromTypedData throws on ByteData', () {
      expect(
        () => DType.fromTypedData(ByteData(10)),
        throwsArgumentError,
      );
    });
  });

  group('TypeCastOp', () {
    group('float conversions', () {
      test('uint8 to float32', () {
        final data = Uint8List.fromList([0, 128, 255]);
        final tensor = TensorBuffer.fromUint8List(data, [3]);

        final cast = TypeCastOp.toFloat32();
        final result = cast(tensor);

        expect(result.dtype, equals(DType.float32));
        expect(result[[0]], equals(0.0));
        expect(result[[1]], equals(128.0));
        expect(result[[2]], equals(255.0));
      });

      test('float32 to float64', () {
        final data = Float32List.fromList([1.5, 2.5, 3.5]);
        final tensor = TensorBuffer.fromFloat32List(data, [3]);

        final cast = TypeCastOp.toFloat64();
        final result = cast(tensor);

        expect(result.dtype, equals(DType.float64));
        expect(result[[0]], closeTo(1.5, 0.001));
        expect(result[[1]], closeTo(2.5, 0.001));
        expect(result[[2]], closeTo(3.5, 0.001));
      });

      test('float64 to float32 (precision loss)', () {
        final data = Float64List.fromList([1.123456789012345]);
        final tensor = TensorBuffer(
          storage: TensorStorage(data, DType.float64),
          shape: [1],
        );

        final cast = TypeCastOp.toFloat32();
        final result = cast(tensor);

        expect(result.dtype, equals(DType.float32));
        // Float32 has less precision
        expect(result[[0]], closeTo(1.1234568, 0.0000001));
      });
    });

    group('integer conversions', () {
      test('float32 to int32 (truncation)', () {
        final data = Float32List.fromList([1.9, -2.1, 3.5, -4.9]);
        final tensor = TensorBuffer.fromFloat32List(data, [4]);

        final cast = TypeCastOp.toInt32();
        final result = cast(tensor);

        expect(result.dtype, equals(DType.int32));
        expect(result[[0]], equals(2.0)); // rounded
        expect(result[[1]], equals(-2.0)); // rounded
        expect(result[[2]], equals(4.0)); // rounded
        expect(result[[3]], equals(-5.0)); // rounded
      });

      test('int32 to int64', () {
        final data = Int32List.fromList([2147483647, -2147483648]);
        final tensor = TensorBuffer(
          storage: TensorStorage(data, DType.int32),
          shape: [2],
        );

        final cast = TypeCastOp.toInt64();
        final result = cast(tensor);

        expect(result.dtype, equals(DType.int64));
        expect(result[[0]], equals(2147483647.0));
        expect(result[[1]], equals(-2147483648.0));
      });
    });

    group('clamping conversions', () {
      test('float32 to uint8 clamps to [0, 255]', () {
        final data = Float32List.fromList([-100.0, 0.0, 128.0, 255.0, 500.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [5]);

        final cast = TypeCastOp.toUint8();
        final result = cast(tensor);

        expect(result.dtype, equals(DType.uint8));
        expect(result[[0]], equals(0.0)); // clamped from -100
        expect(result[[1]], equals(0.0));
        expect(result[[2]], equals(128.0));
        expect(result[[3]], equals(255.0));
        expect(result[[4]], equals(255.0)); // clamped from 500
      });

      test('large float to int8 clamps to [-128, 127]', () {
        final data = Float32List.fromList([-1000.0, 0.0, 127.0, 1000.0]);
        final tensor = TensorBuffer.fromFloat32List(data, [4]);

        final cast = TypeCastOp(DType.int8);
        final result = cast(tensor);

        expect(result.dtype, equals(DType.int8));
        expect(result[[0]], equals(-128.0)); // clamped
        expect(result[[1]], equals(0.0));
        expect(result[[2]], equals(127.0));
        expect(result[[3]], equals(127.0)); // clamped
      });
    });

    group('same type casting', () {
      test('casting to same type returns input', () {
        final tensor = TensorBuffer.zeros([2, 3], dtype: DType.float32);
        final cast = TypeCastOp.toFloat32();
        final result = cast(tensor);

        expect(identical(result, tensor), isTrue);
      });
    });

    group('preserves shape', () {
      test('type cast preserves tensor shape', () {
        final tensor = TensorBuffer.zeros([2, 3, 4], dtype: DType.float32);
        final cast = TypeCastOp.toFloat64();
        final result = cast(tensor);

        expect(result.shape, equals([2, 3, 4]));
      });
    });
  });

  group('TensorStorage Type Handling', () {
    test('storage validates dtype matches data', () {
      final float32Data = Float32List(10);

      expect(
        () => TensorStorage(float32Data, DType.float64),
        throwsArgumentError,
      );
    });

    test('getAsDouble works for all numeric types', () {
      for (final dtype in DType.values) {
        final buffer = dtype.createBuffer(1);
        _setTypedDataValue(buffer, 0, 42);
        final storage = TensorStorage(buffer, dtype);

        expect(storage.getAsDouble(0), equals(42.0));
      }
    });

    test('setFromDouble works for all numeric types', () {
      for (final dtype in DType.values) {
        final buffer = dtype.createBuffer(1);
        final storage = TensorStorage(buffer, dtype);

        storage.setFromDouble(0, 42.5);

        if (dtype.isFloatingPoint) {
          expect(storage.getAsDouble(0), closeTo(42.5, 0.001));
        } else {
          // Integer types truncate/round
          expect(storage.getAsDouble(0), anyOf(equals(42.0), equals(43.0)));
        }
      }
    });

    test('uint8 storage clamps values', () {
      final buffer = Uint8List(3);
      final storage = TensorStorage(buffer, DType.uint8);

      storage.setFromDouble(0, -10.0);
      storage.setFromDouble(1, 128.0);
      storage.setFromDouble(2, 300.0);

      expect(storage.getAsDouble(0), equals(0.0));
      expect(storage.getAsDouble(1), equals(128.0));
      expect(storage.getAsDouble(2), equals(255.0));
    });

    test('uint16 storage clamps values', () {
      final buffer = Uint16List(2);
      final storage = TensorStorage(buffer, DType.uint16);

      storage.setFromDouble(0, -10.0);
      storage.setFromDouble(1, 70000.0);

      expect(storage.getAsDouble(0), equals(0.0));
      expect(storage.getAsDouble(1), equals(65535.0));
    });
  });

  group('ToTensorOp Type Conversion', () {
    test('converts uint8 HWC to float32 CHW', () {
      final data = Uint8List.fromList([255, 0, 128, 0, 255, 64]);
      final hwc = TensorBuffer.fromUint8List(data, [1, 2, 3]);

      final op = ToTensorOp(normalize: true);
      final result = op(hwc);

      expect(result.dtype, equals(DType.float32));
      expect(result.shape, equals([3, 1, 2]));
    });

    test('normalizes to [0, 1] range', () {
      final data = Uint8List.fromList([0, 128, 255]);
      final hwc = TensorBuffer.fromUint8List(data, [1, 1, 3]);

      final op = ToTensorOp(normalize: true);
      final result = op(hwc);

      expect(result[[0, 0, 0]], closeTo(0.0, 0.01));
      expect(result[[1, 0, 0]], closeTo(0.502, 0.01));
      expect(result[[2, 0, 0]], closeTo(1.0, 0.01));
    });

    test('without normalization preserves values', () {
      final data = Uint8List.fromList([0, 128, 255]);
      final hwc = TensorBuffer.fromUint8List(data, [1, 1, 3]);

      final op = ToTensorOp(normalize: false);
      final result = op(hwc);

      expect(result[[0, 0, 0]], equals(0.0));
      expect(result[[1, 0, 0]], equals(128.0));
      expect(result[[2, 0, 0]], equals(255.0));
    });
  });

  group('ToImageOp Type Conversion', () {
    test('converts float32 CHW to uint8 HWC', () {
      final data = Float32List.fromList([1.0, 0.0, 0.5, 0.0, 1.0, 0.25]);
      final chw = TensorBuffer.fromFloat32List(data, [3, 1, 2]);

      final op = ToImageOp(denormalize: true);
      final result = op(chw);

      expect(result.dtype, equals(DType.uint8));
      expect(result.shape, equals([1, 2, 3]));
    });

    test('denormalizes from [0, 1] to [0, 255]', () {
      final data = Float32List.fromList([0.0, 0.5, 1.0]);
      final chw = TensorBuffer.fromFloat32List(data, [3, 1, 1]);

      final op = ToImageOp(denormalize: true);
      final result = op(chw);

      expect(result[[0, 0, 0]], equals(0.0));
      expect(result[[0, 0, 1]], equals(128.0));
      expect(result[[0, 0, 2]], equals(255.0));
    });

    test('clamps out-of-range values', () {
      final data = Float32List.fromList([-0.5, 1.5, 0.5]);
      final chw = TensorBuffer.fromFloat32List(data, [3, 1, 1]);

      final op = ToImageOp(denormalize: true);
      final result = op(chw);

      expect(result[[0, 0, 0]], equals(0.0)); // clamped from -127.5
      expect(result[[0, 0, 1]], equals(255.0)); // clamped from 382.5
      expect(result[[0, 0, 2]], equals(128.0));
    });
  });
}

// Helper functions
int _getTypedDataLength(TypedData data) {
  return switch (data) {
    Float32List list => list.length,
    Float64List list => list.length,
    Int8List list => list.length,
    Int16List list => list.length,
    Int32List list => list.length,
    Int64List list => list.length,
    Uint8List list => list.length,
    Uint16List list => list.length,
    Uint32List list => list.length,
    Uint64List list => list.length,
    _ => throw StateError('Unknown TypedData type'),
  };
}

num _getTypedDataValue(TypedData data, int index) {
  return switch (data) {
    Float32List list => list[index],
    Float64List list => list[index],
    Int8List list => list[index],
    Int16List list => list[index],
    Int32List list => list[index],
    Int64List list => list[index],
    Uint8List list => list[index],
    Uint16List list => list[index],
    Uint32List list => list[index],
    Uint64List list => list[index],
    _ => throw StateError('Unknown TypedData type'),
  };
}

void _setTypedDataValue(TypedData data, int index, num value) {
  switch (data) {
    case Float32List list:
      list[index] = value.toDouble();
    case Float64List list:
      list[index] = value.toDouble();
    case Int8List list:
      list[index] = value.toInt();
    case Int16List list:
      list[index] = value.toInt();
    case Int32List list:
      list[index] = value.toInt();
    case Int64List list:
      list[index] = value.toInt();
    case Uint8List list:
      list[index] = value.toInt();
    case Uint16List list:
      list[index] = value.toInt();
    case Uint32List list:
      list[index] = value.toInt();
    case Uint64List list:
      list[index] = value.toInt();
  }
}
