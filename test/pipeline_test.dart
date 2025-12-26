import 'dart:typed_data';

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('TransformOp', () {
    group('NormalizeOp', () {
      test('normalizes with ImageNet values', () {
        // Create 3-channel tensor with known values
        final data = Float32List(3 * 2 * 2);
        // Fill with 0.5 (after [0,1] scaling)
        for (int i = 0; i < data.length; i++) {
          data[i] = 0.5;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [3, 2, 2]);

        final normalize = NormalizeOp.imagenet();
        final result = normalize(tensor);

        expect(result.shape, equals([3, 2, 2]));
        expect(result.dtype, equals(DType.float32));

        // Check channel 0: (0.5 - 0.485) / 0.229 ≈ 0.0655
        expect(result[[0, 0, 0]], closeTo(0.0655, 0.01));
      });

      test('symmetric normalization', () {
        final data = Float32List.fromList([0.0, 0.5, 1.0, 0.25, 0.75, 0.5]);
        final tensor = TensorBuffer.fromFloat32List(data, [3, 1, 2]);

        final normalize = NormalizeOp.symmetric();
        final result = normalize(tensor);

        // (0.0 - 0.5) / 0.5 = -1.0
        expect(result[[0, 0, 0]], closeTo(-1.0, 0.001));
        // (1.0 - 0.5) / 0.5 = 1.0
        expect(result[[0, 0, 1]], closeTo(0.0, 0.001));
      });
    });

    group('ScaleOp', () {
      test('scales to unit range', () {
        final data = Float32List.fromList([0, 127.5, 255]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 1, 3]);

        final scale = ScaleOp.toUnit();
        final result = scale(tensor);

        expect(result[[0, 0, 0]], closeTo(0.0, 0.001));
        expect(result[[0, 0, 1]], closeTo(0.5, 0.001));
        expect(result[[0, 0, 2]], closeTo(1.0, 0.001));
      });
    });

    group('PermuteOp', () {
      test('permutes NCHW to NHWC', () {
        final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
        final permute = PermuteOp.nchwToNhwc();
        final result = permute(tensor);

        expect(result.shape, equals([1, 224, 224, 3]));
      });

      test('permutes CHW to HWC', () {
        final tensor = TensorBuffer.zeros([3, 224, 224]);
        final permute = PermuteOp.chwToHwc();
        final result = permute(tensor);

        expect(result.shape, equals([224, 224, 3]));
      });
    });

    group('UnsqueezeOp', () {
      test('adds batch dimension', () {
        final tensor = TensorBuffer.zeros([3, 224, 224]);
        final unsqueeze = UnsqueezeOp.batch();
        final result = unsqueeze(tensor);

        expect(result.shape, equals([1, 3, 224, 224]));
      });
    });

    group('SqueezeOp', () {
      test('removes batch dimension', () {
        final tensor = TensorBuffer.zeros([1, 3, 224, 224]);
        final squeeze = SqueezeOp.batch();
        final result = squeeze(tensor);

        expect(result.shape, equals([3, 224, 224]));
      });

      test('removes all size-1 dimensions', () {
        final tensor = TensorBuffer.zeros([1, 3, 1, 224, 1, 224]);
        final squeeze = SqueezeOp.all();
        final result = squeeze(tensor);

        expect(result.shape, equals([3, 224, 224]));
      });
    });

    group('ReshapeOp', () {
      test('reshapes with explicit dimensions', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final reshape = ReshapeOp([6, 4]);
        final result = reshape(tensor);

        expect(result.shape, equals([6, 4]));
      });

      test('reshapes with -1 dimension', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);
        final reshape = ReshapeOp([2, -1]);
        final result = reshape(tensor);

        expect(result.shape, equals([2, 12]));
      });
    });

    group('FlattenOp', () {
      test('flattens from startDim', () {
        final tensor = TensorBuffer.zeros([2, 3, 4, 5]);
        final flatten = FlattenOp(startDim: 1);
        final result = flatten(tensor);

        expect(result.shape, equals([2, 60]));
      });
    });

    group('TypeCastOp', () {
      test('casts to float32', () {
        final data = Uint8List.fromList([0, 128, 255]);
        final tensor = TensorBuffer.fromUint8List(data, [3]);

        final cast = TypeCastOp.toFloat32();
        final result = cast(tensor);

        expect(result.dtype, equals(DType.float32));
        expect(result[[0]], equals(0.0));
        expect(result[[1]], equals(128.0));
        expect(result[[2]], equals(255.0));
      });
    });

    group('ToTensorOp', () {
      test('converts HWC uint8 to CHW float32', () {
        // Create a 2x2 RGB image
        final data = Uint8List.fromList([
          255, 0, 0, // Red pixel
          0, 255, 0, // Green pixel
          0, 0, 255, // Blue pixel
          128, 128, 128, // Gray pixel
        ]);
        final hwc = TensorBuffer.fromUint8List(data, [2, 2, 3]);

        final toTensor = ToTensorOp(normalize: true);
        final result = toTensor(hwc);

        expect(result.shape, equals([3, 2, 2]));
        expect(result.dtype, equals(DType.float32));

        // Check R channel values
        expect(result[[0, 0, 0]], closeTo(1.0, 0.01)); // Red pixel R
        expect(result[[0, 0, 1]], closeTo(0.0, 0.01)); // Green pixel R
      });
    });
  });

  group('TensorPipeline', () {
    test('chains multiple operations', () {
      final pipeline = TensorPipeline([
        UnsqueezeOp.batch(),
        SqueezeOp.batch(),
      ]);

      final tensor = TensorBuffer.zeros([3, 224, 224]);
      final result = pipeline.run(tensor);

      expect(result.shape, equals([3, 224, 224]));
    });

    test('computes output shape', () {
      final pipeline = TensorPipeline([
        UnsqueezeOp.batch(),
        ReshapeOp([1, 3, -1]),
      ]);

      final outputShape = pipeline.computeOutputShape([3, 224, 224]);
      expect(outputShape, equals([1, 3, 50176]));
    });

    test('validates pipeline', () {
      final pipeline = TensorPipeline([
        NormalizeOp.imagenet(),
        UnsqueezeOp.batch(),
      ]);

      expect(pipeline.validate([3, 224, 224]), isTrue);
      // Note: pipeline with NormalizeOp requires 3 or 4 channels
      // [224, 224] doesn't work for NormalizeOp.imagenet() which expects 3 channels
    });

    test('supports callable syntax', () {
      final pipeline = TensorPipeline([IdentityOp()]);
      final tensor = TensorBuffer.zeros([2, 3]);

      final result = pipeline(tensor);
      expect(result.shape, equals([2, 3]));
    });

    test('supports append operator', () {
      final pipeline = TensorPipeline([IdentityOp()]);
      final extended = pipeline >> UnsqueezeOp.batch();

      expect(extended.length, equals(2));
    });

    test('supports concat operator', () {
      final p1 = TensorPipeline([IdentityOp()]);
      final p2 = TensorPipeline([UnsqueezeOp.batch()]);
      final combined = p1 + p2;

      expect(combined.length, equals(2));
    });
  });

  group('PipelinePresets', () {
    test('minimal preset creates valid pipeline', () {
      final pipeline = PipelinePresets.minimal(height: 224, width: 224);
      // Note: minimal preset expects HWC input for ToTensorOp
      // For this test, we'll just verify the pipeline structure
      expect(pipeline.operations.length, greaterThan(0));
    });

    test('imagenetClassification creates pipeline with correct structure', () {
      final pipeline = PipelinePresets.imagenetClassification();

      // Verify pipeline has expected operations
      expect(pipeline.operations.length, equals(5));
      expect(pipeline.operations[0], isA<ResizeShortestOp>());
      expect(pipeline.operations[1], isA<CenterCropOp>());
      expect(pipeline.operations[2], isA<ToTensorOp>());
      expect(pipeline.operations[3], isA<NormalizeOp>());
      expect(pipeline.operations[4], isA<UnsqueezeOp>());
    });

    test('objectDetection creates pipeline with correct structure', () {
      final pipeline = PipelinePresets.objectDetection();

      expect(pipeline.operations.length, equals(3));
      expect(pipeline.operations[0], isA<ResizeOp>());
      expect(pipeline.operations[1], isA<ToTensorOp>());
      expect(pipeline.operations[2], isA<UnsqueezeOp>());
    });

    test('custom pipeline allows flexible configuration', () {
      final pipeline = PipelinePresets.custom(
        height: 512,
        width: 512,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
      );

      expect(pipeline.operations.length, greaterThan(0));
      expect(pipeline.name, equals('Custom'));
    });
  });

  group('Exceptions', () {
    test('ShapeMismatchException provides details', () {
      final exception = ShapeMismatchException(
        expected: [1, 3, 224, 224],
        actual: [3, 224, 224],
      );

      expect(exception.message, contains('expected'));
      expect(exception.message, contains('got'));
    });

    test('EmptyPipelineException on empty operations', () {
      expect(
        () => TensorPipeline([]),
        throwsA(isA<EmptyPipelineException>()),
      );
    });

    test('InvalidParameterException provides details', () {
      final exception = InvalidParameterException('size', -1, 'Must be positive');

      expect(exception.message, contains('size'));
      expect(exception.message, contains('-1'));
      expect(exception.message, contains('Must be positive'));
    });
  });
}
