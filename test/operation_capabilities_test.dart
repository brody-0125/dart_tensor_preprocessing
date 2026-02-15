import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

void main() {
  group('OperationCapabilities', () {
    test('new fields have sensible defaults', () {
      const caps = OperationCapabilities();

      expect(caps.supportsBroadcast, isFalse);
      expect(caps.supportedDTypes, equals({DType.float32, DType.float64}));
      expect(caps.pytorchEquivalent, isNull);
      expect(caps.onnxOpType, isNull);
      expect(caps.onnxOpsetVersion, isNull);
    });

    test('new fields can be set via constructor', () {
      const caps = OperationCapabilities(
        supportsBroadcast: true,
        supportedDTypes: {DType.float32},
        pytorchEquivalent: 'torch.nn.ReLU',
        onnxOpType: 'Relu',
        onnxOpsetVersion: 13,
      );

      expect(caps.supportsBroadcast, isTrue);
      expect(caps.supportedDTypes, equals({DType.float32}));
      expect(caps.pytorchEquivalent, equals('torch.nn.ReLU'));
      expect(caps.onnxOpType, equals('Relu'));
      expect(caps.onnxOpsetVersion, equals(13));
    });

    test('defaultCapabilities has correct new field values', () {
      const caps = OperationCapabilities.defaultCapabilities;

      expect(caps.supportsBroadcast, isFalse);
      expect(caps.supportedDTypes, equals({DType.float32, DType.float64}));
      expect(caps.pytorchEquivalent, isNull);
      expect(caps.onnxOpType, isNull);
      expect(caps.onnxOpsetVersion, isNull);
    });

    test('existing fields still work unchanged', () {
      const caps = OperationCapabilities(
        supportsInPlace: true,
        requiresContiguous: true,
        preservesShape: false,
        modifiesDType: true,
      );

      expect(caps.supportsInPlace, isTrue);
      expect(caps.requiresContiguous, isTrue);
      expect(caps.preservesShape, isFalse);
      expect(caps.modifiesDType, isTrue);
    });

    test('toString includes new fields', () {
      const caps = OperationCapabilities(
        pytorchEquivalent: 'F.relu',
        onnxOpType: 'Relu',
      );
      final str = caps.toString();

      expect(str, contains('pytorch: F.relu'));
      expect(str, contains('onnx: Relu'));
    });

    test('existing operations capabilities are not broken', () {
      // ReLUOp should still report its capabilities correctly
      final relu = ReLUOp();
      expect(relu.capabilities.supportsInPlace, isTrue);
      expect(relu.capabilities.preservesShape, isTrue);

      // ResizeOp should still report its capabilities
      final resize = ResizeOp(height: 224, width: 224);
      expect(resize.capabilities.requiresContiguous, isTrue);
      expect(resize.capabilities.preservesShape, isFalse);
    });
  });
}
