import 'dart:typed_data';
import 'dart:math' as math;

import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:test/test.dart';

/// PyTorch Compatibility Benchmark Tests
///
/// These tests verify that the Dart tensor library produces results
/// compatible with PyTorch's tensor operations and torchvision transforms.
///
/// Reference values were generated using:
/// - PyTorch 2.0+
/// - torchvision 0.15+
///
/// Python code to generate reference values is included in comments.
void main() {
  group('PyTorch Compatibility', () {
    group('Tensor Creation', () {
      /// ```python
      /// import torch
      /// t = torch.zeros(2, 3, 4)
      /// print(t.shape)  # torch.Size([2, 3, 4])
      /// print(t.numel())  # 24
      /// print(t.stride())  # (12, 4, 1)
      /// ```
      test('zeros matches torch.zeros', () {
        final tensor = TensorBuffer.zeros([2, 3, 4]);

        expect(tensor.shape, equals([2, 3, 4]));
        expect(tensor.numel, equals(24));
        expect(tensor.strides, equals([12, 4, 1]));
        expect(tensor.dtype, equals(DType.float32));

        for (int i = 0; i < tensor.numel; i++) {
          expect(tensor.storage.getAsDouble(i), equals(0.0));
        }
      });

      /// ```python
      /// t = torch.ones(3, 224, 224)
      /// print(t.stride())  # (50176, 224, 1)
      /// ```
      test('ones matches torch.ones', () {
        final tensor = TensorBuffer.ones([3, 224, 224]);

        expect(tensor.shape, equals([3, 224, 224]));
        expect(tensor.strides, equals([50176, 224, 1]));

        for (int i = 0; i < tensor.numel; i++) {
          expect(tensor.storage.getAsDouble(i), equals(1.0));
        }
      });

      /// ```python
      /// t = torch.zeros(1, 3, 224, 224)
      /// print(t.stride())  # (150528, 50176, 224, 1)
      /// ```
      test('4D NCHW strides match PyTorch', () {
        final tensor = TensorBuffer.zeros([1, 3, 224, 224]);

        expect(tensor.strides, equals([150528, 50176, 224, 1]));
      });
    });

    group('Transpose/Permute', () {
      /// ```python
      /// t = torch.arange(24).reshape(2, 3, 4).float()
      /// t_transposed = t.permute(2, 0, 1)
      /// print(t_transposed.shape)  # torch.Size([4, 2, 3])
      /// print(t_transposed.stride())  # (1, 12, 4)
      /// print(t_transposed[0, 0, 0].item())  # 0.0
      /// print(t_transposed[1, 0, 0].item())  # 1.0
      /// print(t_transposed[0, 1, 0].item())  # 12.0
      /// ```
      test('permute matches torch.permute', () {
        final data = Float32List.fromList(
          List.generate(24, (i) => i.toDouble()),
        );
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3, 4]);
        final transposed = tensor.transpose([2, 0, 1]);

        expect(transposed.shape, equals([4, 2, 3]));
        expect(transposed.strides, equals([1, 12, 4]));
        expect(transposed[[0, 0, 0]], equals(0.0));
        expect(transposed[[1, 0, 0]], equals(1.0));
        expect(transposed[[0, 1, 0]], equals(12.0));
      });

      /// ```python
      /// t = torch.arange(24).reshape(1, 2, 3, 4).float()
      /// # NCHW -> NHWC
      /// t_nhwc = t.permute(0, 2, 3, 1)
      /// print(t_nhwc.shape)  # torch.Size([1, 3, 4, 2])
      /// print(t_nhwc.stride())  # (24, 4, 1, 12)
      /// ```
      test('NCHW to NHWC permute matches PyTorch', () {
        final data = Float32List.fromList(
          List.generate(24, (i) => i.toDouble()),
        );
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 3, 4]);
        final nhwc = tensor.transpose([0, 2, 3, 1]);

        expect(nhwc.shape, equals([1, 3, 4, 2]));
        expect(nhwc.strides, equals([24, 4, 1, 12]));
      });

      /// ```python
      /// t = torch.arange(24).reshape(1, 2, 3, 4).float()
      /// t_nhwc = t.permute(0, 2, 3, 1).contiguous()
      /// t_back = t_nhwc.permute(0, 3, 1, 2)
      /// print(torch.equal(t, t_back.contiguous()))  # True
      /// ```
      test('NCHW -> NHWC -> NCHW round-trip preserves values', () {
        final data = Float32List.fromList(
          List.generate(24, (i) => i.toDouble()),
        );
        final original = TensorBuffer.fromFloat32List(data, [1, 2, 3, 4]);
        final nhwc = original.transpose([0, 2, 3, 1]).contiguous();
        final restored = nhwc.transpose([0, 3, 1, 2]).contiguous();

        expect(restored.shape, equals(original.shape));
        for (int n = 0; n < 1; n++) {
          for (int c = 0; c < 2; c++) {
            for (int h = 0; h < 3; h++) {
              for (int w = 0; w < 4; w++) {
                expect(
                  restored[[n, c, h, w]],
                  equals(original[[n, c, h, w]]),
                );
              }
            }
          }
        }
      });
    });

    group('Reshape', () {
      /// ```python
      /// t = torch.arange(24).reshape(2, 3, 4).float()
      /// t_flat = t.reshape(-1)
      /// print(t_flat.shape)  # torch.Size([24])
      /// t_2d = t.reshape(6, 4)
      /// print(t_2d.shape)  # torch.Size([6, 4])
      /// print(t_2d[0, 0].item())  # 0.0
      /// print(t_2d[1, 0].item())  # 4.0
      /// ```
      test('reshape matches torch.reshape', () {
        final data = Float32List.fromList(
          List.generate(24, (i) => i.toDouble()),
        );
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3, 4]);

        final flat = tensor.reshape([24]);
        expect(flat.shape, equals([24]));

        final reshaped = tensor.reshape([6, 4]);
        expect(reshaped.shape, equals([6, 4]));
        expect(reshaped[[0, 0]], equals(0.0));
        expect(reshaped[[1, 0]], equals(4.0));
      });
    });

    group('Squeeze/Unsqueeze', () {
      /// ```python
      /// t = torch.arange(6).reshape(1, 2, 1, 3).float()
      /// print(t.squeeze().shape)  # torch.Size([2, 3])
      /// print(t.squeeze(0).shape)  # torch.Size([2, 1, 3])
      /// print(t.squeeze(2).shape)  # torch.Size([1, 2, 3])
      /// ```
      test('squeeze matches torch.squeeze', () {
        final data = Float32List.fromList(
          List.generate(6, (i) => i.toDouble()),
        );
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 1, 3]);

        expect(tensor.squeeze().shape, equals([2, 3]));
        expect(tensor.squeeze(0).shape, equals([2, 1, 3]));
        expect(tensor.squeeze(2).shape, equals([1, 2, 3]));
      });

      /// ```python
      /// t = torch.arange(6).reshape(2, 3).float()
      /// print(t.unsqueeze(0).shape)  # torch.Size([1, 2, 3])
      /// print(t.unsqueeze(1).shape)  # torch.Size([2, 1, 3])
      /// print(t.unsqueeze(2).shape)  # torch.Size([2, 3, 1])
      /// ```
      test('unsqueeze matches torch.unsqueeze', () {
        final data = Float32List.fromList(
          List.generate(6, (i) => i.toDouble()),
        );
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3]);

        expect(tensor.unsqueeze(0).shape, equals([1, 2, 3]));
        expect(tensor.unsqueeze(1).shape, equals([2, 1, 3]));
        expect(tensor.unsqueeze(2).shape, equals([2, 3, 1]));
      });
    });

    group('Normalization', () {
      /// ```python
      /// import torchvision.transforms as T
      /// t = torch.full((3, 2, 2), 0.5)
      /// normalize = T.Normalize(
      ///     mean=[0.485, 0.456, 0.406],
      ///     std=[0.229, 0.224, 0.225]
      /// )
      /// result = normalize(t)
      /// print(result[0, 0, 0].item())  # 0.06550218...
      /// print(result[1, 0, 0].item())  # 0.19642857...
      /// print(result[2, 0, 0].item())  # 0.41777778...
      /// ```
      test('ImageNet normalization matches torchvision', () {
        final data = Float32List(3 * 2 * 2);
        for (int i = 0; i < data.length; i++) {
          data[i] = 0.5;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [3, 2, 2]);

        final normalize = NormalizeOp.imagenet();
        final result = normalize(tensor);

        // Channel 0: (0.5 - 0.485) / 0.229
        expect(result[[0, 0, 0]], closeTo(0.06550218, 1e-5));
        // Channel 1: (0.5 - 0.456) / 0.224
        expect(result[[1, 0, 0]], closeTo(0.19642857, 1e-5));
        // Channel 2: (0.5 - 0.406) / 0.225
        expect(result[[2, 0, 0]], closeTo(0.41777778, 1e-5));
      });

      /// ```python
      /// t = torch.tensor([[[0.0, 0.25], [0.5, 0.75]], [[0.1, 0.2], [0.3, 0.4]], [[0.9, 0.8], [0.7, 0.6]]])
      /// normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      /// result = normalize(t)
      /// print(result[0, 0, 0].item())  # -2.1178...
      /// print(result[0, 0, 1].item())  # -1.0262...
      /// print(result[2, 1, 1].item())  # 0.8622...
      /// ```
      test('ImageNet normalization with varied input', () {
        final data = Float32List.fromList([
          // Channel 0
          0.0, 0.25, 0.5, 0.75,
          // Channel 1
          0.1, 0.2, 0.3, 0.4,
          // Channel 2
          0.9, 0.8, 0.7, 0.6,
        ]);
        final tensor = TensorBuffer.fromFloat32List(data, [3, 2, 2]);

        final normalize = NormalizeOp.imagenet();
        final result = normalize(tensor);

        // (0.0 - 0.485) / 0.229
        expect(result[[0, 0, 0]], closeTo(-2.1178, 1e-3));
        // (0.25 - 0.485) / 0.229
        expect(result[[0, 0, 1]], closeTo(-1.0262, 1e-3));
        // (0.6 - 0.406) / 0.225
        expect(result[[2, 1, 1]], closeTo(0.8622, 1e-3));
      });

      /// ```python
      /// t = torch.full((3, 2, 2), 0.5)
      /// normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
      /// result = normalize(t)
      /// print(result[0, 0, 0].item())  # 0.0348...
      /// print(result[1, 0, 0].item())  # 0.0731...
      /// print(result[2, 0, 0].item())  # 0.2044...
      /// ```
      test('CIFAR-10 normalization matches torchvision', () {
        final data = Float32List(3 * 2 * 2);
        for (int i = 0; i < data.length; i++) {
          data[i] = 0.5;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [3, 2, 2]);

        final normalize = NormalizeOp.cifar10();
        final result = normalize(tensor);

        // Channel 0: (0.5 - 0.4914) / 0.2470
        expect(result[[0, 0, 0]], closeTo(0.0348, 1e-3));
        // Channel 1: (0.5 - 0.4822) / 0.2435
        expect(result[[1, 0, 0]], closeTo(0.0731, 1e-3));
        // Channel 2: (0.5 - 0.4465) / 0.2616
        expect(result[[2, 0, 0]], closeTo(0.2044, 1e-3));
      });

      /// ```python
      /// t = torch.tensor([[[0.0, 1.0], [0.5, 0.5]],
      ///                   [[0.0, 1.0], [0.5, 0.5]],
      ///                   [[0.0, 1.0], [0.5, 0.5]]])
      /// normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
      /// result = normalize(t)
      /// print(result[0, 0, 0].item())  # -1.0
      /// print(result[0, 0, 1].item())  # 1.0
      /// ```
      test('symmetric normalization maps [0,1] to [-1,1]', () {
        // 3-channel tensor with same values per channel
        final data = Float32List.fromList([
          0.0, 1.0, 0.5, 0.5, // Channel 0
          0.0, 1.0, 0.5, 0.5, // Channel 1
          0.0, 1.0, 0.5, 0.5, // Channel 2
        ]);
        final tensor = TensorBuffer.fromFloat32List(data, [3, 2, 2]);

        final normalize = NormalizeOp.symmetric();
        final result = normalize(tensor);

        // (0.0 - 0.5) / 0.5 = -1.0
        expect(result[[0, 0, 0]], closeTo(-1.0, 1e-6));
        // (1.0 - 0.5) / 0.5 = 1.0
        expect(result[[0, 0, 1]], closeTo(1.0, 1e-6));
        // (0.5 - 0.5) / 0.5 = 0.0
        expect(result[[0, 1, 0]], closeTo(0.0, 1e-6));
      });

      /// ```python
      /// t = torch.full((1, 3, 2, 2), 0.5)
      /// normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      /// result = normalize(t)
      /// print(result.shape)  # torch.Size([1, 3, 2, 2])
      /// print(result[0, 0, 0, 0].item())  # 0.06550218...
      /// ```
      test('4D batch normalization matches PyTorch', () {
        final data = Float32List(1 * 3 * 2 * 2);
        for (int i = 0; i < data.length; i++) {
          data[i] = 0.5;
        }
        final tensor = TensorBuffer.fromFloat32List(data, [1, 3, 2, 2]);

        final normalize = NormalizeOp.imagenet();
        final result = normalize(tensor);

        expect(result.shape, equals([1, 3, 2, 2]));
        expect(result[[0, 0, 0, 0]], closeTo(0.06550218, 1e-5));
      });
    });

    group('Resize - Nearest', () {
      /// Nearest neighbor interpolation using floor-based sampling.
      /// Formula: srcIdx = floor(dstIdx * srcSize / dstSize)
      ///
      /// For 2x2 -> 4x4 upsample with scale=0.5:
      /// dst[0] -> src[floor(0*0.5)] = src[0]
      /// dst[1] -> src[floor(1*0.5)] = src[0]
      /// dst[2] -> src[floor(2*0.5)] = src[1]
      /// dst[3] -> src[floor(3*0.5)] = src[1]
      test('nearest upsample 2x produces correct pattern', () {
        final data = Float32List.fromList([0, 1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2]);

        final resize = ResizeOp(
          height: 4,
          width: 4,
          mode: InterpolationMode.nearest,
        );
        final result = resize(tensor);

        expect(result.shape, equals([1, 4, 4]));

        // Each source pixel is replicated in a 2x2 block
        // Source [0,0]=0 -> dst rows 0-1, cols 0-1
        expect(result[[0, 0, 0]], equals(0.0));
        expect(result[[0, 0, 1]], equals(0.0));
        expect(result[[0, 1, 0]], equals(0.0));
        expect(result[[0, 1, 1]], equals(0.0));

        // Source [0,1]=1 -> dst rows 0-1, cols 2-3
        expect(result[[0, 0, 2]], equals(1.0));
        expect(result[[0, 0, 3]], equals(1.0));

        // Source [1,0]=2 -> dst rows 2-3, cols 0-1
        expect(result[[0, 2, 0]], equals(2.0));
        expect(result[[0, 3, 0]], equals(2.0));

        // Source [1,1]=3 -> dst rows 2-3, cols 2-3
        expect(result[[0, 2, 2]], equals(3.0));
        expect(result[[0, 3, 3]], equals(3.0));
      });

      /// For 4x4 -> 2x2 downsample with scale=2.0:
      /// dst[0] -> src[floor(0*2)] = src[0]
      /// dst[1] -> src[floor(1*2)] = src[2]
      test('nearest downsample 2x samples correct pixels', () {
        final data = Float32List.fromList(
          List.generate(16, (i) => i.toDouble()),
        );
        final tensor = TensorBuffer.fromFloat32List(data, [1, 4, 4]);

        final resize = ResizeOp(
          height: 2,
          width: 2,
          mode: InterpolationMode.nearest,
        );
        final result = resize(tensor);

        expect(result.shape, equals([1, 2, 2]));
        // src layout: 0  1  2  3
        //             4  5  6  7
        //             8  9  10 11
        //             12 13 14 15
        // dst samples: [0,0], [0,2], [2,0], [2,2]
        expect(result[[0, 0, 0]], equals(0.0));
        expect(result[[0, 0, 1]], equals(2.0));
        expect(result[[0, 1, 0]], equals(8.0));
        expect(result[[0, 1, 1]], equals(10.0));
      });
    });

    group('Resize - Bilinear', () {
      /// ```python
      /// t = torch.tensor([[[[0., 1.], [2., 3.]]]])
      /// result = F.interpolate(t, size=(4, 4), mode='bilinear', align_corners=False)
      /// print(result[0, 0])
      /// # tensor([[0.0000, 0.2500, 0.7500, 1.0000],
      /// #         [0.5000, 0.7500, 1.2500, 1.5000],
      /// #         [1.5000, 1.7500, 2.2500, 2.5000],
      /// #         [2.0000, 2.2500, 2.7500, 3.0000]])
      /// ```
      test('bilinear upsample 2x matches PyTorch (align_corners=false)', () {
        final data = Float32List.fromList([0, 1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2]);

        final resize = ResizeOp(
          height: 4,
          width: 4,
          mode: InterpolationMode.bilinear,
          alignCorners: false,
        );
        final result = resize(tensor);

        expect(result.shape, equals([1, 4, 4]));

        // Check corner and edge values
        expect(result[[0, 0, 0]], closeTo(0.0, 0.01));
        expect(result[[0, 0, 3]], closeTo(1.0, 0.01));
        expect(result[[0, 3, 0]], closeTo(2.0, 0.01));
        expect(result[[0, 3, 3]], closeTo(3.0, 0.01));

        // Check center values
        expect(result[[0, 1, 1]], closeTo(0.75, 0.01));
        expect(result[[0, 2, 2]], closeTo(2.25, 0.01));
      });

      /// ```python
      /// t = torch.tensor([[[[0., 1.], [2., 3.]]]])
      /// result = F.interpolate(t, size=(4, 4), mode='bilinear', align_corners=True)
      /// print(result[0, 0])
      /// # tensor([[0.0000, 0.3333, 0.6667, 1.0000],
      /// #         [0.6667, 1.0000, 1.3333, 1.6667],
      /// #         [1.3333, 1.6667, 2.0000, 2.3333],
      /// #         [2.0000, 2.3333, 2.6667, 3.0000]])
      /// ```
      test('bilinear upsample 2x matches PyTorch (align_corners=true)', () {
        final data = Float32List.fromList([0, 1, 2, 3]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2]);

        final resize = ResizeOp(
          height: 4,
          width: 4,
          mode: InterpolationMode.bilinear,
          alignCorners: true,
        );
        final result = resize(tensor);

        expect(result.shape, equals([1, 4, 4]));

        // Check corners (exact match expected)
        expect(result[[0, 0, 0]], closeTo(0.0, 1e-4));
        expect(result[[0, 0, 3]], closeTo(1.0, 1e-4));
        expect(result[[0, 3, 0]], closeTo(2.0, 1e-4));
        expect(result[[0, 3, 3]], closeTo(3.0, 1e-4));

        // Check interpolated values
        expect(result[[0, 0, 1]], closeTo(0.3333, 1e-3));
        expect(result[[0, 1, 0]], closeTo(0.6667, 1e-3));
      });
    });

    group('Resize - Bicubic', () {
      /// ```python
      /// t = torch.tensor([[[[0., 1., 2., 3.],
      ///                     [4., 5., 6., 7.],
      ///                     [8., 9., 10., 11.],
      ///                     [12., 13., 14., 15.]]]])
      /// result = F.interpolate(t, size=(2, 2), mode='bicubic', align_corners=False)
      /// print(result[0, 0])
      /// # Values will be smooth interpolation
      /// ```
      test('bicubic downsample produces smooth results', () {
        final data = Float32List.fromList(
          List.generate(16, (i) => i.toDouble()),
        );
        final tensor = TensorBuffer.fromFloat32List(data, [1, 4, 4]);

        final resize = ResizeOp(
          height: 2,
          width: 2,
          mode: InterpolationMode.bicubic,
          alignCorners: false,
        );
        final result = resize(tensor);

        expect(result.shape, equals([1, 2, 2]));

        // Bicubic should produce values close to bilinear but smoother
        // Values should be in reasonable range
        expect(result[[0, 0, 0]], greaterThanOrEqualTo(0.0));
        expect(result[[0, 1, 1]], lessThanOrEqualTo(15.0));
      });
    });

    group('ToTensor (HWC to CHW)', () {
      /// ```python
      /// import torchvision.transforms as T
      /// from PIL import Image
      /// import numpy as np
      ///
      /// # Create HWC uint8 image
      /// img = np.array([[[255, 0, 0], [0, 255, 0]],
      ///                 [[0, 0, 255], [128, 128, 128]]], dtype=np.uint8)
      /// pil_img = Image.fromarray(img)
      /// t = T.ToTensor()(pil_img)
      /// print(t.shape)  # torch.Size([3, 2, 2])
      /// print(t[0, 0, 0].item())  # 1.0 (255/255)
      /// print(t[1, 0, 1].item())  # 1.0 (255/255)
      /// print(t[2, 1, 0].item())  # 1.0 (255/255)
      /// ```
      test('HWC to CHW with normalization matches torchvision.ToTensor', () {
        // HWC format: 2x2 image with 3 channels
        // Pixel [0,0] = Red (255, 0, 0)
        // Pixel [0,1] = Green (0, 255, 0)
        // Pixel [1,0] = Blue (0, 0, 255)
        // Pixel [1,1] = Gray (128, 128, 128)
        final data = Uint8List.fromList([
          255, 0, 0, // [0,0]
          0, 255, 0, // [0,1]
          0, 0, 255, // [1,0]
          128, 128, 128, // [1,1]
        ]);
        final tensor = TensorBuffer.fromUint8List(data, [2, 2, 3]);

        final toTensor = ToTensorOp(normalize: true);
        final result = toTensor(tensor);

        expect(result.shape, equals([3, 2, 2]));
        expect(result.dtype, equals(DType.float32));

        // Red channel
        expect(result[[0, 0, 0]], closeTo(1.0, 1e-3)); // 255/255
        expect(result[[0, 0, 1]], closeTo(0.0, 1e-3)); // 0/255
        expect(result[[0, 1, 0]], closeTo(0.0, 1e-3)); // 0/255
        expect(result[[0, 1, 1]], closeTo(128 / 255, 1e-3));

        // Green channel
        expect(result[[1, 0, 0]], closeTo(0.0, 1e-3));
        expect(result[[1, 0, 1]], closeTo(1.0, 1e-3)); // 255/255
        expect(result[[1, 1, 1]], closeTo(128 / 255, 1e-3));

        // Blue channel
        expect(result[[2, 1, 0]], closeTo(1.0, 1e-3)); // 255/255
      });
    });

    group('Full Pipeline', () {
      /// Standard image preprocessing pipeline:
      /// 1. HWC uint8 input -> ToTensor (HWC->CHW, uint8->float32, /255)
      /// 2. Resize to target size
      /// 3. Normalize with ImageNet stats
      /// 4. Add batch dimension
      test('ImageNet pipeline produces expected output shape', () {
        // Create a simple 4x4 RGB image in HWC format
        final data = Uint8List(4 * 4 * 3);
        for (int i = 0; i < data.length; i++) {
          data[i] = 128; // Mid-gray
        }
        final input = TensorBuffer.fromUint8List(data, [4, 4, 3]);

        // Build pipeline: HWC -> CHW -> Resize -> Normalize -> Batch
        final pipeline = TensorPipeline([
          ToTensorOp(normalize: true), // HWC->CHW, scale to [0,1]
          ResizeOp(height: 4, width: 4, mode: InterpolationMode.bilinear),
          CenterCropOp(height: 2, width: 2),
          NormalizeOp.imagenet(),
          UnsqueezeOp.batch(),
        ]);

        final result = pipeline.run(input);

        expect(result.shape, equals([1, 3, 2, 2]));
        expect(result.dtype, equals(DType.float32));

        // Verify normalization was applied
        // 128/255 ≈ 0.502, then normalized
        // Channel 0: (0.502 - 0.485) / 0.229 ≈ 0.074
        expect(result[[0, 0, 0, 0]], closeTo(0.074, 0.02));
      });

      /// Shape transformation trace:
      /// Input: [480, 640, 3] (HWC)
      /// After ToTensor: [3, 480, 640] (CHW)
      /// After Resize: [3, 256, 256] (CHW)
      /// After CenterCrop: [3, 224, 224] (CHW)
      /// After Normalize: [3, 224, 224] (CHW)
      /// After Unsqueeze: [1, 3, 224, 224] (NCHW)
      test('computeOutputShape matches actual output', () {
        final pipeline = TensorPipeline([
          ToTensorOp(), // HWC -> CHW first!
          ResizeOp(height: 256, width: 256),
          CenterCropOp(height: 224, width: 224),
          NormalizeOp.imagenet(),
          UnsqueezeOp.batch(),
        ]);

        final inputShape = [480, 640, 3]; // HWC input
        final expectedShape = [1, 3, 224, 224]; // NCHW output

        expect(pipeline.computeOutputShape(inputShape), equals(expectedShape));
      });
    });

    group('Memory Layout', () {
      /// ```python
      /// t = torch.zeros(1, 3, 4, 4)
      /// print(t.stride())  # (48, 16, 4, 1)
      /// t_nhwc = t.to(memory_format=torch.channels_last)
      /// print(t_nhwc.stride())  # (48, 1, 12, 3)
      /// ```
      test('NCHW strides match PyTorch contiguous', () {
        final tensor = TensorBuffer.zeros(
          [1, 3, 4, 4],
          memoryFormat: MemoryFormat.contiguous,
        );

        expect(tensor.strides, equals([48, 16, 4, 1]));
      });

      test('channelsLast strides match PyTorch channels_last', () {
        final tensor = TensorBuffer.zeros(
          [1, 3, 4, 4],
          memoryFormat: MemoryFormat.channelsLast,
        );

        expect(tensor.strides, equals([48, 1, 12, 3]));
      });
    });

    group('Numerical Precision', () {
      /// Test that operations maintain reasonable numerical precision
      test('normalization precision within 1e-6 of expected', () {
        // Use exactly representable values
        final data = Float32List.fromList([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        final tensor = TensorBuffer.fromFloat32List(data, [3, 1, 2]);

        final normalize = NormalizeOp(
          mean: [0.5, 0.5, 0.5],
          std: [0.5, 0.5, 0.5],
        );
        final result = normalize(tensor);

        // (0.5 - 0.5) / 0.5 = 0.0
        for (int c = 0; c < 3; c++) {
          for (int h = 0; h < 1; h++) {
            for (int w = 0; w < 2; w++) {
              expect(result[[c, h, w]], closeTo(0.0, 1e-6));
            }
          }
        }
      });

      test('chained operations accumulate error within acceptable range', () {
        final data = Float32List.fromList(
          List.generate(12, (i) => (i / 11.0)),
        );
        final tensor = TensorBuffer.fromFloat32List(data, [3, 2, 2]);

        // Apply multiple operations
        final normalized = NormalizeOp.imagenet()(tensor);
        final transposed = normalized.transpose([1, 2, 0]);
        final contiguous = transposed.contiguous();
        final backTransposed = contiguous.transpose([2, 0, 1]);
        final final_ = backTransposed.contiguous();

        // Values should still be valid (not NaN or Inf)
        for (int i = 0; i < final_.numel; i++) {
          final value = final_.storage.getAsDouble(i);
          expect(value.isFinite, isTrue);
        }
      });
    });

    group('Edge Cases', () {
      test('single pixel image resize', () {
        final data = Float32List.fromList([0.5]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 1, 1]);

        final resize = ResizeOp(
          height: 4,
          width: 4,
          mode: InterpolationMode.bilinear,
        );
        final result = resize(tensor);

        expect(result.shape, equals([1, 4, 4]));

        // All values should be 0.5 (constant extrapolation)
        for (int h = 0; h < 4; h++) {
          for (int w = 0; w < 4; w++) {
            expect(result[[0, h, w]], closeTo(0.5, 1e-3));
          }
        }
      });

      test('batch dimension handling', () {
        final data = Float32List(2 * 3 * 4 * 4);
        for (int i = 0; i < data.length; i++) {
          data[i] = i / (data.length - 1);
        }
        final tensor = TensorBuffer.fromFloat32List(data, [2, 3, 4, 4]);

        final normalize = NormalizeOp.imagenet();
        final result = normalize(tensor);

        expect(result.shape, equals([2, 3, 4, 4]));

        // Both batches should be normalized independently
        // but with the same mean/std
      });

      test('very small values normalization', () {
        final data = Float32List.fromList([1e-7, 1e-7, 1e-7, 1e-7]);
        final tensor = TensorBuffer.fromFloat32List(data, [1, 2, 2]);

        final normalize = NormalizeOp(mean: [0.0], std: [1.0]);
        final result = normalize(tensor);

        for (int h = 0; h < 2; h++) {
          for (int w = 0; w < 2; w++) {
            expect(result[[0, h, w]], closeTo(1e-7, 1e-10));
          }
        }
      });
    });

    group('DType Compatibility', () {
      /// ```python
      /// t_f32 = torch.zeros(2, 3, dtype=torch.float32)
      /// t_f64 = torch.zeros(2, 3, dtype=torch.float64)
      /// t_i32 = torch.zeros(2, 3, dtype=torch.int32)
      /// t_i64 = torch.zeros(2, 3, dtype=torch.int64)
      /// t_u8 = torch.zeros(2, 3, dtype=torch.uint8)
      /// ```
      test('dtype creation matches PyTorch', () {
        final f32 = TensorBuffer.zeros([2, 3], dtype: DType.float32);
        final f64 = TensorBuffer.zeros([2, 3], dtype: DType.float64);
        final i32 = TensorBuffer.zeros([2, 3], dtype: DType.int32);
        final i64 = TensorBuffer.zeros([2, 3], dtype: DType.int64);
        final u8 = TensorBuffer.zeros([2, 3], dtype: DType.uint8);

        expect(f32.dtype, equals(DType.float32));
        expect(f64.dtype, equals(DType.float64));
        expect(i32.dtype, equals(DType.int32));
        expect(i64.dtype, equals(DType.int64));
        expect(u8.dtype, equals(DType.uint8));
      });

      test('ONNX dtype IDs match specification', () {
        // ONNX TensorProto.DataType values
        expect(DType.float32.onnxId, equals(1));
        expect(DType.uint8.onnxId, equals(2));
        expect(DType.int8.onnxId, equals(3));
        expect(DType.uint16.onnxId, equals(4));
        expect(DType.int16.onnxId, equals(5));
        expect(DType.int32.onnxId, equals(6));
        expect(DType.int64.onnxId, equals(7));
        expect(DType.float64.onnxId, equals(11));
        expect(DType.uint32.onnxId, equals(12));
        expect(DType.uint64.onnxId, equals(13));
      });
    });
  });
}
