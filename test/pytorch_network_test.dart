import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:dart_tensor_preprocessing/dart_tensor_preprocessing.dart';
import 'package:image/image.dart' as img;
import 'package:test/test.dart';

import 'pytorch_fixture_support.dart';

const directory = 'test/fixtures/pytorch';

void main() {
  final sources =
      jsonDecode(File('$directory/network-manifest.json').readAsStringSync())
          as List;
  final goldens =
      jsonDecode(File('$directory/network-goldens.json').readAsStringSync())
          as List;
  final enabled = Platform.environment['RUN_PYTORCH_NETWORK_TESTS'] == '1';
  for (final source in sources) {
    group(
      source['name'] as String,
      () {
        late TensorBuffer input;
        setUpAll(() async {
          // Always download: a cached file must not turn a network gate into an offline test.
          final client = HttpClient()
            ..connectionTimeout = const Duration(seconds: 30);
          try {
            final request = await client
                .getUrl(Uri.parse(source['url'] as String))
                .timeout(const Duration(seconds: 30));
            final response = await request.close().timeout(
              const Duration(seconds: 60),
            );
            expect(response.statusCode, HttpStatus.ok);
            final bytes = await response
                .fold<List<int>>([], (all, part) => all..addAll(part))
                .timeout(const Duration(seconds: 60));
            expect(
              sha256.convert(bytes).toString(),
              source['sha256'],
              reason: 'remote PNG',
            );
            final decoded = img.decodePng(Uint8List.fromList(bytes))!;
            final shape = (source['shape'] as List).cast<int>();
            expect([decoded.height, decoded.width, decoded.numChannels], shape);
            final pixels = decoded.getBytes();
            expect(
              sha256.convert(pixels).toString(),
              source['decoded_sha256'],
              reason: 'decoder parity before resize',
            );
            input = TensorBuffer.fromUint8List(pixels, shape);
          } finally {
            client.close(force: true);
          }
        });
        if (source['expected_error'] != null) {
          test('rejects non-RGB input explicitly', () {
            expect(
              () => PipelinePresets.imagenetClassification().run(input),
              throwsA(
                isA<ShapeMismatchException>().having(
                  (e) => e.message,
                  'message',
                  contains(source['expected_error']),
                ),
              ),
            );
          });
        }
        for (final golden in goldens.where(
          (g) => g['source'] == source['name'],
        )) {
          test('${golden['recipe']}', () async {
            final bytes = Uint8List.fromList(
              gzip.decode(
                File('$directory/${golden['file']}').readAsBytesSync(),
              ),
            );
            expect(
              sha256.convert(bytes).toString(),
              golden['sha256'],
              reason: 'golden payload',
            );
            final buffer = ByteData.sublistView(bytes);
            final values = List<double>.generate(
              bytes.length ~/ 4,
              (i) => buffer.getFloat32(i * 4, Endian.little),
            );
            final expected = <String, dynamic>{
              'shape': golden['shape'],
              'dtype': golden['dtype'],
              'values': values,
            };
            final pipeline = fixturePipeline(
              golden['recipe'] as Map<String, dynamic>,
            );
            expect(pipeline.computeOutputShape(input.shape), golden['shape']);
            for (final result in [
              pipeline.run(input),
              await pipeline.runAsync(input, isolateThreshold: 0),
            ]) {
              expectFixture(
                result,
                expected,
                atol: fixtureNumber(golden['atol']),
                rtol: fixtureNumber(golden['rtol']),
              );
            }
          });
        }
      },
      skip: enabled
          ? false
          : 'Set RUN_PYTORCH_NETWORK_TESTS=1 to verify pinned remote images',
    );
  }
}
