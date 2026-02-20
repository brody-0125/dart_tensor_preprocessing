import 'package:dart_tensor_preprocessing/src/service/launchd/launchd_service_manager.dart';
import 'package:dart_tensor_preprocessing/src/service/service_config.dart';
import 'package:test/test.dart';

void main() {
  group('LaunchdServiceManager', () {
    late LaunchdServiceManager manager;

    setUp(() {
      manager = LaunchdServiceManager();
    });

    test('platform returns macOS', () {
      expect(manager.platform, equals('macOS'));
    });

    test('fileExtension returns .plist', () {
      expect(manager.fileExtension, equals('.plist'));
    });

    group('generateServiceFile', () {
      test('generates valid plist XML with required fields', () {
        const config = ServiceConfig(
          identifier: 'com.example.test',
          description: 'Test service',
          executablePath: '/usr/local/bin/myapp',
          arguments: ['--config', '/etc/myapp/config.yaml'],
          workingDirectory: '/var/lib/myapp',
          standardOutPath: '/var/log/myapp/out.log',
          standardErrorPath: '/var/log/myapp/err.log',
        );

        final result = manager.generateServiceFile(config);

        expect(result, contains('<?xml version="1.0" encoding="UTF-8"?>'));
        expect(result, contains('<!DOCTYPE plist'));
        expect(result, contains('<plist version="1.0">'));
        expect(result, contains('<key>Label</key>'));
        expect(result, contains('<string>com.example.test</string>'));
        expect(result, contains('<key>ProgramArguments</key>'));
        expect(result, contains('<string>/usr/local/bin/myapp</string>'));
        expect(result, contains('<string>--config</string>'));
        expect(
          result,
          contains('<string>/etc/myapp/config.yaml</string>'),
        );
        expect(result, contains('<key>WorkingDirectory</key>'));
        expect(result, contains('<string>/var/lib/myapp</string>'));
        expect(result, contains('<key>StandardOutPath</key>'));
        expect(result, contains('<string>/var/log/myapp/out.log</string>'));
        expect(result, contains('<key>StandardErrorPath</key>'));
        expect(result, contains('<string>/var/log/myapp/err.log</string>'));
        expect(result, contains('</dict>'));
        expect(result, contains('</plist>'));
      });

      test('includes KeepAlive when restartOnFailure is true', () {
        const config = ServiceConfig(
          identifier: 'com.example.test',
          description: 'Test service',
          executablePath: '/usr/local/bin/myapp',
          workingDirectory: '/var/lib/myapp',
          standardOutPath: '/var/log/myapp/out.log',
          standardErrorPath: '/var/log/myapp/err.log',
          restartOnFailure: true,
        );

        final result = manager.generateServiceFile(config);

        expect(result, contains('<key>KeepAlive</key>'));
        expect(result, contains('<true/>'));
      });

      test('omits KeepAlive when restartOnFailure is false', () {
        const config = ServiceConfig(
          identifier: 'com.example.test',
          description: 'Test service',
          executablePath: '/usr/local/bin/myapp',
          workingDirectory: '/var/lib/myapp',
          standardOutPath: '/var/log/myapp/out.log',
          standardErrorPath: '/var/log/myapp/err.log',
          restartOnFailure: false,
        );

        final result = manager.generateServiceFile(config);

        expect(result, isNot(contains('<key>KeepAlive</key>')));
      });

      test('includes environment variables when provided', () {
        const config = ServiceConfig(
          identifier: 'com.example.test',
          description: 'Test service',
          executablePath: '/usr/local/bin/myapp',
          workingDirectory: '/var/lib/myapp',
          standardOutPath: '/var/log/myapp/out.log',
          standardErrorPath: '/var/log/myapp/err.log',
          environmentVariables: {
            'APP_ENV': 'production',
            'LOG_LEVEL': 'info',
          },
        );

        final result = manager.generateServiceFile(config);

        expect(result, contains('<key>EnvironmentVariables</key>'));
        expect(result, contains('<key>APP_ENV</key>'));
        expect(result, contains('<string>production</string>'));
        expect(result, contains('<key>LOG_LEVEL</key>'));
        expect(result, contains('<string>info</string>'));
      });

      test('omits environment variables when empty', () {
        const config = ServiceConfig(
          identifier: 'com.example.test',
          description: 'Test service',
          executablePath: '/usr/local/bin/myapp',
          workingDirectory: '/var/lib/myapp',
          standardOutPath: '/var/log/myapp/out.log',
          standardErrorPath: '/var/log/myapp/err.log',
        );

        final result = manager.generateServiceFile(config);

        expect(result, isNot(contains('<key>EnvironmentVariables</key>')));
      });

      test('escapes XML special characters', () {
        const config = ServiceConfig(
          identifier: 'com.example.<test>&"app"',
          description: 'Test & service',
          executablePath: '/usr/local/bin/myapp',
          workingDirectory: '/var/lib/myapp',
          standardOutPath: '/var/log/myapp/out.log',
          standardErrorPath: '/var/log/myapp/err.log',
        );

        final result = manager.generateServiceFile(config);

        expect(
          result,
          contains('com.example.&lt;test&gt;&amp;&quot;app&quot;'),
        );
      });

      test('handles empty arguments list', () {
        const config = ServiceConfig(
          identifier: 'com.example.test',
          description: 'Test service',
          executablePath: '/usr/local/bin/myapp',
          workingDirectory: '/var/lib/myapp',
          standardOutPath: '/var/log/myapp/out.log',
          standardErrorPath: '/var/log/myapp/err.log',
        );

        final result = manager.generateServiceFile(config);

        expect(result, contains('<key>ProgramArguments</key>'));
        expect(result, contains('<string>/usr/local/bin/myapp</string>'));
      });
    });

    group('convenience methods', () {
      test('generateSyncServiceFile creates sync configuration', () {
        final result = manager.generateSyncServiceFile(
          identifier: 'com.example.sync',
          executablePath: '/usr/local/bin/ccgateway',
          configPath: '/etc/ccgateway/sync.yaml',
          workingDirectory: '/var/lib/ccgateway',
          logDirectory: '/var/log/ccgateway',
        );

        expect(result, contains('<string>com.example.sync</string>'));
        expect(
          result,
          contains('<string>/usr/local/bin/ccgateway</string>'),
        );
        expect(result, contains('<string>--config</string>'));
        expect(
          result,
          contains('<string>/etc/ccgateway/sync.yaml</string>'),
        );
      });

      test('generateProxyServiceFile creates proxy configuration', () {
        final result = manager.generateProxyServiceFile(
          identifier: 'com.example.proxy',
          executablePath: '/usr/local/bin/ccgateway',
          configPath: '/etc/ccgateway/proxy.yaml',
          workingDirectory: '/var/lib/ccgateway',
          logDirectory: '/var/log/ccgateway',
        );

        expect(result, contains('<string>com.example.proxy</string>'));
        expect(
          result,
          contains('<string>/usr/local/bin/ccgateway</string>'),
        );
        expect(result, contains('<string>--config</string>'));
        expect(
          result,
          contains('<string>/etc/ccgateway/proxy.yaml</string>'),
        );
      });
    });

    group('outputFileName', () {
      test('returns identifier with .plist extension', () {
        const config = ServiceConfig(
          identifier: 'com.example.test',
          description: 'Test',
          executablePath: '/usr/bin/test',
          workingDirectory: '/tmp',
          standardOutPath: '/tmp/out.log',
          standardErrorPath: '/tmp/err.log',
        );

        expect(
          manager.outputFileName(config),
          equals('com.example.test.plist'),
        );
      });
    });
  });
}
