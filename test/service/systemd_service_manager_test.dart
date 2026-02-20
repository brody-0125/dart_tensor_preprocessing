import 'package:dart_tensor_preprocessing/src/service/systemd/systemd_service_manager.dart';
import 'package:dart_tensor_preprocessing/src/service/service_config.dart';
import 'package:test/test.dart';

void main() {
  group('SystemdServiceManager', () {
    late SystemdServiceManager manager;

    setUp(() {
      manager = SystemdServiceManager();
    });

    test('platform returns Linux', () {
      expect(manager.platform, equals('Linux'));
    });

    test('fileExtension returns .service', () {
      expect(manager.fileExtension, equals('.service'));
    });

    group('generateServiceFile', () {
      test('generates valid systemd unit file with required sections', () {
        const config = ServiceConfig(
          identifier: 'ccgateway-proxy',
          description: 'ccgateway proxy - test',
          executablePath: '/usr/local/bin/ccgateway',
          arguments: ['--config', '/etc/ccgateway/config.yaml'],
          workingDirectory: '/var/lib/ccgateway',
          standardOutPath: '/var/log/ccgateway/out.log',
          standardErrorPath: '/var/log/ccgateway/err.log',
        );

        final result = manager.generateServiceFile(config);

        expect(result, contains('[Unit]'));
        expect(result, contains('Description=ccgateway proxy - test'));
        expect(result, contains('[Service]'));
        expect(
          result,
          contains(
            'ExecStart=/usr/local/bin/ccgateway '
            '--config /etc/ccgateway/config.yaml',
          ),
        );
        expect(
          result,
          contains('WorkingDirectory=/var/lib/ccgateway'),
        );
        expect(result, contains('Restart=always'));
        expect(
          result,
          contains('StandardOutput=append:/var/log/ccgateway/out.log'),
        );
        expect(
          result,
          contains('StandardError=append:/var/log/ccgateway/err.log'),
        );
        expect(result, contains('[Install]'));
        expect(result, contains('WantedBy=default.target'));
      });

      test('matches expected systemd template structure from issue', () {
        const config = ServiceConfig(
          identifier: 'ccgateway-proxy',
          description: 'ccgateway proxy - myservice',
          executablePath: '/usr/local/bin/ccgateway',
          arguments: ['--config', '/etc/ccgateway/config.yaml'],
          workingDirectory: '/var/lib/ccgateway',
          standardOutPath: '/var/log/ccgateway/out.log',
          standardErrorPath: '/var/log/ccgateway/err.log',
        );

        final result = manager.generateServiceFile(config);
        final lines = result.split('\n');

        // Verify section ordering: [Unit] -> [Service] -> [Install]
        final unitIndex = lines.indexWhere((l) => l.trim() == '[Unit]');
        final serviceIndex =
            lines.indexWhere((l) => l.trim() == '[Service]');
        final installIndex =
            lines.indexWhere((l) => l.trim() == '[Install]');

        expect(unitIndex, lessThan(serviceIndex));
        expect(serviceIndex, lessThan(installIndex));
      });

      test('uses Restart=always when restartOnFailure is true', () {
        const config = ServiceConfig(
          identifier: 'test-service',
          description: 'Test',
          executablePath: '/usr/bin/test',
          workingDirectory: '/tmp',
          standardOutPath: '/tmp/out.log',
          standardErrorPath: '/tmp/err.log',
          restartOnFailure: true,
        );

        final result = manager.generateServiceFile(config);

        expect(result, contains('Restart=always'));
      });

      test('uses Restart=no when restartOnFailure is false', () {
        const config = ServiceConfig(
          identifier: 'test-service',
          description: 'Test',
          executablePath: '/usr/bin/test',
          workingDirectory: '/tmp',
          standardOutPath: '/tmp/out.log',
          standardErrorPath: '/tmp/err.log',
          restartOnFailure: false,
        );

        final result = manager.generateServiceFile(config);

        expect(result, contains('Restart=no'));
      });

      test('includes environment variables when provided', () {
        const config = ServiceConfig(
          identifier: 'test-service',
          description: 'Test',
          executablePath: '/usr/bin/test',
          workingDirectory: '/tmp',
          standardOutPath: '/tmp/out.log',
          standardErrorPath: '/tmp/err.log',
          environmentVariables: {
            'APP_ENV': 'production',
            'LOG_LEVEL': 'info',
          },
        );

        final result = manager.generateServiceFile(config);

        expect(result, contains('Environment="APP_ENV=production"'));
        expect(result, contains('Environment="LOG_LEVEL=info"'));
      });

      test('omits environment variables when empty', () {
        const config = ServiceConfig(
          identifier: 'test-service',
          description: 'Test',
          executablePath: '/usr/bin/test',
          workingDirectory: '/tmp',
          standardOutPath: '/tmp/out.log',
          standardErrorPath: '/tmp/err.log',
        );

        final result = manager.generateServiceFile(config);

        expect(result, isNot(contains('Environment=')));
      });

      test('handles executable with no arguments', () {
        const config = ServiceConfig(
          identifier: 'test-service',
          description: 'Test',
          executablePath: '/usr/bin/test',
          workingDirectory: '/tmp',
          standardOutPath: '/tmp/out.log',
          standardErrorPath: '/tmp/err.log',
        );

        final result = manager.generateServiceFile(config);

        expect(result, contains('ExecStart=/usr/bin/test'));
        // Should not have trailing space
        final execLine = result
            .split('\n')
            .firstWhere((l) => l.startsWith('ExecStart='));
        expect(execLine, equals('ExecStart=/usr/bin/test'));
      });

      test('includes RestartSec when configured', () {
        final customManager = SystemdServiceManager(restartSec: 5);
        const config = ServiceConfig(
          identifier: 'test-service',
          description: 'Test',
          executablePath: '/usr/bin/test',
          workingDirectory: '/tmp',
          standardOutPath: '/tmp/out.log',
          standardErrorPath: '/tmp/err.log',
        );

        final result = customManager.generateServiceFile(config);

        expect(result, contains('RestartSec=5'));
      });

      test('omits RestartSec when not configured', () {
        const config = ServiceConfig(
          identifier: 'test-service',
          description: 'Test',
          executablePath: '/usr/bin/test',
          workingDirectory: '/tmp',
          standardOutPath: '/tmp/out.log',
          standardErrorPath: '/tmp/err.log',
        );

        final result = manager.generateServiceFile(config);

        expect(result, isNot(contains('RestartSec=')));
      });
    });

    group('custom configuration', () {
      test('uses custom wantedBy target', () {
        final customManager =
            SystemdServiceManager(wantedBy: 'multi-user.target');
        const config = ServiceConfig(
          identifier: 'test-service',
          description: 'Test',
          executablePath: '/usr/bin/test',
          workingDirectory: '/tmp',
          standardOutPath: '/tmp/out.log',
          standardErrorPath: '/tmp/err.log',
        );

        final result = customManager.generateServiceFile(config);

        expect(result, contains('WantedBy=multi-user.target'));
      });

      test('uses custom restart policy', () {
        final customManager =
            SystemdServiceManager(restartPolicy: 'on-failure');
        const config = ServiceConfig(
          identifier: 'test-service',
          description: 'Test',
          executablePath: '/usr/bin/test',
          workingDirectory: '/tmp',
          standardOutPath: '/tmp/out.log',
          standardErrorPath: '/tmp/err.log',
        );

        final result = customManager.generateServiceFile(config);

        expect(result, contains('Restart=on-failure'));
      });
    });

    group('convenience methods', () {
      test('generateSyncServiceFile creates sync configuration', () {
        final result = manager.generateSyncServiceFile(
          identifier: 'ccgateway-sync',
          executablePath: '/usr/local/bin/ccgateway',
          configPath: '/etc/ccgateway/sync.yaml',
          workingDirectory: '/var/lib/ccgateway',
          logDirectory: '/var/log/ccgateway',
        );

        expect(result, contains('Description=ccgateway sync'));
        expect(
          result,
          contains(
            'ExecStart=/usr/local/bin/ccgateway '
            '--config /etc/ccgateway/sync.yaml',
          ),
        );
        expect(
          result,
          contains('WorkingDirectory=/var/lib/ccgateway'),
        );
      });

      test('generateProxyServiceFile creates proxy configuration', () {
        final result = manager.generateProxyServiceFile(
          identifier: 'ccgateway-proxy',
          executablePath: '/usr/local/bin/ccgateway',
          configPath: '/etc/ccgateway/proxy.yaml',
          workingDirectory: '/var/lib/ccgateway',
          logDirectory: '/var/log/ccgateway',
        );

        expect(result, contains('Description=ccgateway proxy'));
        expect(
          result,
          contains(
            'ExecStart=/usr/local/bin/ccgateway '
            '--config /etc/ccgateway/proxy.yaml',
          ),
        );
      });
    });

    group('outputFileName', () {
      test('returns identifier with .service extension', () {
        const config = ServiceConfig(
          identifier: 'ccgateway-proxy',
          description: 'Test',
          executablePath: '/usr/bin/test',
          workingDirectory: '/tmp',
          standardOutPath: '/tmp/out.log',
          standardErrorPath: '/tmp/err.log',
        );

        expect(
          manager.outputFileName(config),
          equals('ccgateway-proxy.service'),
        );
      });
    });
  });
}
