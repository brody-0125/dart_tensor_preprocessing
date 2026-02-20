import 'package:dart_tensor_preprocessing/src/service/service_config.dart';
import 'package:dart_tensor_preprocessing/src/service/service_manager.dart';
import 'package:test/test.dart';

void main() {
  group('ServiceConfig', () {
    test('creates configuration with required fields', () {
      const config = ServiceConfig(
        identifier: 'com.example.test',
        description: 'Test service',
        executablePath: '/usr/bin/test',
        workingDirectory: '/tmp',
        standardOutPath: '/tmp/out.log',
        standardErrorPath: '/tmp/err.log',
      );

      expect(config.identifier, equals('com.example.test'));
      expect(config.description, equals('Test service'));
      expect(config.executablePath, equals('/usr/bin/test'));
      expect(config.arguments, isEmpty);
      expect(config.workingDirectory, equals('/tmp'));
      expect(config.standardOutPath, equals('/tmp/out.log'));
      expect(config.standardErrorPath, equals('/tmp/err.log'));
      expect(config.restartOnFailure, isTrue);
      expect(config.environmentVariables, isEmpty);
    });

    test('creates configuration with all fields', () {
      const config = ServiceConfig(
        identifier: 'com.example.test',
        description: 'Test service',
        executablePath: '/usr/bin/test',
        arguments: ['--flag', 'value'],
        workingDirectory: '/tmp',
        standardOutPath: '/tmp/out.log',
        standardErrorPath: '/tmp/err.log',
        restartOnFailure: false,
        environmentVariables: {'KEY': 'VALUE'},
      );

      expect(config.arguments, equals(['--flag', 'value']));
      expect(config.restartOnFailure, isFalse);
      expect(config.environmentVariables, equals({'KEY': 'VALUE'}));
    });

    group('sync factory', () {
      test('creates sync service configuration', () {
        final config = ServiceConfig.sync(
          identifier: 'com.example.sync',
          executablePath: '/usr/bin/app',
          configPath: '/etc/app/config.yaml',
          workingDirectory: '/var/lib/app',
          logDirectory: '/var/log/app',
        );

        expect(config.identifier, equals('com.example.sync'));
        expect(config.description, contains('sync'));
        expect(config.executablePath, equals('/usr/bin/app'));
        expect(
          config.arguments,
          equals(['--config', '/etc/app/config.yaml']),
        );
        expect(config.workingDirectory, equals('/var/lib/app'));
        expect(
          config.standardOutPath,
          equals('/var/log/app/com.example.sync.out.log'),
        );
        expect(
          config.standardErrorPath,
          equals('/var/log/app/com.example.sync.err.log'),
        );
      });
    });

    group('proxy factory', () {
      test('creates proxy service configuration', () {
        final config = ServiceConfig.proxy(
          identifier: 'com.example.proxy',
          executablePath: '/usr/bin/app',
          configPath: '/etc/app/proxy.yaml',
          workingDirectory: '/var/lib/app',
          logDirectory: '/var/log/app',
        );

        expect(config.identifier, equals('com.example.proxy'));
        expect(config.description, contains('proxy'));
        expect(config.executablePath, equals('/usr/bin/app'));
        expect(
          config.arguments,
          equals(['--config', '/etc/app/proxy.yaml']),
        );
        expect(config.workingDirectory, equals('/var/lib/app'));
        expect(
          config.standardOutPath,
          equals('/var/log/app/com.example.proxy.out.log'),
        );
        expect(
          config.standardErrorPath,
          equals('/var/log/app/com.example.proxy.err.log'),
        );
      });
    });
  });

  group('ServiceManager', () {
    test('generateBashScript produces valid shell script', () {
      final manager = _TestServiceManager();
      const config = ServiceConfig(
        identifier: 'com.example.test',
        description: 'Test service',
        executablePath: '/usr/bin/test',
        arguments: ['--config', '/etc/config.yaml'],
        workingDirectory: '/var/lib/app',
        standardOutPath: '/var/log/app/out.log',
        standardErrorPath: '/var/log/app/err.log',
        environmentVariables: {'APP_ENV': 'production'},
      );

      final result = manager.generateBashScript(config);

      expect(result, startsWith('#!/bin/bash'));
      expect(result, contains('com.example.test'));
      expect(result, contains('export APP_ENV="production"'));
      expect(result, contains('cd "/var/lib/app"'));
      expect(result, contains('exec "/usr/bin/test"'));
      expect(result, contains('--config /etc/config.yaml'));
      expect(result, contains('>> "/var/log/app/out.log"'));
      expect(result, contains('2>> "/var/log/app/err.log"'));
    });

    test('generateBashScript omits env section when no vars', () {
      final manager = _TestServiceManager();
      const config = ServiceConfig(
        identifier: 'com.example.test',
        description: 'Test service',
        executablePath: '/usr/bin/test',
        workingDirectory: '/var/lib/app',
        standardOutPath: '/var/log/app/out.log',
        standardErrorPath: '/var/log/app/err.log',
      );

      final result = manager.generateBashScript(config);

      expect(result, isNot(contains('export')));
    });
  });
}

/// Test implementation of ServiceManager for testing base class methods.
class _TestServiceManager extends ServiceManager {
  @override
  String get platform => 'Test';

  @override
  String get fileExtension => '.test';

  @override
  String generateServiceFile(ServiceConfig config) => '';
}
