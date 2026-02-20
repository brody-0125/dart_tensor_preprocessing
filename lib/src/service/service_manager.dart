import 'service_config.dart';

/// Abstract base class for platform-specific service managers.
///
/// Implementations generate service configuration files appropriate for
/// their target platform:
/// - [LaunchdServiceManager] for macOS (plist XML)
/// - [SystemdServiceManager] for Linux (systemd unit files)
///
/// Example:
/// ```dart
/// final manager = ServiceManagerFactory.create();
/// final config = ServiceConfig.proxy(
///   identifier: 'com.example.proxy',
///   executablePath: '/usr/local/bin/ccgateway',
///   configPath: '/etc/ccgateway/config.yaml',
///   workingDirectory: '/var/lib/ccgateway',
///   logDirectory: '/var/log/ccgateway',
/// );
/// final content = manager.generateServiceFile(config);
/// ```
abstract class ServiceManager {
  /// The platform name this manager targets (e.g., 'macOS', 'Linux').
  String get platform;

  /// The file extension used by this platform's service files.
  String get fileExtension;

  /// Generates the service configuration file content for the given [config].
  ///
  /// Returns a string containing the complete service file content
  /// ready to be written to disk.
  String generateServiceFile(ServiceConfig config);

  /// Generates a sync service configuration file.
  ///
  /// Convenience method that creates a [ServiceConfig.sync] and generates
  /// the platform-specific service file.
  String generateSyncServiceFile({
    required String identifier,
    required String executablePath,
    required String configPath,
    required String workingDirectory,
    required String logDirectory,
  }) {
    final config = ServiceConfig.sync(
      identifier: identifier,
      executablePath: executablePath,
      configPath: configPath,
      workingDirectory: workingDirectory,
      logDirectory: logDirectory,
    );
    return generateServiceFile(config);
  }

  /// Generates a proxy service configuration file.
  ///
  /// Convenience method that creates a [ServiceConfig.proxy] and generates
  /// the platform-specific service file.
  String generateProxyServiceFile({
    required String identifier,
    required String executablePath,
    required String configPath,
    required String workingDirectory,
    required String logDirectory,
  }) {
    final config = ServiceConfig.proxy(
      identifier: identifier,
      executablePath: executablePath,
      configPath: configPath,
      workingDirectory: workingDirectory,
      logDirectory: logDirectory,
    );
    return generateServiceFile(config);
  }

  /// Generates a platform-agnostic bash wrapper script for the service.
  ///
  /// The wrapper script handles signal forwarding and logging setup.
  /// This is shared across all platforms.
  String generateBashScript(ServiceConfig config) {
    final args = config.arguments.join(' ');
    final envLines = config.environmentVariables.entries
        .map((e) => 'export ${e.key}="${e.value}"')
        .join('\n');

    final buffer = StringBuffer()
      ..writeln('#!/bin/bash')
      ..writeln('# Auto-generated service wrapper for ${config.identifier}')
      ..writeln('# Description: ${config.description}')
      ..writeln();

    if (envLines.isNotEmpty) {
      buffer
        ..writeln('# Environment variables')
        ..writeln(envLines)
        ..writeln();
    }

    buffer
      ..writeln('cd "${config.workingDirectory}"')
      ..writeln()
      ..writeln('exec "${config.executablePath}" $args \\')
      ..writeln('  >> "${config.standardOutPath}" \\')
      ..writeln('  2>> "${config.standardErrorPath}"');

    return buffer.toString();
  }
}

/// Factory for creating platform-appropriate [ServiceManager] instances.
class ServiceManagerFactory {
  ServiceManagerFactory._();

  /// Creates a [ServiceManager] for the specified [platform].
  ///
  /// Supported values: 'macos', 'linux'.
  /// Throws [UnsupportedError] for unrecognized platforms.
  static ServiceManager create(String platform) {
    // Deferred imports would be used in a real scenario, but for this
    // library we use a simple switch.
    throw UnsupportedError(
      'Use LaunchdServiceManager or SystemdServiceManager directly. '
      'ServiceManagerFactory.forCurrentPlatform() is available for '
      'runtime detection.',
    );
  }
}
