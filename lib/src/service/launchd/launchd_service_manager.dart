import '../service_config.dart';
import '../service_manager.dart';

/// macOS launchd service manager that generates plist XML files.
///
/// Generates Apple property list (plist) XML files compatible with
/// `launchctl` for managing background services on macOS.
///
/// The generated plist files follow the standard launchd format:
/// - `Label`: unique service identifier
/// - `ProgramArguments`: executable path and arguments
/// - `WorkingDirectory`: service working directory
/// - `StandardOutPath` / `StandardErrorPath`: log file paths
/// - `KeepAlive`: restart behavior
///
/// Example:
/// ```dart
/// final manager = LaunchdServiceManager();
/// final config = ServiceConfig.proxy(
///   identifier: 'com.example.proxy',
///   executablePath: '/usr/local/bin/ccgateway',
///   configPath: '/etc/ccgateway/config.yaml',
///   workingDirectory: '/var/lib/ccgateway',
///   logDirectory: '/var/log/ccgateway',
/// );
/// final plistXml = manager.generateServiceFile(config);
/// // Write plistXml to ~/Library/LaunchAgents/com.example.proxy.plist
/// ```
class LaunchdServiceManager extends ServiceManager {
  /// Creates a launchd service manager for macOS.
  LaunchdServiceManager();

  @override
  String get platform => 'macOS';

  @override
  String get fileExtension => '.plist';

  @override
  String generateServiceFile(ServiceConfig config) {
    final buffer = StringBuffer()
      ..writeln('<?xml version="1.0" encoding="UTF-8"?>')
      ..writeln(
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
      )
      ..writeln('<plist version="1.0">')
      ..writeln('<dict>')
      ..writeln('  <key>Label</key>')
      ..writeln('  <string>${_escapeXml(config.identifier)}</string>')
      ..writeln('  <key>ProgramArguments</key>')
      ..writeln('  <array>');

    buffer.writeln(
      '    <string>${_escapeXml(config.executablePath)}</string>',
    );
    for (final arg in config.arguments) {
      buffer.writeln('    <string>${_escapeXml(arg)}</string>');
    }

    buffer
      ..writeln('  </array>')
      ..writeln('  <key>WorkingDirectory</key>')
      ..writeln(
        '  <string>${_escapeXml(config.workingDirectory)}</string>',
      )
      ..writeln('  <key>StandardOutPath</key>')
      ..writeln(
        '  <string>${_escapeXml(config.standardOutPath)}</string>',
      )
      ..writeln('  <key>StandardErrorPath</key>')
      ..writeln(
        '  <string>${_escapeXml(config.standardErrorPath)}</string>',
      );

    if (config.restartOnFailure) {
      buffer
        ..writeln('  <key>KeepAlive</key>')
        ..writeln('  <true/>');
    }

    if (config.environmentVariables.isNotEmpty) {
      buffer
        ..writeln('  <key>EnvironmentVariables</key>')
        ..writeln('  <dict>');
      for (final entry in config.environmentVariables.entries) {
        buffer
          ..writeln('    <key>${_escapeXml(entry.key)}</key>')
          ..writeln('    <string>${_escapeXml(entry.value)}</string>');
      }
      buffer.writeln('  </dict>');
    }

    buffer
      ..writeln('</dict>')
      ..writeln('</plist>');

    return buffer.toString();
  }

  /// Generates the output file name for the given [config].
  String outputFileName(ServiceConfig config) {
    return '${config.identifier}$fileExtension';
  }

  /// Escapes special XML characters in [value].
  static String _escapeXml(String value) {
    return value
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&apos;');
  }
}
