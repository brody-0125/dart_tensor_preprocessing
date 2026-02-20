import '../service_config.dart';
import '../service_manager.dart';

/// Linux systemd service manager that generates unit files.
///
/// Generates systemd unit files for managing background services
/// on Linux systems. The generated files are compatible with both
/// system-level (`systemctl`) and user-level (`systemctl --user`)
/// service management.
///
/// The generated unit files follow the standard systemd format:
/// ```ini
/// [Unit]
/// Description=service description
///
/// [Service]
/// ExecStart=/path/to/executable --args
/// WorkingDirectory=/path/to/workdir
/// Restart=always
/// StandardOutput=append:/path/to/stdout.log
/// StandardError=append:/path/to/stderr.log
///
/// [Install]
/// WantedBy=default.target
/// ```
///
/// Example:
/// ```dart
/// final manager = SystemdServiceManager();
/// final config = ServiceConfig.proxy(
///   identifier: 'ccgateway-proxy',
///   executablePath: '/usr/local/bin/ccgateway',
///   configPath: '/etc/ccgateway/config.yaml',
///   workingDirectory: '/var/lib/ccgateway',
///   logDirectory: '/var/log/ccgateway',
/// );
/// final unitFile = manager.generateServiceFile(config);
/// // Write to ~/.config/systemd/user/ccgateway-proxy.service
/// // Then: systemctl --user enable --now ccgateway-proxy
/// ```
class SystemdServiceManager extends ServiceManager {
  /// The systemd target to install the service under.
  ///
  /// Defaults to `default.target` for user services.
  final String wantedBy;

  /// The restart policy for the service.
  ///
  /// Common values: 'always', 'on-failure', 'on-abnormal', 'no'.
  /// Defaults to 'always'.
  final String restartPolicy;

  /// The delay before restarting the service after a failure, in seconds.
  final int? restartSec;

  /// Creates a systemd service manager for Linux.
  ///
  /// [wantedBy] specifies the install target (default: 'default.target').
  /// [restartPolicy] controls restart behavior (default: 'always').
  /// [restartSec] optionally sets the restart delay in seconds.
  SystemdServiceManager({
    this.wantedBy = 'default.target',
    this.restartPolicy = 'always',
    this.restartSec,
  });

  @override
  String get platform => 'Linux';

  @override
  String get fileExtension => '.service';

  @override
  String generateServiceFile(ServiceConfig config) {
    final execStart = _buildExecStart(config);
    final restart = config.restartOnFailure ? restartPolicy : 'no';

    final buffer = StringBuffer()
      ..writeln('[Unit]')
      ..writeln('Description=${config.description}')
      ..writeln()
      ..writeln('[Service]')
      ..writeln('ExecStart=$execStart')
      ..writeln('WorkingDirectory=${config.workingDirectory}')
      ..writeln('Restart=$restart')
      ..writeln('StandardOutput=append:${config.standardOutPath}')
      ..writeln('StandardError=append:${config.standardErrorPath}');

    if (restartSec != null) {
      buffer.writeln('RestartSec=$restartSec');
    }

    if (config.environmentVariables.isNotEmpty) {
      for (final entry in config.environmentVariables.entries) {
        buffer.writeln('Environment="${entry.key}=${entry.value}"');
      }
    }

    buffer
      ..writeln()
      ..writeln('[Install]')
      ..writeln('WantedBy=$wantedBy');

    return buffer.toString();
  }

  /// Generates the output file name for the given [config].
  String outputFileName(ServiceConfig config) {
    return '${config.identifier}$fileExtension';
  }

  /// Builds the ExecStart line from the config.
  String _buildExecStart(ServiceConfig config) {
    if (config.arguments.isEmpty) {
      return config.executablePath;
    }
    final args = config.arguments.join(' ');
    return '${config.executablePath} $args';
  }
}
