/// Configuration for a managed service.
///
/// Contains all the information needed to generate platform-specific
/// service configuration files (macOS launchd plist or Linux systemd unit).
class ServiceConfig {
  /// A unique identifier for the service (e.g., 'com.example.sync').
  final String identifier;

  /// A human-readable description of the service.
  final String description;

  /// The full path to the executable to run.
  final String executablePath;

  /// Command-line arguments passed to the executable.
  final List<String> arguments;

  /// The working directory for the service process.
  final String workingDirectory;

  /// Path to the standard output log file.
  final String standardOutPath;

  /// Path to the standard error log file.
  final String standardErrorPath;

  /// Whether the service should automatically restart on failure.
  final bool restartOnFailure;

  /// Optional environment variables for the service.
  final Map<String, String> environmentVariables;

  /// Creates a service configuration.
  const ServiceConfig({
    required this.identifier,
    required this.description,
    required this.executablePath,
    this.arguments = const [],
    required this.workingDirectory,
    required this.standardOutPath,
    required this.standardErrorPath,
    this.restartOnFailure = true,
    this.environmentVariables = const {},
  });

  /// Creates a sync service configuration.
  ///
  /// Convenience factory for creating a sync-type service with the
  /// given [identifier], [executablePath], and [configPath].
  factory ServiceConfig.sync({
    required String identifier,
    required String executablePath,
    required String configPath,
    required String workingDirectory,
    required String logDirectory,
  }) {
    return ServiceConfig(
      identifier: identifier,
      description: 'ccgateway sync - $identifier',
      executablePath: executablePath,
      arguments: ['--config', configPath],
      workingDirectory: workingDirectory,
      standardOutPath: '$logDirectory/$identifier.out.log',
      standardErrorPath: '$logDirectory/$identifier.err.log',
    );
  }

  /// Creates a proxy service configuration.
  ///
  /// Convenience factory for creating a proxy-type service with the
  /// given [identifier], [executablePath], and [configPath].
  factory ServiceConfig.proxy({
    required String identifier,
    required String executablePath,
    required String configPath,
    required String workingDirectory,
    required String logDirectory,
  }) {
    return ServiceConfig(
      identifier: identifier,
      description: 'ccgateway proxy - $identifier',
      executablePath: executablePath,
      arguments: ['--config', configPath],
      workingDirectory: workingDirectory,
      standardOutPath: '$logDirectory/$identifier.out.log',
      standardErrorPath: '$logDirectory/$identifier.err.log',
    );
  }
}
