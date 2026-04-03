/// Centralized error message formatting for tensor operations.
///
/// Provides consistent, informative error messages across the library.
/// All methods are pure string formatters with no dependencies.
class ErrorMessages {
  ErrorMessages._();

  /// Message for rank mismatch errors.
  static String rankMismatch(
    String operationName,
    int expectedRank,
    int actualRank,
  ) => '$operationName: expected ${expectedRank}D tensor, got ${actualRank}D';

  /// Message for rank range errors.
  static String rankRange(
    String operationName,
    int minRank,
    int maxRank,
    int actualRank,
  ) =>
      '$operationName: expected $minRank-${maxRank}D tensor, got ${actualRank}D';

  /// Message for channel count mismatch errors.
  static String channelMismatch(
    String operationName,
    int expectedChannels,
    int actualChannels,
  ) =>
      '$operationName: expected $expectedChannels channels, got $actualChannels';

  /// Message for shape mismatch errors.
  static String shapeMismatch(
    String operationName,
    List<int> expectedShape,
    List<int> actualShape,
  ) => '$operationName: expected shape $expectedShape, got $actualShape';

  /// Message for axis out of bounds errors.
  static String axisBounds(String operationName, int axis, int rank) =>
      '$operationName: axis $axis is out of bounds for ${rank}D tensor';

  /// Message for dtype mismatch errors.
  static String dtypeMismatch(
    String operationName,
    String expectedDType,
    String actualDType,
  ) => '$operationName: expected dtype $expectedDType, got $actualDType';

  /// Message for parameter out of range errors.
  static String parameterRange(
    String operationName,
    String paramName,
    num value,
    num min,
    num max,
  ) => '$operationName: $paramName $value is out of range [$min, $max]';

  /// Message for broadcast-incompatible shapes.
  static String broadcastIncompatible(
    String operationName,
    List<int> shape1,
    List<int> shape2,
  ) =>
      '$operationName: shapes $shape1 and $shape2 are not broadcast-compatible';
}
