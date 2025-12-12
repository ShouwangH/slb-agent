/**
 * Loading spinner component for async operations.
 */

interface LoadingSpinnerProps {
  message?: string;
}

export function LoadingSpinner({ message = "Loading..." }: LoadingSpinnerProps) {
  return (
    <div className="loading-spinner">
      <div className="spinner" aria-hidden="true" />
      <span>{message}</span>
    </div>
  );
}
