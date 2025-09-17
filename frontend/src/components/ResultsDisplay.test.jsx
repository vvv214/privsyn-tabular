import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import ResultsDisplay from './ResultsDisplay';

const baseFormData = { dataset_name: 'adult' };

const renderComponent = (overrides = {}) => {
  const retrySpy = overrides.onRetryEvaluate || vi.fn();
  const props = {
    formData: baseFormData,
    downloadUrl: overrides.downloadUrl || '#',
    synthesizedDataPreview: overrides.synthesizedDataPreview || [],
    synthesizedDataHeaders: overrides.synthesizedDataHeaders || [],
    evaluationResults: overrides.evaluationResults || {},
    sessionId: overrides.sessionId || '1234567890abcdef',
    isEvaluating: overrides.isEvaluating || false,
    evaluationError: overrides.evaluationError || '',
    onRetryEvaluate: retrySpy,
  };
  render(<ResultsDisplay {...props} />);
  return { retrySpy };
};

describe('ResultsDisplay', () => {
  afterEach(() => cleanup());

  it('shows evaluation warning and retries when requested', () => {
    const { retrySpy } = renderComponent({ evaluationError: 'Something went wrong.' });

    expect(screen.getByText(/evaluation warning/i)).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: /retry evaluation/i }));
    expect(retrySpy).toHaveBeenCalledTimes(1);
  });

  it('indicates evaluation progress', () => {
    renderComponent({ isEvaluating: true, evaluationResults: {}, evaluationError: '' });
    expect(screen.getByText(/evaluating data fidelity/i)).toBeVisible();
  });
});
