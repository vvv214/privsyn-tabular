import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import ResultsDisplay from './ResultsDisplay';

const baseFormData = { dataset_name: 'adult' };

const renderComponent = (overrides = {}) => {
  const props = {
    formData: baseFormData,
    downloadUrl: overrides.downloadUrl || '#',
    synthesizedDataPreview: overrides.synthesizedDataPreview || [],
    synthesizedDataHeaders: overrides.synthesizedDataHeaders || [],
    evaluationResults: overrides.evaluationResults || {},
    isEvaluating: overrides.isEvaluating || false,
  };
  render(<ResultsDisplay {...props} />);
};

describe('ResultsDisplay', () => {
  afterEach(() => cleanup());

  it('renders a download button for the synthesized dataset', () => {
    renderComponent();
    expect(screen.getByRole('link', { name: /download synthesized data/i })).toBeVisible();
  });

  it('indicates evaluation progress', () => {
    renderComponent({ isEvaluating: true, evaluationResults: {} });
    expect(screen.getByText(/evaluating data fidelity/i)).toBeVisible();
  });
});
